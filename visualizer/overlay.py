"""Visual grounding: draw evidence on top of the source image with OpenCV."""

from __future__ import annotations

import math

import cv2
import numpy as np

from core.registry import (
    AnatomyPayload,
    AnomalyType,
    EvidenceProtocol,
    GeometryPayload,
    LightingPayload,
    Point2D,
    Severity,
)

RED = (0, 0, 255)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
_COORD_LIMIT = 1e6


def render_overlay(image: np.ndarray, protocol: EvidenceProtocol) -> np.ndarray:
    canvas = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for evidence in protocol.evidences:
        if evidence.geometry is not None:
            draw_geometry_evidence(canvas, evidence.geometry)
        elif evidence.anatomy is not None:
            draw_anatomy_evidence(canvas, evidence.anatomy)
        elif evidence.lighting is not None:
            draw_lighting_evidence(canvas, evidence.lighting)
    _draw_banner(canvas, protocol)
    return canvas


def draw_geometry_evidence(canvas: np.ndarray, payload: GeometryPayload) -> None:
    """Extend each involved line up to its erroneous crossing point (thin red lines)."""
    for crossing in payload.crossings:
        point = crossing.crossing
        for segment in (crossing.segment_a, crossing.segment_b):
            _draw_line(canvas, segment.p1, segment.p2, RED, 1)
            near = min((segment.p1, segment.p2), key=lambda p: _distance(p, point))
            _draw_line(canvas, near, point, RED, 1)
        _draw_marker(canvas, point, RED, cv2.MARKER_CROSS, 10)


def draw_anatomy_evidence(canvas: np.ndarray, payload: AnatomyPayload) -> None:
    """Red boxes around deformed joints with short English labels."""
    for joint in payload.flagged_joints:
        x1, y1, x2, y2 = (int(round(v)) for v in joint.bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), RED, 2)
        label = f"{joint.joint_name} [{joint.reason}]"
        cv2.putText(canvas, label, (x1, max(y1 - 6, 12)), FONT, 0.45, RED, 1, cv2.LINE_AA)


def draw_lighting_evidence(canvas: np.ndarray, payload: LightingPayload) -> None:
    """Yellow rays from each shadow tip through its vertex toward the estimated light."""
    far_length = 2.0 * payload.image_diagonal
    for ray in payload.rays:
        shadow = np.asarray(ray.shadow_tip, dtype=np.float64)
        vertex = np.asarray(ray.top_vertex, dtype=np.float64)
        if payload.estimated_light is not None:
            target = payload.estimated_light
        else:
            direction = vertex - shadow
            direction /= max(float(np.linalg.norm(direction)), 1e-9)
            target = tuple(shadow + direction * far_length)
        _draw_line(canvas, ray.shadow_tip, target, YELLOW, 2)
        _draw_marker(canvas, ray.top_vertex, YELLOW, cv2.MARKER_DIAMOND, 10)
    for point in payload.pairwise_intersections:
        _draw_marker(canvas, point, YELLOW, cv2.MARKER_TILTED_CROSS, 8)
    if payload.estimated_light is not None:
        _draw_marker(canvas, payload.estimated_light, YELLOW, cv2.MARKER_CROSS, 18)


def _distance(a: Point2D, b: Point2D) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _draw_line(canvas: np.ndarray, p1: Point2D, p2: Point2D, color: tuple[int, int, int], thickness: int) -> None:
    """Clip to the image first, so far-away endpoints never overflow OpenCV integer arithmetic."""
    height, width = canvas.shape[:2]
    a = np.clip(np.asarray(p1, dtype=np.float64), -_COORD_LIMIT, _COORD_LIMIT)
    b = np.clip(np.asarray(p2, dtype=np.float64), -_COORD_LIMIT, _COORD_LIMIT)
    pa = (int(round(a[0])), int(round(a[1])))
    pb = (int(round(b[0])), int(round(b[1])))
    accepted, ca, cb = cv2.clipLine((0, 0, width, height), pa, pb)
    if accepted:
        cv2.line(canvas, ca, cb, color, thickness, cv2.LINE_AA)


def _draw_marker(
    canvas: np.ndarray, point: Point2D, color: tuple[int, int, int], marker: int, size: int
) -> None:
    height, width = canvas.shape[:2]
    x, y = point
    if 0 <= x < width and 0 <= y < height:
        cv2.drawMarker(canvas, (int(round(x)), int(round(y))), color, marker, size, 1)


def _draw_banner(canvas: np.ndarray, protocol: EvidenceProtocol) -> None:
    critical = sum(1 for e in protocol.evidences if e.severity is Severity.CRITICAL)
    warning = sum(1 for e in protocol.evidences if e.severity is Severity.WARNING)
    text = (
        f"Gavel AI | {protocol.image.file_name} | evidences: {len(protocol.evidences)} "
        f"| CRITICAL: {critical} | WARNING: {warning}"
    )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 26), BLACK, -1)
    cv2.putText(canvas, text, (8, 18), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
