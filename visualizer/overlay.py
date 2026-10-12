"""Visual grounding: draw evidence on top of the source image with OpenCV.

Legend used on every overlay:
    cyan                  anatomy map: face contours (not a verdict)
    green                 anatomy map: pose skeleton (not a verdict)
    amber thin            anatomy map: hand skeleton (not a verdict)
    magenta               anatomy anomaly (rule violation → layer evidence)
    red with white halo   geometry findings
    yellow                lighting findings
    top bar               Final decision + L0, then symbolic finding counts
"""

from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np

from core.registry import (
    AnatomyMap,
    AnatomyPayload,
    EvidenceProtocol,
    GeometryPayload,
    LightingPayload,
    Point2D,
    Severity,
)

RED = (0, 0, 255)
YELLOW = (0, 255, 255)
AMBER = (0, 165, 255)
# Magenta: anatomy *anomaly* marks (distinct from amber hand map).
MAGENTA = (255, 0, 255)
CYAN = (255, 255, 0)
GREEN = (0, 220, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
_COORD_LIMIT = 1e6
MAX_GEOMETRY_CROSSINGS = 4
MAX_CROSSING_LABELS = 2
MAX_CLUSTER_SEGMENTS = 24
BANNER_HEIGHT = 48

# MediaPipe Pose and Hands connection graphs (stable across 0.10.x).
POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
)
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)


@lru_cache(maxsize=1)
def _face_contour_edges() -> tuple[tuple[int, int], ...]:
    try:
        import mediapipe as mp

        return tuple(sorted((min(a, b), max(a, b)) for a, b in mp.solutions.face_mesh.FACEMESH_CONTOURS))
    except Exception:
        return ()


def render_overlay(image: np.ndarray, protocol: EvidenceProtocol) -> np.ndarray:
    canvas = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if protocol.anatomy_map is not None:
        draw_anatomy_map(canvas, protocol.anatomy_map)
    draw_l0_probes(canvas, protocol)
    draw_halves_seam(canvas, protocol)
    for evidence in protocol.evidences:
        if evidence.geometry is not None:
            draw_geometry_evidence(canvas, evidence.geometry, evidence.evidence_id)
        elif evidence.anatomy is not None:
            draw_anatomy_evidence(canvas, evidence.anatomy)
        elif evidence.lighting is not None:
            draw_lighting_evidence(canvas, evidence.lighting)
        elif evidence.artifacts is not None:
            draw_artifacts_badge(canvas, evidence.artifacts.score)
    _draw_banner(canvas, protocol)
    return canvas


def draw_l0_probes(canvas: np.ndarray, protocol: EvidenceProtocol) -> None:
    """Outline L0 tile/hand crops with scores. White=tiles, blue=hand probes, thick=winner."""
    detector = protocol.detector
    if detector is None or not detector.view_boxes:
        return
    h, w = canvas.shape[:2]
    for name, box in detector.view_boxes.items():
        if name == "full" and len(detector.view_boxes) > 1:
            continue  # full frame outline is noise when tiles/hands exist
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        score = detector.view_scores.get(name)
        is_hand = name.startswith("hand_")
        is_win = name == detector.view
        color = (255, 0, 0) if is_hand else WHITE  # BGR blue = hand L0 probe
        thickness = 3 if is_win else 1
        cv2.rectangle(canvas, (x1, y1), (x2, y2), BLACK, thickness + 2, cv2.LINE_AA)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        label = name if score is None else f"{name} L0={score:.2f}"
        _draw_boxed_label(canvas, label, x1 + 4, min(y2 - 8, h - 8), color)


def draw_halves_seam(canvas: np.ndarray, protocol: EvidenceProtocol) -> None:
    """Vertical midline when left/right appearance disagree (diptych cue)."""
    metrics = protocol.layers.global_layer.metrics or {}
    if float(metrics.get("halves_disagree", 0.0)) < 1.0:
        return
    mid = int(round(float(metrics.get("half_mid_x", canvas.shape[1] // 2))))
    h = canvas.shape[0]
    cv2.line(canvas, (mid, BANNER_HEIGHT), (mid, h - 1), BLACK, 3, cv2.LINE_AA)
    cv2.line(canvas, (mid, BANNER_HEIGHT), (mid, h - 1), MAGENTA, 1, cv2.LINE_AA)
    ratio = metrics.get("half_energy_ratio")
    bhat = metrics.get("half_hist_bhatta")
    label = "halves DISAGREE"
    if ratio is not None and bhat is not None:
        label = f"halves DISAGREE E={ratio:.1f} bhat={bhat:.2f}"
    _draw_boxed_label(canvas, label, max(mid - 40, 4), BANNER_HEIGHT + 18, MAGENTA)


def draw_artifacts_badge(canvas: np.ndarray, score: float) -> None:
    """Corner badge for soft artifact cue (does not paint a full heatmap)."""
    h, w = canvas.shape[:2]
    text = f"ART soft {score:.2f}"
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.5, 1)
    pad = 6
    x1 = w - tw - 2 * pad - 8
    y1 = BANNER_HEIGHT + 8
    x2 = w - 8
    y2 = y1 + th + 2 * pad
    cv2.rectangle(canvas, (x1, y1), (x2, y2), BLACK, -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), YELLOW, 1, cv2.LINE_AA)
    cv2.putText(canvas, text, (x1 + pad, y2 - pad), FONT, 0.5, YELLOW, 1, cv2.LINE_AA)


def draw_anatomy_map(canvas: np.ndarray, anatomy_map: AnatomyMap) -> None:
    """Draw face, pose and hand landmarks. Map only: never means a generation verdict."""
    face = anatomy_map.face_landmarks_px
    if face:
        for a, b in _face_contour_edges():
            if a < len(face) and b < len(face):
                _draw_line(canvas, face[a], face[b], CYAN, 1)

    pose = anatomy_map.pose_landmarks_px
    if pose:
        for a, b in POSE_CONNECTIONS:
            if a < len(pose) and b < len(pose) and pose[a] is not None and pose[b] is not None:
                _draw_halo_line(canvas, pose[a], pose[b], GREEN, 2)
        for point in pose:
            if point is not None:
                _draw_marker(canvas, point, GREEN, cv2.MARKER_DIAMOND, 6)

    for hand in anatomy_map.hands:
        pts = hand.landmarks_px
        if len(pts) < 21:
            continue
        for a, b in HAND_CONNECTIONS:
            _draw_halo_line(canvas, pts[a], pts[b], AMBER, 1)
        for point in pts:
            x, y = int(round(point[0])), int(round(point[1]))
            if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:
                cv2.circle(canvas, (x, y), 2, AMBER, -1, cv2.LINE_AA)


def draw_geometry_evidence(canvas: np.ndarray, payload: GeometryPayload, evidence_id: str = "") -> None:
    """Draw the flagged line family and its vanishing point even when no outlier crossings exist.

    Earlier overlays only drew `crossings`. On many CRITICAL findings the outlier list is empty
    (spread is high, but no pair exceeds outlier_sigma * MAD), so nothing was painted.
    """
    for segment in payload.cluster_segments[:MAX_CLUSTER_SEGMENTS]:
        _draw_halo_line(canvas, segment.p1, segment.p2, RED, 2)

    if payload.vanishing_point is not None:
        _draw_marker(canvas, payload.vanishing_point, WHITE, cv2.MARKER_CROSS, 22)
        _draw_marker(canvas, payload.vanishing_point, RED, cv2.MARKER_CROSS, 16)
        _draw_point_label(canvas, payload.vanishing_point, evidence_id or "VP", RED)

    for index, crossing in enumerate(payload.crossings[:MAX_GEOMETRY_CROSSINGS]):
        point = crossing.crossing
        for segment in (crossing.segment_a, crossing.segment_b):
            _draw_halo_line(canvas, segment.p1, segment.p2, RED, 3)
            near = min((segment.p1, segment.p2), key=lambda p: _distance(p, point))
            _draw_halo_line(canvas, near, point, RED, 1)
        if index < MAX_CROSSING_LABELS:
            label = f"{evidence_id}.{index + 1}" if evidence_id else str(index + 1)
            _draw_point_label(canvas, point, label, RED)
        else:
            height, width = canvas.shape[:2]
            x, y = point
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(canvas, (int(round(x)), int(round(y))), 5, WHITE, 2, cv2.LINE_AA)
                cv2.circle(canvas, (int(round(x)), int(round(y))), 3, RED, -1, cv2.LINE_AA)


def draw_anatomy_evidence(canvas: np.ndarray, payload: AnatomyPayload) -> None:
    """Magenta box for hand rule violations (distinct from amber map skeleton)."""
    if payload.landmarks_px:
        xs = [p[0] for p in payload.landmarks_px]
        ys = [p[1] for p in payload.landmarks_px]
    elif payload.flagged_joints:
        xs = [(j.bbox[0] + j.bbox[2]) / 2 for j in payload.flagged_joints]
        ys = [(j.bbox[1] + j.bbox[3]) / 2 for j in payload.flagged_joints]
    else:
        return

    pad = 12.0
    x1 = int(round(max(min(xs) - pad, 0)))
    y1 = int(round(max(min(ys) - pad, 0)))
    x2 = int(round(min(max(xs) + pad, canvas.shape[1] - 1)))
    y2 = int(round(min(max(ys) + pad, canvas.shape[0] - 1)))
    cv2.rectangle(canvas, (x1, y1), (x2, y2), BLACK, 3, cv2.LINE_AA)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), MAGENTA, 2, cv2.LINE_AA)
    for joint in payload.flagged_joints:
        jx1, jy1, jx2, jy2 = (int(round(v)) for v in joint.bbox)
        cv2.rectangle(canvas, (jx1, jy1), (jx2, jy2), MAGENTA, 1, cv2.LINE_AA)
    n = len(payload.flagged_joints)
    label = f"ANATOMY anomaly ({payload.handedness}): {n} flag(s)"
    _draw_boxed_label(canvas, label, x1, max(y1 - 8, 40), MAGENTA)


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
        _draw_halo_line(canvas, ray.shadow_tip, target, YELLOW, 2)
        _draw_marker(canvas, ray.top_vertex, YELLOW, cv2.MARKER_DIAMOND, 10)
    for point in payload.pairwise_intersections:
        _draw_marker(canvas, point, YELLOW, cv2.MARKER_TILTED_CROSS, 8)
    if payload.estimated_light is not None:
        _draw_marker(canvas, payload.estimated_light, YELLOW, cv2.MARKER_CROSS, 18)


def _distance(a: Point2D, b: Point2D) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _draw_halo_line(
    canvas: np.ndarray, p1: Point2D, p2: Point2D, color: tuple[int, int, int], thickness: int
) -> None:
    """White underlay then colored line, so marks stay visible on neon and dark photos."""
    _draw_line(canvas, p1, p2, WHITE, thickness + 2)
    _draw_line(canvas, p1, p2, color, thickness)


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
        cv2.drawMarker(canvas, (int(round(x)), int(round(y))), color, marker, size, 2)


def _draw_boxed_label(
    canvas: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]
) -> None:
    height, width = canvas.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.5, 1)
    tx = min(max(x, 2), max(width - tw - 6, 2))
    ty = min(max(y, th + 4), height - 4)
    cv2.rectangle(canvas, (tx - 2, ty - th - 4), (tx + tw + 2, ty + 4), BLACK, -1)
    cv2.putText(canvas, text, (tx, ty), FONT, 0.5, color, 1, cv2.LINE_AA)


def _draw_point_label(canvas: np.ndarray, point: Point2D, text: str, color: tuple[int, int, int]) -> None:
    """Ring at the point with a short label on a dark background. Skipped if outside the image."""
    height, width = canvas.shape[:2]
    x, y = point
    if not (0 <= x < width and 0 <= y < height):
        return
    center = (int(round(x)), int(round(y)))
    cv2.circle(canvas, center, 9, WHITE, 2, cv2.LINE_AA)
    cv2.circle(canvas, center, 7, color, 2, cv2.LINE_AA)
    _draw_boxed_label(canvas, text, center[0] + 10, center[1] - 10, color)


def _detector_banner_text(protocol: EvidenceProtocol) -> str:
    from core.verdict import fuse_layers

    _, decision, fusion = fuse_layers(protocol)
    detector = protocol.detector
    if detector is None:
        return f"Final: {decision} ({fusion})  |  L0: not run  |  marks below are symbolic only"
    return (
        f"Final: {decision} ({fusion})  |  L0={detector.probability:.3f} "
        f"thr={detector.high_threshold:.3f} raw={detector.decision}  |  "
        f"marks below are symbolic only"
    )


def _draw_banner(canvas: np.ndarray, protocol: EvidenceProtocol) -> None:
    critical = sum(1 for e in protocol.evidences if e.severity is Severity.CRITICAL)
    warning = sum(1 for e in protocol.evidences if e.severity is Severity.WARNING)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], BANNER_HEIGHT), BLACK, -1)
    cv2.putText(canvas, _detector_banner_text(protocol), (8, 18), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
    map_note = "map=on" if protocol.anatomy_map is not None else "map=off"
    second = (
        f"{protocol.image.file_name}  |  symbolic findings: {len(protocol.evidences)}  "
        f"(CRITICAL {critical}, WARNING {warning})  |  {map_note}  |  "
        f"cyan=face green=pose amber=map magenta=anomaly white=L0-tile blue=L0-hand yellow=art/light"
    )
    cv2.putText(canvas, second, (8, 38), FONT, 0.45, WHITE, 1, cv2.LINE_AA)
