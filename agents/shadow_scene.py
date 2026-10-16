"""Classical cast-shadow scene provider for LightAgent.

Finds dark elongated blobs (HSV V-channel) and pairs each with a brighter
object contour above the tip. Conservative: returns [] when fewer than two
valid pairs (pipeline skips physics rather than hard-voting FP).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.observations import LightObservation
from core.registry import BBox2D


@dataclass(frozen=True)
class ShadowSceneConfig:
    v_percentile: float = 18.0
    min_blob_area_frac: float = 0.0015
    max_blob_area_frac: float = 0.12
    min_aspect: float = 1.8
    max_blobs: int = 8
    object_search_up_frac: float = 0.35
    min_object_area_frac: float = 0.0008
    min_brightness_delta: float = 25.0


def extract_shadow_observations(
    image_bgr: np.ndarray, config: ShadowSceneConfig | None = None
) -> list[LightObservation]:
    cfg = config or ShadowSceneConfig()
    if image_bgr is None or image_bgr.size == 0:
        return []
    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    h, w = image_bgr.shape[:2]
    area = float(max(h * w, 1))
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    thr = float(np.percentile(v, cfg.v_percentile))
    mask = (v <= thr).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[float, np.ndarray]] = []
    for cnt in contours:
        blob_area = float(cv2.contourArea(cnt))
        frac = blob_area / area
        if frac < cfg.min_blob_area_frac or frac > cfg.max_blob_area_frac:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        if aspect < cfg.min_aspect:
            continue
        candidates.append((blob_area, cnt))
    candidates.sort(key=lambda t: t[0], reverse=True)
    candidates = candidates[: cfg.max_blobs]

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    edges = cv2.dilate(edges, kernel, iterations=1)
    obj_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    observations: list[LightObservation] = []
    for idx, (_, shadow_cnt) in enumerate(candidates):
        sx, sy, sw, sh = cv2.boundingRect(shadow_cnt)
        tip = _farthest_point_from_top(shadow_cnt, sx + sw / 2.0, sy)
        tip_xy = (float(tip[0]), float(tip[1]))
        tip_v = float(v[int(np.clip(tip[1], 0, h - 1)), int(np.clip(tip[0], 0, w - 1))])

        search_top = max(0, int(tip[1] - cfg.object_search_up_frac * h))
        best: tuple[float, BBox2D, tuple[float, float], tuple[float, float]] | None = None
        for oc in obj_contours:
            oa = float(cv2.contourArea(oc))
            if oa / area < cfg.min_object_area_frac:
                continue
            ox, oy, ow, oh = cv2.boundingRect(oc)
            if oy + oh > tip[1] + 4:
                continue
            if oy < search_top:
                continue
            cx = ox + ow / 2.0
            if abs(cx - tip[0]) > 0.45 * w:
                continue
            mean_v = float(np.mean(v[oy : oy + oh, ox : ox + ow])) if oh > 0 and ow > 0 else 0.0
            if mean_v - tip_v < cfg.min_brightness_delta:
                continue
            top_vertex = (cx, float(oy))
            base = (cx, float(oy + oh))
            dist = float(np.hypot(top_vertex[0] - tip_xy[0], top_vertex[1] - tip_xy[1]))
            if dist < 0.02 * min(h, w):
                continue
            score = mean_v - tip_v + dist * 0.01
            bbox: BBox2D = (float(ox), float(oy), float(ox + ow), float(oy + oh))
            if best is None or score > best[0]:
                best = (score, bbox, top_vertex, base)

        if best is None:
            continue
        _, bbox, top_vertex, base = best
        observations.append(
            LightObservation(
                object_id=f"shadow_{idx}",
                bbox=bbox,
                base=base,
                top_vertex=top_vertex,
                shadow_tip=tip_xy,
                confidence=0.5,
            )
        )

    if len(observations) < 2:
        return []
    return observations


def _farthest_point_from_top(contour: np.ndarray, top_x: float, top_y: float) -> np.ndarray:
    pts = contour.reshape(-1, 2).astype(np.float64)
    anchor = np.array([top_x, top_y], dtype=np.float64)
    d = np.linalg.norm(pts - anchor, axis=1)
    return pts[int(np.argmax(d))]
