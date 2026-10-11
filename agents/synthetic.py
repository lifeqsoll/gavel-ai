"""Synthetic generators for tests and the mock run. No neural networks involved."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from core.observations import HandObservation, LightObservation
from core.registry import BBox2D
from core.validators.anatomy import ANATOMICAL_PHALANX_MM, FINGER_BASE

_MCP_POSITIONS_M = {
    "index": (-0.035, 0.085, 0.0),
    "middle": (-0.012, 0.090, 0.0),
    "ring": (0.012, 0.087, 0.0),
    "pinky": (0.035, 0.080, 0.0),
}
_THUMB_CMC_M = (0.035, 0.02, 0.0)
_THUMB_METACARPAL_M = 0.045


def _bend(axis: np.ndarray, normal: np.ndarray, angle_deg: float) -> np.ndarray:
    a = math.radians(angle_deg)
    return axis * math.cos(a) + normal * math.sin(a)


def build_synthetic_hand(
    *,
    hand_index: int = 0,
    handedness: str = "Right",
    flex_deg: dict[str, float] | None = None,
    length_scale: dict[str, float] | None = None,
    center_px: tuple[float, float] = (640.0, 360.0),
    px_per_m: float = 2400.0,
) -> HandObservation:
    """Build a planar hand. Keys: 'index.PIP', 'index.DIP', 'thumb.MCP', 'thumb.IP'.

    length_scale keys: '<finger>.<bone>', e.g. 'index.distal'.
    """
    flex = dict(flex_deg or {})
    scale = dict(length_scale or {})
    f = np.array([0.0, 1.0, 0.0])
    n = np.array([0.0, 0.0, -1.0])
    lm = np.zeros((21, 3))

    for finger, base in FINGER_BASE.items():
        lm[base] = _MCP_POSITIONS_M[finger]
        pip = flex.get(f"{finger}.PIP", 0.0)
        dip = flex.get(f"{finger}.DIP", 0.0)
        cumulative = (0.0, pip, pip + dip)
        cursor = lm[base].copy()
        for k, bone in enumerate(("proximal", "middle", "distal")):
            length = ANATOMICAL_PHALANX_MM[finger][bone] * 1e-3 * scale.get(f"{finger}.{bone}", 1.0)
            cursor = cursor + _bend(f, n, cumulative[k]) * length
            lm[base + k + 1] = cursor

    thumb_dir = np.array([0.7, 0.7, 0.0])
    thumb_dir /= np.linalg.norm(thumb_dir)
    lm[1] = _THUMB_CMC_M
    lm[2] = lm[1] + thumb_dir * _THUMB_METACARPAL_M
    mcp = flex.get("thumb.MCP", 0.0)
    ip = flex.get("thumb.IP", 0.0)
    prox_len = ANATOMICAL_PHALANX_MM["thumb"]["proximal"] * 1e-3 * scale.get("thumb.proximal", 1.0)
    dist_len = ANATOMICAL_PHALANX_MM["thumb"]["distal"] * 1e-3 * scale.get("thumb.distal", 1.0)
    lm[3] = lm[2] + _bend(thumb_dir, n, mcp) * prox_len
    lm[4] = lm[3] + _bend(thumb_dir, n, mcp + ip) * dist_len

    if handedness.strip().lower() == "left":
        lm[:, 0] *= -1.0

    cx, cy = center_px
    px = np.stack([cx + lm[:, 0] * px_per_m, cy - lm[:, 1] * px_per_m], axis=1)
    return HandObservation(
        hand_index=hand_index,
        handedness=handedness,
        landmarks_3d=lm,
        landmarks_px=px,
        score=0.99,
    )


@dataclass(frozen=True)
class ShadowObjectSpec:
    object_id: str
    bbox: BBox2D
    shadow_length_px: float
    deviation_deg: float = 0.0
    """Rotation of the light ray around the vertex, used to inject inconsistency."""


def simulate_shadow_scene(
    light_xy: tuple[float, float], objects: Sequence[ShadowObjectSpec]
) -> list[LightObservation]:
    """2D model: shadow tip lies on the ray from the light through the vertex."""
    light = np.asarray(light_xy, dtype=np.float64)
    observations: list[LightObservation] = []
    for spec in objects:
        x1, y1, x2, y2 = spec.bbox
        cx = (x1 + x2) / 2.0
        top = np.array([cx, y1], dtype=np.float64)
        direction = top - light
        direction /= np.linalg.norm(direction)
        theta = math.radians(spec.deviation_deg)
        rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        direction = rotation @ direction
        tip = top + direction * spec.shadow_length_px
        observations.append(
            LightObservation(
                object_id=spec.object_id,
                bbox=spec.bbox,
                base=(cx, float(y2)),
                top_vertex=(float(top[0]), float(top[1])),
                shadow_tip=(float(tip[0]), float(tip[1])),
            )
        )
    return observations
