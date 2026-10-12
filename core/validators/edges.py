"""Edge-coherence probe (H10): top-k LSD segment length / image area.

AI frames with over-clean geometry (ai_05-like) score high; real streets stay lower.
Soft cue only — fusion may lean generated when L0 is uncertain.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Calibrated on samples/: florence/corridor dens≈0.0037; ai_05≈0.012; ai_09≈0.0087.
EDGE_DENS_UNCERTAIN_THR = 0.009


@dataclass(frozen=True)
class EdgeReport:
    density: float
    long_count: float
    segment_count: float
    disagree: bool


def analyze_edges(image_bgr: np.ndarray, top_k: int = 50) -> EdgeReport:
    if image_bgr is None or image_bgr.size == 0:
        return EdgeReport(0.0, 0.0, 0.0, False)
    h, w = image_bgr.shape[:2]
    area = float(max(h * w, 1))
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    try:
        lsd = cv2.createLineSegmentDetector()
    except cv2.error:
        return EdgeReport(0.0, 0.0, 0.0, False)
    lines = lsd.detect(gray)[0]
    if lines is None or len(lines) == 0:
        return EdgeReport(0.0, 0.0, 0.0, False)
    arr = np.asarray(lines, dtype=np.float64).reshape(-1, 4)
    lengths = np.sqrt((arr[:, 2] - arr[:, 0]) ** 2 + (arr[:, 3] - arr[:, 1]) ** 2)
    lengths_sorted = np.sort(lengths)[::-1]
    dens = float(lengths_sorted[:top_k].sum() / area)
    long_thr = 0.08 * float(min(h, w))
    long_n = float(np.sum(lengths >= long_thr))
    return EdgeReport(
        density=dens,
        long_count=long_n,
        segment_count=float(len(lengths)),
        disagree=dens >= EDGE_DENS_UNCERTAIN_THR,
    )
