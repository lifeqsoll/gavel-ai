"""Wide-image left/right appearance probe (diptych / painting|photo composites).

L0 CLIP often scores both halves of a painting|AI demo as "real". This probe
compares residual energy and color histograms across the vertical midline.
It does not replace L0; fusion uses the flag when L0 says real.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Calibrated on samples/ai_01 (painting|photo): E_ratio≈2.82, bhat≈0.56.
# Real single scenes rarely exceed both at once on aspect≥1.6.
DEFAULT_MIN_ASPECT = 1.6
DEFAULT_ENERGY_RATIO = 2.5
DEFAULT_HIST_BHATTA = 0.50


@dataclass(frozen=True)
class HalvesConfig:
    min_aspect: float = DEFAULT_MIN_ASPECT
    energy_ratio_thr: float = DEFAULT_ENERGY_RATIO
    hist_bhatta_thr: float = DEFAULT_HIST_BHATTA


@dataclass(frozen=True)
class HalvesReport:
    applicable: bool
    disagree: bool
    energy_ratio: float
    hist_bhatta: float
    energy_left: float
    energy_right: float
    mid_x: float


def _half_energy_and_hist(crop_bgr: np.ndarray) -> tuple[float, np.ndarray]:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0, 0), 2.0)
    resid = gray - blur
    energy = float(np.mean(resid * resid))
    hist = cv2.calcHist([crop_bgr], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return energy, hist


def analyze_halves(image_bgr: np.ndarray, config: HalvesConfig | None = None) -> HalvesReport:
    cfg = config or HalvesConfig()
    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    h, w = image_bgr.shape[:2]
    aspect = w / max(h, 1)
    mid = w // 2
    empty = HalvesReport(
        applicable=False,
        disagree=False,
        energy_ratio=1.0,
        hist_bhatta=0.0,
        energy_left=0.0,
        energy_right=0.0,
        mid_x=float(mid),
    )
    if aspect < cfg.min_aspect or w < 64:
        return empty

    e_l, hist_l = _half_energy_and_hist(image_bgr[:, :mid])
    e_r, hist_r = _half_energy_and_hist(image_bgr[:, mid:])
    lo = max(min(e_l, e_r), 1e-6)
    ratio = max(e_l, e_r) / lo
    bhatta = float(
        cv2.compareHist(hist_l.reshape(-1, 1), hist_r.reshape(-1, 1), cv2.HISTCMP_BHATTACHARYYA)
    )
    disagree = ratio >= cfg.energy_ratio_thr and bhatta >= cfg.hist_bhatta_thr
    return HalvesReport(
        applicable=True,
        disagree=disagree,
        energy_ratio=float(ratio),
        hist_bhatta=float(bhatta),
        energy_left=float(e_l),
        energy_right=float(e_r),
        mid_x=float(mid),
    )
