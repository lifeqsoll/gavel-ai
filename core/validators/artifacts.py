"""L2 artifact layer: flat-region residual kurtosis as a soft generation cue.

Absolute high-frequency energy is a texture detector (cobblestones, stone, paint),
not an AI detector. Real samples hit energy 190–938 while some AI frames sit at 10.

This layer instead measures residual *spikiness* on low-gradient patches (kurtosis).
On the project sample set that separates better; vote threshold is calibrated there
and stored in models/layer_thresholds.json.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.layer_thresholds import load_layer_thresholds
from core.registry import LayerScore


@dataclass(frozen=True)
class ArtifactConfig:
    # Default overridden by models/layer_thresholds.json (calibrate_artifacts.py).
    signal_threshold: float = 0.95
    flat_percentile: float = 25.0
    kurtosis_floor: float = 22.0
    kurtosis_span: float = 55.0
    # Heavy texture + low kurtosis → photographic grain/structure, dampen score.
    texture_energy_gate: float = 80.0
    texture_kurtosis_gate: float = 32.0
    texture_dampen: float = 0.25


@dataclass
class ArtifactReport:
    layer: LayerScore
    residual_energy: float
    flat_kurtosis: float
    heatmap: np.ndarray | None = None


class ArtifactValidator:
    def __init__(self, config: ArtifactConfig | None = None) -> None:
        if config is None:
            thr = load_layer_thresholds().artifacts
            config = ArtifactConfig(signal_threshold=thr)
        self.config = config

    def analyze(self, image_bgr: np.ndarray) -> ArtifactReport:
        cfg = self.config
        if image_bgr.ndim == 2:
            gray = image_bgr.astype(np.float32)
        else:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        blur = cv2.GaussianBlur(gray, (0, 0), 2.0)
        resid = gray - blur
        energy = float(np.mean(resid * resid))

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        flat = mag <= float(np.percentile(mag, cfg.flat_percentile))
        sample = resid[flat] if int(flat.sum()) >= 200 else resid.ravel()
        sample = sample - float(sample.mean())
        m2 = float(np.mean(sample * sample)) + 1e-9
        kurtosis = float(np.mean(sample**4) / (m2 * m2))

        score = float(np.clip((kurtosis - cfg.kurtosis_floor) / cfg.kurtosis_span, 0.0, 1.0))
        if kurtosis < cfg.texture_kurtosis_gate and energy > cfg.texture_energy_gate:
            score *= cfg.texture_dampen

        signal = score >= cfg.signal_threshold
        heatmap = np.clip(np.abs(resid) / 40.0, 0.0, 1.0)
        layer = LayerScore(
            name="artifacts",
            status="ran",
            score=score,
            signal=signal,
            reason=f"flat_kurtosis={kurtosis:.1f} residual_energy={energy:.2f}",
        )
        return ArtifactReport(
            layer=layer,
            residual_energy=energy,
            flat_kurtosis=kurtosis,
            heatmap=heatmap,
        )
