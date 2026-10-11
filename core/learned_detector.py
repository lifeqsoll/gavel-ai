"""Learned generated-image detector: frozen CLIP ViT-L/14 features and a linear head.

The head weights and decision thresholds are produced by
training/train_two_real.py and stored as models/detector_weights.npz and
models/thresholds.json. Only the head and thresholds ship with the project.
The CLIP backbone is downloaded by open_clip on first use and requires torch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from core.normalize import normalize_pil
from core.registry import Decision

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "models" / "detector_weights.npz"
DEFAULT_THRESHOLDS = ROOT / "models" / "thresholds.json"
MODEL_NAME = "ViT-L-14"
PRETRAINED = "laion2b_s32b_b82k"
MODEL_ID = f"{MODEL_NAME}/{PRETRAINED}+linear-probe"


@dataclass(frozen=True)
class DetectorThresholds:
    high: float
    low: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.low < self.high <= 1.0:
            raise ValueError(f"expected 0 <= low < high <= 1, got low={self.low}, high={self.high}")

    def decide(self, probability: float) -> Decision:
        if probability >= self.high:
            return "generated"
        if probability < self.low:
            return "real"
        return "uncertain"


@dataclass(frozen=True)
class DetectorOutput:
    probability: float
    logit: float
    decision: Decision
    thresholds: DetectorThresholds
    model: str = MODEL_ID


class LearnedDetector:
    def __init__(
        self,
        weights_path: Path | None = None,
        thresholds_path: Path | None = None,
        device: str | None = None,
    ) -> None:
        weights = Path(weights_path) if weights_path else DEFAULT_WEIGHTS
        thresholds = Path(thresholds_path) if thresholds_path else DEFAULT_THRESHOLDS
        if not weights.exists():
            raise FileNotFoundError(f"detector weights not found: {weights}")
        if not thresholds.exists():
            raise FileNotFoundError(f"detector thresholds not found: {thresholds}")
        data = np.load(weights)
        self._coef = data["coef"].astype(np.float32)
        self._intercept = float(data["intercept"])
        raw = json.loads(thresholds.read_text(encoding="utf-8"))
        self.thresholds = DetectorThresholds(high=float(raw["high"]), low=float(raw["low"]))
        self._device = device
        self._model = None
        self._preprocess = None

    def _ensure_backbone(self) -> None:
        if self._model is not None:
            return
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise RuntimeError("open_clip and torch are required for the learned detector") from exc
        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
        self._model = model.to(device).eval()
        self._preprocess = preprocess
        self._device = device

    def score(self, image_bgr: np.ndarray) -> DetectorOutput:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("expected a BGR image with shape (H, W, 3)")
        self._ensure_backbone()
        import torch
        from PIL import Image

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        normalized = normalize_pil(Image.fromarray(rgb))
        tensor = self._preprocess(normalized).unsqueeze(0).to(self._device)
        with torch.no_grad():
            emb = self._model.encode_image(tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        logit = float(emb.float().cpu().numpy()[0] @ self._coef + self._intercept)
        probability = float(1.0 / (1.0 + np.exp(-logit)))
        return DetectorOutput(
            probability=probability,
            logit=logit,
            decision=self.thresholds.decide(probability),
            thresholds=self.thresholds,
        )
