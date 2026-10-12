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


# Wide diptychs (painting|AI side-by-side) center-crop onto the seam and look "real".
# Score left/right halves too and keep the strongest generation signal.
WIDE_ASPECT_RATIO = 1.6
# Non-wide frames: 2x2 quadrants catch local AI regions that full-frame averages out
# (validated on samples/ai_05: full 0.28 → quadrant 0.79; reals do not rise).
MIN_QUAD_SIDE = 64


@dataclass(frozen=True)
class DetectorOutput:
    probability: float
    logit: float
    decision: Decision
    thresholds: DetectorThresholds
    model: str = MODEL_ID
    view: str = "full"
    # Per-tile / probe probabilities (full/left/right/hand_*).
    view_scores: dict[str, float] | None = None
    views_disagree: bool = False
    # Image-pixel boxes for scored views (overlay grounding).
    view_boxes: dict[str, tuple[float, float, float, float]] | None = None


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

    def _score_rgb(self, rgb: np.ndarray) -> tuple[float, float]:
        """Return (probability, logit) for one RGB uint8 array."""
        import torch
        from PIL import Image

        normalized = normalize_pil(Image.fromarray(rgb))
        tensor = self._preprocess(normalized).unsqueeze(0).to(self._device)
        with torch.no_grad():
            emb = self._model.encode_image(tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        logit = float(emb.float().cpu().numpy()[0] @ self._coef + self._intercept)
        probability = float(1.0 / (1.0 + np.exp(-logit)))
        return probability, logit

    @staticmethod
    def _views(
        image_bgr: np.ndarray,
        *,
        include_quads: bool = True,
        include_center: bool = False,
    ) -> list[tuple[str, np.ndarray, tuple[float, float, float, float]]]:
        h, w = image_bgr.shape[:2]
        views: list[tuple[str, np.ndarray, tuple[float, float, float, float]]] = [
            ("full", image_bgr, (0.0, 0.0, float(w - 1), float(h - 1)))
        ]
        aspect = w / max(h, 1)
        if aspect >= WIDE_ASPECT_RATIO and w >= MIN_QUAD_SIDE:
            mid = w // 2
            views.append(("left", image_bgr[:, :mid], (0.0, 0.0, float(mid - 1), float(h - 1))))
            views.append(
                ("right", image_bgr[:, mid:], (float(mid), 0.0, float(w - 1), float(h - 1)))
            )
        # Center 60% crop (H8): subject often sits mid-frame; full averages edges.
        if include_center and min(h, w) >= MIN_QUAD_SIDE:
            m = 0.2  # keep central 60%
            x1, y1 = int(w * m), int(h * m)
            x2, y2 = int(w * (1.0 - m)), int(h * (1.0 - m))
            if x2 - x1 >= MIN_QUAD_SIDE and y2 - y1 >= MIN_QUAD_SIDE:
                views.append(
                    ("center", image_bgr[y1:y2, x1:x2], (float(x1), float(y1), float(x2 - 1), float(y2 - 1)))
                )
        # 2x2 quadrants: local AI regions that full-frame / half averages miss
        # (samples/ai_05). H14: score() may skip these when L0 is already decisive.
        if include_quads and min(h, w) >= MIN_QUAD_SIDE:
            for i in range(2):
                for j in range(2):
                    y1, y2 = i * h // 2, (i + 1) * h // 2
                    x1, x2 = j * w // 2, (j + 1) * w // 2
                    box = (float(x1), float(y1), float(x2 - 1), float(y2 - 1))
                    views.append((f"q{i}{j}", image_bgr[y1:y2, x1:x2], box))
        return views

    def score_bbox(
        self, image_bgr: np.ndarray, bbox: tuple[float, float, float, float]
    ) -> float:
        """Score one axis-aligned crop; used for anatomy hand probes."""
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("expected a BGR image with shape (H, W, 3)")
        self._ensure_backbone()
        h, w = image_bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        xa, xb = int(max(0, min(x1, x2))), int(min(w, max(x1, x2)))
        ya, yb = int(max(0, min(y1, y2))), int(min(h, max(y1, y2)))
        if xb - xa < 16 or yb - ya < 16:
            raise ValueError("bbox too small to score")
        crop = image_bgr[ya:yb, xa:xb]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        prob, _ = self._score_rgb(rgb)
        return prob

    def score(self, image_bgr: np.ndarray) -> DetectorOutput:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("expected a BGR image with shape (H, W, 3)")
        self._ensure_backbone()

        best_prob = -1.0
        best_logit = 0.0
        best_view = "full"
        view_scores: dict[str, float] = {}
        view_boxes: dict[str, tuple[float, float, float, float]] = {}

        def _score_tile(name: str, tile_bgr: np.ndarray, box: tuple[float, float, float, float]) -> None:
            nonlocal best_prob, best_logit, best_view
            rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
            prob, logit = self._score_rgb(rgb)
            view_scores[name] = prob
            view_boxes[name] = box
            if prob > best_prob:
                best_prob, best_logit, best_view = prob, logit, name

        # Pass 1: full (+ left/right if wide). Center crop (H8) rejected: raises
        # corridor L0 past high without helping AI margin on the gate set.
        for name, tile_bgr, box in self._views(image_bgr, include_quads=False, include_center=False):
            _score_tile(name, tile_bgr, box)

        # H14: quadrants only when L0 still in the uncertain band.
        if self.thresholds.low <= best_prob < self.thresholds.high:
            for name, tile_bgr, box in self._views(image_bgr, include_quads=True, include_center=False):
                if name.startswith("q") and name not in view_scores:
                    _score_tile(name, tile_bgr, box)

        # Disagree is only for left|right diptych halves. Quadrants intentionally
        # disagree with full-frame; that is a feature (take max), not uncertainty.
        disagree = False
        if "left" in view_scores and "right" in view_scores:
            disagree = abs(view_scores["left"] - view_scores["right"]) >= 0.45
        return DetectorOutput(
            probability=best_prob,
            logit=best_logit,
            decision=self.thresholds.decide(best_prob),
            thresholds=self.thresholds,
            view=best_view,
            view_scores=view_scores,
            views_disagree=disagree,
            view_boxes=view_boxes,
        )
