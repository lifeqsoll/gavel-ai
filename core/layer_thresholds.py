"""Per-layer vote thresholds for fusion A.

L0 high/low live in models/thresholds.json (calibrated probabilities).
Other layers use score-space thresholds here; geometry does not vote yet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "models" / "layer_thresholds.json"


@dataclass(frozen=True)
class LayerThresholds:
    # L0: mirror of calibrated detector high (informational for UI/docs).
    global_high: float = 0.5962293148040771
    anatomy: float = 0.5
    artifacts: float = 0.95
    physics: float = 0.5
    geometry: float = 0.55

    def for_name(self, name: str) -> float | None:
        if name == "global":
            return self.global_high
        if name == "anatomy":
            return self.anatomy
        if name == "artifacts":
            return self.artifacts
        if name == "physics":
            return self.physics
        if name == "geometry":
            return self.geometry
        return None


def load_layer_thresholds(path: Path | None = None) -> LayerThresholds:
    p = path or DEFAULT_PATH
    if not p.exists():
        return LayerThresholds()
    raw = json.loads(p.read_text(encoding="utf-8"))
    return LayerThresholds(
        global_high=float(raw.get("global", {}).get("high", 0.5962293148040771)),
        anatomy=float(raw.get("anatomy", {}).get("signal_threshold", 0.5)),
        artifacts=float(raw.get("artifacts", {}).get("signal_threshold", 0.95)),
        physics=float(raw.get("physics", {}).get("signal_threshold", 0.5)),
        geometry=float(raw.get("geometry", {}).get("signal_threshold", 0.55) or 0.55),
    )
