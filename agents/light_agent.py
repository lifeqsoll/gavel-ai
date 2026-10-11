"""Light extractor. Produces shadow observations from a pluggable scene provider.

A real segmentation backend can be plugged in as `scene_provider`. Without one,
extract() raises NotImplementedError and the pipeline records the check as skipped.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from core.observations import LightObservation

SceneProvider = Callable[[np.ndarray], Sequence[LightObservation]]


class LightAgent:
    def __init__(self, scene_provider: SceneProvider | None = None) -> None:
        self._provider = scene_provider

    def extract(self, image: np.ndarray) -> list[LightObservation]:
        if self._provider is None:
            raise NotImplementedError("LightAgent has no scene provider configured")
        return list(self._provider(image))
