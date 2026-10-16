"""Light extractor. Produces shadow observations from a pluggable scene provider.

Default provider is agents.shadow_scene.extract_shadow_observations.
Pass scene_provider=None to force skip.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from agents.shadow_scene import extract_shadow_observations
from core.observations import LightObservation

SceneProvider = Callable[[np.ndarray], Sequence[LightObservation]]


class LightAgent:
    def __init__(
        self, scene_provider: SceneProvider | None = extract_shadow_observations
    ) -> None:
        self._provider = scene_provider

    def extract(self, image: np.ndarray) -> list[LightObservation]:
        if self._provider is None:
            raise NotImplementedError("LightAgent has no scene provider configured")
        return list(self._provider(image))
