"""Anatomy extractor backed by MediaPipe Hands (legacy `mp.solutions` API)."""

from __future__ import annotations

import cv2
import numpy as np

from core.observations import HandObservation


class AnatomyAgent:
    def __init__(self, max_hands: int = 2, min_detection_confidence: float = 0.5) -> None:
        self._max_hands = max_hands
        self._min_detection_confidence = min_detection_confidence
        self._hands = None

    def _ensure_model(self):
        if self._hands is not None:
            return self._hands
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("mediapipe is not installed; anatomy check disabled") from exc
        solutions = getattr(mp, "solutions", None)
        if solutions is None:
            raise RuntimeError("installed mediapipe has no legacy `solutions` API")
        self._hands = solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=self._max_hands,
            min_detection_confidence=self._min_detection_confidence,
        )
        return self._hands

    def extract(self, image: np.ndarray) -> list[HandObservation]:
        model = self._ensure_model()
        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        result = model.process(rgb)
        if not result.multi_hand_world_landmarks:
            return []

        observations: list[HandObservation] = []
        for idx, (world, image_lm, handed) in enumerate(
            zip(result.multi_hand_world_landmarks, result.multi_hand_landmarks, result.multi_handedness)
        ):
            landmarks_3d = np.array([[p.x, p.y, p.z] for p in world.landmark], dtype=np.float64)
            landmarks_px = np.array(
                [[p.x * width, p.y * height] for p in image_lm.landmark], dtype=np.float64
            )
            classification = handed.classification[0]
            observations.append(
                HandObservation(
                    hand_index=idx,
                    handedness=classification.label,
                    landmarks_3d=landmarks_3d,
                    landmarks_px=landmarks_px,
                    score=float(classification.score),
                )
            )
        return observations

    def close(self) -> None:
        if self._hands is not None:
            self._hands.close()
            self._hands = None
