"""Contracts between perception agents and symbolic validators."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.registry import BBox2D, Point2D

Segment = tuple[Point2D, Point2D]

NUM_HAND_LANDMARKS = 21


@dataclass(frozen=True)
class HandObservation:
    """One detected hand.

    landmarks_3d: (21, 3) world coordinates in meters.
    landmarks_px: (21, 2) image coordinates in pixels.
    """

    hand_index: int
    handedness: str
    landmarks_3d: np.ndarray
    landmarks_px: np.ndarray
    score: float = 1.0

    def __post_init__(self) -> None:
        if self.landmarks_3d.shape != (NUM_HAND_LANDMARKS, 3):
            raise ValueError(f"landmarks_3d must have shape (21, 3), got {self.landmarks_3d.shape}")
        if self.landmarks_px.shape != (NUM_HAND_LANDMARKS, 2):
            raise ValueError(f"landmarks_px must have shape (21, 2), got {self.landmarks_px.shape}")


@dataclass(frozen=True)
class BodyScene:
    """Perception output for the anatomy map: hands plus optional face and pose landmarks.

    face_landmarks_px: MediaPipe Face Mesh points in pixels (empty if no face).
    pose_landmarks_px: 33 pose points; a point is None when visibility is below the agent threshold.
    """

    hands: list[HandObservation] = field(default_factory=list)
    face_landmarks_px: list[Point2D] = field(default_factory=list)
    pose_landmarks_px: list[Point2D | None] = field(default_factory=list)


@dataclass(frozen=True)
class LightObservation:
    """Object with a visible shadow.

    top_vertex is the object vertex that casts shadow_tip (the farthest shadow point).
    """

    object_id: str
    bbox: BBox2D
    base: Point2D
    top_vertex: Point2D
    shadow_tip: Point2D
    confidence: float = 1.0
