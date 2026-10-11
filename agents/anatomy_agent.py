"""Body anatomy extractor: MediaPipe Face Mesh + Pose + Hands.

Holistic alone often misses close-up hands (no body/face in frame). Separate models
cover that case. Hands that lie on a face are dropped. The landmark map is for the
overlay; hand deformation evidence stays off by default in the validator.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from core.observations import BodyScene, HandObservation
from core.registry import Point2D

logger = logging.getLogger(__name__)

FACE_OVERLAP_FRACTION = 0.5
FACE_BOX_PAD = 0.2
POSE_VISIBILITY_MIN = 0.5


class AnatomyAgent:
    def __init__(
        self,
        max_hands: int = 2,
        min_detection_confidence: float = 0.5,
    ) -> None:
        self._max_hands = max_hands
        self._min_detection_confidence = min_detection_confidence
        self._hands = None
        self._faces = None
        self._pose = None

    def _solutions(self):
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("mediapipe is not installed; anatomy check disabled") from exc
        solutions = getattr(mp, "solutions", None)
        if solutions is None:
            raise RuntimeError("installed mediapipe has no legacy `solutions` API")
        return solutions

    def _ensure_hands(self):
        if self._hands is None:
            solutions = self._solutions()
            self._hands = solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=self._max_hands,
                min_detection_confidence=self._min_detection_confidence,
            )
        return self._hands

    def _ensure_faces(self):
        if self._faces is None:
            solutions = self._solutions()
            self._faces = solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=self._min_detection_confidence,
            )
        return self._faces

    def _ensure_pose(self):
        if self._pose is None:
            solutions = self._solutions()
            self._pose = solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                min_detection_confidence=self._min_detection_confidence,
            )
        return self._pose

    @staticmethod
    def _face_box(face_pts: list[Point2D]) -> tuple[float, float, float, float] | None:
        if not face_pts:
            return None
        xs = [p[0] for p in face_pts]
        ys = [p[1] for p in face_pts]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        pad_x, pad_y = FACE_BOX_PAD * (x2 - x1), FACE_BOX_PAD * (y2 - y1)
        return (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y)

    @staticmethod
    def _is_face(points: np.ndarray, boxes: list[tuple[float, float, float, float]]) -> bool:
        for x1, y1, x2, y2 in boxes:
            inside = (
                (points[:, 0] >= x1)
                & (points[:, 0] <= x2)
                & (points[:, 1] >= y1)
                & (points[:, 1] <= y2)
            )
            if float(np.mean(inside)) >= FACE_OVERLAP_FRACTION:
                return True
        return False

    def extract(self, image: np.ndarray) -> BodyScene:
        height, width = image.shape[:2]
        rgb = (
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if image.ndim == 3
            else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        )

        face_landmarks_px: list[Point2D] = []
        try:
            face_result = self._ensure_faces().process(rgb)
            if face_result.multi_face_landmarks:
                face_landmarks_px = [
                    (p.x * width, p.y * height)
                    for p in face_result.multi_face_landmarks[0].landmark
                ]
        except Exception as exc:
            logger.warning("face mesh skipped: %s", exc)

        pose_landmarks_px: list[Point2D | None] = []
        try:
            pose_result = self._ensure_pose().process(rgb)
            if pose_result.pose_landmarks is not None:
                for p in pose_result.pose_landmarks.landmark:
                    if p.visibility < POSE_VISIBILITY_MIN:
                        pose_landmarks_px.append(None)
                    else:
                        pose_landmarks_px.append((p.x * width, p.y * height))
        except Exception as exc:
            logger.warning("pose skipped: %s", exc)

        face_boxes: list[tuple[float, float, float, float]] = []
        box = self._face_box(face_landmarks_px)
        if box is not None:
            face_boxes.append(box)

        hands: list[HandObservation] = []
        try:
            hand_result = self._ensure_hands().process(rgb)
        except Exception as exc:
            logger.warning("hands skipped: %s", exc)
            hand_result = None

        if hand_result is not None and hand_result.multi_hand_landmarks:
            worlds = hand_result.multi_hand_world_landmarks or [None] * len(
                hand_result.multi_hand_landmarks
            )
            handed_list = hand_result.multi_handedness or []
            for idx, (image_lm, world, handed) in enumerate(
                zip(hand_result.multi_hand_landmarks, worlds, handed_list)
            ):
                landmarks_px = np.array(
                    [[p.x * width, p.y * height] for p in image_lm.landmark], dtype=np.float64
                )
                if face_boxes and self._is_face(landmarks_px, face_boxes):
                    logger.info("dropped a hand detection that lies on a face")
                    continue
                if world is not None:
                    landmarks_3d = np.array([[p.x, p.y, p.z] for p in world.landmark], dtype=np.float64)
                else:
                    landmarks_3d = np.column_stack(
                        [
                            landmarks_px[:, 0] / max(width, 1),
                            landmarks_px[:, 1] / max(height, 1),
                            np.zeros(len(landmarks_px)),
                        ]
                    )
                label = handed.classification[0].label
                score = float(handed.classification[0].score)
                hands.append(
                    HandObservation(
                        hand_index=len(hands),
                        handedness=label,
                        landmarks_3d=landmarks_3d,
                        landmarks_px=landmarks_px,
                        score=score,
                    )
                )

        return BodyScene(
            hands=hands,
            face_landmarks_px=face_landmarks_px,
            pose_landmarks_px=pose_landmarks_px,
        )

    def close(self) -> None:
        for attr in ("_hands", "_faces", "_pose"):
            model = getattr(self, attr)
            if model is not None:
                model.close()
                setattr(self, attr, None)
