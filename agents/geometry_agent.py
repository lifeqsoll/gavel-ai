"""Geometry extractor: long straight segments via LSD (or Hough as fallback)."""

from __future__ import annotations

import logging
from typing import Literal

import cv2
import numpy as np

from core.observations import Segment

logger = logging.getLogger(__name__)


class GeometryAgent:
    def __init__(
        self,
        method: Literal["lsd", "hough"] = "lsd",
        max_segments: int = 400,
        hough_threshold: int = 60,
        hough_max_gap: int = 4,
        canny_low: int = 50,
        canny_high: int = 150,
    ) -> None:
        if method == "lsd" and not hasattr(cv2, "createLineSegmentDetector"):
            logger.warning("OpenCV has no LSD detector, falling back to Hough")
            method = "hough"
        self.method = method
        self.max_segments = max_segments
        self.hough_threshold = hough_threshold
        self.hough_max_gap = hough_max_gap
        self.canny_low = canny_low
        self.canny_high = canny_high

    def extract(self, image: np.ndarray) -> list[Segment]:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        diag = float(np.hypot(width, height))

        if self.method == "lsd":
            raw = self._detect_lsd(gray)
        else:
            raw = self._detect_hough(gray, min_length=0.02 * diag)
        if raw.size == 0:
            return []

        lengths = np.hypot(raw[:, 2] - raw[:, 0], raw[:, 3] - raw[:, 1])
        order = np.argsort(-lengths)[: self.max_segments]
        return [
            ((float(x1), float(y1)), (float(x2), float(y2)))
            for x1, y1, x2, y2 in raw[order]
        ]

    @staticmethod
    def _detect_lsd(gray: np.ndarray) -> np.ndarray:
        detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        lines = detector.detect(gray)[0]
        if lines is None:
            return np.empty((0, 4), dtype=np.float64)
        return lines.reshape(-1, 4).astype(np.float64)

    def _detect_hough(self, gray: np.ndarray, min_length: float) -> np.ndarray:
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180.0,
            self.hough_threshold,
            minLineLength=min_length,
            maxLineGap=self.hough_max_gap,
        )
        if lines is None:
            return np.empty((0, 4), dtype=np.float64)
        return lines.reshape(-1, 4).astype(np.float64)
