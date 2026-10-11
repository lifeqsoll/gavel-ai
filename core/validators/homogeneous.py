"""Homogeneous coordinate helpers for 2D points and lines.

Conventions:
    point p = (x, y) -> (x, y, 1)
    line  l = (a, b, c) for a*x + b*y + c = 0, normalized so that a^2 + b^2 = 1.
    Then |l . p| / |p_w| is the Euclidean distance in pixels.
"""

from __future__ import annotations

import numpy as np

from core.registry import Point2D

_EPS = 1e-12


def to_homogeneous(point: Point2D) -> np.ndarray:
    return np.array([point[0], point[1], 1.0], dtype=np.float64)


def segment_to_line(p1: Point2D, p2: Point2D) -> np.ndarray | None:
    """Normalized line through two points, or None if the points coincide."""
    line = np.cross(to_homogeneous(p1), to_homogeneous(p2))
    norm = float(np.hypot(line[0], line[1]))
    if norm < _EPS:
        return None
    return line / norm


def homogeneous_to_affine(h: np.ndarray, eps: float = 1e-9) -> Point2D | None:
    """Homogeneous point to affine coordinates, None if at infinity."""
    norm = float(np.linalg.norm(h))
    if norm < _EPS:
        return None
    hx, hy, hw = h / norm
    if abs(hw) < eps:
        return None
    return (float(hx / hw), float(hy / hw))


def point_line_distances(lines: np.ndarray, point_h: np.ndarray) -> np.ndarray:
    """Pixel distances from one finite homogeneous point to normalized lines (N, 3)."""
    return np.abs(lines @ point_h) / abs(point_h[2])
