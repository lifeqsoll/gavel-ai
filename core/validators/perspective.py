"""Perspective validator: RANSAC vanishing points and intersection variance.

Pipeline:
    1. Segments become normalized homogeneous lines.
    2. Short segments are dropped (relative to the image diagonal).
    3. Iterative RANSAC (vectorized NumPy) proposes vanishing points v = l_i x l_j.
       A line is an inlier if its distance to v is below the threshold.
       Consensus is weighted by segment length.
    4. Each cluster is refined with DLT (smallest right singular vector of the inlier matrix).
    5. All pairwise intersections inside a cluster are analyzed. The robust spread
       (MAD per axis, scaled by 1.4826) is normalized by the image diagonal. The fraction of
       pairs that intersect far outside the admissible region is an independent criterion.
    6. Threshold exceeded -> Evidence(GEOMETRY_PERSPECTIVE_VIOLATION).

Segments that join no cluster are reported as rejected (treated as random).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.stats import median_abs_deviation

from core.observations import Segment
from core.registry import (
    AnomalyType,
    CrossingPair,
    Evidence,
    GeometryPayload,
    LineSegment,
    Point2D,
    Severity,
)
from core.validators.homogeneous import homogeneous_to_affine, segment_to_line

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PerspectiveConfig:
    min_segment_length_ratio: float = 0.03
    """Minimum segment length as a fraction of the image diagonal."""
    inlier_threshold_px: float = 4.0
    min_sample_angle_deg: float = 8.0
    min_inliers: int = 4
    ransac_iterations: int = 2000
    max_vanishing_points: int = 3
    max_region_factor: float = 10.0
    """Admissible vanishing point radius, in diagonals from the image center."""
    variance_threshold_norm: float = 0.02
    """Empirical: consistent corridor ~0.003, planted conflicting family ~0.02-0.04."""
    divergent_fraction_threshold: float = 0.5
    warning_fraction: float = 0.5
    outlier_sigma: float = 3.0
    max_crossings_reported: int = 12
    random_seed: int | None = 42


@dataclass
class VanishingPointCluster:
    point: Point2D
    segment_indices: list[int]
    rms_residual_px: float
    n_pairs: int
    divergent_fraction: float
    robust_std_px: float
    robust_std_norm: float
    variance_px2: float
    crossings: list[CrossingPair] = field(default_factory=list)


@dataclass
class PerspectiveReport:
    image_diagonal: float
    n_input_segments: int
    n_valid_segments: int
    clusters: list[VanishingPointCluster] = field(default_factory=list)
    rejected_segment_indices: list[int] = field(default_factory=list)
    evidences: list[Evidence] = field(default_factory=list)
    skipped_reason: str | None = None

    @property
    def has_critical(self) -> bool:
        return any(e.severity is Severity.CRITICAL for e in self.evidences)


class PerspectiveValidator:
    def __init__(self, config: PerspectiveConfig | None = None) -> None:
        self.config = config or PerspectiveConfig()
        self._sin_min_sample = math.sin(math.radians(self.config.min_sample_angle_deg))

    def validate(self, segments: Sequence[Segment], image_size: tuple[int, int]) -> PerspectiveReport:
        """Run the perspective check.

        Args:
            segments: ((x1, y1), (x2, y2)) pairs in pixels.
            image_size: (width, height) of the frame.
        """
        cfg = self.config
        width, height = image_size
        diag = math.hypot(width, height)
        center = np.array([width / 2.0, height / 2.0])
        region_radius = cfg.max_region_factor * diag

        kept_orig: list[int] = []
        lines: list[np.ndarray] = []
        lengths: list[float] = []
        for idx, (p1, p2) in enumerate(segments):
            length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            if length < cfg.min_segment_length_ratio * diag:
                continue
            line = segment_to_line(p1, p2)
            if line is None:
                continue
            kept_orig.append(idx)
            lines.append(line)
            lengths.append(length)

        report = PerspectiveReport(
            image_diagonal=diag,
            n_input_segments=len(segments),
            n_valid_segments=len(lines),
        )
        if len(lines) < cfg.min_inliers:
            report.skipped_reason = f"too few valid segments: {len(lines)} < {cfg.min_inliers}"
            return report

        line_arr = np.asarray(lines, dtype=np.float64)
        len_arr = np.asarray(lengths, dtype=np.float64)
        kept_orig_arr = np.asarray(kept_orig, dtype=np.int64)
        remaining = np.ones(len(lines), dtype=bool)
        rng = np.random.default_rng(cfg.random_seed)

        for _ in range(cfg.max_vanishing_points):
            pool = np.flatnonzero(remaining)
            if pool.size < cfg.min_inliers:
                break
            found = self._ransac_vanishing_point(line_arr[pool], len_arr[pool], rng, center, region_radius)
            if found is None:
                break
            local_mask, vp_h = found
            members = pool[local_mask]
            remaining[members] = False
            report.clusters.append(
                self._analyze_cluster(
                    vp_h=vp_h,
                    members=members,
                    lines=line_arr,
                    segments=segments,
                    kept_orig=kept_orig_arr,
                    center=center,
                    diag=diag,
                    region_radius=region_radius,
                )
            )

        report.rejected_segment_indices = [int(kept_orig_arr[k]) for k in np.flatnonzero(remaining)]

        for k, cluster in enumerate(report.clusters):
            severity = self._classify(cluster)
            if severity is None:
                continue
            report.evidences.append(self._build_evidence(k, cluster, severity, segments, diag))
        return report

    def _ransac_vanishing_point(
        self,
        lines: np.ndarray,
        weights: np.ndarray,
        rng: np.random.Generator,
        center: np.ndarray,
        region_radius: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Find the dominant consensus set of concurrent lines.

        Returns (inlier mask over `lines`, homogeneous vanishing point) or None.
        """
        cfg = self.config
        n = len(lines)
        if n < cfg.min_inliers:
            return None

        i = rng.integers(0, n, size=cfg.ransac_iterations)
        j = rng.integers(0, n, size=cfg.ransac_iterations)
        sin_ij = np.abs(lines[i, 0] * lines[j, 1] - lines[i, 1] * lines[j, 0])
        ok = (i != j) & (sin_ij > self._sin_min_sample)
        if not np.any(ok):
            return None
        hyps = np.cross(lines[i[ok]], lines[j[ok]])
        hyps /= np.linalg.norm(hyps, axis=1, keepdims=True)

        hyps = hyps[self._valid_points_mask(hyps, center, region_radius)]
        if hyps.shape[0] == 0:
            return None

        dist = np.abs(lines @ hyps.T) / np.abs(hyps[:, 2])[None, :]
        inlier = dist < cfg.inlier_threshold_px
        counts = inlier.sum(axis=0)
        scores = weights @ inlier.astype(np.float64)
        scores = np.where(counts >= cfg.min_inliers, scores, -1.0)
        best = int(np.argmax(scores))
        if scores[best] < 0.0:
            return None

        vp = hyps[best]
        mask = inlier[:, best]
        for _ in range(3):
            refined = self._fit_vanishing_point(lines[mask], weights[mask])
            if not self._valid_points_mask(refined[None, :], center, region_radius)[0]:
                break
            new_mask = np.abs(lines @ refined) / abs(refined[2]) < cfg.inlier_threshold_px
            if new_mask.sum() < cfg.min_inliers:
                break
            vp = refined
            if np.array_equal(new_mask, mask):
                mask = new_mask
                break
            mask = new_mask
        return mask, vp

    @staticmethod
    def _fit_vanishing_point(lines: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """DLT: minimize sum w_i (l_i . v)^2 subject to ||v|| = 1."""
        a = lines * np.sqrt(weights)[:, None]
        _, _, vt = np.linalg.svd(a, full_matrices=True)
        v = vt[-1]
        return v / np.linalg.norm(v)

    @staticmethod
    def _valid_points_mask(hyps: np.ndarray, center: np.ndarray, region_radius: float) -> np.ndarray:
        hw = hyps[:, 2]
        finite = np.abs(hw) > 1e-9
        safe_w = np.where(finite, hw, 1.0)
        x = hyps[:, 0] / safe_w
        y = hyps[:, 1] / safe_w
        return finite & (np.hypot(x - center[0], y - center[1]) <= region_radius)

    def _analyze_cluster(
        self,
        *,
        vp_h: np.ndarray,
        members: np.ndarray,
        lines: np.ndarray,
        segments: Sequence[Segment],
        kept_orig: np.ndarray,
        center: np.ndarray,
        diag: float,
        region_radius: float,
    ) -> VanishingPointCluster:
        cfg = self.config
        cluster_lines = lines[members]
        n = len(members)

        ii, jj = np.triu_indices(n, k=1)
        sin_ij = np.abs(
            cluster_lines[ii, 0] * cluster_lines[jj, 1] - cluster_lines[ii, 1] * cluster_lines[jj, 0]
        )
        keep = sin_ij > 1e-9
        ii, jj = ii[keep], jj[keep]
        inter = np.cross(cluster_lines[ii], cluster_lines[jj])
        inter /= np.linalg.norm(inter, axis=1, keepdims=True)

        hw = inter[:, 2]
        finite = np.abs(hw) > 1e-9
        pts = np.zeros((len(inter), 2))
        pts[finite] = inter[finite, :2] / hw[finite, None]
        dist_center = np.hypot(pts[:, 0] - center[0], pts[:, 1] - center[1])
        in_region = finite & (dist_center <= region_radius)
        pair_ids = np.flatnonzero(in_region)
        conv = pts[pair_ids]

        n_pairs = len(inter)
        divergent_fraction = 1.0 - len(conv) / n_pairs if n_pairs else 1.0

        if len(conv) >= 3:
            med = np.median(conv, axis=0)
            mad = median_abs_deviation(conv, axis=0, scale="normal")
            robust_std_px = float(np.sqrt(np.sum(mad**2)))
            variance_px2 = float(np.sum(np.var(conv, axis=0)))
            conv_dist = np.linalg.norm(conv - med, axis=1)
        else:
            robust_std_px = 0.0
            variance_px2 = 0.0
            conv_dist = np.zeros(len(conv))

        residuals = np.abs(cluster_lines @ vp_h) / abs(vp_h[2])
        rms_residual = float(np.sqrt(np.mean(residuals**2)))

        crossings: list[CrossingPair] = []
        if len(conv) >= 3 and robust_std_px > 0.0:
            outlier_rel = np.flatnonzero(conv_dist > cfg.outlier_sigma * robust_std_px)
            order = outlier_rel[np.argsort(-conv_dist[outlier_rel])][: cfg.max_crossings_reported]
            for rel in order:
                pair = pair_ids[rel]
                seg_a = segments[kept_orig[members[ii[pair]]]]
                seg_b = segments[kept_orig[members[jj[pair]]]]
                crossings.append(
                    CrossingPair(
                        segment_a=_to_line_segment(seg_a),
                        segment_b=_to_line_segment(seg_b),
                        crossing=(float(pts[pair, 0]), float(pts[pair, 1])),
                    )
                )

        vp_affine = homogeneous_to_affine(vp_h)
        if vp_affine is None:
            raise ValueError("Vanishing point is at infinity; the region mask should have rejected it")

        return VanishingPointCluster(
            point=vp_affine,
            segment_indices=[int(kept_orig[k]) for k in members],
            rms_residual_px=rms_residual,
            n_pairs=n_pairs,
            divergent_fraction=float(divergent_fraction),
            robust_std_px=robust_std_px,
            robust_std_norm=robust_std_px / diag,
            variance_px2=variance_px2,
            crossings=crossings,
        )

    def _classify(self, cluster: VanishingPointCluster) -> Severity | None:
        cfg = self.config
        if cluster.n_pairs < 3:
            return None
        var_thr = cfg.variance_threshold_norm
        div_thr = cfg.divergent_fraction_threshold
        if cluster.robust_std_norm > var_thr or cluster.divergent_fraction > div_thr:
            return Severity.CRITICAL
        if (
            cluster.robust_std_norm > cfg.warning_fraction * var_thr
            or cluster.divergent_fraction > cfg.warning_fraction * div_thr
        ):
            return Severity.WARNING
        return None

    def _build_evidence(
        self,
        index: int,
        cluster: VanishingPointCluster,
        severity: Severity,
        segments: Sequence[Segment],
        diag: float,
    ) -> Evidence:
        cfg = self.config
        vx, vy = cluster.point
        description = (
            f"Линии, претендующие на параллельность ({len(cluster.segment_indices)} шт.), "
            f"сходятся в точку ({vx:.0f}, {vy:.0f}), но робастный разброс их попарных пересечений "
            f"составляет {cluster.robust_std_norm * 100:.2f}% диагонали кадра "
            f"(порог {cfg.variance_threshold_norm * 100:.2f}%). Доля пар, пересекающихся вне "
            f"допустимой области: {cluster.divergent_fraction * 100:.0f}%."
        )
        metrics = {
            "vp_x": vx,
            "vp_y": vy,
            "n_lines": float(len(cluster.segment_indices)),
            "rms_residual_px": cluster.rms_residual_px,
            "n_pairs": float(cluster.n_pairs),
            "divergent_fraction": cluster.divergent_fraction,
            "robust_std_px": cluster.robust_std_px,
            "robust_std_norm": cluster.robust_std_norm,
            "variance_px2": cluster.variance_px2,
            "threshold_variance_norm": cfg.variance_threshold_norm,
            "threshold_divergent_fraction": cfg.divergent_fraction_threshold,
        }
        payload = GeometryPayload(
            vanishing_point=cluster.point,
            cluster_segments=[_to_line_segment(segments[i]) for i in cluster.segment_indices],
            crossings=cluster.crossings,
            image_diagonal=diag,
        )
        return Evidence(
            evidence_id=f"GEO-{index + 1:02d}",
            anomaly_type=AnomalyType.GEOMETRY_PERSPECTIVE_VIOLATION,
            severity=severity,
            description=description,
            metrics=metrics,
            geometry=payload,
        )


def _to_line_segment(seg: Segment) -> LineSegment:
    (x1, y1), (x2, y2) = seg
    return LineSegment(p1=(float(x1), float(y1)), p2=(float(x2), float(y2)))
