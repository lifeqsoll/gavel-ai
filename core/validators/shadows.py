"""Light source consistency validator.

Each shadow gives a 2D line through the shadow tip S and the object vertex V.
For a point light L, the ray L -> V -> S is straight, so L lies on every such line.

Checks:
    1. Least-squares intersection of all lines (SVD). RMS residual, normalized by the
       image diagonal, above the threshold -> CRITICAL.
    2. Light must lie beyond each vertex as seen from its shadow tip:
       (L - V) . (V - S) > 0. Otherwise the configuration is physically impossible.
    3. If the estimate is at infinity (directional light), all shadow directions must agree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from core.observations import LightObservation
from core.registry import (
    AnomalyType,
    Evidence,
    LightingPayload,
    Point2D,
    Severity,
    ShadowRay,
)
from core.validators.homogeneous import segment_to_line


@dataclass(frozen=True)
class ShadowConfig:
    min_shadow_length_ratio: float = 0.02
    consistency_threshold_norm: float = 0.02
    warning_fraction: float = 0.5
    directional_threshold_deg: float = 5.0
    far_light_factor: float = 20.0
    max_pairwise_reported: int = 30


@dataclass
class LightReport:
    n_objects: int
    n_rays: int
    estimated_light: Point2D | None = None
    directional: bool = False
    rms_residual_px: float = 0.0
    rms_residual_norm: float = 0.0
    frontal_violations: int = 0
    max_direction_deviation_deg: float = 0.0
    pairwise_intersections: list[Point2D] = field(default_factory=list)
    evidences: list[Evidence] = field(default_factory=list)
    skipped_reason: str | None = None

    @property
    def has_critical(self) -> bool:
        return any(e.severity is Severity.CRITICAL for e in self.evidences)


@dataclass(frozen=True)
class _Ray:
    obs: LightObservation
    shadow: np.ndarray
    vertex: np.ndarray
    line: np.ndarray


class ShadowValidator:
    def __init__(self, config: ShadowConfig | None = None) -> None:
        self.config = config or ShadowConfig()

    def validate(self, observations: Sequence[LightObservation], image_size: tuple[int, int]) -> LightReport:
        cfg = self.config
        width, height = image_size
        diag = math.hypot(width, height)
        center = np.array([width / 2.0, height / 2.0])
        far_radius = cfg.far_light_factor * diag

        rays: list[_Ray] = []
        for obs in observations:
            shadow = np.asarray(obs.shadow_tip, dtype=np.float64)
            vertex = np.asarray(obs.top_vertex, dtype=np.float64)
            if np.linalg.norm(vertex - shadow) < cfg.min_shadow_length_ratio * diag:
                continue
            line = segment_to_line(obs.shadow_tip, obs.top_vertex)
            if line is None:
                continue
            rays.append(_Ray(obs=obs, shadow=shadow, vertex=vertex, line=line))

        report = LightReport(n_objects=len(observations), n_rays=len(rays))
        if len(rays) < 2:
            report.skipped_reason = f"need at least 2 valid shadow rays, got {len(rays)}"
            return report

        lines = np.stack([r.line for r in rays])
        _, _, vt = np.linalg.svd(lines, full_matrices=True)
        h_vec = vt[-1]
        hw = h_vec[2]

        estimate: np.ndarray | None = None
        if abs(hw) > 1e-9:
            candidate = h_vec[:2] / hw
            if np.linalg.norm(candidate - center) <= far_radius:
                estimate = candidate

        if estimate is not None:
            residuals = np.abs(lines @ h_vec) / abs(hw)
            rms = float(np.sqrt(np.mean(residuals**2)))
            rms_norm = rms / diag
            frontal = sum(
                1 for r in rays if float(np.dot(estimate - r.vertex, r.vertex - r.shadow)) <= 0.0
            )
            report.estimated_light = (float(estimate[0]), float(estimate[1]))
            report.rms_residual_px = rms
            report.rms_residual_norm = rms_norm
            report.frontal_violations = frontal
            report.pairwise_intersections = self._pairwise(lines, center, far_radius)
            violation = rms_norm > cfg.consistency_threshold_norm or frontal > 0
            warning = (not violation) and rms_norm > cfg.warning_fraction * cfg.consistency_threshold_norm
        else:
            report.directional = True
            dirs = np.stack(
                [(r.vertex - r.shadow) / np.linalg.norm(r.vertex - r.shadow) for r in rays]
            )
            mean = dirs.mean(axis=0)
            mean_norm = float(np.linalg.norm(mean))
            if mean_norm < 1e-9:
                max_dev = 180.0
            else:
                cosines = np.clip(dirs @ (mean / mean_norm), -1.0, 1.0)
                max_dev = float(np.degrees(np.arccos(cosines.min())))
            report.max_direction_deviation_deg = max_dev
            violation = max_dev > cfg.directional_threshold_deg
            warning = (not violation) and max_dev > cfg.warning_fraction * cfg.directional_threshold_deg

        if violation or warning:
            severity = Severity.CRITICAL if violation else Severity.WARNING
            report.evidences.append(self._build_evidence(report, rays, diag, severity))
        return report

    def _pairwise(self, lines: np.ndarray, center: np.ndarray, far_radius: float) -> list[Point2D]:
        points: list[Point2D] = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                p = np.cross(lines[i], lines[j])
                if abs(p[2]) < 1e-9:
                    continue
                x, y = p[0] / p[2], p[1] / p[2]
                if math.hypot(x - center[0], y - center[1]) <= far_radius:
                    points.append((float(x), float(y)))
        return points[: self.config.max_pairwise_reported]

    def _build_evidence(self, report: LightReport, rays: list[_Ray], diag: float, severity: Severity) -> Evidence:
        cfg = self.config
        thr_pct = cfg.consistency_threshold_norm * 100.0
        parts: list[str] = []
        metrics: dict[str, float] = {
            "n_rays": float(len(rays)),
            "directional": 1.0 if report.directional else 0.0,
            "threshold_consistency_norm": cfg.consistency_threshold_norm,
        }
        if report.directional:
            parts.append(
                f"направления теней расходятся на {report.max_direction_deviation_deg:.1f} deg "
                f"(порог {cfg.directional_threshold_deg:.1f} deg)"
            )
            metrics["max_direction_deviation_deg"] = report.max_direction_deviation_deg
        else:
            ex, ey = report.estimated_light or (0.0, 0.0)
            parts.append(
                f"лучи от {len(rays)} теней, проведенные через вершины объектов, не сходятся "
                f"в единый источник света: RMS ошибка {report.rms_residual_px:.1f} px "
                f"({report.rms_residual_norm * 100:.2f}% диагонали, порог {thr_pct:.2f}%)"
            )
            if report.frontal_violations:
                parts.append(
                    f"оценка источника ({ex:.0f}, {ey:.0f}) лежит за вершинами "
                    f"{report.frontal_violations} объект(ов) относительно их теней"
                )
            metrics.update(
                {
                    "rms_residual_px": report.rms_residual_px,
                    "rms_residual_norm": report.rms_residual_norm,
                    "frontal_violations": float(report.frontal_violations),
                    "estimated_light_x": ex,
                    "estimated_light_y": ey,
                }
            )

        payload = LightingPayload(
            estimated_light=report.estimated_light,
            directional=report.directional,
            rays=[
                ShadowRay(
                    object_id=r.obs.object_id,
                    bbox=tuple(float(v) for v in r.obs.bbox),  # type: ignore[arg-type]
                    base=(float(r.obs.base[0]), float(r.obs.base[1])),
                    top_vertex=(float(r.vertex[0]), float(r.vertex[1])),
                    shadow_tip=(float(r.shadow[0]), float(r.shadow[1])),
                )
                for r in rays
            ],
            pairwise_intersections=report.pairwise_intersections,
            image_diagonal=diag,
        )
        return Evidence(
            evidence_id="LIGHT-01",
            anomaly_type=AnomalyType.LIGHT_CONSISTENCY_VIOLATION,
            severity=severity,
            description="Несогласованность источника света: " + "; ".join(parts) + ".",
            metrics=metrics,
            lighting=payload,
        )
