"""Gavel AI orchestrator: agents -> validators -> EvidenceProtocol -> overlay."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from agents.anatomy_agent import AnatomyAgent
from agents.geometry_agent import GeometryAgent
from agents.light_agent import LightAgent
from core.learned_detector import DetectorOutput
from core.observations import BodyScene, LightObservation, Segment
from core.registry import (
    AnatomyMap,
    CheckRecord,
    DetectorResult,
    Evidence,
    EvidenceProtocol,
    HandMap,
    ImageMetadata,
    LayerScore,
    LayerScores,
    Severity,
)
from core.validators.anatomy import AnatomyValidator
from core.validators.artifacts import ArtifactValidator
from core.validators.perspective import PerspectiveValidator
from core.validators.shadows import ShadowValidator
from visualizer.overlay import render_overlay

logger = logging.getLogger(__name__)

_STAGE_ERRORS = (RuntimeError, NotImplementedError, ValueError)


class GeometryExtractor(Protocol):
    def extract(self, image: np.ndarray) -> list[Segment]: ...


class AnatomyExtractor(Protocol):
    def extract(self, image: np.ndarray) -> BodyScene: ...


class LightExtractor(Protocol):
    def extract(self, image: np.ndarray) -> list[LightObservation]: ...


class DetectorScorer(Protocol):
    def score(self, image: np.ndarray) -> DetectorOutput: ...


@dataclass
class PipelineResult:
    protocol: EvidenceProtocol
    annotated_image: np.ndarray


@dataclass
class GavelPipeline:
    geometry_agent: GeometryExtractor = field(default_factory=GeometryAgent)
    perspective_validator: PerspectiveValidator = field(default_factory=PerspectiveValidator)
    anatomy_agent: AnatomyExtractor = field(default_factory=AnatomyAgent)
    anatomy_validator: AnatomyValidator = field(default_factory=AnatomyValidator)
    light_agent: LightExtractor = field(default_factory=LightAgent)
    shadow_validator: ShadowValidator = field(default_factory=ShadowValidator)
    artifact_validator: ArtifactValidator = field(default_factory=ArtifactValidator)
    learned_detector: DetectorScorer | None = None

    def analyze(self, image: np.ndarray, image_name: str = "unknown") -> PipelineResult:
        if image.ndim not in (2, 3):
            raise ValueError(f"expected a 2D or 3D image array, got ndim={image.ndim}")
        height, width = image.shape[:2]
        size = (width, height)
        checks: list[CheckRecord] = []
        evidences: list[Evidence] = []
        layers = LayerScores()

        layers.geometry = self._run_geometry(image, size, checks, evidences)
        anatomy_map, layers.anatomy = self._run_anatomy(image, checks, evidences)
        layers.physics = self._run_lighting(image, size, checks, evidences)
        layers.artifacts = self.artifact_validator.analyze(image).layer
        checks.append(
            CheckRecord(
                check="artifacts.residual",
                status="passed" if not layers.artifacts.signal else "violation",
                reason=layers.artifacts.reason,
            )
        )
        detector, layers.global_layer = self._run_detector(image, checks)

        protocol = EvidenceProtocol(
            image=ImageMetadata(file_name=image_name, width=width, height=height),
            checks=checks,
            evidences=evidences,
            detector=detector,
            anatomy_map=anatomy_map,
            layers=layers,
        )
        return PipelineResult(protocol=protocol, annotated_image=render_overlay(image, protocol))

    def _run_geometry(
        self,
        image: np.ndarray,
        size: tuple[int, int],
        checks: list[CheckRecord],
        evidences: list[Evidence],
    ) -> LayerScore:
        name = "geometry.perspective"
        try:
            segments = self.geometry_agent.extract(image)
            report = self.perspective_validator.validate(segments, size)
        except _STAGE_ERRORS as exc:
            logger.warning("%s skipped: %s", name, exc)
            checks.append(CheckRecord(check=name, status="skipped", reason=str(exc)))
            return LayerScore(name="geometry", status="skipped", reason=str(exc))
        if report.skipped_reason:
            checks.append(CheckRecord(check=name, status="skipped", reason=report.skipped_reason))
            return LayerScore(name="geometry", status="skipped", reason=report.skipped_reason)
        evidences.extend(report.evidences)
        checks.append(
            CheckRecord(
                check=name,
                status="violation" if report.evidences else "passed",
                reason=f"segments={report.n_valid_segments}, vanishing_points={len(report.clusters)}",
            )
        )
        n_crit = sum(1 for e in report.evidences if e.severity is Severity.CRITICAL)
        n_warn = sum(1 for e in report.evidences if e.severity is Severity.WARNING)
        # Score only from evidences. Raw MAD without a WARNING/CRITICAL is noise
        # (near-parallel real edges), not a generation signal.
        score = float(min(1.0, 0.55 * n_crit + 0.25 * n_warn))
        signal = n_crit > 0
        return LayerScore(
            name="geometry",
            status="ran",
            score=score,
            signal=signal,
            reason=f"evidences={len(report.evidences)} critical={n_crit} warning={n_warn}",
        )

    def _run_anatomy(
        self, image: np.ndarray, checks: list[CheckRecord], evidences: list[Evidence]
    ) -> tuple[AnatomyMap | None, LayerScore]:
        name = "anatomy.body"
        try:
            scene = self.anatomy_agent.extract(image)
        except _STAGE_ERRORS as exc:
            logger.warning("%s skipped: %s", name, exc)
            checks.append(CheckRecord(check=name, status="skipped", reason=str(exc)))
            return None, LayerScore(name="anatomy", status="skipped", reason=str(exc))

        anatomy_map = AnatomyMap(
            face_landmarks_px=list(scene.face_landmarks_px),
            pose_landmarks_px=list(scene.pose_landmarks_px),
            hands=[
                HandMap(
                    handedness=hand.handedness,
                    landmarks_px=[(float(x), float(y)) for x, y in hand.landmarks_px],
                )
                for hand in scene.hands
            ],
        )
        has_map = bool(anatomy_map.face_landmarks_px or anatomy_map.pose_landmarks_px or anatomy_map.hands)
        if not has_map:
            checks.append(CheckRecord(check=name, status="skipped", reason="no face, pose or hands detected"))
            return None, LayerScore(name="anatomy", status="skipped", reason="no face, pose or hands detected")

        report = self.anatomy_validator.validate(scene.hands)
        evidences.extend(report.evidences)
        checks.append(
            CheckRecord(
                check=name,
                status="violation" if report.evidences else "passed",
                reason=(
                    f"face={len(anatomy_map.face_landmarks_px)} "
                    f"pose={sum(1 for p in anatomy_map.pose_landmarks_px if p is not None)} "
                    f"hands={len(anatomy_map.hands)} "
                    f"hand_evidence={'on' if self.anatomy_validator.config.emit_hand_evidence else 'off'}"
                ),
            )
        )
        score = float(min(1.0, 0.5 * len(report.evidences)))
        # Vote threshold 0.5 == one calibrated hand WARNING (models/layer_thresholds.json).
        hand_span = 0.0
        if anatomy_map.hands:
            h, w = image.shape[:2]
            min_side = float(min(h, w)) or 1.0
            for hand in anatomy_map.hands:
                if len(hand.landmarks_px) < 3:
                    continue
                xs = [p[0] for p in hand.landmarks_px]
                ys = [p[1] for p in hand.landmarks_px]
                span = max(max(xs) - min(xs), max(ys) - min(ys)) / min_side
                hand_span = max(hand_span, span)
        return anatomy_map, LayerScore(
            name="anatomy",
            status="ran",
            score=score,
            signal=score >= 0.5,
            reason=(
                f"hand_warnings={len(report.evidences)} "
                f"hands={len(anatomy_map.hands)} "
                f"face={len(anatomy_map.face_landmarks_px)} "
                f"hand_span={hand_span:.3f}"
            ),
        )

    def _run_lighting(
        self,
        image: np.ndarray,
        size: tuple[int, int],
        checks: list[CheckRecord],
        evidences: list[Evidence],
    ) -> LayerScore:
        name = "lighting.consistency"
        try:
            observations = self.light_agent.extract(image)
        except _STAGE_ERRORS as exc:
            logger.warning("%s skipped: %s", name, exc)
            checks.append(CheckRecord(check=name, status="skipped", reason=str(exc)))
            return LayerScore(name="physics", status="skipped", reason=str(exc))
        if not observations:
            checks.append(CheckRecord(check=name, status="skipped", reason="no shadow objects detected"))
            return LayerScore(name="physics", status="skipped", reason="no shadow objects detected")
        report = self.shadow_validator.validate(observations, size)
        if report.skipped_reason:
            checks.append(CheckRecord(check=name, status="skipped", reason=report.skipped_reason))
            return LayerScore(name="physics", status="skipped", reason=report.skipped_reason)
        evidences.extend(report.evidences)
        checks.append(
            CheckRecord(
                check=name,
                status="violation" if report.evidences else "passed",
                reason=f"rays={report.n_rays}",
            )
        )
        score = float(min(1.0, 0.5 * len(report.evidences)))
        return LayerScore(
            name="physics",
            status="ran",
            score=score,
            signal=score >= 0.5,
            reason=f"evidences={len(report.evidences)}",
        )

    def _run_detector(
        self, image: np.ndarray, checks: list[CheckRecord]
    ) -> tuple[DetectorResult | None, LayerScore]:
        if self.learned_detector is None:
            return None, LayerScore(name="global", status="skipped", reason="no detector")
        name = "detector.learned"
        try:
            output = self.learned_detector.score(image)
        except _STAGE_ERRORS as exc:
            logger.warning("%s skipped: %s", name, exc)
            checks.append(CheckRecord(check=name, status="skipped", reason=str(exc)))
            return None, LayerScore(name="global", status="skipped", reason=str(exc))
        detector = DetectorResult(
            model=output.model,
            probability=output.probability,
            logit=output.logit,
            decision=output.decision,
            high_threshold=output.thresholds.high,
            low_threshold=output.thresholds.low,
        )
        view = getattr(output, "view", "full")
        reason = f"decision={output.decision}"
        if view != "full":
            reason += f" view={view}"
        view_scores = getattr(output, "view_scores", None) or {}
        if len(view_scores) > 1:
            parts = ",".join(f"{k}={v:.3f}" for k, v in sorted(view_scores.items()))
            reason += f" tiles=[{parts}]"
        if getattr(output, "views_disagree", False):
            reason += " tiles_disagree"
        return detector, LayerScore(
            name="global",
            status="ran",
            score=float(output.probability),
            signal=output.decision == "generated",
            reason=reason,
        )
