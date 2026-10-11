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
from core.observations import HandObservation, LightObservation, Segment
from core.registry import CheckRecord, DetectorResult, Evidence, EvidenceProtocol, ImageMetadata
from core.validators.anatomy import AnatomyValidator
from core.validators.perspective import PerspectiveValidator
from core.validators.shadows import ShadowValidator
from visualizer.overlay import render_overlay

logger = logging.getLogger(__name__)

_STAGE_ERRORS = (RuntimeError, NotImplementedError, ValueError)


class GeometryExtractor(Protocol):
    def extract(self, image: np.ndarray) -> list[Segment]: ...


class AnatomyExtractor(Protocol):
    def extract(self, image: np.ndarray) -> list[HandObservation]: ...


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
    learned_detector: DetectorScorer | None = None

    def analyze(self, image: np.ndarray, image_name: str = "unknown") -> PipelineResult:
        if image.ndim not in (2, 3):
            raise ValueError(f"expected a 2D or 3D image array, got ndim={image.ndim}")
        height, width = image.shape[:2]
        size = (width, height)
        checks: list[CheckRecord] = []
        evidences: list[Evidence] = []

        self._run_geometry(image, size, checks, evidences)
        self._run_anatomy(image, checks, evidences)
        self._run_lighting(image, size, checks, evidences)
        detector = self._run_detector(image, checks)

        protocol = EvidenceProtocol(
            image=ImageMetadata(file_name=image_name, width=width, height=height),
            checks=checks,
            evidences=evidences,
            detector=detector,
        )
        return PipelineResult(protocol=protocol, annotated_image=render_overlay(image, protocol))

    def _run_geometry(
        self,
        image: np.ndarray,
        size: tuple[int, int],
        checks: list[CheckRecord],
        evidences: list[Evidence],
    ) -> None:
        name = "geometry.perspective"
        try:
            segments = self.geometry_agent.extract(image)
            report = self.perspective_validator.validate(segments, size)
        except _STAGE_ERRORS as exc:
            logger.warning("%s skipped: %s", name, exc)
            checks.append(CheckRecord(check=name, status="skipped", reason=str(exc)))
            return
        if report.skipped_reason:
            checks.append(CheckRecord(check=name, status="skipped", reason=report.skipped_reason))
            return
        evidences.extend(report.evidences)
        checks.append(
            CheckRecord(
                check=name,
                status="violation" if report.evidences else "passed",
                reason=f"segments={report.n_valid_segments}, vanishing_points={len(report.clusters)}",
            )
        )

    def _run_anatomy(self, image: np.ndarray, checks: list[CheckRecord], evidences: list[Evidence]) -> None:
        name = "anatomy.hands"
        try:
            hands = self.anatomy_agent.extract(image)
        except _STAGE_ERRORS as exc:
            logger.warning("%s skipped: %s", name, exc)
            checks.append(CheckRecord(check=name, status="skipped", reason=str(exc)))
            return
        if not hands:
            checks.append(CheckRecord(check=name, status="skipped", reason="no hands detected"))
            return
        report = self.anatomy_validator.validate(hands)
        evidences.extend(report.evidences)
        checks.append(
            CheckRecord(
                check=name,
                status="violation" if report.evidences else "passed",
                reason=f"hands={len(hands)}",
            )
        )

    def _run_lighting(
        self,
        image: np.ndarray,
        size: tuple[int, int],
        checks: list[CheckRecord],
        evidences: list[Evidence],
    ) -> None:
        name = "lighting.consistency"
        try:
            observations = self.light_agent.extract(image)
        except _STAGE_ERRORS as exc:
            logger.warning("%s skipped: %s", name, exc)
            checks.append(CheckRecord(check=name, status="skipped", reason=str(exc)))
            return
        if not observations:
            checks.append(CheckRecord(check=name, status="skipped", reason="no shadow objects detected"))
            return
        report = self.shadow_validator.validate(observations, size)
        if report.skipped_reason:
            checks.append(CheckRecord(check=name, status="skipped", reason=report.skipped_reason))
            return
        evidences.extend(report.evidences)
        checks.append(
            CheckRecord(
                check=name,
                status="violation" if report.evidences else "passed",
                reason=f"rays={report.n_rays}",
            )
        )

    def _run_detector(self, image: np.ndarray, checks: list[CheckRecord]) -> DetectorResult | None:
        if self.learned_detector is None:
            return None
        name = "detector.learned"
        try:
            output = self.learned_detector.score(image)
        except _STAGE_ERRORS as exc:
            logger.warning("%s skipped: %s", name, exc)
            checks.append(CheckRecord(check=name, status="skipped", reason=str(exc)))
            return None
        return DetectorResult(
            model=output.model,
            probability=output.probability,
            logit=output.logit,
            decision=output.decision,
            high_threshold=output.thresholds.high,
            low_threshold=output.thresholds.low,
        )
