"""Typed data schemas (Pydantic v2) for Gavel AI evidence aggregation.

Structure passed to the text LLM:

    EvidenceProtocol
    ├── image: ImageMetadata
    ├── checks: CheckRecord[]      (which checks ran and their status)
    ├── evidences: Evidence[]
    │   └── payload                (geometry | anatomy | lighting, matching the anomaly type)
    └── detector: DetectorResult?  (learned probability, decision and thresholds)
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROTOCOL_VERSION = "1.4.1"

Point2D = tuple[float, float]
BBox2D = tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels


class _Schema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"


class AnomalyType(str, Enum):
    GEOMETRY_PERSPECTIVE_VIOLATION = "GEOMETRY_PERSPECTIVE_VIOLATION"
    ANATOMY_HAND_DEFORMATION = "ANATOMY_HAND_DEFORMATION"
    LIGHT_CONSISTENCY_VIOLATION = "LIGHT_CONSISTENCY_VIOLATION"
    ARTIFACTS_RESIDUAL = "ARTIFACTS_RESIDUAL"


class LineSegment(_Schema):
    p1: Point2D
    p2: Point2D


class CrossingPair(_Schema):
    """Two segments meeting at an erroneous crossing point."""

    segment_a: LineSegment
    segment_b: LineSegment
    crossing: Point2D


class GeometryPayload(_Schema):
    vanishing_point: Point2D | None = None
    cluster_segments: list[LineSegment] = Field(default_factory=list)
    crossings: list[CrossingPair] = Field(default_factory=list)
    image_diagonal: float = Field(gt=0)


class FlaggedJoint(_Schema):
    hand_index: int = Field(ge=0)
    landmark_index: int = Field(ge=0, le=20)
    joint_name: str
    reason: Literal["angle", "proportion", "adjacent_angle"]
    severity: Severity
    bbox: BBox2D


class AnatomyPayload(_Schema):
    hand_index: int = Field(ge=0)
    handedness: str
    flagged_joints: list[FlaggedJoint] = Field(default_factory=list)
    landmarks_px: list[Point2D] = Field(default_factory=list)


class ShadowRay(_Schema):
    object_id: str
    bbox: BBox2D
    base: Point2D
    top_vertex: Point2D
    shadow_tip: Point2D


class LightingPayload(_Schema):
    estimated_light: Point2D | None = None
    directional: bool = False
    rays: list[ShadowRay] = Field(default_factory=list)
    pairwise_intersections: list[Point2D] = Field(default_factory=list)
    image_diagonal: float = Field(gt=0)


class ArtifactsPayload(_Schema):
    """Soft cue metrics from flat-region residual kurtosis (no spatial marks)."""

    flat_kurtosis: float
    residual_energy: float
    score: float = Field(ge=0.0, le=1.0)


_PAYLOAD_FIELD: dict[AnomalyType, str] = {
    AnomalyType.GEOMETRY_PERSPECTIVE_VIOLATION: "geometry",
    AnomalyType.ANATOMY_HAND_DEFORMATION: "anatomy",
    AnomalyType.LIGHT_CONSISTENCY_VIOLATION: "lighting",
    AnomalyType.ARTIFACTS_RESIDUAL: "artifacts",
}


class Evidence(_Schema):
    """A single anomaly with its status, exact metrics, description and payload."""

    evidence_id: str
    anomaly_type: AnomalyType
    severity: Severity
    description: str
    metrics: dict[str, float] = Field(default_factory=dict)
    geometry: GeometryPayload | None = None
    anatomy: AnatomyPayload | None = None
    lighting: LightingPayload | None = None
    artifacts: ArtifactsPayload | None = None

    @field_validator("metrics")
    @classmethod
    def _metrics_are_finite(cls, value: dict[str, float]) -> dict[str, float]:
        bad = [key for key, val in value.items() if not math.isfinite(val)]
        if bad:
            raise ValueError(f"Non-finite metrics: {bad}")
        return value

    @model_validator(mode="after")
    def _payload_matches_anomaly(self) -> Evidence:
        expected = _PAYLOAD_FIELD[self.anomaly_type]
        for name in ("geometry", "anatomy", "lighting", "artifacts"):
            present = getattr(self, name) is not None
            if present != (name == expected):
                raise ValueError(f"{self.anomaly_type.value} requires only the '{expected}' payload")
        return self


class ImageMetadata(_Schema):
    file_name: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    analyzed_at_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CheckRecord(_Schema):
    """Records that a check ran, so the LLM can tell 'not found' from 'not checked'."""

    check: str
    status: Literal["passed", "violation", "skipped"]
    reason: str = ""


Decision = Literal["generated", "uncertain", "real"]


class DetectorResult(_Schema):
    """Output of the learned detector, with the thresholds that produced the decision."""

    model: str
    probability: float = Field(ge=0.0, le=1.0)
    logit: float
    decision: Decision
    high_threshold: float = Field(ge=0.0, le=1.0)
    low_threshold: float = Field(ge=0.0, le=1.0)
    view: str = "full"
    view_scores: dict[str, float] = Field(default_factory=dict)
    views_disagree: bool = False
    # Pixel boxes for scored views (full/left/right/hand_*), for overlay only.
    view_boxes: dict[str, BBox2D] = Field(default_factory=dict)


class HandMap(_Schema):
    handedness: str
    landmarks_px: list[Point2D] = Field(default_factory=list)


class AnatomyMap(_Schema):
    """Landmark map for the overlay only. Not evidence and not part of the verdict."""

    face_landmarks_px: list[Point2D] = Field(default_factory=list)
    pose_landmarks_px: list[Point2D | None] = Field(default_factory=list)
    hands: list[HandMap] = Field(default_factory=list)


class LayerScore(_Schema):
    """One classifier layer: score is a generation signal in [0, 1], not a probability."""

    name: Literal["global", "anatomy", "artifacts", "geometry", "physics"]
    status: Literal["ran", "skipped"] = "skipped"
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    signal: bool = False
    reason: str = ""
    # Machine-readable layer context for fusion (hand_span, face_count, n_critical, …).
    metrics: dict[str, float] = Field(default_factory=dict)

    @field_validator("metrics")
    @classmethod
    def _layer_metrics_are_finite(cls, value: dict[str, float]) -> dict[str, float]:
        bad = [key for key, val in value.items() if not math.isfinite(val)]
        if bad:
            raise ValueError(f"Non-finite layer metrics: {bad}")
        return value


class LayerScores(_Schema):
    global_layer: LayerScore = Field(default_factory=lambda: LayerScore(name="global"))
    anatomy: LayerScore = Field(default_factory=lambda: LayerScore(name="anatomy"))
    artifacts: LayerScore = Field(default_factory=lambda: LayerScore(name="artifacts"))
    geometry: LayerScore = Field(default_factory=lambda: LayerScore(name="geometry"))
    physics: LayerScore = Field(default_factory=lambda: LayerScore(name="physics"))


class EvidenceProtocol(_Schema):
    protocol_version: str = PROTOCOL_VERSION
    image: ImageMetadata
    checks: list[CheckRecord] = Field(default_factory=list)
    evidences: list[Evidence] = Field(default_factory=list)
    detector: DetectorResult | None = None
    anatomy_map: AnatomyMap | None = None
    layers: LayerScores = Field(default_factory=LayerScores)

    @property
    def has_critical(self) -> bool:
        return any(e.severity is Severity.CRITICAL for e in self.evidences)

    def to_json(self, indent: int | None = 2) -> str:
        return self.model_dump_json(indent=indent)
