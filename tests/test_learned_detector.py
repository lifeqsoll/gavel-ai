import json

import numpy as np
import pytest
from PIL import Image

from core.learned_detector import (
    DEFAULT_THRESHOLDS,
    MODEL_ID,
    DetectorOutput,
    DetectorThresholds,
    LearnedDetector,
)
from core.normalize import NORMALIZE_CROP, normalize_pil
from core.pipeline import GavelPipeline
from core.registry import (
    AnomalyType,
    CheckRecord,
    DetectorResult,
    Evidence,
    EvidenceProtocol,
    GeometryPayload,
    ImageMetadata,
    Severity,
)
from core.verdict import (
    LABEL_GENERATION_SIGNAL,
    LABEL_INCONCLUSIVE,
    LABEL_NO_SIGNAL,
    compute_verdict,
)

THRESHOLDS = DetectorThresholds(high=0.42, low=0.02)


@pytest.mark.parametrize(
    ("probability", "decision"),
    [(0.99, "generated"), (0.42, "generated"), (0.41, "uncertain"), (0.02, "uncertain"), (0.019, "real"), (0.0, "real")],
)
def test_decision_boundaries(probability, decision):
    assert THRESHOLDS.decide(probability) == decision


def test_thresholds_must_be_ordered():
    with pytest.raises(ValueError):
        DetectorThresholds(high=0.1, low=0.5)


def test_detector_loads_weights_and_thresholds(tmp_path):
    weights = tmp_path / "w.npz"
    np.savez(weights, coef=np.zeros(768, np.float32), intercept=np.float32(0.0))
    thresholds = tmp_path / "t.json"
    thresholds.write_text(json.dumps({"high": 0.5, "low": 0.01}), encoding="utf-8")

    detector = LearnedDetector(weights_path=weights, thresholds_path=thresholds)
    assert detector.thresholds == DetectorThresholds(high=0.5, low=0.01)


def test_detector_requires_thresholds_file(tmp_path):
    weights = tmp_path / "w.npz"
    np.savez(weights, coef=np.zeros(768, np.float32), intercept=np.float32(0.0))
    with pytest.raises(FileNotFoundError):
        LearnedDetector(weights_path=weights, thresholds_path=tmp_path / "missing.json")


def test_shipped_thresholds_are_valid():
    raw = json.loads(DEFAULT_THRESHOLDS.read_text(encoding="utf-8"))
    DetectorThresholds(high=raw["high"], low=raw["low"])


def test_normalization_produces_fixed_size_jpeg_round_trip():
    img = Image.new("RGB", (1024, 768), color=(120, 80, 40))
    out = normalize_pil(img)
    assert out.size == (NORMALIZE_CROP, NORMALIZE_CROP)
    assert out.mode == "RGB"


def test_wide_image_adds_left_right_views():
    wide = np.zeros((200, 400, 3), dtype=np.uint8)
    names = [name for name, _tile, _box in LearnedDetector._views(wide)]
    assert names == ["full", "left", "right"]
    tall = np.zeros((400, 200, 3), dtype=np.uint8)
    assert [name for name, _tile, _box in LearnedDetector._views(tall)] == ["full"]


def _fake_detection(decision: str) -> DetectorResult:
    probability = {"generated": 0.9, "uncertain": 0.1, "real": 0.001}[decision]
    return DetectorResult(
        model=MODEL_ID,
        probability=probability,
        logit=0.0,
        decision=decision,
        high_threshold=0.42,
        low_threshold=0.02,
    )


@pytest.mark.parametrize(
    ("decision", "label"),
    [
        ("generated", LABEL_GENERATION_SIGNAL),
        ("uncertain", LABEL_INCONCLUSIVE),
        ("real", LABEL_NO_SIGNAL),
    ],
)
def test_verdict_follows_detector_decision(decision, label):
    protocol = EvidenceProtocol(
        image=ImageMetadata(file_name="t.png", width=10, height=10),
        checks=[CheckRecord(check="geometry.perspective", status="passed")],
        detector=_fake_detection(decision),
    )
    verdict = compute_verdict(protocol)
    assert verdict.label == label
    assert verdict.decision == decision
    assert verdict.probability == pytest.approx(_fake_detection(decision).probability)


def test_symbolic_findings_do_not_change_detector_label():
    evidence = Evidence(
        evidence_id="GEO-01",
        anomaly_type=AnomalyType.GEOMETRY_PERSPECTIVE_VIOLATION,
        severity=Severity.CRITICAL,
        description="synthetic",
        geometry=GeometryPayload(image_diagonal=100.0),
    )
    protocol = EvidenceProtocol(
        image=ImageMetadata(file_name="t.png", width=10, height=10),
        checks=[CheckRecord(check="anatomy.hands", status="violation")],
        evidences=[evidence],
        detector=_fake_detection("real"),
    )
    verdict = compute_verdict(protocol)
    assert verdict.label == LABEL_NO_SIGNAL
    assert len(verdict.findings) == 1


class _FakeDetector:
    def __init__(self, decision: str):
        self._decision = decision

    def score(self, image: np.ndarray) -> DetectorOutput:
        probability = {"generated": 0.9, "uncertain": 0.1, "real": 0.001}[self._decision]
        return DetectorOutput(
            probability=probability,
            logit=float(np.log(probability / (1 - probability))),
            decision=self._decision,
            thresholds=THRESHOLDS,
        )


class _BrokenDetector:
    def score(self, image: np.ndarray) -> DetectorOutput:
        raise RuntimeError("open_clip and torch are required for the learned detector")


def _noise_image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8)


def test_pipeline_attaches_detector_result():
    pipeline = GavelPipeline(learned_detector=_FakeDetector("generated"))
    result = pipeline.analyze(_noise_image(), image_name="noise.png")
    detector = result.protocol.detector
    assert detector is not None
    assert detector.decision == "generated"
    assert detector.high_threshold == THRESHOLDS.high
    assert compute_verdict(result.protocol).label == LABEL_GENERATION_SIGNAL


def test_pipeline_falls_back_when_detector_fails():
    pipeline = GavelPipeline(learned_detector=_BrokenDetector())
    result = pipeline.analyze(_noise_image(), image_name="noise.png")
    assert result.protocol.detector is None
    skipped = [c for c in result.protocol.checks if c.check == "detector.learned"]
    assert skipped and skipped[0].status == "skipped"
    assert compute_verdict(result.protocol).probability is None


def test_pipeline_without_detector_keeps_heuristic_path():
    result = GavelPipeline().analyze(_noise_image(), image_name="noise.png")
    assert result.protocol.detector is None
    assert all(c.check != "detector.learned" for c in result.protocol.checks)
