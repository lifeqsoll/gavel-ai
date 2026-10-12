import pytest

from core.registry import (
    AnomalyType,
    CheckRecord,
    Evidence,
    EvidenceProtocol,
    GeometryPayload,
    ImageMetadata,
    Severity,
)
from core.verdict import (
    LABEL_ANOMALOUS,
    LABEL_CLEAN,
    LABEL_CLEAN_PARTIAL,
    LABEL_NOT_ANALYZED,
    LABEL_SUSPICIOUS,
    compute_verdict,
)


def _evidence(index: int, severity: Severity) -> Evidence:
    return Evidence(
        evidence_id=f"GEO-{index:02d}",
        anomaly_type=AnomalyType.GEOMETRY_PERSPECTIVE_VIOLATION,
        severity=severity,
        description="synthetic",
        geometry=GeometryPayload(image_diagonal=100.0),
    )


def _protocol(checks: list[CheckRecord], evidences: list[Evidence]) -> EvidenceProtocol:
    return EvidenceProtocol(
        image=ImageMetadata(file_name="t.png", width=10, height=10),
        checks=checks,
        evidences=evidences,
    )


PASSED = [
    CheckRecord(check="geometry.perspective", status="passed"),
    CheckRecord(check="anatomy.hands", status="passed"),
    CheckRecord(check="lighting.consistency", status="passed"),
]


def test_clean_when_all_checks_pass():
    verdict = compute_verdict(_protocol(PASSED, []))
    assert verdict.score == 0.0
    assert verdict.label == LABEL_CLEAN
    assert verdict.checks_run == 3 and verdict.checks_total == 3


def test_partial_coverage_when_some_checks_skipped():
    checks = [PASSED[0], CheckRecord(check="anatomy.hands", status="skipped", reason="no hands")]
    verdict = compute_verdict(_protocol(checks, []))
    assert verdict.label == LABEL_CLEAN_PARTIAL
    assert verdict.coverage_note == "1 of 2 checks ran"


def test_not_analyzed_when_nothing_ran():
    checks = [CheckRecord(check="anatomy.hands", status="skipped", reason="x")]
    assert compute_verdict(_protocol(checks, [])).label == LABEL_NOT_ANALYZED


@pytest.mark.parametrize(
    ("severities", "expected_score", "expected_label"),
    [
        ([Severity.WARNING], 0.25, LABEL_SUSPICIOUS),
        ([Severity.CRITICAL], 0.6, LABEL_ANOMALOUS),
        ([Severity.WARNING, Severity.WARNING], 0.4375, LABEL_SUSPICIOUS),
        ([Severity.CRITICAL, Severity.CRITICAL], 0.84, LABEL_ANOMALOUS),
    ],
)
def test_score_combines_independent_evidence(severities, expected_score, expected_label):
    evidences = [_evidence(i + 1, sev) for i, sev in enumerate(severities)]
    verdict = compute_verdict(_protocol(PASSED, evidences))
    assert verdict.score == pytest.approx(expected_score, abs=1e-4)
    assert verdict.label == expected_label
    assert len(verdict.findings) == len(severities)


def test_score_never_exceeds_one():
    evidences = [_evidence(i + 1, Severity.CRITICAL) for i in range(12)]
    assert compute_verdict(_protocol(PASSED, evidences)).score <= 1.0


def test_fusion_a_hard_layer_beats_l0_real():
    """Geometry CRITICAL (hard) can overturn L0_real; soft artifacts cannot."""
    from core.registry import DetectorResult, LayerScore, LayerScores
    from core.verdict import LABEL_GENERATION_SIGNAL, fuse_layers

    protocol = EvidenceProtocol(
        image=ImageMetadata(file_name="t.png", width=10, height=10),
        checks=PASSED,
        evidences=[],
        detector=DetectorResult(
            model="test",
            probability=0.022,
            logit=-3.8,
            decision="real",
            high_threshold=0.596,
            low_threshold=0.066,
        ),
        layers=LayerScores(
            global_layer=LayerScore(name="global", status="ran", score=0.022, signal=False),
            geometry=LayerScore(
                name="geometry", status="ran", score=0.55, signal=True, reason="critical=1"
            ),
        ),
    )
    label, decision, fusion = fuse_layers(protocol)
    assert decision == "generated"
    assert label == LABEL_GENERATION_SIGNAL
    assert "geometry" in fusion


def test_anatomy_hand_veto_suppresses_l0_only_fp():
    from core.registry import DetectorResult, LayerScore, LayerScores
    from core.verdict import LABEL_NO_SIGNAL, fuse_layers

    protocol = EvidenceProtocol(
        image=ImageMetadata(file_name="hand.png", width=100, height=100),
        checks=PASSED,
        evidences=[],
        detector=DetectorResult(
            model="test",
            probability=0.996,
            logit=5.5,
            decision="generated",
            high_threshold=0.596,
            low_threshold=0.066,
        ),
        layers=LayerScores(
            global_layer=LayerScore(
                name="global", status="ran", score=0.996, signal=True, reason="decision=generated"
            ),
            anatomy=LayerScore(
                name="anatomy",
                status="ran",
                score=0.0,
                signal=False,
                reason="hand_warnings=0 hands=1 face=0 hand_span=0.194",
            ),
        ),
    )
    label, decision, fusion = fuse_layers(protocol)
    assert decision == "real"
    assert label == LABEL_NO_SIGNAL
    assert fusion == "anatomy_hand_veto"


def test_anatomy_hand_veto_reads_metrics_without_reason():
    """Fusion veto must work from LayerScore.metrics alone (no reason regex)."""
    from core.registry import DetectorResult, LayerScore, LayerScores
    from core.verdict import LABEL_NO_SIGNAL, fuse_layers

    protocol = EvidenceProtocol(
        image=ImageMetadata(file_name="hand.png", width=100, height=100),
        checks=PASSED,
        evidences=[],
        detector=DetectorResult(
            model="test",
            probability=0.99,
            logit=5.0,
            decision="generated",
            high_threshold=0.596,
            low_threshold=0.066,
        ),
        layers=LayerScores(
            global_layer=LayerScore(name="global", status="ran", score=0.99, signal=True),
            anatomy=LayerScore(
                name="anatomy",
                status="ran",
                score=0.0,
                signal=False,
                reason="",
                metrics={"hand_count": 1.0, "face_count": 0.0, "hand_span": 0.25},
            ),
        ),
    )
    _, decision, fusion = fuse_layers(protocol)
    assert decision == "real"
    assert fusion == "anatomy_hand_veto"


def test_artifacts_cannot_overturn_l0_real():
    from core.registry import DetectorResult, LayerScore, LayerScores
    from core.verdict import LABEL_NO_SIGNAL, fuse_layers

    protocol = EvidenceProtocol(
        image=ImageMetadata(file_name="train.png", width=100, height=100),
        checks=PASSED,
        evidences=[],
        detector=DetectorResult(
            model="test",
            probability=0.006,
            logit=-5.0,
            decision="real",
            high_threshold=0.596,
            low_threshold=0.066,
        ),
        layers=LayerScores(
            global_layer=LayerScore(name="global", status="ran", score=0.006, signal=False),
            artifacts=LayerScore(
                name="artifacts", status="ran", score=1.0, signal=True, reason="flat_kurtosis=91.8"
            ),
        ),
    )
    label, decision, fusion = fuse_layers(protocol)
    assert decision == "real"
    assert label == LABEL_NO_SIGNAL
    assert "L0_real" in fusion
    assert "artifacts" in fusion


def test_anatomy_hand_veto_does_not_kill_ai_face():
    from core.registry import DetectorResult, LayerScore, LayerScores
    from core.verdict import LABEL_GENERATION_SIGNAL, fuse_layers

    protocol = EvidenceProtocol(
        image=ImageMetadata(file_name="face.png", width=100, height=100),
        checks=PASSED,
        evidences=[],
        detector=DetectorResult(
            model="test",
            probability=0.99,
            logit=5.0,
            decision="generated",
            high_threshold=0.596,
            low_threshold=0.066,
        ),
        layers=LayerScores(
            global_layer=LayerScore(name="global", status="ran", score=0.99, signal=True),
            anatomy=LayerScore(
                name="anatomy",
                status="ran",
                score=0.0,
                signal=False,
                reason="hand_warnings=0 hands=0 face=478 hand_span=0.000",
                metrics={"hand_count": 0.0, "face_count": 478.0, "hand_span": 0.0},
            ),
        ),
    )
    _, decision, fusion = fuse_layers(protocol)
    assert decision == "generated"
    assert "L0" in fusion
    assert fusion != "anatomy_hand_veto"


def test_tiles_disagree_forces_uncertain_without_hard_votes():
    from core.registry import DetectorResult, LayerScore, LayerScores
    from core.verdict import LABEL_INCONCLUSIVE, fuse_layers

    protocol = EvidenceProtocol(
        image=ImageMetadata(file_name="diptych.png", width=200, height=100),
        checks=PASSED,
        evidences=[],
        detector=DetectorResult(
            model="test",
            probability=0.92,
            logit=2.4,
            decision="generated",
            high_threshold=0.596,
            low_threshold=0.066,
            views_disagree=True,
            view_scores={"full": 0.1, "left": 0.92, "right": 0.05},
        ),
        layers=LayerScores(
            global_layer=LayerScore(
                name="global",
                status="ran",
                score=0.92,
                signal=True,
                reason="decision=generated tiles_disagree",
                metrics={"views_disagree": 1.0},
            ),
            artifacts=LayerScore(
                name="artifacts", status="ran", score=1.0, signal=True, reason="soft"
            ),
        ),
    )
    label, decision, fusion = fuse_layers(protocol)
    assert decision == "uncertain"
    assert label == LABEL_INCONCLUSIVE
    assert fusion == "L0_tiles_disagree"


def test_tiles_disagree_yields_to_hard_geometry():
    from core.registry import DetectorResult, LayerScore, LayerScores
    from core.verdict import LABEL_GENERATION_SIGNAL, fuse_layers

    protocol = EvidenceProtocol(
        image=ImageMetadata(file_name="diptych.png", width=200, height=100),
        checks=PASSED,
        evidences=[],
        detector=DetectorResult(
            model="test",
            probability=0.1,
            logit=-2.0,
            decision="real",
            high_threshold=0.596,
            low_threshold=0.066,
            views_disagree=True,
        ),
        layers=LayerScores(
            global_layer=LayerScore(name="global", status="ran", score=0.1, signal=False),
            geometry=LayerScore(
                name="geometry", status="ran", score=0.55, signal=True, reason="critical=1"
            ),
        ),
    )
    label, decision, fusion = fuse_layers(protocol)
    assert decision == "generated"
    assert label == LABEL_GENERATION_SIGNAL
    assert "geometry" in fusion


def test_soft_artifacts_evidence_does_not_overturn_l0_real():
    """Artifacts may appear as Evidence WARNING but remain soft vs L0_real."""
    from core.registry import (
        ArtifactsPayload,
        DetectorResult,
        LayerScore,
        LayerScores,
    )
    from core.verdict import LABEL_NO_SIGNAL, compute_verdict

    evidences = [
        Evidence(
            evidence_id="ART-01",
            anomaly_type=AnomalyType.ARTIFACTS_RESIDUAL,
            severity=Severity.WARNING,
            description="soft cue",
            metrics={"score": 1.0, "flat_kurtosis": 90.0, "residual_energy": 10.0},
            artifacts=ArtifactsPayload(flat_kurtosis=90.0, residual_energy=10.0, score=1.0),
        )
    ]
    protocol = EvidenceProtocol(
        image=ImageMetadata(file_name="train.png", width=100, height=100),
        checks=PASSED,
        evidences=evidences,
        detector=DetectorResult(
            model="test",
            probability=0.006,
            logit=-5.0,
            decision="real",
            high_threshold=0.596,
            low_threshold=0.066,
        ),
        layers=LayerScores(
            global_layer=LayerScore(name="global", status="ran", score=0.006, signal=False),
            artifacts=LayerScore(
                name="artifacts", status="ran", score=1.0, signal=True, reason="flat_kurtosis=90"
            ),
        ),
    )
    verdict = compute_verdict(protocol)
    assert verdict.decision == "real"
    assert verdict.label == LABEL_NO_SIGNAL
    assert "L0_real" in (verdict.fusion or "")
    assert "artifacts" in (verdict.fusion or "")
