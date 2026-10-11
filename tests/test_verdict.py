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
