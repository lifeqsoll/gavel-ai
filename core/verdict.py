"""Aggregate an EvidenceProtocol into a verdict.

When the protocol carries a learned detector result, the verdict is the
detector decision (generation signal, inconclusive or no signal) at the calibrated thresholds.
Symbolic evidence is listed as explanation and does not change that label.

Without a detector result, the verdict falls back to a heuristic anomaly score
built from symbolic evidence. That score is not a calibrated probability.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.registry import EvidenceProtocol, Severity

# Fallback only: the score is a heuristic, not a calibrated probability.
# Evidence weights combine as 1 - prod(1 - w), so independent findings add up
# without exceeding 1.0.


@dataclass(frozen=True)
class VerdictConfig:
    critical_weight: float = 0.6
    warning_weight: float = 0.25
    anomalous_threshold: float = 0.5


@dataclass(frozen=True)
class Verdict:
    score: float
    label: str
    checks_run: int
    checks_total: int
    findings: list[str] = field(default_factory=list)
    probability: float | None = None
    decision: str | None = None

    @property
    def coverage_note(self) -> str:
        return f"{self.checks_run} of {self.checks_total} checks ran"


LABEL_NOT_ANALYZED = "Not analyzed (no check ran)"
LABEL_CLEAN = "No anomalies found"
LABEL_CLEAN_PARTIAL = "No anomalies found (limited coverage)"
LABEL_SUSPICIOUS = "Suspicious"
LABEL_ANOMALOUS = "Anomalous"

# The learned detector is not reliable enough to name a photo as AI-generated, because it
# flags some real photos with a polished composition. The labels describe a signal only.
LABEL_GENERATION_SIGNAL = "Possible generation signal (verify manually)"
LABEL_INCONCLUSIVE = "Inconclusive"
LABEL_NO_SIGNAL = "No generation signal"

_DECISION_LABEL = {
    "generated": LABEL_GENERATION_SIGNAL,
    "uncertain": LABEL_INCONCLUSIVE,
    "real": LABEL_NO_SIGNAL,
}


def compute_verdict(protocol: EvidenceProtocol, config: VerdictConfig | None = None) -> Verdict:
    config = config or VerdictConfig()

    remaining = 1.0
    findings: list[str] = []
    for evidence in protocol.evidences:
        weight = config.critical_weight if evidence.severity is Severity.CRITICAL else config.warning_weight
        remaining *= 1.0 - weight
        findings.append(f"{evidence.evidence_id} [{evidence.severity.value}] {evidence.description}")
    score = round(1.0 - remaining, 4)

    checks_total = len(protocol.checks)
    checks_run = sum(1 for check in protocol.checks if check.status != "skipped")

    if protocol.detector is not None:
        return Verdict(
            score=score,
            label=_DECISION_LABEL[protocol.detector.decision],
            checks_run=checks_run,
            checks_total=checks_total,
            findings=findings,
            probability=protocol.detector.probability,
            decision=protocol.detector.decision,
        )

    if checks_run == 0:
        label = LABEL_NOT_ANALYZED
    elif not protocol.evidences:
        label = LABEL_CLEAN if checks_run == checks_total else LABEL_CLEAN_PARTIAL
    elif score >= config.anomalous_threshold:
        label = LABEL_ANOMALOUS
    else:
        label = LABEL_SUSPICIOUS

    return Verdict(
        score=score,
        label=label,
        checks_run=checks_run,
        checks_total=checks_total,
        findings=findings,
    )
