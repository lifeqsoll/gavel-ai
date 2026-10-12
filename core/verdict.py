"""Aggregate an EvidenceProtocol into a verdict.

Fusion A (layered classifier):
    L0 global detector decision OR any layer with signal=True -> generation signal.
    L0 real and no layer signal -> no signal.
    Otherwise inconclusive.

Anatomy hand veto: a large clean hand map (no face, no hand WARNING) can suppress an
L0-only `generated` vote. Stock close-up hands are a known CLIP FP; AI portraits keep
faces so they are not vetoed. Does not suppress artifacts/geometry votes.

Each layer sets `signal` with its own threshold (models/layer_thresholds.json):
    L0 ~0.596 (calibrated), artifacts 0.7 (flat-kurtosis), anatomy/physics 0.5,
    geometry votes on CRITICAL perspective evidence.
Symbolic evidence is listed as explanation. Layer scores are on protocol.layers.
Without a detector result, falls back to the heuristic anomaly score.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from core.registry import EvidenceProtocol, LayerScore, Severity

# Fallback only: the score is a heuristic, not a calibrated probability.
# Evidence weights combine as 1 - prod(1 - w), so independent findings add up
# without exceeding 1.0.


@dataclass(frozen=True)
class VerdictConfig:
    critical_weight: float = 0.6
    warning_weight: float = 0.25
    anomalous_threshold: float = 0.5
    # Suppress L0-only generation when anatomy shows a clean dominant hand (no face).
    anatomy_veto_max_score: float = 0.1
    anatomy_veto_min_hand_span: float = 0.12


@dataclass(frozen=True)
class Verdict:
    score: float
    label: str
    checks_run: int
    checks_total: int
    findings: list[str] = field(default_factory=list)
    probability: float | None = None
    decision: str | None = None
    high_threshold: float | None = None
    fusion: str | None = None

    @property
    def coverage_note(self) -> str:
        return f"{self.checks_run} of {self.checks_total} checks ran"


LABEL_NOT_ANALYZED = "Not analyzed (no check ran)"
LABEL_CLEAN = "No anomalies found"
LABEL_CLEAN_PARTIAL = "No anomalies found (limited coverage)"
LABEL_SUSPICIOUS = "Suspicious"
LABEL_ANOMALOUS = "Anomalous"

LABEL_GENERATION_SIGNAL = "Possible generation signal (verify manually)"
LABEL_INCONCLUSIVE = "Inconclusive"
LABEL_NO_SIGNAL = "No generation signal"

_DECISION_LABEL = {
    "generated": LABEL_GENERATION_SIGNAL,
    "uncertain": LABEL_INCONCLUSIVE,
    "real": LABEL_NO_SIGNAL,
}


def _layer_list(protocol: EvidenceProtocol) -> list[LayerScore]:
    layers = protocol.layers
    return [
        layers.global_layer,
        layers.anatomy,
        layers.artifacts,
        layers.geometry,
        layers.physics,
    ]


def _layer_votes(layer: LayerScore) -> bool:
    """Vote only when the layer itself raised signal (per-layer threshold already applied)."""
    return layer.status == "ran" and bool(layer.signal)


def _parse_hand_span(reason: str) -> float:
    match = re.search(r"hand_span=([0-9.]+)", reason or "")
    return float(match.group(1)) if match else 0.0


def _parse_face_count(reason: str) -> int:
    match = re.search(r"face=(\d+)", reason or "")
    return int(match.group(1)) if match else -1


def _parse_hand_count(reason: str) -> int:
    match = re.search(r"hands=(\d+)", reason or "")
    return int(match.group(1)) if match else 0


def _metric_float(metrics: dict[str, float], key: str) -> float | None:
    if key not in metrics:
        return None
    return float(metrics[key])


def _anatomy_hand_context(
    protocol: EvidenceProtocol, config: VerdictConfig
) -> tuple[bool, float, float]:
    """Return (eligible_face_free_hand, hand_span, face_count) for veto helpers."""
    anatomy = protocol.layers.anatomy
    if anatomy.status != "ran" or anatomy.score is None:
        return False, 0.0, 0.0
    if anatomy.score > config.anatomy_veto_max_score or anatomy.signal:
        return False, 0.0, 0.0

    metrics = anatomy.metrics or {}
    hand_n = _metric_float(metrics, "hand_count")
    if hand_n is None:
        hand_n = float(_parse_hand_count(anatomy.reason))
    if hand_n < 1:
        return False, 0.0, 0.0

    face_n = _metric_float(metrics, "face_count")
    if face_n is None:
        parsed = _parse_face_count(anatomy.reason)
        if parsed < 0:
            amap = protocol.anatomy_map
            face_n = float(len(amap.face_landmarks_px) if amap else 0)
        else:
            face_n = float(parsed)
    if face_n > 0:
        return False, 0.0, face_n

    hand_span = _metric_float(metrics, "hand_span")
    if hand_span is None:
        hand_span = _parse_hand_span(anatomy.reason)
    if hand_span < config.anatomy_veto_min_hand_span:
        return False, hand_span, face_n
    return True, hand_span, face_n


def anatomy_hand_vetoes_l0(protocol: EvidenceProtocol, config: VerdictConfig) -> bool:
    """True when a clean, face-free hand close-up should override L0 alone.

    AI portraits almost always have a face mesh, so they keep the L0 vote.
    Prefers LayerScore.metrics; falls back to reason-string parsing for older protocols.
    """
    eligible, _, _ = _anatomy_hand_context(protocol, config)
    return eligible


def _hand_crop_min(protocol: EvidenceProtocol) -> float | None:
    detector = protocol.detector
    if detector is None:
        return None
    hand_scores = [v for k, v in detector.view_scores.items() if k.startswith("hand_")]
    if hand_scores:
        return float(min(hand_scores))
    return _metric_float(protocol.layers.global_layer.metrics or {}, "hand_crop_min")


def hand_crop_vetoes_l0(protocol: EvidenceProtocol, config: VerdictConfig) -> bool:
    """True when an L0 hand-crop probe looks real while full-frame L0 says generated.

    Extra detector feature (not a symbolic layer): MediaPipe hand bbox re-scored by CLIP.
    Face-free hand close-ups only (AI portraits keep the L0 vote).
    """
    detector = protocol.detector
    if detector is None or detector.decision != "generated":
        return False
    eligible, _, _ = _anatomy_hand_context(protocol, config)
    if not eligible:
        return False
    hand_min = _hand_crop_min(protocol)
    if hand_min is None:
        return False
    return hand_min < detector.low_threshold


def hand_crop_disagrees_l0(protocol: EvidenceProtocol, config: VerdictConfig) -> bool:
    """Full-frame L0 generated but hand crop firmly real (often selfie+hands with a face).

    Does not claim real — fusion should force uncertain when no hard layer votes.
    Gap mirrors tiles_disagree (0.45) so AI faces with weird hands rarely trigger.
    """
    detector = protocol.detector
    if detector is None or detector.decision != "generated":
        return False
    anatomy = protocol.layers.anatomy
    if anatomy.status != "ran" or anatomy.signal:
        return False
    metrics = anatomy.metrics or {}
    hand_n = _metric_float(metrics, "hand_count")
    if hand_n is None:
        hand_n = float(_parse_hand_count(anatomy.reason))
    if hand_n < 1:
        return False
    hand_span = _metric_float(metrics, "hand_span")
    if hand_span is None:
        hand_span = _parse_hand_span(anatomy.reason)
    if hand_span < config.anatomy_veto_min_hand_span:
        return False
    hand_min = _hand_crop_min(protocol)
    if hand_min is None or hand_min >= detector.low_threshold:
        return False
    return (detector.probability - hand_min) >= 0.45


# Soft cues: useful with L0 uncertain/generated, but alone they FP on real night/texture
# photos (train platforms, cobblestone). Cannot overturn a calibrated L0_real.
_SOFT_LAYER_VOTES = frozenset({"artifacts"})


def fuse_layers(
    protocol: EvidenceProtocol, config: VerdictConfig | None = None
) -> tuple[str, str, str]:
    """Return (label, decision, fusion_reason) for fusion A."""
    config = config or VerdictConfig()
    detector = protocol.detector
    layer_hits = [
        layer.name
        for layer in _layer_list(protocol)
        if _layer_votes(layer) and layer.name != "global"
    ]
    hard_hits = [name for name in layer_hits if name not in _SOFT_LAYER_VOTES]
    soft_hits = [name for name in layer_hits if name in _SOFT_LAYER_VOTES]
    l0_generated = detector is not None and detector.decision == "generated"
    l0_real = detector is not None and detector.decision == "real"
    if l0_real:
        # Soft cues cannot overturn calibrated L0_real.
        layer_hits = list(hard_hits)
    map_veto = l0_generated and anatomy_hand_vetoes_l0(protocol, config) and not layer_hits
    crop_veto = l0_generated and hand_crop_vetoes_l0(protocol, config) and not layer_hits
    crop_disagree = (
        l0_generated and hand_crop_disagrees_l0(protocol, config) and not hard_hits
    )
    veto = map_veto or crop_veto
    tiles_disagree = bool(detector is not None and detector.views_disagree)

    if detector is None:
        if layer_hits:
            return LABEL_GENERATION_SIGNAL, "generated", "layers:" + ",".join(layer_hits)
        return LABEL_INCONCLUSIVE, "uncertain", "no_detector"

    if crop_veto and map_veto:
        return LABEL_NO_SIGNAL, "real", "anatomy_hand_veto+L0_hand_crop"
    if crop_veto:
        return LABEL_NO_SIGNAL, "real", "L0_hand_crop_real"
    if map_veto:
        return LABEL_NO_SIGNAL, "real", "anatomy_hand_veto"

    # Wide diptychs OR full-vs-hand-crop clash: do not trust max L0 alone.
    if tiles_disagree and not hard_hits and not veto:
        return LABEL_INCONCLUSIVE, "uncertain", "L0_tiles_disagree"
    if crop_disagree and not veto:
        return LABEL_INCONCLUSIVE, "uncertain", "L0_hand_crop_disagree"

    if l0_generated or layer_hits:
        reasons = []
        if l0_generated:
            reasons.append("L0")
        if layer_hits:
            reasons.append("layers:" + ",".join(layer_hits))
        return LABEL_GENERATION_SIGNAL, "generated", "+".join(reasons)

    # Plastic face skin (very low Laplacian) + L0 uncertain → generation lean.
    # Extreme threshold (lap<20) calibrated on ai_08 (lap≈5); avoids normal photos.
    # H11 expands plastic via face clipping in pipeline metrics.
    if (
        detector is not None
        and detector.decision == "uncertain"
        and not hard_hits
        and _plastic_skin_cue(protocol)
    ):
        return LABEL_GENERATION_SIGNAL, "generated", "L0_uncertain+plastic_skin"

    # Painting|AI diptychs: halves diverge; L0 may be real OR uncertain after probes.
    halves = _halves_composite_cue(protocol)
    if halves and not hard_hits and detector.decision in {"real", "uncertain"}:
        face_n = _face_count_for_fusion(protocol)
        if face_n > 0:
            return LABEL_GENERATION_SIGNAL, "generated", f"L0_{detector.decision}+halves_composite"
        if l0_real:
            return LABEL_INCONCLUSIVE, "uncertain", "L0_real_halves_disagree"

    # H10: over-coherent straight edges + L0 uncertain → generation lean.
    if (
        detector is not None
        and detector.decision == "uncertain"
        and not hard_hits
        and _edge_coherent_cue(protocol)
    ):
        return LABEL_GENERATION_SIGNAL, "generated", "L0_uncertain+edge_coherent"

    if l0_real and not layer_hits:
        if soft_hits:
            return LABEL_NO_SIGNAL, "real", "L0_real_soft_ignored:" + ",".join(soft_hits)
        return LABEL_NO_SIGNAL, "real", "L0_real"

    return LABEL_INCONCLUSIVE, "uncertain", "L0_" + detector.decision


def _face_count_for_fusion(protocol: EvidenceProtocol) -> int:
    anatomy = protocol.layers.anatomy
    face_n = _metric_float(anatomy.metrics or {}, "face_count")
    if face_n is not None:
        return int(face_n)
    parsed = _parse_face_count(anatomy.reason)
    if parsed >= 0:
        return parsed
    amap = protocol.anatomy_map
    return len(amap.face_landmarks_px) if amap else 0


def _halves_composite_cue(protocol: EvidenceProtocol) -> bool:
    metrics = protocol.layers.global_layer.metrics or {}
    flag = _metric_float(metrics, "halves_disagree")
    return bool(flag and flag >= 1.0)


def _plastic_skin_cue(protocol: EvidenceProtocol) -> bool:
    metrics = protocol.layers.global_layer.metrics or {}
    flag = _metric_float(metrics, "plastic_skin")
    return bool(flag and flag >= 1.0)


def _edge_coherent_cue(protocol: EvidenceProtocol) -> bool:
    metrics = protocol.layers.global_layer.metrics or {}
    flag = _metric_float(metrics, "edge_coherent")
    return bool(flag and flag >= 1.0)


def compute_verdict(protocol: EvidenceProtocol, config: VerdictConfig | None = None) -> Verdict:
    config = config or VerdictConfig()

    remaining = 1.0
    findings: list[str] = []
    for evidence in protocol.evidences:
        weight = config.critical_weight if evidence.severity is Severity.CRITICAL else config.warning_weight
        remaining *= 1.0 - weight
        findings.append(f"{evidence.evidence_id} [{evidence.severity.value}] {evidence.description}")
    for layer in _layer_list(protocol):
        if layer.status == "ran" and layer.score is not None:
            votes = _layer_votes(layer) and not (
                layer.name == "global"
                and (
                    anatomy_hand_vetoes_l0(protocol, config)
                    or hand_crop_vetoes_l0(protocol, config)
                )
            )
            findings.append(
                f"layer.{layer.name} [{('SIGNAL' if votes else 'ok')}] "
                f"score={layer.score:.3f} {layer.reason}"
            )
    score = round(1.0 - remaining, 4)

    checks_total = len(protocol.checks)
    checks_run = sum(1 for check in protocol.checks if check.status != "skipped")

    layers_ready = any(l.status == "ran" for l in _layer_list(protocol))
    if protocol.detector is not None or layers_ready:
        label, decision, fusion = fuse_layers(protocol, config)
        return Verdict(
            score=score,
            label=label,
            checks_run=checks_run,
            checks_total=checks_total,
            findings=findings,
            probability=protocol.detector.probability if protocol.detector else None,
            decision=decision,
            high_threshold=protocol.detector.high_threshold if protocol.detector else None,
            fusion=fusion,
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
