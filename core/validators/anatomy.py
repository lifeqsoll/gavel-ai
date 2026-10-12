"""Hand anatomy validator.

Evidence uses 2D image-plane rules only (calibrated against real curled hands):
    1. Non-adjacent finger bones that properly intersect in the image -> WARNING.
    2. At least `min_bone_crossings` such intersections on one hand.

Legacy 3D flexion/proportion checks remain in `analyze_hand` for unit tests and debug
(`emit_legacy_3d_evidence=True`). They are off by default because MediaPipe world
landmarks fire on normal hands (real_hand_gym: 24 flags).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

from core.observations import HandObservation
from core.registry import (
    AnatomyPayload,
    AnomalyType,
    Evidence,
    FlaggedJoint,
    Severity,
)

# Approximate average adult phalanx lengths in millimeters.
# Only the ratios are used as norms; values are configurable.
ANATOMICAL_PHALANX_MM: dict[str, dict[str, float]] = {
    "thumb": {"proximal": 32.0, "distal": 25.0},
    "index": {"proximal": 45.0, "middle": 25.0, "distal": 17.0},
    "middle": {"proximal": 50.0, "middle": 30.0, "distal": 19.0},
    "ring": {"proximal": 46.0, "middle": 28.0, "distal": 18.0},
    "pinky": {"proximal": 36.0, "middle": 21.0, "distal": 16.0},
}

# MediaPipe Hands landmark indices of the MCP joints of the four long fingers.
FINGER_BASE: dict[str, int] = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
ADJACENT_PAIRS: tuple[tuple[str, str], ...] = (
    ("index", "middle"),
    ("middle", "ring"),
    ("ring", "pinky"),
)


@dataclass(frozen=True)
class JointSpec:
    """Joint flexion is the angle between bones prev->landmark and landmark->nxt."""

    name: str
    landmark: int
    prev: int
    nxt: int
    min_flex_deg: float
    max_flex_deg: float


@dataclass(frozen=True)
class FingerLayout:
    name: str
    phalanges: dict[str, tuple[int, int]]
    landmark_names: dict[int, str]
    joints: tuple[JointSpec, ...]


def _finger_layout(name: str, base: int) -> FingerLayout:
    """Long fingers: MCP=base, PIP=base+1, DIP=base+2, TIP=base+3."""
    return FingerLayout(
        name=name,
        phalanges={
            "proximal": (base, base + 1),
            "middle": (base + 1, base + 2),
            "distal": (base + 2, base + 3),
        },
        landmark_names={base: "MCP", base + 1: "PIP", base + 2: "DIP", base + 3: "TIP"},
        joints=(
            JointSpec("PIP", base + 1, base, base + 2, min_flex_deg=-15.0, max_flex_deg=110.0),
            JointSpec("DIP", base + 2, base + 1, base + 3, min_flex_deg=-20.0, max_flex_deg=90.0),
        ),
    )


FINGER_LAYOUTS: tuple[FingerLayout, ...] = (
    FingerLayout(
        name="thumb",
        phalanges={"proximal": (2, 3), "distal": (3, 4)},
        landmark_names={1: "CMC", 2: "MCP", 3: "IP", 4: "TIP"},
        joints=(
            JointSpec("MCP", 2, 1, 3, min_flex_deg=-20.0, max_flex_deg=70.0),
            JointSpec("IP", 3, 2, 4, min_flex_deg=-20.0, max_flex_deg=90.0),
        ),
    ),
    _finger_layout("index", FINGER_BASE["index"]),
    _finger_layout("middle", FINGER_BASE["middle"]),
    _finger_layout("ring", FINGER_BASE["ring"]),
    _finger_layout("pinky", FINGER_BASE["pinky"]),
)


# Distal tip bones only. Proximal crossings are common in real grips (dumbbells, phones).
DISTAL_BONES: tuple[tuple[str, int, int], ...] = (
    ("thumb", 3, 4),
    ("index", 7, 8),
    ("middle", 11, 12),
    ("ring", 15, 16),
    ("pinky", 19, 20),
)
ADJACENT_FINGERS = {
    frozenset(("index", "middle")),
    frozenset(("middle", "ring")),
    frozenset(("ring", "pinky")),
}


@dataclass(frozen=True)
class AnatomyConfig:
    proportion_warning_deviation: float = 0.20
    adjacent_max_deg: float = 45.0
    bbox_padding_ratio: float = 0.12
    """Half-size of the joint box as a fraction of the hand size (wrist to middle MCP)."""
    min_bbox_padding_px: float = 6.0
    # 2D self-crossing evidence (default on). Legacy 3D evidence stays off.
    emit_hand_evidence: bool = True
    emit_legacy_3d_evidence: bool = False
    min_bone_crossings: int = 2


@dataclass(frozen=True)
class AnatomyFlag:
    landmark_index: int
    joint_name: str
    reason: Literal["angle", "proportion", "adjacent_angle"]
    severity: Severity
    message: str
    value: float


@dataclass
class HandAnalysis:
    hand_index: int
    handedness: str
    metrics: dict[str, float]
    flags: list[AnatomyFlag] = field(default_factory=list)


@dataclass
class AnatomyReport:
    analyses: list[HandAnalysis]
    evidences: list[Evidence]

    @property
    def has_critical(self) -> bool:
        # Anatomy never produces CRITICAL evidence; kept for interface compatibility.
        return any(e.severity is Severity.CRITICAL for e in self.evidences)


def _unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v / max(norm, 1e-12)


def palm_frame(landmarks: np.ndarray, handedness: str) -> tuple[np.ndarray, np.ndarray]:
    """Hand axis f and palm normal n.

    n = (index_MCP - wrist) x (pinky_MCP - wrist), projected orthogonal to f.
    For left hands n is flipped so that it points to the same side of the palm.
    The palm side convention was verified on synthetic data only.
    """
    wrist = landmarks[0]
    f = _unit(landmarks[9] - wrist)
    n_raw = np.cross(landmarks[5] - wrist, landmarks[17] - wrist)
    n = _unit(n_raw - np.dot(n_raw, f) * f)
    if handedness.strip().lower() == "left":
        n = -n
    return f, n


def signed_flexion_deg(
    prev: np.ndarray, joint: np.ndarray, nxt: np.ndarray, f: np.ndarray, n: np.ndarray
) -> float:
    """Signed flexion in degrees, 0 when the bones are collinear.

    alpha = arccos(a . b / (|a||b|)) with a = joint - prev, b = nxt - joint.
    Sign = sign((a x b) . (f x n)). Positive means bending toward the palm.
    """
    a = joint - prev
    b = nxt - joint
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    cos_alpha = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    angle = math.degrees(math.acos(cos_alpha))
    sign = float(np.dot(np.cross(a, b), np.cross(f, n)))
    return angle if sign >= 0.0 else -angle


def angle_between_deg(u: np.ndarray, v: np.ndarray) -> float:
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    cos_angle = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    return math.degrees(math.acos(cos_angle))


def analyze_hand(hand: HandObservation, config: AnatomyConfig | None = None) -> HandAnalysis:
    cfg = config or AnatomyConfig()
    lm = hand.landmarks_3d
    f, n = palm_frame(lm, hand.handedness)
    metrics: dict[str, float] = {}
    flags: list[AnatomyFlag] = []

    for layout in FINGER_LAYOUTS:
        lengths = {
            bone: float(np.linalg.norm(lm[end] - lm[start]))
            for bone, (start, end) in layout.phalanges.items()
        }
        proximal_len = max(lengths["proximal"], 1e-9)
        norm_mm = ANATOMICAL_PHALANX_MM[layout.name]

        for bone, (start, end) in layout.phalanges.items():
            if bone == "proximal":
                continue
            ratio = lengths[bone] / proximal_len
            expected = norm_mm[bone] / norm_mm["proximal"]
            deviation = abs(ratio / expected - 1.0)
            key = f"{layout.name}.{bone}_over_proximal"
            metrics[f"{key}.measured"] = ratio
            metrics[f"{key}.expected"] = expected
            metrics[f"{key}.deviation"] = deviation

            severity = _classify_deviation(deviation, cfg)
            if severity is None:
                continue
            message = (
                f"proportion {layout.name}: {bone}/proximal = {ratio:.2f} "
                f"(norm {expected:.2f}, deviation {deviation * 100:.0f}%)"
            )
            for landmark in (start, end):
                flags.append(
                    AnatomyFlag(
                        landmark_index=landmark,
                        joint_name=f"{layout.name}.{layout.landmark_names[landmark]}",
                        reason="proportion",
                        severity=severity,
                        message=message,
                        value=deviation,
                    )
                )

        for joint in layout.joints:
            angle = signed_flexion_deg(lm[joint.prev], lm[joint.landmark], lm[joint.nxt], f, n)
            metrics[f"{layout.name}.{joint.name}.flexion_deg"] = angle
            if joint.min_flex_deg <= angle <= joint.max_flex_deg:
                continue
            kind = "hyperextension" if angle < joint.min_flex_deg else "excessive flexion"
            flags.append(
                AnatomyFlag(
                    landmark_index=joint.landmark,
                    joint_name=f"{layout.name}.{joint.name}",
                    reason="angle",
                    severity=Severity.WARNING,
                    message=(
                        f"{kind} at {layout.name}.{joint.name}: {angle:.0f} deg "
                        f"(allowed {joint.min_flex_deg:.0f}..{joint.max_flex_deg:.0f} deg)"
                    ),
                    value=angle,
                )
            )

    for left, right in ADJACENT_PAIRS:
        base_l, base_r = FINGER_BASE[left], FINGER_BASE[right]
        spread = angle_between_deg(lm[base_l + 1] - lm[base_l], lm[base_r + 1] - lm[base_r])
        metrics[f"adjacent.{left}_{right}.deg"] = spread
        if spread <= cfg.adjacent_max_deg:
            continue
        message = f"angle between {left} and {right}: {spread:.0f} deg (max {cfg.adjacent_max_deg:.0f} deg)"
        for base, finger in ((base_l, left), (base_r, right)):
            flags.append(
                AnatomyFlag(
                    landmark_index=base,
                    joint_name=f"{finger}.MCP",
                    reason="adjacent_angle",
                    severity=Severity.WARNING,
                    message=message,
                    value=spread,
                )
            )

    return HandAnalysis(hand_index=hand.hand_index, handedness=hand.handedness, metrics=metrics, flags=flags)


def _classify_deviation(deviation: float, cfg: AnatomyConfig) -> Severity | None:
    if deviation > cfg.proportion_warning_deviation:
        return Severity.WARNING
    return None


def _orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _segments_cross(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> bool:
    """True when open segments p1-p2 and p3-p4 properly intersect (shared endpoints ignored)."""
    if (
        np.allclose(p1, p3)
        or np.allclose(p1, p4)
        or np.allclose(p2, p3)
        or np.allclose(p2, p4)
    ):
        return False
    o1, o2 = _orient(p1, p2, p3), _orient(p1, p2, p4)
    o3, o4 = _orient(p3, p4, p1), _orient(p3, p4, p2)
    return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)


def analyze_hand_2d(hand: HandObservation, config: AnatomyConfig | None = None) -> HandAnalysis:
    """2D evidence: distal tip bones of non-adjacent fingers that cross in the image plane.

    Calibrated so a real gripping hand (samples/real/real_hand_gym.jpg) yields 0 crossings.
    H13 tip-order rejected: real grips reverse DIP→TIP while proximal chain stays
    extended (gym middle/ring), and samples lack broken-AI hands to justify shipping.
    """
    cfg = config or AnatomyConfig()
    px = np.asarray(hand.landmarks_px, dtype=np.float64)
    flags: list[AnatomyFlag] = []
    crossings = 0
    for i, (fa, a0, a1) in enumerate(DISTAL_BONES):
        for fb, b0, b1 in DISTAL_BONES[i + 1 :]:
            if frozenset((fa, fb)) in ADJACENT_FINGERS:
                continue
            if not _segments_cross(px[a0], px[a1], px[b0], px[b1]):
                continue
            crossings += 1
            flags.append(
                AnatomyFlag(
                    landmark_index=a1,
                    joint_name=f"distal_{fa}_x_{fb}",
                    reason="angle",
                    severity=Severity.WARNING,
                    message=f"distal tips of {fa} and {fb} cross in the image plane",
                    value=float(crossings),
                )
            )
    metrics = {"bone_crossings_2d": float(crossings)}
    # One clear non-adjacent distal crossing is enough; grips do not produce these.
    if crossings < 1:
        flags = []
    return HandAnalysis(
        hand_index=hand.hand_index, handedness=hand.handedness, metrics=metrics, flags=flags
    )


class AnatomyValidator:
    def __init__(self, config: AnatomyConfig | None = None) -> None:
        self.config = config or AnatomyConfig()

    def validate(self, hands: Sequence[HandObservation]) -> AnatomyReport:
        analyses: list[HandAnalysis] = []
        evidences: list[Evidence] = []
        for hand in hands:
            analysis_2d = analyze_hand_2d(hand, self.config)
            analysis_3d = analyze_hand(hand, self.config)
            merged_flags = list(analysis_2d.flags)
            if self.config.emit_legacy_3d_evidence:
                merged_flags.extend(analysis_3d.flags)
            metrics = {**analysis_3d.metrics, **analysis_2d.metrics}
            analysis = HandAnalysis(
                hand_index=hand.hand_index,
                handedness=hand.handedness,
                metrics=metrics,
                flags=merged_flags,
            )
            analyses.append(analysis)
            if analysis_2d.flags and self.config.emit_hand_evidence:
                evidences.append(self._build_evidence(hand, analysis_2d))
            elif analysis_3d.flags and self.config.emit_legacy_3d_evidence:
                evidences.append(self._build_evidence(hand, analysis_3d))
        return AnatomyReport(analyses=analyses, evidences=evidences)

    def _build_evidence(self, hand: HandObservation, analysis: HandAnalysis) -> Evidence:
        cfg = self.config
        px = np.asarray(hand.landmarks_px, dtype=np.float64)
        hand_size = max(float(np.linalg.norm(px[9] - px[0])), 1.0)
        pad = max(cfg.bbox_padding_ratio * hand_size, cfg.min_bbox_padding_px)

        joints: list[FlaggedJoint] = []
        seen: set[tuple[int, str]] = set()
        for flag in analysis.flags:
            key = (flag.landmark_index, flag.reason)
            if key in seen:
                continue
            seen.add(key)
            x, y = px[flag.landmark_index]
            joints.append(
                FlaggedJoint(
                    hand_index=hand.hand_index,
                    landmark_index=flag.landmark_index,
                    joint_name=flag.joint_name,
                    reason=flag.reason,
                    severity=flag.severity,
                    bbox=(float(x - pad), float(y - pad), float(x + pad), float(y + pad)),
                )
            )

        messages = "; ".join(dict.fromkeys(flag.message for flag in analysis.flags))
        description = f"Рука #{hand.hand_index + 1} ({hand.handedness}): {messages}."

        return Evidence(
            evidence_id=f"ANAT-{hand.hand_index + 1:02d}",
            anomaly_type=AnomalyType.ANATOMY_HAND_DEFORMATION,
            severity=Severity.WARNING,
            description=description,
            metrics={k: float(v) for k, v in analysis.metrics.items()},
            anatomy=AnatomyPayload(
                hand_index=hand.hand_index,
                handedness=hand.handedness,
                flagged_joints=joints,
                landmarks_px=[(float(x), float(y)) for x, y in px],
            ),
        )
