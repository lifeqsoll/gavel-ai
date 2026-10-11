import math

import numpy as np

from agents.synthetic import ShadowObjectSpec, build_synthetic_hand, simulate_shadow_scene
from core.registry import AnomalyType, Evidence, GeometryPayload, Severity
from core.validators.anatomy import AnatomyValidator, analyze_hand
from core.validators.perspective import PerspectiveValidator
from core.validators.shadows import ShadowValidator

SIZE = (1280, 720)


def _segments_through(point, angles_deg, offsets, half_length=300.0):
    segs = []
    for angle, off in zip(angles_deg, offsets):
        d = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
        normal = np.array([-d[1], d[0]])
        anchor = np.asarray(point) + normal * off
        p1 = anchor - half_length * d
        p2 = anchor + half_length * d
        segs.append(((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))))
    return segs


def test_ransac_recovers_converging_family_without_evidence():
    vp = (640.0, 250.0)
    rng = np.random.default_rng(0)
    segs = []
    for x in np.linspace(-200, 1480, 10):
        target = np.array([x, 719.0])
        start = np.array(vp) + 0.25 * (target - np.array(vp))
        segs.append(((float(start[0]), float(start[1])), (float(target[0]), float(target[1]))))
    segs += [
        ((float(a), float(b)), (float(c), float(d)))
        for a, b, c, d in rng.uniform(0, 700, size=(6, 4))
    ]
    report = PerspectiveValidator().validate(segs, SIZE)
    assert report.clusters, "a dominant vanishing point should be found"
    px, py = report.clusters[0].point
    assert math.hypot(px - vp[0], py - vp[1]) < 15.0
    assert not report.has_critical


def test_inconsistent_parallel_family_is_critical():
    rng = np.random.default_rng(0)
    angles = 70.0 + rng.uniform(-8.0, 8.0, size=8)
    offsets = rng.uniform(-4.0, 4.0, size=8)
    segs = _segments_through((1000.0, 150.0), angles, offsets)
    report = PerspectiveValidator().validate(segs, SIZE)
    geo = [e for e in report.evidences if e.anomaly_type is AnomalyType.GEOMETRY_PERSPECTIVE_VIOLATION]
    assert geo and geo[0].severity is Severity.CRITICAL
    assert isinstance(geo[0].geometry, GeometryPayload)
    assert geo[0].geometry.crossings


def test_normal_hand_has_no_anatomy_evidence():
    hand = build_synthetic_hand()
    report = AnatomyValidator().validate([hand])
    assert report.evidences == []


def test_distal_proportion_violation_is_critical():
    hand = build_synthetic_hand(length_scale={"index.distal": 1.6})
    analysis = analyze_hand(hand)
    assert any(f.reason == "proportion" and f.severity is Severity.CRITICAL for f in analysis.flags)
    report = AnatomyValidator().validate([hand])
    ev = report.evidences[0]
    assert ev.anomaly_type is AnomalyType.ANATOMY_HAND_DEFORMATION
    assert ev.anatomy is not None and ev.anatomy.flagged_joints


def test_hyperextension_is_detected_as_negative_flexion():
    hand = build_synthetic_hand(flex_deg={"index.PIP": -30.0})
    analysis = analyze_hand(hand)
    assert math.isclose(analysis.metrics["index.PIP.flexion_deg"], -30.0, abs_tol=0.5)
    assert any(f.reason == "angle" for f in analysis.flags)


def test_positive_flexion_is_measured_correctly():
    hand = build_synthetic_hand(flex_deg={"middle.DIP": 45.0})
    analysis = analyze_hand(hand)
    assert math.isclose(analysis.metrics["middle.DIP.flexion_deg"], 45.0, abs_tol=0.5)
    assert not any(f.reason == "angle" for f in analysis.flags)


def test_consistent_light_has_no_evidence():
    specs = [
        ShadowObjectSpec("a", (270, 380, 330, 560), 170.0),
        ShadowObjectSpec("b", (650, 360, 710, 560), 160.0),
        ShadowObjectSpec("c", (1000, 370, 1060, 560), 150.0),
    ]
    obs = simulate_shadow_scene((100.0, -400.0), specs)
    report = ShadowValidator().validate(obs, SIZE)
    assert report.evidences == []
    assert report.estimated_light is not None
    assert math.hypot(report.estimated_light[0] - 100.0, report.estimated_light[1] + 400.0) < 1.0


def test_inconsistent_light_is_critical():
    specs = [
        ShadowObjectSpec("a", (270, 380, 330, 560), 170.0),
        ShadowObjectSpec("b", (650, 360, 710, 560), 160.0),
        ShadowObjectSpec("c", (1000, 370, 1060, 560), 150.0, deviation_deg=25.0),
    ]
    report = ShadowValidator().validate(simulate_shadow_scene((100.0, -400.0), specs), SIZE)
    assert report.evidences and report.evidences[0].severity is Severity.CRITICAL


def test_light_behind_objects_is_critical():
    from core.observations import LightObservation

    obs = [
        LightObservation("a", (290, 380, 310, 560), (300, 560), (300, 400), (300, 500)),
        LightObservation("b", (590, 380, 610, 560), (600, 560), (600, 400), (700, 300)),
    ]
    report = ShadowValidator().validate(obs, SIZE)
    assert report.frontal_violations > 0
    assert report.evidences and report.evidences[0].severity is Severity.CRITICAL


def test_evidence_payload_must_match_anomaly_type():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Evidence(
            evidence_id="X",
            anomaly_type=AnomalyType.LIGHT_CONSISTENCY_VIOLATION,
            severity=Severity.CRITICAL,
            description="mismatch",
            geometry=GeometryPayload(image_diagonal=10.0),
        )
