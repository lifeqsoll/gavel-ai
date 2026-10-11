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


def test_mild_angle_jitter_is_not_critical():
    """Near-parallel real edges (scooters) look like mild MAD; must not CRITICAL."""
    rng = np.random.default_rng(0)
    angles = 70.0 + rng.uniform(-8.0, 8.0, size=8)
    offsets = rng.uniform(-4.0, 4.0, size=8)
    segs = _segments_through((1000.0, 150.0), angles, offsets)
    report = PerspectiveValidator().validate(segs, SIZE)
    critical = [
        e
        for e in report.evidences
        if e.anomaly_type is AnomalyType.GEOMETRY_PERSPECTIVE_VIOLATION
        and e.severity is Severity.CRITICAL
    ]
    assert not critical


def test_classify_extreme_spread_is_critical():
    from core.validators.perspective import VanishingPointCluster

    validator = PerspectiveValidator()
    extreme = VanishingPointCluster(
        point=(100.0, 100.0),
        segment_indices=list(range(8)),
        rms_residual_px=1.0,
        n_pairs=28,
        divergent_fraction=0.0,
        robust_std_px=200.0,
        robust_std_norm=0.20,
        variance_px2=1.0,
        vp_distance_norm=0.25,
    )
    assert validator._classify(extreme) is Severity.CRITICAL

    mild_street_like = VanishingPointCluster(
        point=(640.0, 360.0),
        segment_indices=list(range(11)),
        rms_residual_px=1.75,
        n_pairs=55,
        divergent_fraction=0.0,
        robust_std_px=40.0,
        robust_std_norm=0.028,
        variance_px2=1.0,
        vp_distance_norm=0.24,
    )
    assert validator._classify(mild_street_like) is None


def test_normal_hand_has_no_anatomy_evidence():
    hand = build_synthetic_hand()
    report = AnatomyValidator().validate([hand])
    assert report.evidences == []


def test_legacy_3d_flags_do_not_emit_evidence_by_default():
    hand = build_synthetic_hand(length_scale={"index.distal": 1.6})
    analysis = analyze_hand(hand)
    assert any(f.reason == "proportion" for f in analysis.flags)
    report = AnatomyValidator().validate([hand])
    assert report.evidences == []


def test_2d_distal_crossing_emits_warning():
    from core.observations import HandObservation
    from core.validators.anatomy import analyze_hand_2d

    hand = build_synthetic_hand()
    # Cross index tip (7-8) with pinky tip (19-20): non-adjacent distal bones.
    px = np.asarray(hand.landmarks_px, dtype=np.float64).copy()
    px[7] = np.array([100.0, 100.0])
    px[8] = np.array([200.0, 100.0])
    px[19] = np.array([150.0, 50.0])
    px[20] = np.array([150.0, 150.0])
    crossed = HandObservation(
        hand_index=0,
        handedness=hand.handedness,
        landmarks_3d=hand.landmarks_3d,
        landmarks_px=px,
    )
    analysis = analyze_hand_2d(crossed)
    assert analysis.metrics["bone_crossings_2d"] >= 1
    report = AnatomyValidator().validate([crossed])
    assert report.evidences
    assert report.evidences[0].severity is Severity.WARNING
    assert report.evidences[0].anomaly_type is AnomalyType.ANATOMY_HAND_DEFORMATION


def test_hand_mostly_inside_face_box_is_dropped():
    from agents.anatomy_agent import AnatomyAgent

    face = [(100.0, 100.0, 200.0, 220.0)]
    on_face = np.array([[150.0, 150.0], [160.0, 170.0], [140.0, 190.0], [120.0, 210.0]])
    off_face = np.array([[20.0, 20.0], [40.0, 40.0], [60.0, 60.0], [80.0, 80.0]])
    mixed = np.array([[150.0, 150.0], [20.0, 20.0], [40.0, 40.0], [60.0, 60.0]])
    assert AnatomyAgent._is_face(on_face, face)
    assert not AnatomyAgent._is_face(off_face, face)
    assert not AnatomyAgent._is_face(mixed, face)


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


def test_geometry_overlay_draws_cluster_segments_without_crossings():
    """CRITICAL findings with empty outlier crossings must still paint the line family."""
    from core.registry import GeometryPayload, LineSegment
    from visualizer.overlay import draw_geometry_evidence

    canvas = np.zeros((200, 200, 3), dtype=np.uint8)
    payload = GeometryPayload(
        vanishing_point=(100.0, 50.0),
        cluster_segments=[
            LineSegment(p1=(10.0, 180.0), p2=(90.0, 60.0)),
            LineSegment(p1=(190.0, 180.0), p2=(110.0, 60.0)),
        ],
        crossings=[],
        image_diagonal=282.0,
    )
    draw_geometry_evidence(canvas, payload, "GEO-01")
    assert canvas.sum() > 0
