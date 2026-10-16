"""Unit tests for soft probes, classical shadows, and physics fusion vote."""

from __future__ import annotations

import cv2
import numpy as np

from agents.light_agent import LightAgent
from agents.shadow_scene import extract_shadow_observations
from agents.synthetic import ShadowObjectSpec, simulate_shadow_scene
from core.pipeline import GavelPipeline
from core.registry import (
    AnomalyType,
    CheckRecord,
    DetectorResult,
    Evidence,
    EvidenceProtocol,
    ImageMetadata,
    LayerScore,
    LayerScores,
    LightingPayload,
    Severity,
)
from core.validators.edges import analyze_edges
from core.validators.halves import HalvesConfig, analyze_halves
from core.verdict import LABEL_GENERATION_SIGNAL, compute_verdict


def test_analyze_halves_flags_diptych_energy_gap():
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    img[:, :200] = (30, 30, 30)
    noise = np.random.default_rng(0).integers(0, 255, size=(200, 200, 3), dtype=np.uint8)
    img[:, 200:] = noise
    report = analyze_halves(img, HalvesConfig(min_aspect=1.5, energy_ratio_thr=2.0, hist_bhatta_thr=0.3))
    assert report.applicable
    assert report.disagree


def test_analyze_halves_skips_square_frame():
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    report = analyze_halves(img)
    assert not report.applicable
    assert not report.disagree


def test_analyze_edges_returns_finite_density():
    img = np.full((240, 320, 3), 200, dtype=np.uint8)
    cv2.line(img, (20, 20), (300, 200), (0, 0, 0), 2)
    cv2.line(img, (40, 200), (280, 40), (0, 0, 0), 2)
    report = analyze_edges(img)
    assert report.density >= 0.0
    assert report.segment_count >= 0.0


def test_extract_shadow_observations_empty_on_flat_image():
    img = np.full((240, 320, 3), 180, dtype=np.uint8)
    assert extract_shadow_observations(img) == []


def test_light_agent_default_provider_runs():
    agent = LightAgent()
    img = np.full((240, 320, 3), 180, dtype=np.uint8)
    assert agent.extract(img) == []


def test_physics_hard_vote_via_injected_provider():
    light = (200.0, 80.0)
    objects = [
        ShadowObjectSpec("a", (400.0, 200.0, 480.0, 360.0), 180.0, deviation_deg=25.0),
        ShadowObjectSpec("b", (700.0, 220.0, 780.0, 380.0), 160.0, deviation_deg=-20.0),
    ]
    obs = simulate_shadow_scene(light, objects)
    pipe = GavelPipeline(
        light_agent=LightAgent(lambda _img: obs),
        learned_detector=None,
    )
    image = np.full((720, 1280, 3), 40, dtype=np.uint8)
    result = pipe.analyze(image, image_name="shadow_inconsistent.png")
    assert result.protocol.layers.physics.status == "ran"
    assert result.protocol.layers.physics.signal
    assert any(e.anomaly_type is AnomalyType.LIGHT_CONSISTENCY_VIOLATION for e in result.protocol.evidences)
    verdict = compute_verdict(result.protocol)
    assert verdict.decision == "generated"
    assert "physics" in (verdict.fusion or "")


def test_fusion_physics_layer_vote_without_l0():
    protocol = EvidenceProtocol(
        image=ImageMetadata(file_name="x", width=100, height=100),
        checks=[CheckRecord(check="lighting.consistency", status="violation")],
        evidences=[
            Evidence(
                evidence_id="LIT-01",
                anomaly_type=AnomalyType.LIGHT_CONSISTENCY_VIOLATION,
                severity=Severity.CRITICAL,
                description="inconsistent shadows",
                lighting=LightingPayload(image_diagonal=100.0),
            )
        ],
        layers=LayerScores(
            physics=LayerScore(name="physics", status="ran", score=0.5, signal=True, reason="rays=2")
        ),
        detector=DetectorResult(
            model="t",
            probability=0.1,
            logit=-2.0,
            decision="real",
            high_threshold=0.9,
            low_threshold=0.01,
        ),
    )
    verdict = compute_verdict(protocol)
    assert verdict.label == LABEL_GENERATION_SIGNAL
    assert verdict.decision == "generated"
    assert "physics" in (verdict.fusion or "")
