"""Mock run of the Gavel AI pipeline without neural networks.

Scenarios:
    mixed: all three anomaly types are planted on the frame.
    clean: all cues are consistent, no evidence expected.

Usage:
    python mock_run.py --scenario mixed --out-dir output
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from agents.geometry_agent import GeometryAgent
from agents.light_agent import LightAgent
from agents.synthetic import ShadowObjectSpec, build_synthetic_hand, simulate_shadow_scene
from core.observations import HandObservation
from core.pipeline import GavelPipeline
from core.validators.anatomy import AnatomyValidator
from core.validators.perspective import PerspectiveValidator
from core.validators.shadows import ShadowValidator

IMAGE_SIZE = (1280, 720)
LIGHT_XY = (100.0, -400.0)
BASE_OBJECTS: tuple[ShadowObjectSpec, ...] = (
    ShadowObjectSpec("object_1", (270.0, 380.0, 330.0, 560.0), shadow_length_px=170.0),
    ShadowObjectSpec("object_2", (650.0, 360.0, 710.0, 560.0), shadow_length_px=160.0),
    ShadowObjectSpec("object_3", (1000.0, 370.0, 1060.0, 560.0), shadow_length_px=150.0),
)
HAND_CENTER = (180.0, 540.0)
HAND_BONES = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
    (5, 9, 13, 17),
)


class StaticAnatomyAgent:
    """Returns predefined hand observations instead of running MediaPipe."""

    def __init__(self, hands: list[HandObservation]) -> None:
        self._hands = hands

    def extract(self, image: np.ndarray) -> list[HandObservation]:
        return list(self._hands)


def _draw_corridor(canvas: np.ndarray) -> None:
    """Consistent perspective: segments converging to one vanishing point."""
    width, height = IMAGE_SIZE
    vp = np.array([640.0, 250.0])
    for x in np.linspace(-200.0, 1480.0, 10):
        target = np.array([x, height - 1.0])
        start = vp + 0.25 * (target - vp)
        cv2.line(canvas, tuple(int(round(v)) for v in start), tuple(int(round(v)) for v in target),
                 (220, 220, 220), 2, cv2.LINE_AA)


def _draw_conflicting_family(canvas: np.ndarray, rng: np.random.Generator) -> None:
    """Near-parallel lines passing near one point with slope jitter: wide intersection spread."""
    anchor_point = np.array([1000.0, 150.0])
    base_angle = math.radians(70.0)
    direction_base = np.array([math.cos(base_angle), math.sin(base_angle)])
    perp = np.array([-direction_base[1], direction_base[0]])
    for _ in range(8):
        angle = base_angle + math.radians(rng.uniform(-8.0, 8.0))
        d = np.array([math.cos(angle), math.sin(angle)])
        anchor = anchor_point + perp * rng.uniform(-4.0, 4.0)
        p1 = anchor - 260.0 * d
        p2 = anchor + 260.0 * d
        cv2.line(canvas, tuple(int(round(v)) for v in p1), tuple(int(round(v)) for v in p2),
                 (200, 200, 200), 2, cv2.LINE_AA)


def _draw_hand(canvas: np.ndarray, hand: HandObservation) -> None:
    pts = hand.landmarks_px
    for chain in HAND_BONES:
        for a, b in zip(chain[:-1], chain[1:]):
            cv2.line(canvas, tuple(int(round(v)) for v in pts[a]), tuple(int(round(v)) for v in pts[b]),
                     (150, 150, 150), 2, cv2.LINE_AA)


def build_scene(scenario: Literal["mixed", "clean"]) -> tuple[np.ndarray, list[HandObservation], list[ShadowObjectSpec]]:
    rng = np.random.default_rng(7)
    canvas = np.full((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), 30, dtype=np.uint8)
    _draw_corridor(canvas)

    if scenario == "mixed":
        _draw_conflicting_family(canvas, rng)
        hand = build_synthetic_hand(
            hand_index=0,
            handedness="Right",
            flex_deg={"index.PIP": -30.0},
            length_scale={"index.distal": 1.6},
            center_px=HAND_CENTER,
        )
        objects = [
            dataclasses.replace(spec, deviation_deg=25.0) if spec.object_id == "object_3" else spec
            for spec in BASE_OBJECTS
        ]
    else:
        hand = build_synthetic_hand(hand_index=0, handedness="Right", center_px=HAND_CENTER)
        objects = list(BASE_OBJECTS)

    _draw_hand(canvas, hand)
    return canvas, [hand], objects


def run(scenario: Literal["mixed", "clean"], out_dir: Path) -> None:
    image, hands, objects = build_scene(scenario)

    pipeline = GavelPipeline(
        geometry_agent=GeometryAgent(method="lsd"),
        perspective_validator=PerspectiveValidator(),
        anatomy_agent=StaticAnatomyAgent(hands),
        anatomy_validator=AnatomyValidator(),
        light_agent=LightAgent(lambda _img: simulate_shadow_scene(LIGHT_XY, objects)),
        shadow_validator=ShadowValidator(),
    )
    result = pipeline.analyze(image, image_name=f"mock_{scenario}.png")

    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"input_{scenario}.png"), image)
    cv2.imwrite(str(out_dir / f"overlay_{scenario}.png"), result.annotated_image)
    (out_dir / f"evidence_{scenario}.json").write_text(result.protocol.to_json(), encoding="utf-8")

    print(f"scenario: {scenario}")
    for check in result.protocol.checks:
        print(f"  check {check.check}: {check.status} ({check.reason})")
    for evidence in result.protocol.evidences:
        print(f"  {evidence.evidence_id} {evidence.anomaly_type.value} {evidence.severity.value}")
        print(f"    {evidence.description}")
    print(json.dumps({"has_critical": result.protocol.has_critical,
                      "evidences": len(result.protocol.evidences)}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Gavel AI mock run")
    parser.add_argument("--scenario", choices=["mixed", "clean"], default="mixed")
    parser.add_argument("--out-dir", type=Path, default=Path("output"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(args.scenario, args.out_dir)


if __name__ == "__main__":
    main()
