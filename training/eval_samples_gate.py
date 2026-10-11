"""Gate candidate L0 weights against samples/ before shipping.

Does not copy weights. Exit 0 only if all hard gates pass.

Usage:
    python training/eval_samples_gate.py
    python training/eval_samples_gate.py --weights models/exp_luna_plan/detector_weights.npz \\
        --thresholds models/exp_luna_plan/thresholds.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.geometry_agent import GeometryAgent  # noqa: E402
from agents.anatomy_agent import AnatomyAgent  # noqa: E402
from core.learned_detector import LearnedDetector  # noqa: E402
from core.pipeline import GavelPipeline  # noqa: E402
from core.validators.perspective import PerspectiveConfig, PerspectiveValidator  # noqa: E402
from core.verdict import compute_verdict  # noqa: E402

# Hard gates on checked-in samples (no new data collection).
EXPECT_REAL_DECISION = {
    "real_hand_gym.jpg": {"real", "uncertain"},  # prefer real; uncertain ok, generated fails
    "real_street_florence.jpg": {"real", "uncertain"},
    "real_corridor_jama_masjid.jpg": {"real", "uncertain"},
}
EXPECT_NOT_REAL = {
    # Prefer generated; uncertain allowed; real fails (FN).
    "ai_03.jpg",
    "ai_05.jpg",
    "ai_08.jpg",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "detector_weights.npz")
    parser.add_argument("--thresholds", type=Path, default=ROOT / "models" / "thresholds.json")
    parser.add_argument("--samples", type=Path, default=ROOT / "samples")
    parser.add_argument("--strict-hand", action="store_true", help="require real_hand_gym == real")
    args = parser.parse_args()

    detector = LearnedDetector(weights_path=args.weights, thresholds_path=args.thresholds)
    pipe = GavelPipeline(
        geometry_agent=GeometryAgent(method="lsd"),
        perspective_validator=PerspectiveValidator(PerspectiveConfig()),
        anatomy_agent=AnatomyAgent(),
        learned_detector=detector,
    )

    failures: list[str] = []
    print(f"weights={args.weights}")
    print(f"thresholds={args.thresholds}")
    print(f"{'sample':35} {'L0':8} {'decision':10} fusion")

    for split, rules in (("real", EXPECT_REAL_DECISION), ("ai_generated", None)):
        for path in sorted((args.samples / split).glob("*")):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            result = pipe.analyze(cv2.imread(str(path)), path.name)
            verdict = compute_verdict(result.protocol)
            l0 = result.protocol.layers.global_layer.score
            l0_s = "n/a" if l0 is None else f"{l0:.3f}"
            print(f"{path.name:35} {l0_s:8} {verdict.decision or 'n/a':10} {verdict.fusion}")

            if path.name in EXPECT_REAL_DECISION:
                allowed = set(EXPECT_REAL_DECISION[path.name])
                if args.strict_hand and path.name == "real_hand_gym.jpg":
                    allowed = {"real"}
                if verdict.decision not in allowed:
                    failures.append(
                        f"{path.name}: decision={verdict.decision} not in {sorted(allowed)} "
                        f"(fusion={verdict.fusion})"
                    )
            if path.name in EXPECT_NOT_REAL and verdict.decision == "real":
                failures.append(f"{path.name}: FN decision=real (fusion={verdict.fusion})")

    if failures:
        print("\nGATE FAIL")
        for line in failures:
            print(" -", line)
        return 1
    print("\nGATE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
