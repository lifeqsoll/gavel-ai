"""Local hard-neg regression pack (samples + bushon_eval on disk-E + wiki hands).

Does not ship weights. Exit 0 only if hard rules pass.

Hard fails:
  - any samples/ai_generated → decision == real
  - real_street_florence.jpg → decision == generated
  - real_hand_gym.jpg → decision == generated (unless --allow-gym-uncertain)

Soft (printed, non-fatal unless --strict-bushon):
  - bushon_eval / wiki hands → decision == generated

Usage:
    ~/gavel/train-venv/bin/python training/eval_hardneg_local.py
    ~/gavel/train-venv/bin/python training/eval_hardneg_local.py --wiki-n 12 --strict-bushon
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.anatomy_agent import AnatomyAgent  # noqa: E402
from agents.geometry_agent import GeometryAgent  # noqa: E402
from core.learned_detector import LearnedDetector  # noqa: E402
from core.pipeline import GavelPipeline  # noqa: E402
from core.validators.perspective import PerspectiveConfig, PerspectiveValidator  # noqa: E402
from core.verdict import compute_verdict  # noqa: E402

DEFAULT_BUSHON = Path("/mnt/disk-e/gavel-data/real/hands/bushon_eval")
DEFAULT_WIKI = Path("/mnt/disk-e/gavel-data/real/hands/wikimedia")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _iter_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    paths = [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(paths)


def _build_pipe(weights: Path, thresholds: Path) -> GavelPipeline:
    return GavelPipeline(
        geometry_agent=GeometryAgent(method="lsd"),
        perspective_validator=PerspectiveValidator(PerspectiveConfig()),
        anatomy_agent=AnatomyAgent(),
        learned_detector=LearnedDetector(weights_path=weights, thresholds_path=thresholds),
    )


def _run_one(pipe: GavelPipeline, path: Path) -> tuple[str | None, str | None, float | None]:
    img = cv2.imread(str(path))
    if img is None:
        return None, None, None
    result = pipe.analyze(img, path.name)
    verdict = compute_verdict(result.protocol)
    l0 = result.protocol.layers.global_layer.score
    return verdict.decision, verdict.fusion, l0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "detector_weights.npz")
    parser.add_argument("--thresholds", type=Path, default=ROOT / "models" / "thresholds.json")
    parser.add_argument("--samples", type=Path, default=ROOT / "samples")
    parser.add_argument("--bushon-eval", type=Path, default=DEFAULT_BUSHON)
    parser.add_argument("--wiki", type=Path, default=DEFAULT_WIKI)
    parser.add_argument("--wiki-n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict-bushon", action="store_true")
    parser.add_argument("--allow-gym-uncertain", action="store_true")
    args = parser.parse_args()

    pipe = _build_pipe(args.weights, args.thresholds)
    failures: list[str] = []
    soft: list[str] = []

    print(f"weights={args.weights}")
    print(f"{'tag':12} {'file':40} {'L0':8} {'decision':10} fusion")

    # --- checked-in samples ---
    for split, expect in (("real", "realish"), ("ai_generated", "ai")):
        for path in sorted((args.samples / split).glob("*")):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            decision, fusion, l0 = _run_one(pipe, path)
            l0_s = "n/a" if l0 is None else f"{l0:.3f}"
            print(f"{split[:12]:12} {path.name:40} {l0_s:8} {decision or 'n/a':10} {fusion}")
            if decision is None:
                failures.append(f"{path.name}: undecodable")
                continue
            if expect == "ai" and decision == "real":
                failures.append(f"{path.name}: FN decision=real fusion={fusion}")
            if path.name == "real_street_florence.jpg" and decision == "generated":
                failures.append(f"{path.name}: FP decision=generated fusion={fusion}")
            if path.name == "real_hand_gym.jpg" and decision == "generated":
                failures.append(f"{path.name}: FP decision=generated fusion={fusion}")
            if (
                path.name == "real_hand_gym.jpg"
                and decision == "uncertain"
                and not args.allow_gym_uncertain
            ):
                soft.append(f"{path.name}: uncertain (prefer real)")

    # --- bushon leave-out on E ---
    for path in _iter_images(args.bushon_eval):
        decision, fusion, l0 = _run_one(pipe, path)
        l0_s = "n/a" if l0 is None else f"{l0:.3f}"
        print(f"{'bushon_eval':12} {path.name:40} {l0_s:8} {decision or 'n/a':10} {fusion}")
        if decision == "generated":
            msg = f"bushon {path.name}: decision=generated fusion={fusion}"
            if args.strict_bushon:
                failures.append(msg)
            else:
                soft.append(msg)

    # --- random wiki hands ---
    wiki_paths = _iter_images(args.wiki / "images") or _iter_images(args.wiki)
    rng = random.Random(args.seed)
    if wiki_paths and args.wiki_n > 0:
        pick = wiki_paths if len(wiki_paths) <= args.wiki_n else rng.sample(wiki_paths, args.wiki_n)
        for path in sorted(pick, key=lambda p: p.name):
            decision, fusion, l0 = _run_one(pipe, path)
            l0_s = "n/a" if l0 is None else f"{l0:.3f}"
            print(f"{'wiki':12} {path.name:40} {l0_s:8} {decision or 'n/a':10} {fusion}")
            if decision == "generated":
                soft.append(f"wiki {path.name}: decision=generated fusion={fusion}")

    if soft:
        print("\nSOFT")
        for line in soft:
            print(" -", line)
    if failures:
        print("\nHARD FAIL")
        for line in failures:
            print(" -", line)
        return 1
    print("\nHARD PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
