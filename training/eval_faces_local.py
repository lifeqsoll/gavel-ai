"""R3: calibrate plastic_skin / face_clip on real faces (disk-E) vs samples AI.

Does not ship weights. Exit 0 if no real-face plastic FP under current thresholds
and ai_08 still trips plastic when a face crop exists.

Usage:
    ~/gavel/train-venv/bin/python training/eval_faces_local.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.anatomy_agent import AnatomyAgent  # noqa: E402
from agents.geometry_agent import GeometryAgent  # noqa: E402
from core.learned_detector import LearnedDetector  # noqa: E402
from core.pipeline import GavelPipeline  # noqa: E402
from core.validators.perspective import PerspectiveConfig, PerspectiveValidator  # noqa: E402
from core.verdict import compute_verdict  # noqa: E402

DEFAULT_FACES = Path("/mnt/disk-e/gavel-data/real/wikimedia_faces")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faces", type=Path, default=DEFAULT_FACES)
    parser.add_argument("--samples", type=Path, default=ROOT / "samples")
    parser.add_argument("--limit-real", type=int, default=40)
    args = parser.parse_args()

    pipe = GavelPipeline(
        geometry_agent=GeometryAgent(method="lsd"),
        perspective_validator=PerspectiveValidator(PerspectiveConfig()),
        anatomy_agent=AnatomyAgent(),
        learned_detector=LearnedDetector(),
    )

    failures: list[str] = []
    real_laps: list[float] = []
    real_plastic = 0
    real_faces = 0

    print(f"{'tag':10} {'file':40} {'lap':8} {'clip':8} plastic decision fusion")
    for path in _images(args.faces)[: args.limit_real]:
        img = cv2.imread(str(path))
        if img is None:
            continue
        result = pipe.analyze(img, path.name)
        metrics = result.protocol.layers.global_layer.metrics or {}
        if "face_laplacian" not in metrics:
            continue
        real_faces += 1
        lap = float(metrics["face_laplacian"])
        clip = float(metrics.get("face_clip_frac") or 0.0)
        plastic = float(metrics.get("plastic_skin") or 0.0)
        real_laps.append(lap)
        if plastic >= 1.0:
            real_plastic += 1
            failures.append(f"real face plastic FP: {path.name} lap={lap:.1f}")
        verdict = compute_verdict(result.protocol)
        print(
            f"{'real':10} {path.name:40} {lap:8.1f} {clip:8.4f} "
            f"{int(plastic):7d} {verdict.decision or 'n/a':10} {verdict.fusion}"
        )

    ai08 = args.samples / "ai_generated" / "ai_08.jpg"
    if ai08.is_file():
        result = pipe.analyze(cv2.imread(str(ai08)), ai08.name)
        metrics = result.protocol.layers.global_layer.metrics or {}
        lap = metrics.get("face_laplacian")
        plastic = float(metrics.get("plastic_skin") or 0.0)
        verdict = compute_verdict(result.protocol)
        print(
            f"{'ai_08':10} {ai08.name:40} {float(lap or -1):8.1f} "
            f"{float(metrics.get('face_clip_frac') or 0):8.4f} "
            f"{int(plastic):7d} {verdict.decision or 'n/a':10} {verdict.fusion}"
        )
        if lap is not None and plastic < 1.0 and float(lap) < 20.0:
            failures.append("ai_08 lost plastic_skin cue")

    if real_laps:
        arr = np.asarray(real_laps, dtype=np.float64)
        print(
            f"\nreal faces with crop: {real_faces}; "
            f"lap p10/med/p90={np.percentile(arr,10):.1f}/"
            f"{np.median(arr):.1f}/{np.percentile(arr,90):.1f}; "
            f"plastic_fp={real_plastic}"
        )
    else:
        print("\nNo MediaPipe face crops on real set — plastic kept by prior ai_08 cal.")

    if failures:
        print("\nFACES FAIL")
        for line in failures:
            print(" -", line)
        return 1
    print("\nFACES PASS — keep plastic_skin (lap<20 or lap<40&clip≥0.05)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
