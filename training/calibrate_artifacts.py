"""Calibrate L2 artifact vote threshold on already-downloaded real photos.

Uses samples/real plus up to N images from on-disk Wikimedia/COCO dirs.
Writes suggested threshold into models/layer_thresholds.json (artifacts only).

Usage:
    python training/calibrate_artifacts.py --root /mnt/disk-e/gavel-data
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from core.validators.artifacts import ArtifactConfig, ArtifactValidator  # noqa: E402


def _iter_images(dirs: list[Path], limit: int, rng: random.Random) -> list[Path]:
    paths: list[Path] = []
    for d in dirs:
        if not d.is_dir():
            continue
        for p in d.rglob("*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                paths.append(p)
    rng.shuffle(paths)
    return paths[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/mnt/disk-e/gavel-data"))
    parser.add_argument("--limit-real", type=int, default=400)
    parser.add_argument("--target-fpr", type=float, default=0.02)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "models" / "layer_thresholds.json",
        help="layer_thresholds.json to update (artifacts.signal_threshold)",
    )
    args = parser.parse_args()

    sample_reals = sorted((ROOT / "samples" / "real").glob("*"))
    disk_reals = _iter_images(
        [
            args.root / "real" / "wikimedia" / "images",
            args.root / "real" / "wikimedia_faces" / "images",
            args.root / "real" / "wikimedia_extra" / "images",
            args.root / "real" / "wikimedia_scenes" / "images",
        ],
        limit=args.limit_real,
        rng=random.Random(0),
    )
    paths = [p for p in sample_reals if p.suffix.lower() in {".jpg", ".jpeg", ".png"}] + disk_reals

    # Score with dampening disabled mapping only (raw ArtifactValidator score).
    validator = ArtifactValidator(ArtifactConfig(signal_threshold=1.1))
    scores: list[float] = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        scores.append(validator.analyze(img).layer.score or 0.0)

    if len(scores) < 20:
        raise SystemExit(f"too few real images scored: {len(scores)}")

    arr = np.asarray(scores, dtype=np.float64)
    # Threshold above the (1 - target_fpr) quantile of real scores.
    thr = float(np.quantile(arr, 1.0 - args.target_fpr))
    thr = min(0.95, max(0.55, thr + 1e-3))

    print(f"n_real={len(arr)} mean={arr.mean():.3f} p50={np.median(arr):.3f} "
          f"p{100*(1-args.target_fpr):.0f}={np.quantile(arr, 1-args.target_fpr):.3f}")
    print(f"suggested artifacts.signal_threshold={thr:.3f}")

    # Show sample real scores
    for p in sample_reals:
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        img = cv2.imread(str(p))
        s = validator.analyze(img).layer.score or 0.0
        print(f"  sample {p.name}: score={s:.3f} vote={s >= thr}")

    if args.out.exists():
        raw = json.loads(args.out.read_text(encoding="utf-8"))
    else:
        raw = {}
    art = raw.setdefault("artifacts", {})
    art["signal_threshold"] = thr
    art["source"] = (
        f"calibrate_artifacts.py on {len(arr)} on-disk reals; "
        f"target_fpr={args.target_fpr}"
    )
    art["votes_via"] = f"score >= {thr:.3f}"
    args.out.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    print(f"updated {args.out}")


if __name__ == "__main__":
    main()
