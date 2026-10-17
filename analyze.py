"""Run the Gavel AI pipeline on real image files.

Usage:
    python analyze.py samples/real/*.jpg samples/ai_generated/*.jpg --out-dir results

For every input the script writes:
    <out-dir>/<name>_overlay.png   annotated image
    <out-dir>/<name>.json          EvidenceProtocol
and prints final decision, fusion, L0, checks.

Anatomy needs mediapipe==0.10.14 (Python 3.11 or lower); without it anatomy is skipped.
Physics uses the classical shadow scene provider; skips when fewer than two rays.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2

from core.learned_detector import LearnedDetector
from core.pipeline import GavelPipeline
from core.verdict import compute_verdict


def analyze_file(pipeline: GavelPipeline, path: Path, out_dir: Path) -> dict:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    result = pipeline.analyze(image, image_name=path.name)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem
    cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), result.annotated_image)
    (out_dir / f"{stem}.json").write_text(result.protocol.to_json(), encoding="utf-8")

    verdict = compute_verdict(result.protocol)
    detector = result.protocol.detector
    return {
        "file": str(path),
        "size": f"{image.shape[1]}x{image.shape[0]}",
        "decision": verdict.decision,
        "label": verdict.label,
        "lean": verdict.lean,
        "guidance": verdict.guidance,
        "fusion": verdict.fusion,
        "probability": verdict.probability,
        "detector": None if detector is None else f"{detector.decision} (p={detector.probability:.3f})",
        "checks": {c.check: c.status for c in result.protocol.checks},
        "evidences": [
            (e.evidence_id, e.anomaly_type.value, e.severity.value) for e in result.protocol.evidences
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gavel AI on real images")
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--log-level", default="WARNING")
    parser.add_argument("--no-detector", action="store_true", help="skip the learned detector")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")

    detector = None if args.no_detector else LearnedDetector()
    pipeline = GavelPipeline(learned_detector=detector)
    for path in args.images:
        summary = analyze_file(pipeline, path, args.out_dir)
        print(f"\n{summary['file']} ({summary['size']})")
        print(f"  final                  {summary['decision']} ({summary['label']})")
        print(f"  fusion                 {summary['fusion']}")
        if summary.get("lean"):
            print(f"  lean                   {summary['lean']}")
        if summary.get("guidance"):
            print(f"  guidance               {summary['guidance']}")
        if summary["probability"] is not None:
            print(f"  L0 probability         {summary['probability']:.3f}")
        if summary["detector"]:
            print(f"  learned detector       {summary['detector']}")
        for check, status in summary["checks"].items():
            print(f"  {check:<22} {status}")
        if not summary["evidences"]:
            print("  evidences: none")
        for evidence_id, anomaly, severity in summary["evidences"]:
            print(f"  {evidence_id:<8} {anomaly} [{severity}]")


if __name__ == "__main__":
    main()
