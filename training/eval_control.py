"""Control evaluation: score a trained probe on real photos from a different source.

The probe is trained with COCO as the only real class. If it is detecting
generation rather than dataset style, real photos from another source
(Open Images) should be rarely flagged, and generated images should still
score high on them.

Usage:
    python training/eval_control.py --root /mnt/disk-e/gavel-data --weights models/detector_weights.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from train_head import SEED, load, split_indices, tpr_at_fpr


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=Path("models/detector_weights.npz"))
    parser.add_argument("--out", type=Path, default=Path("models/control_metrics.json"))
    args = parser.parse_args()
    features_dir = args.features_dir or args.root / "features_normalized"

    weights = np.load(args.weights)
    coef, intercept = weights["coef"], float(weights["intercept"])

    def score(x: np.ndarray) -> np.ndarray:
        return _sigmoid(x @ coef + intercept)

    # Reproduce the exact train/calibration/test split used by train_head.py.
    rng = np.random.default_rng(SEED)
    nano_x, nano_y, nano_g, nano_r = load(features_dir, "nano_banana")
    cf_x, cf_y, cf_g, cf_r = load(features_dir, "cf_commercial_0")
    coco_x, _, _, _ = load(features_dir, "coco")
    train_mask = nano_r == "train"
    cf_train = cf_r == "train"
    pos_x = np.concatenate([nano_x[train_mask], cf_x[cf_train & (cf_y == 1)]])
    pos_tr, pos_ca, pos_te = split_indices(len(pos_x), rng)
    neg_tr, neg_ca, neg_te = split_indices(len(coco_x), rng)

    # Threshold at 1% FPR on COCO held-out real photos (the calibration split).
    coco_ca_scores = score(coco_x[neg_ca])
    threshold_1pct = float(np.quantile(coco_ca_scores, 0.99))

    ctrl_x, _, _, _ = load(features_dir, "open_images")
    ctrl_scores = score(ctrl_x)
    coco_test_scores = score(coco_x[neg_te])

    result = {
        "threshold_at_1pct_fpr_on_coco": threshold_1pct,
        "real_source_check": {
            "coco_test_flagged_rate": float(np.mean(coco_test_scores >= threshold_1pct)),
            "open_images_flagged_rate": float(np.mean(ctrl_scores >= threshold_1pct)),
            "n_open_images": int(len(ctrl_x)),
        },
        "generated_vs_open_images": {},
    }

    pos_te_x = pos_x[pos_te]
    pos_te_scores = score(pos_te_x)
    result["generated_vs_open_images"]["in_distribution"] = _pair_metrics(pos_te_scores, ctrl_scores)

    eval_mask = cf_r == "eval_only"
    for gen in sorted(set(cf_g[eval_mask])):
        gen_x = cf_x[eval_mask & (cf_g == gen)]
        gen_scores = score(gen_x)
        result["generated_vs_open_images"][gen] = _pair_metrics(gen_scores, ctrl_scores)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def _pair_metrics(pos_scores: np.ndarray, neg_scores: np.ndarray) -> dict:
    y = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    s = np.concatenate([pos_scores, neg_scores])
    return {
        "auc": float(roc_auc_score(y, s)),
        "tpr_at_1pct_fpr": tpr_at_fpr(y, s, 0.01),
        "n_generated": int(len(pos_scores)),
        "n_real": int(len(neg_scores)),
    }


if __name__ == "__main__":
    main()
