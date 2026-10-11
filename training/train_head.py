"""Train a linear probe on cached CLIP features and evaluate on held-out generators.

Training uses only samples with role "train" (permissively licensed generators
and COCO real photos). Held-out evaluation adds every "eval_only" generator
paired with COCO photos that were never seen in training.

Outputs:
    <out-dir>/detector_weights.npz   coefficients, intercept, metadata
    <out-dir>/metrics.json           AUC, TPR at 1% and 0.1% FPR, per generator

Usage:
    python training/train_head.py --root /mnt/disk-e/gavel-data --out-dir models
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

SEED = 42


def load(features_dir: Path, source: str):
    data = np.load(features_dir / f"{source}.npz")
    return data["features"].astype(np.float32), data["labels"], data["generators"], data["roles"]


def tpr_at_fpr(y: np.ndarray, score: np.ndarray, target: float) -> float:
    fpr, tpr, _ = roc_curve(y, score)
    return float(tpr[fpr <= target].max()) if np.any(fpr <= target) else 0.0


def split_indices(n: int, rng: np.random.Generator):
    idx = rng.permutation(n)
    n_train, n_calib = int(0.7 * n), int(0.1 * n)
    return idx[:n_train], idx[n_train : n_train + n_calib], idx[n_train + n_calib :]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    features_dir = args.features_dir or args.root / "features_normalized"
    rng = np.random.default_rng(SEED)

    nano_x, nano_y, nano_g, nano_r = load(features_dir, "nano_banana")
    cf_x, cf_y, cf_g, cf_r = load(features_dir, "cf_commercial_0")
    coco_x, coco_y, coco_g, coco_r = load(features_dir, "coco")

    train_mask = nano_r == "train"
    cf_train = cf_r == "train"
    eval_mask = cf_r == "eval_only"

    # In-distribution pools: permissive generators and COCO real photos.
    pos_x = np.concatenate([nano_x[train_mask], cf_x[cf_train & (cf_y == 1)]])
    pos_g = np.concatenate([nano_g[train_mask], cf_g[cf_train & (cf_y == 1)]])
    neg_x, neg_g = coco_x, coco_g

    pos_tr, pos_ca, pos_te = split_indices(len(pos_x), rng)
    neg_tr, neg_ca, neg_te = split_indices(len(neg_x), rng)

    X_train = np.concatenate([pos_x[pos_tr], neg_x[neg_tr]])
    y_train = np.concatenate([np.ones(len(pos_tr)), np.zeros(len(neg_tr))])
    X_calib = np.concatenate([pos_x[pos_ca], neg_x[neg_ca]])
    y_calib = np.concatenate([np.ones(len(pos_ca)), np.zeros(len(neg_ca))])
    X_test = np.concatenate([pos_x[pos_te], neg_x[neg_te]])
    y_test = np.concatenate([np.ones(len(pos_te)), np.zeros(len(neg_te))])
    g_test = np.concatenate([pos_g[pos_te], neg_g[neg_te]])

    best = None
    for c in (0.1, 0.3, 1.0, 3.0, 10.0):
        clf = LogisticRegression(C=c, max_iter=2000, class_weight="balanced")
        clf.fit(X_train, y_train)
        auc = roc_auc_score(y_calib, clf.predict_proba(X_calib)[:, 1])
        if best is None or auc > best[0]:
            best = (auc, c, clf)
    _, chosen_c, clf = best

    metrics = {
        "C": chosen_c,
        "in_distribution": {},
        "held_out_generators": {},
    }
    score_test = clf.predict_proba(X_test)[:, 1]
    metrics["in_distribution"] = {
        "auc": float(roc_auc_score(y_test, score_test)),
        "tpr_at_1pct_fpr": tpr_at_fpr(y_test, score_test, 0.01),
        "tpr_at_0.1pct_fpr": tpr_at_fpr(y_test, score_test, 0.001),
        "n_positive": int(y_test.sum()),
        "n_negative": int((1 - y_test).sum()),
    }

    neg_eval_x = neg_x[neg_te]
    for gen in sorted(set(cf_g[eval_mask])):
        gen_x = cf_x[eval_mask & (cf_g == gen)]
        gen_x = gen_x[: len(neg_eval_x)]
        X = np.concatenate([gen_x, neg_eval_x])
        y = np.concatenate([np.ones(len(gen_x)), np.zeros(len(neg_eval_x))])
        score = clf.predict_proba(X)[:, 1]
        metrics["held_out_generators"][gen] = {
            "auc": float(roc_auc_score(y, score)),
            "tpr_at_1pct_fpr": tpr_at_fpr(y, score, 0.01),
            "n_generated": int(len(gen_x)),
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out_dir / "detector_weights.npz",
        coef=clf.coef_[0].astype(np.float32),
        intercept=np.float32(clf.intercept_[0]),
        train_generators=np.array(sorted(set(pos_g.tolist()))),
        model_name="ViT-L-14",
        pretrained="laion2b_s32b_b82k",
    )
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
