"""Train the linear probe with any set of real-photo sources and calibrate decision thresholds.

Each real source is split 70/10/20 independently. Training uses the train splits of all
real sources and the permissive generators. Thresholds are calibrated on the calibration
splits only. Test numbers are reported per real source, so a source that the probe still
over-flags shows up directly.

Outputs in --out-dir:
    detector_weights.npz   coefficients and intercept
    thresholds.json        high (1% FPR on real calibration) and low (0.1% of generated calibration)
    metrics.json           AUC, TPR, and decision shares per source and per generator

Usage:
    python training/train_detector.py --root /mnt/disk-e/gavel-data \
        --real-sources coco,open_images,wikimedia --out-dir models/exp_three_real
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from train_head import SEED, split_indices, tpr_at_fpr

# Layer-mode files are named after the source without the shard suffix.
LAYER_FILE_NAMES = {"cf_commercial_0": "cf_commercial"}


def load_features(features_dir: Path, name: str, feature_kind: str, layer_indices: list[int]):
    """Return (X, labels, generators, roles) for a source in either feature layout."""
    if feature_kind == "normalized":
        data = np.load(features_dir / f"{name}.npz")
        x = data["features"].astype(np.float32)
    else:
        data = np.load(features_dir / f"{LAYER_FILE_NAMES.get(name, name)}.npz")
        layers = data["layers"][:, layer_indices].astype(np.float32)
        x = layers.reshape(len(layers), -1)
    return x, data["labels"], data["generators"], data["roles"]


def _pair(pos: np.ndarray, neg: np.ndarray, clf) -> dict:
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    s = clf.predict_proba(np.concatenate([pos, neg]))[:, 1]
    return {
        "auc": float(roc_auc_score(y, s)),
        "tpr_at_1pct_fpr": tpr_at_fpr(y, s, 0.01),
        "tpr_at_0.1pct_fpr": tpr_at_fpr(y, s, 0.001),
        "n_generated": int(len(pos)),
        "n_real": int(len(neg)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--real-sources", default="coco,open_images")
    parser.add_argument(
        "--gen-sources",
        default="nano_banana,cf_commercial_0",
        help="comma-separated generated feature files used for training",
    )
    parser.add_argument("--feature-kind", choices=["normalized", "layers"], default="normalized")
    parser.add_argument("--layer-indices", default="0,1,2,3", help="layer positions in the layers file")
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument(
        "--max-per-real-source",
        type=int,
        default=0,
        help="if >0, randomly subsample each real source to at most N rows before split "
        "(keeps HaGRID from drowning coco/scenes; 0 = use all)",
    )
    parser.add_argument(
        "--max-real-source",
        action="append",
        default=[],
        metavar="NAME=N",
        help="per-source cap NAME=N (overrides --max-per-real-source for that name); repeatable",
    )
    parser.add_argument(
        "--calibrate-real-sources",
        default="",
        help="comma-separated real sources used for the high-threshold 1%% FPR quantile; "
        "empty = all real sources in --real-sources. Train still uses all real sources. "
        "Example: coco,open_images,coco_scenes (exclude real_hands from thr flood).",
    )
    parser.add_argument(
        "--real-source-weight",
        action="append",
        default=[],
        metavar="NAME=W",
        help="multiply sample_weight for that real source's train rows (default 1.0); repeatable",
    )
    parser.add_argument(
        "--no-class-weight",
        action="store_true",
        help="disable LogisticRegression class_weight='balanced'",
    )
    parser.add_argument(
        "--C",
        default="",
        help="comma-separated C values to try (default: 0.1,0.3,1,3,10); "
        "single value forces that C",
    )
    args = parser.parse_args()
    features_dir = args.features_dir or args.root / (
        "features_normalized" if args.feature_kind == "normalized" else "features_layers"
    )
    layer_indices = [int(i) for i in args.layer_indices.split(",") if i.strip()]
    real_names = [name.strip() for name in args.real_sources.split(",") if name.strip()]
    gen_names = [name.strip() for name in args.gen_sources.split(",") if name.strip()]
    rng = np.random.default_rng(SEED)

    def load(name: str):
        return load_features(features_dir, name, args.feature_kind, layer_indices)

    pos_parts = []
    cf_x = cf_y = cf_g = cf_r = None
    used_gen: list[str] = []
    for name in gen_names:
        try:
            x, y, g, r = load(name)
        except FileNotFoundError:
            print(f"skip missing gen source: {name}", flush=True)
            continue
        if name.startswith("cf_commercial"):
            cf_x, cf_y, cf_g, cf_r = x, y, g, r
            mask = (r == "train") & (y == 1)
        else:
            mask = (r == "train") & (y == 1) if (r == "train").any() else (y == 1)
        if mask.any():
            pos_parts.append(x[mask])
            used_gen.append(name)
    if not pos_parts:
        raise RuntimeError(f"no generated training samples from {gen_names}")
    pos_x = np.concatenate(pos_parts)

    pos_tr, pos_ca, pos_te = split_indices(len(pos_x), rng)
    per_source_cap: dict[str, int] = {}
    for item in args.max_real_source:
        if "=" not in item:
            raise RuntimeError(f"--max-real-source expected NAME=N, got {item!r}")
        src, _, raw_n = item.partition("=")
        per_source_cap[src.strip()] = int(raw_n)

    real = {}
    for name in real_names:
        try:
            x = load(name)[0]
        except FileNotFoundError:
            print(f"skip missing real source: {name}", flush=True)
            continue
        cap = per_source_cap.get(name, args.max_per_real_source)
        if cap and len(x) > cap:
            pick = rng.choice(len(x), size=cap, replace=False)
            pick.sort()
            x = x[pick]
            print(f"subsample {name}: {cap} kept", flush=True)
        real[name] = x
    real_names = list(real)
    if not real_names:
        raise RuntimeError("no real training sources available")
    splits = {name: split_indices(len(x), rng) for name, x in real.items()}
    gen_names = used_gen

    scaler = None
    if args.standardize:
        raw_train = np.concatenate([pos_x[pos_tr]] + [real[n][splits[n][0]] for n in real_names])
        scaler = StandardScaler().fit(raw_train)
        pos_x = scaler.transform(pos_x)
        if cf_x is not None:
            cf_x = scaler.transform(cf_x)
        real = {n: scaler.transform(x) for n, x in real.items()}

    source_w: dict[str, float] = {}
    for item in args.real_source_weight:
        if "=" not in item:
            raise RuntimeError(f"--real-source-weight expected NAME=W, got {item!r}")
        src, _, raw_w = item.partition("=")
        source_w[src.strip()] = float(raw_w)
        print(f"real-source-weight {src.strip()}={raw_w}", flush=True)

    train_blocks = [pos_x[pos_tr]] + [real[n][splits[n][0]] for n in real_names]
    X_train = np.concatenate(train_blocks)
    y_train = np.concatenate([np.ones(len(pos_tr))] + [np.zeros(len(splits[n][0])) for n in real_names])
    sw_parts = [np.ones(len(pos_tr), dtype=np.float64)]
    for n in real_names:
        w = float(source_w.get(n, 1.0))
        sw_parts.append(np.full(len(splits[n][0]), w, dtype=np.float64))
    sample_weight = np.concatenate(sw_parts)
    # Model-selection calib: all reals. High-threshold calib: optional subset.
    real_calib = np.concatenate([real[n][splits[n][1]] for n in real_names])
    calib_names = [n.strip() for n in args.calibrate_real_sources.split(",") if n.strip()]
    if calib_names:
        missing = [n for n in calib_names if n not in real]
        if missing:
            raise RuntimeError(f"--calibrate-real-sources not in loaded reals: {missing}")
        real_thr_calib = np.concatenate([real[n][splits[n][1]] for n in calib_names])
        print(f"high-threshold calib sources: {calib_names} (n={len(real_thr_calib)})", flush=True)
    else:
        calib_names = list(real_names)
        real_thr_calib = real_calib
    X_calib = np.concatenate([pos_x[pos_ca], real_calib])
    y_calib = np.concatenate([np.ones(len(pos_ca)), np.zeros(len(real_calib))])

    cw = None if args.no_class_weight else "balanced"
    if args.C.strip():
        c_grid = tuple(float(x) for x in args.C.split(",") if x.strip())
    else:
        c_grid = (0.1, 0.3, 1.0, 3.0, 10.0)
    best = None
    for c in c_grid:
        clf = LogisticRegression(C=c, max_iter=2000, class_weight=cw)
        clf.fit(X_train, y_train, sample_weight=sample_weight)
        auc = roc_auc_score(y_calib, clf.predict_proba(X_calib)[:, 1])
        if best is None or auc > best[0]:
            best = (auc, c, clf)
    _, chosen_c, clf = best
    print(f"chosen C={chosen_c} class_weight={cw!r} grid={c_grid}", flush=True)

    # Thresholds come from calibration splits only, never from test splits.
    gen_calib_scores = clf.predict_proba(pos_x[pos_ca])[:, 1]
    real_calib_scores = clf.predict_proba(real_thr_calib)[:, 1]
    thresholds = {
        "high": float(np.quantile(real_calib_scores, 0.99)),
        "low": float(np.quantile(gen_calib_scores, 0.001)),
        "real_sources": real_names,
        "calibrate_real_sources": calib_names,
        "gen_sources": gen_names,
        "high_meaning": "probability at or above which 1% of real calibration photos are flagged",
        "low_meaning": "probability below which 0.1% of generated calibration images fall",
    }
    if not thresholds["low"] < thresholds["high"]:
        raise RuntimeError(f"invalid thresholds: {thresholds}")

    def decide(scores: np.ndarray) -> dict:
        shares = {
            "generated": float(np.mean(scores >= thresholds["high"])),
            "uncertain": float(np.mean((scores < thresholds["high"]) & (scores >= thresholds["low"]))),
            "real": float(np.mean(scores < thresholds["low"])),
        }
        return {k: round(v, 4) for k, v in shares.items()}

    metrics = {
        "C": chosen_c,
        "thresholds": thresholds,
        "in_distribution": {},
        "decisions_on_test": {
            "generated_in_distribution": decide(clf.predict_proba(pos_x[pos_te])[:, 1]),
        },
        "held_out_generators": {},
    }
    real_test = []
    for name in real_names:
        test_x = real[name][splits[name][2]]
        real_test.append(test_x)
        metrics["in_distribution"][f"vs_{name}"] = _pair(pos_x[pos_te], test_x, clf)
        metrics["decisions_on_test"][f"{name}_real"] = decide(clf.predict_proba(test_x)[:, 1])
    real_test = np.concatenate(real_test)

    if cf_x is not None and cf_r is not None:
        eval_mask = cf_r == "eval_only"
        for gen in sorted(set(cf_g[eval_mask])):
            gen_x = cf_x[eval_mask & (cf_g == gen)]
            metrics["held_out_generators"][gen] = _pair(gen_x, real_test, clf)
            metrics["decisions_on_test"][f"held_out_{gen}"] = decide(clf.predict_proba(gen_x)[:, 1])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics["feature_kind"] = args.feature_kind
    metrics["layer_indices"] = layer_indices
    metrics["standardized"] = bool(args.standardize)
    np.savez(
        args.out_dir / "detector_weights.npz",
        coef=clf.coef_[0].astype(np.float32),
        intercept=np.float32(clf.intercept_[0]),
        model_name="ViT-L-14",
        pretrained="laion2b_s32b_b82k",
    )
    (args.out_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
