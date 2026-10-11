"""Fine-tune an ImageNet ResNet-50 to separate generated from real photos.

Input comes from the native-resolution JPEG caches built by training/build_cnn_cache.py.
Training uses random 224-pixel crops with horizontal flips. Evaluation uses the center crop.
Splits and held-out generators are the same as in training/train_detector.py.

Outputs in --out-dir:
    detector.pt        state dict (not committed, *.pt is ignored)
    thresholds.json    high (1% FPR on real validation) and low (0.1% of generated validation)
    metrics.json       AUC and decision shares per source and per held-out generator

Usage:
    python training/train_cnn.py --root /mnt/disk-e/gavel-data --out-dir models/cnn --epochs 3
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_head import SEED, split_indices, tpr_at_fpr  # noqa: E402

CROP = 224
REAL_SOURCES = ["coco", "open_images", "wikimedia"]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class CacheSource:
    """Read-only access to one cache directory."""

    def __init__(self, cache_dir: Path) -> None:
        self.offsets = np.load(cache_dir / "offsets.npy")
        self.labels = np.load(cache_dir / "labels.npy")
        self.generators = np.load(cache_dir / "generators.npy")
        self.roles = np.load(cache_dir / "roles.npy")
        self.path = cache_dir / "data.bin"
        self._data: np.memmap | None = None

    def __len__(self) -> int:
        return len(self.labels)

    def read(self, index: int) -> bytes:
        if self._data is None:
            self._data = np.memmap(self.path, dtype=np.uint8, mode="r")
        return bytes(self._data[self.offsets[index] : self.offsets[index + 1]])


class ItemDataset(Dataset):
    def __init__(self, sources: dict[str, CacheSource], items: list[tuple[str, int, int]], train: bool):
        self.sources = sources
        self.items = items
        normalize = transforms.Normalize(MEAN, STD)
        if train:
            self.transform = transforms.Compose(
                [
                    transforms.RandomCrop(CROP, pad_if_needed=True),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            self.transform = transforms.Compose(
                [transforms.CenterCrop(CROP), transforms.ToTensor(), normalize]
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        name, index, label = self.items[i]
        img = Image.open(io.BytesIO(self.sources[name].read(index))).convert("RGB")
        return self.transform(img), float(label)


def build_splits(sources: dict[str, CacheSource], rng: np.random.Generator):
    """Return train, val and test item lists plus held-out generator items."""

    pos_pool: list[tuple[str, int, int]] = []
    for name in ("nano_banana", "cf_commercial_0"):
        src = sources[name]
        for idx in np.flatnonzero((src.labels == 1) & (src.roles == "train")):
            pos_pool.append((name, int(idx), 1))
    pos_tr, pos_va, pos_te = split_indices(len(pos_pool), rng)

    real_pools: dict[str, list[tuple[str, int, int]]] = {}
    for name in REAL_SOURCES:
        src = sources[name]
        real_pools[name] = [(name, int(i), 0) for i in range(len(src))]

    train, val, test = [], [], []
    for pos_idx, sink in ((pos_tr, train), (pos_va, val), (pos_te, test)):
        sink.extend(pos_pool[i] for i in pos_idx)
    real_test: dict[str, list] = {}
    for name in REAL_SOURCES:
        tr, va, te = split_indices(len(real_pools[name]), rng)
        train.extend(real_pools[name][i] for i in tr)
        val.extend(real_pools[name][i] for i in va)
        real_test[name] = [real_pools[name][i] for i in te]
        test.extend(real_test[name])

    held_out: dict[str, list] = {}
    cf = sources["cf_commercial_0"]
    for idx in np.flatnonzero(cf.roles == "eval_only"):
        gen = str(cf.generators[idx])
        held_out.setdefault(gen, []).append(("cf_commercial_0", int(idx), 1))
    return train, val, test, real_test, held_out, pos_pool, pos_te


def score_items(model, device, sources, items, batch_size=128) -> np.ndarray:
    loader = DataLoader(ItemDataset(sources, items, train=False), batch_size=batch_size, num_workers=6)
    scores = []
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for x, _ in loader:
            scores.append(torch.sigmoid(model(x.to(device)).float().squeeze(1)).cpu().numpy())
    return np.concatenate(scores) if scores else np.zeros(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("models/cnn"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--limit-train", type=int, default=0, help="smoke test: cap training items")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    cache = args.root / "cnn_cache"
    names = ["nano_banana", "cf_commercial_0", *REAL_SOURCES]
    sources = {n: CacheSource(cache / ("cf_commercial" if n == "cf_commercial_0" else n)) for n in names}

    train, val, test, real_test, held_out, pos_pool, pos_te = build_splits(sources, rng)
    if args.limit_train:
        train = train[: args.limit_train]
    print(f"train {len(train)}  val {len(val)}  test {len(test)}", flush=True)

    device = "cuda"
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)

    n_pos = sum(1 for it in train if it[2] == 1)
    n_neg = len(train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_loader = DataLoader(
        ItemDataset(sources, train, train=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x).float().squeeze(1)
                loss = loss_fn(logits, y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running = 0.98 * running + 0.02 * loss.item() if step else loss.item()
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss {running:.4f}", flush=True)
        val_scores = score_items(model, device, sources, val)
        val_labels = np.array([it[2] for it in val])
        print(f"epoch {epoch} val AUC {roc_auc_score(val_labels, val_scores):.4f}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out_dir / "detector.pt")

    val_real = val_scores[val_labels == 0]
    val_gen = val_scores[val_labels == 1]
    thresholds = {
        "high": float(np.quantile(val_real, 0.99)),
        "low": float(np.quantile(val_gen, 0.001)),
        "real_sources": REAL_SOURCES,
        "input": "native-resolution JPEG q90 cache, center crop 224",
        "high_meaning": "score at or above which 1% of real validation photos are flagged",
        "low_meaning": "score below which 0.1% of generated validation images fall",
    }
    (args.out_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

    def decide(scores: np.ndarray) -> dict:
        shares = {
            "generated": float(np.mean(scores >= thresholds["high"])),
            "uncertain": float(np.mean((scores < thresholds["high"]) & (scores >= thresholds["low"]))),
            "real": float(np.mean(scores < thresholds["low"])),
        }
        return {k: round(v, 4) for k, v in shares.items()}

    def pair_metrics(pos: np.ndarray, neg: np.ndarray) -> dict:
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        s = np.concatenate([pos, neg])
        return {
            "auc": float(roc_auc_score(y, s)),
            "tpr_at_1pct_fpr": tpr_at_fpr(y, s, 0.01),
            "n_generated": int(len(pos)),
            "n_real": int(len(neg)),
        }

    pos_test_scores = score_items(model, device, sources, [pos_pool[i] for i in pos_te])
    metrics = {
        "thresholds": thresholds,
        "decisions_on_test": {"generated_in_distribution": decide(pos_test_scores)},
        "in_distribution": {},
        "held_out_generators": {},
    }
    real_test_all = []
    for name in REAL_SOURCES:
        scores = score_items(model, device, sources, real_test[name])
        real_test_all.append(scores)
        metrics["in_distribution"][f"vs_{name}"] = pair_metrics(pos_test_scores, scores)
        metrics["decisions_on_test"][f"{name}_real"] = decide(scores)
    real_test_all = np.concatenate(real_test_all)
    for gen, items in sorted(held_out.items()):
        scores = score_items(model, device, sources, items)
        metrics["held_out_generators"][gen] = pair_metrics(scores, real_test_all)
        metrics["decisions_on_test"][f"held_out_{gen}"] = decide(scores)

    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
