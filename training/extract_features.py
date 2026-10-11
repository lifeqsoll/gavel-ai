"""Extract frozen CLIP ViT-L/14 embeddings from the parquet datasets on disk.

Each source is streamed from parquet and cached as one .npz file with
features (float16, L2-normalized), labels (1 = generated, 0 = real),
generator names, and a role per sample: "train" (permissively licensed
generators and real photos) or "eval_only" (generators with restrictive
output terms, used only for held-out testing).

Usage:
    python training/extract_features.py --source nano_banana --root /mnt/disk-e/gavel-data
    python training/extract_features.py --source cf_commercial --root /mnt/disk-e/gavel-data
    python training/extract_features.py --source coco --root /mnt/disk-e/gavel-data
"""

from __future__ import annotations

import argparse
import glob
import io
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.normalize import normalize_pil  # noqa: E402

MODEL_NAME = "ViT-L-14"
PRETRAINED = "laion2b_s32b_b82k"

# Generators whose outputs carry permissive terms and may be used for training.
TRAIN_GENERATORS = {"nano-banana", "FLUX-schnell"}
# FLUX-dev is non-commercial; it is excluded from all use.
EXCLUDED_GENERATORS = {"FLUX-dev"}

_PREPROCESS = None


def _image_bytes(cell) -> bytes:
    if isinstance(cell, dict):
        return cell["bytes"]
    return cell


def iter_nano_banana(root: Path) -> Iterator[tuple[bytes, int, str]]:
    for path in sorted(glob.glob(str(root / "generated/nano-banana/data/*.parquet"))):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=["image"]):
            for cell in batch.column("image").to_pylist():
                yield _image_bytes(cell), 1, "nano-banana"


CF_FILE_PATTERN = "*.parquet"


def iter_cf_commercial(root: Path) -> Iterator[tuple[bytes, int, str]]:
    pattern = str(root / "generated/community-forensics-commercial/data/commercial" / CF_FILE_PATTERN)
    for path in sorted(glob.glob(pattern)):
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=64, columns=["image_data", "model_name", "label"]
        ):
            for img, model, label in zip(
                batch.column("image_data").to_pylist(),
                batch.column("model_name").to_pylist(),
                batch.column("label").to_pylist(),
            ):
                yield _image_bytes(img), int(label), model


def iter_coco(root: Path) -> Iterator[tuple[bytes, int, str]]:
    for path in sorted(glob.glob(str(root / "real/coco/data/*.parquet"))):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=["image"]):
            for cell in batch.column("image").to_pylist():
                yield _image_bytes(cell), 0, "coco"


def iter_open_images(root: Path) -> Iterator[tuple[bytes, int, str]]:
    for path in sorted(glob.glob(str(root / "real/open-images/data/*.parquet"))):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=["image"]):
            for cell in batch.column("image").to_pylist():
                yield _image_bytes(cell), 0, "open-images"


def iter_wikimedia(root: Path) -> Iterator[tuple[bytes, int, str]]:
    import csv

    base = root / "real/wikimedia"
    with (base / "manifest.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            yield (base / "images" / row["file"]).read_bytes(), 0, "wikimedia"


SOURCES = {
    "nano_banana": iter_nano_banana,
    "cf_commercial": iter_cf_commercial,
    "coco": iter_coco,
    "open_images": iter_open_images,
    "wikimedia": iter_wikimedia,
}


def _role(generator: str, label: int) -> str:
    if label == 0 or generator in TRAIN_GENERATORS:
        return "train"
    return "eval_only"


def _init_worker() -> None:
    global _PREPROCESS
    from open_clip import image_transform, get_model_config

    image_size = get_model_config(MODEL_NAME)["vision_cfg"]["image_size"]
    _PREPROCESS = image_transform(image_size, is_train=False)


def _normalize(raw: bytes) -> Image.Image:
    return normalize_pil(Image.open(io.BytesIO(raw)))


def _prepare(item):
    raw, label, gen = item
    try:
        img = _normalize(raw)
    except Exception:
        return None
    return _PREPROCESS(img), label, gen


CHUNK_SIZE = 256


def _chunks(iterable, size: int):
    buf = []
    for item in iterable:
        buf.append(item)
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf


def _labelled_stream(root: Path, source: str) -> Iterator[tuple[bytes, int, str]]:
    for raw, label, gen in SOURCES[source](root):
        if gen not in EXCLUDED_GENERATORS:
            yield raw, label, gen


def main() -> None:
    import open_clip

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=sorted(SOURCES), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0, help="stop after N images (0 = all)")
    parser.add_argument("--cf-file", default="*.parquet", help="glob for CommunityForensics shards")
    parser.add_argument("--out-name", default=None, help="output file stem (defaults to source name)")
    args = parser.parse_args()
    global CF_FILE_PATTERN
    CF_FILE_PATTERN = args.cf_file

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model = model.to(device).eval().half()

    feats: list[np.ndarray] = []
    labels: list[int] = []
    generators: list[str] = []
    roles: list[str] = []

    def encode(tensors: list[torch.Tensor], metas: list[tuple[int, str]]) -> None:
        batch = torch.stack(tensors).to(device).half()
        with torch.no_grad():
            emb = model.encode_image(batch)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        feats.append(emb.float().cpu().numpy().astype(np.float16))
        for label, gen in metas:
            labels.append(label)
            generators.append(gen)
            roles.append(_role(gen, label))

    tensors: list[torch.Tensor] = []
    metas: list[tuple[int, str]] = []
    seen = 0
    stream = _labelled_stream(args.root, args.source)
    ctx = mp.get_context("spawn")
    done = False
    with ctx.Pool(args.workers, initializer=_init_worker) as pool:
        for chunk in _chunks(stream, CHUNK_SIZE):
            for result in pool.imap(_prepare, chunk, chunksize=16):
                if result is None:
                    continue
                tensor, label, gen = result
                tensors.append(tensor)
                metas.append((label, gen))
                seen += 1
                if len(tensors) >= args.batch_size:
                    encode(tensors, metas)
                    tensors, metas = [], []
                    if seen % 2560 == 0:
                        print(f"{args.source}: {seen} images", flush=True)
                if args.limit and seen >= args.limit:
                    done = True
                    break
            if done:
                break
    if tensors:
        encode(tensors, metas)

    out_dir = args.root / "features_normalized"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.out_name or args.source}.npz"
    np.savez_compressed(
        out_path,
        features=np.concatenate(feats) if feats else np.zeros((0, 768), np.float16),
        labels=np.asarray(labels, dtype=np.int8),
        generators=np.asarray(generators),
        roles=np.asarray(roles),
    )
    print(f"saved {len(labels)} samples to {out_path}", flush=True)


if __name__ == "__main__":
    main()
