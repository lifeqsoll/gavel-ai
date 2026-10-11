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
import re
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
TRAIN_GENERATORS = {
    "nano-banana",
    "FLUX-schnell",
    "ecommerce-cc0",
    "hmnsh-fake",
    "omnigen",
    "ai-vs-real",
    "diffusiondb",
    "synthetic-characters",
}
# FLUX-dev is non-commercial; it is excluded from all use.
EXCLUDED_GENERATORS = {"FLUX-dev"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

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


def _iter_wikimedia_dir(base: Path, gen: str) -> Iterator[tuple[bytes, int, str]]:
    import csv

    with (base / "manifest.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            yield (base / "images" / row["file"]).read_bytes(), 0, gen


def iter_wikimedia(root: Path) -> Iterator[tuple[bytes, int, str]]:
    yield from _iter_wikimedia_dir(root / "real/wikimedia", "wikimedia")


def iter_wikimedia_scenes(root: Path) -> Iterator[tuple[bytes, int, str]]:
    """Scene photos, minus any file already in the base wikimedia manifest (by title)."""
    import csv

    base = root / "real/wikimedia"
    with (base / "manifest.csv").open(encoding="utf-8") as handle:
        known_titles = {row["title"] for row in csv.DictReader(handle)}
    scenes = root / "real/wikimedia_scenes"
    with (scenes / "manifest.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["title"] in known_titles:
                continue
            yield (scenes / "images" / row["file"]).read_bytes(), 0, "wikimedia-scenes"


def iter_wikimedia_faces(root: Path) -> Iterator[tuple[bytes, int, str]]:
    """Wikimedia face/portrait close-ups (hard negatives for polished-skin FP)."""
    yield from _iter_wikimedia_dir(root / "real/wikimedia_faces", "wikimedia-faces")


def iter_wikimedia_extra(root: Path) -> Iterator[tuple[bytes, int, str]]:
    """Extra Wikimedia reals (image dir; no manifest required)."""
    yield from _iter_image_dir(root / "real/wikimedia_extra/images", "wikimedia-extra")


def iter_real_hands(root: Path) -> Iterator[tuple[bytes, int, str]]:
    """User-provided real hand / close-up hard negatives.

    Drop images under ``real/hands/`` (any nesting). Used after you collect data;
    see ``training/retrain_when_hands_ready.sh``.
    """
    yield from _iter_image_dir(root / "real/hands", "real-hands")


# COCO rows used by the "coco" source (--limit 20000 in the first extraction).
COCO_BASE_ROWS = 20000

# Caption keywords for real scenes that the first COCO sample under-represents:
# people, streets and roads, houses and buildings.
SCENE_PATTERN = re.compile(
    r"\b(person|people|man|men|woman|women|child|children|kid|kids|boy|boys|girl|girls|family|"
    r"street|streets|road|roads|highway|house|houses|home|homes|building|buildings|apartment|"
    r"apartments|town|city|neighborhood|village)\b"
)


def iter_coco_scenes(root: Path) -> Iterator[tuple[bytes, int, str]]:
    """Real COCO photos whose caption mentions a scene keyword, excluding the "coco" source rows.

    The "coco" source is the first COCO_BASE_ROWS rows in shard order, so those rows and any
    other row of the same image are skipped. Each image is yielded at most once.
    """
    paths = sorted(glob.glob(str(root / "real/coco/data/*.parquet")))
    used_ids: set[int] = set()
    seen = 0
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=["cocoid"]):
            for cid in batch.column("cocoid").to_pylist():
                if seen >= COCO_BASE_ROWS:
                    break
                used_ids.add(cid)
                seen += 1
    seen_ids: set[int] = set(used_ids)
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["cocoid", "sentences", "image"]):
            cocoids = batch.column("cocoid").to_pylist()
            captions = batch.column("sentences").to_pylist()
            images = batch.column("image").to_pylist()
            for cid, sent, img in zip(cocoids, captions, images):
                if cid in seen_ids:
                    continue
                caption = (sent or {}).get("raw", "") or ""
                if not SCENE_PATTERN.search(caption.lower()):
                    continue
                seen_ids.add(cid)
                yield _image_bytes(img), 0, "coco-scenes"


def _iter_image_dir(directory: Path, generator: str, min_side: int = 64) -> Iterator[tuple[bytes, int, str]]:
    """Yield RGB-capable images from a directory tree. Tiny files (below min_side) are skipped."""
    if not directory.is_dir():
        return
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
            continue
        try:
            with Image.open(path) as img:
                width, height = img.size
                if min(width, height) < min_side:
                    continue
            yield path.read_bytes(), 1, generator
        except Exception:
            continue


def iter_ecommerce_cc0(root: Path) -> Iterator[tuple[bytes, int, str]]:
    """AI-generated ecommerce photos (CC0). Skips README and annotation files."""
    base = root / "generated/ecommerce-cc0"
    for child in sorted(base.iterdir()) if base.is_dir() else []:
        if child.is_dir() and child.name not in {".cache", ".git"}:
            yield from _iter_image_dir(child, "ecommerce-cc0")


def iter_hmnsh_fake(root: Path) -> Iterator[tuple[bytes, int, str]]:
    """FAKE split of hmnshudhmn24/real-fake-ai-generated-art-images (MIT)."""
    yield from _iter_image_dir(root / "generated/hmnsh-fake/Data/FAKE", "hmnsh-fake")


def iter_omnigen(root: Path) -> Iterator[tuple[bytes, int, str]]:
    """orbisAI/OmniGen AI images (Apache-2.0)."""
    yield from _iter_image_dir(root / "generated/omnigen/images", "omnigen")


def iter_ai_vs_real_fake(root: Path) -> Iterator[tuple[bytes, int, str]]:
    """MIT Parveshiiii/AI-vs-Real, binary_label=1 (AI)."""
    for path in sorted(glob.glob(str(root / "generated/ai-vs-real/data/*.parquet"))):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=["image", "binary_label"]):
            for cell, label in zip(
                batch.column("image").to_pylist(), batch.column("binary_label").to_pylist()
            ):
                if int(label) != 1:
                    continue
                yield _image_bytes(cell), 1, "ai-vs-real"


def iter_ai_vs_real_real(root: Path) -> Iterator[tuple[bytes, int, str]]:
    """MIT Parveshiiii/AI-vs-Real, binary_label=0 (real photos)."""
    for path in sorted(glob.glob(str(root / "generated/ai-vs-real/data/*.parquet"))):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=["image", "binary_label"]):
            for cell, label in zip(
                batch.column("image").to_pylist(), batch.column("binary_label").to_pylist()
            ):
                if int(label) != 0:
                    continue
                yield _image_bytes(cell), 0, "ai-vs-real-real"


def iter_diffusiondb(root: Path) -> Iterator[tuple[bytes, int, str]]:
    """DiffusionDB / svjack random subset (CC0): images/ or parquet shards."""
    base = root / "generated/diffusiondb-10k"
    images = base / "images"
    if images.is_dir() and any(images.iterdir()):
        yield from _iter_image_dir(images, "diffusiondb")
        return
    for path in sorted(glob.glob(str(base / "data/*.parquet"))):
        pf = pq.ParquetFile(path)
        cols = set(pf.schema_arrow.names)
        if "image" not in cols:
            continue
        for batch in pf.iter_batches(batch_size=32, columns=["image"]):
            for cell in batch.column("image").to_pylist():
                yield _image_bytes(cell), 1, "diffusiondb"


def iter_synthetic_characters(root: Path) -> Iterator[tuple[bytes, int, str]]:
    """AbstractPhil/synthetic-characters (MIT), generated faces/characters."""
    base = root / "generated/synthetic-characters"
    for path in sorted(glob.glob(str(base / "**/*.parquet"), recursive=True)):
        pf = pq.ParquetFile(path)
        cols = set(pf.schema_arrow.names)
        image_col = "image" if "image" in cols else ("img" if "img" in cols else None)
        if image_col is None:
            continue
        for batch in pf.iter_batches(batch_size=32, columns=[image_col]):
            for cell in batch.column(image_col).to_pylist():
                yield _image_bytes(cell), 1, "synthetic-characters"


SOURCES = {
    "nano_banana": iter_nano_banana,
    "cf_commercial": iter_cf_commercial,
    "coco": iter_coco,
    "coco_scenes": iter_coco_scenes,
    "open_images": iter_open_images,
    "wikimedia": iter_wikimedia,
    "wikimedia_scenes": iter_wikimedia_scenes,
    "wikimedia_faces": iter_wikimedia_faces,
    "wikimedia_extra": iter_wikimedia_extra,
    "real_hands": iter_real_hands,
    "ecommerce_cc0": iter_ecommerce_cc0,
    "hmnsh_fake": iter_hmnsh_fake,
    "omnigen": iter_omnigen,
    "ai_vs_real_fake": iter_ai_vs_real_fake,
    "ai_vs_real_real": iter_ai_vs_real_real,
    "diffusiondb": iter_diffusiondb,
    "synthetic_characters": iter_synthetic_characters,
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
