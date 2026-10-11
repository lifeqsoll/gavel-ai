"""Build a JPEG cache at native resolution for CNN training.

Every image is decoded, converted to RGB and re-encoded as JPEG with quality 90 at its
original size. All sources therefore share one container format, while resolution and
fine detail are kept. The CNN trains on random 224-pixel crops from these bytes.

Output per source in <root>/cnn_cache/<source>/:
    data.bin         concatenated JPEG bytes
    offsets.npy      int64, length N+1, byte offsets into data.bin
    labels.npy, generators.npy, roles.npy

Usage:
    python training/build_cnn_cache.py --root /mnt/disk-e/gavel-data --source nano_banana
"""

from __future__ import annotations

import argparse
import io
import itertools
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_features as ef  # noqa: E402

LIMITS = {"coco": 20000, "open_images": 15000}
CACHE_QUALITY = 90


def _reencode(item):
    raw, label, gen = item
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return None
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=CACHE_QUALITY)
    return buf.getvalue(), label, gen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", choices=sorted(ef.SOURCES), required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--cf-file", default="HFCF_Commercial_0.parquet")
    args = parser.parse_args()
    ef.CF_FILE_PATTERN = args.cf_file

    out_dir = args.root / "cnn_cache" / args.source
    out_dir.mkdir(parents=True, exist_ok=True)
    stream = ef._labelled_stream(args.root, args.source)
    if args.source in LIMITS:
        stream = itertools.islice(stream, LIMITS[args.source])

    offsets = [0]
    labels: list[int] = []
    generators: list[str] = []
    roles: list[str] = []
    skipped = 0
    ctx = mp.get_context("spawn")
    with (out_dir / "data.bin").open("wb") as data_file, ctx.Pool(args.workers) as pool:
        for result in pool.imap(_reencode, stream, chunksize=16):
            if result is None:
                skipped += 1
                continue
            encoded, label, gen = result
            data_file.write(encoded)
            offsets.append(offsets[-1] + len(encoded))
            labels.append(label)
            generators.append(gen)
            roles.append(ef._role(gen, label))
            if len(labels) % 2000 == 0:
                print(f"{args.source}: {len(labels)} images", flush=True)

    np.save(out_dir / "offsets.npy", np.asarray(offsets, dtype=np.int64))
    np.save(out_dir / "labels.npy", np.asarray(labels, dtype=np.int8))
    np.save(out_dir / "generators.npy", np.asarray(generators))
    np.save(out_dir / "roles.npy", np.asarray(roles))
    print(f"saved {len(labels)} images to {out_dir} (skipped {skipped})", flush=True)


if __name__ == "__main__":
    main()
