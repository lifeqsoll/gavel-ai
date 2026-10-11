"""Extract mean-pooled intermediate ViT-L/14 features for every labelled source.

The final CLIP embedding is semantic. Intermediate residual blocks keep more low-level
texture and rendering statistics, which may separate generated and real photos
where the final embedding does not. Images go through the same normalization as
training/extract_features.py (core/normalize.py).

Output: <root>/features_layers/<source>.npz with
    layers       float16, shape (N, len(LAYER_INDICES), 1024), mean over all tokens
    labels, generators, roles  as in extract_features.py

Usage:
    python training/extract_layers.py --source nano_banana --root /mnt/disk-e/gavel-data
"""

from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_features as ef  # noqa: E402

LAYER_INDICES = (7, 11, 15, 19)  # zero-based residual block indices in ViT-L/14 (24 blocks)


def main() -> None:
    import open_clip

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=sorted(ef.SOURCES), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--cf-file", default="HFCF_Commercial_0.parquet")
    parser.add_argument("--limit", type=int, default=0, help="stop after N images (0 = all)")
    args = parser.parse_args()
    ef.CF_FILE_PATTERN = args.cf_file

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(ef.MODEL_NAME, pretrained=ef.PRETRAINED)
    model = model.to(device).eval().half()

    captured: dict[int, torch.Tensor] = {}
    blocks = model.visual.transformer.resblocks
    for index in LAYER_INDICES:
        blocks[index].register_forward_hook(
            lambda _m, _i, out, index=index: captured.__setitem__(index, out)
        )

    feats: list[np.ndarray] = []
    labels: list[int] = []
    generators: list[str] = []
    roles: list[str] = []

    def encode(tensors: list[torch.Tensor], metas: list[tuple[int, str]]) -> None:
        batch = torch.stack(tensors).to(device).half()
        with torch.no_grad():
            model.encode_image(batch)
        layer_means = [captured[i].float().mean(dim=1) for i in LAYER_INDICES]
        feats.append(torch.stack(layer_means, dim=1).cpu().numpy().astype(np.float16))
        for label, gen in metas:
            labels.append(label)
            generators.append(gen)
            roles.append(ef._role(gen, label))

    tensors: list[torch.Tensor] = []
    metas: list[tuple[int, str]] = []
    stream = ef._labelled_stream(args.root, args.source)
    if args.limit:
        stream = itertools.islice(stream, args.limit)
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=ef._init_worker) as pool:
        for chunk in ef._chunks(stream, ef.CHUNK_SIZE):
            for result in pool.imap(ef._prepare, chunk, chunksize=16):
                if result is None:
                    continue
                tensor, label, gen = result
                tensors.append(tensor)
                metas.append((label, gen))
                if len(tensors) >= args.batch_size:
                    encode(tensors, metas)
                    tensors, metas = [], []
    if tensors:
        encode(tensors, metas)

    out_dir = args.root / "features_layers"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.source}.npz"
    np.savez_compressed(
        out_path,
        layers=np.concatenate(feats) if feats else np.zeros((0, len(LAYER_INDICES), 1024), np.float16),
        labels=np.asarray(labels, dtype=np.int8),
        generators=np.asarray(generators),
        roles=np.asarray(roles),
    )
    print(f"saved {len(labels)} samples to {out_path}", flush=True)


if __name__ == "__main__":
    main()
