"""Download open real-hand photo corpora into gavel-data/real/hands.

Sources (permissive / academic fair-use):
  - Wikimedia Commons close-up / hand queries (CC0 / CC BY / CC BY-SA)
  - 11k Hands via Hugging Face (ryushinn/11k-Hands)
  - HaGRIDv2 512px (CC-BY-SA-4.0 variant) — large in-the-wild hands

Usage:
    .venv-data/bin/python training/fetch_hands.py \\
        --root /mnt/disk-e/gavel-data --wiki-target 3000 --hagrid-512
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests

# Reuse Wikimedia helpers from the sibling script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_wikimedia import (  # noqa: E402
    ALLOWED_LICENSES,
    API,
    EXCLUDED_TITLES,
    HEADERS,
    acceptable,
    search_files,
)

HAND_QUERIES = [
    "close-up of hands",
    "hands holding object",
    "human hand photograph",
    "hand gesture photo",
    "palm of hand photo",
    "fingers close-up photo",
    "person holding smartphone hand",
    "handshake photo",
    "hand writing photo",
    "cooking hands kitchen",
    "manicure hands photo",
    "musician hands piano",
    "guitarist hands",
    "boxing gloves hands",
    "yoga mudra hands",
    "sign language hands",
    "worker hands tools",
    "farmer hands soil",
    "surgeon hands gloves",
    "child hands playing",
]

HAGRID_512_URL = (
    "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/"
    "datasets/hagrid_v2/hagridv2_512.zip"
)
HAGRID_ANN_URL = (
    "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/"
    "datasets/hagrid_v2/annotations_with_landmarks/annotations.zip"
)


def fetch_wikimedia_hands(out: Path, target: int, per_query: int) -> int:
    image_dir = out / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.csv"
    seen: set[str] = set()
    rows: list[dict] = []
    if manifest_path.exists():
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                seen.add(row["title"])
                rows.append(row)

    for query in HAND_QUERIES:
        if len(rows) >= target:
            break
        try:
            pages = search_files(query, per_query)
        except requests.RequestException as exc:
            print(f"wiki search fail {query}: {exc}", flush=True)
            continue
        for page in pages:
            if len(rows) >= target:
                break
            title = page.get("title", "")
            if title in seen or title in EXCLUDED_TITLES:
                continue
            seen.add(title)
            picked = acceptable(page)
            if picked is None:
                continue
            url, license_name = picked
            if license_name not in ALLOWED_LICENSES:
                continue
            name = f"{page['pageid']}.jpg"
            path = image_dir / name
            if not path.exists():
                try:
                    response = requests.get(url, headers=HEADERS, timeout=60)
                    response.raise_for_status()
                except requests.RequestException as exc:
                    print(f"skip {title}: {exc}", flush=True)
                    continue
                path.write_bytes(response.content)
                time.sleep(0.25)
            rows.append(
                {"file": name, "title": title, "license": license_name, "query": query}
            )
            print(f"[wiki {len(rows)}/{target}] {title} ({license_name})", flush=True)

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "title", "license", "query"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wikimedia hands: {len(rows)} photos -> {out}", flush=True)
    return len(rows)


def fetch_11k_hands(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    print(f"downloading 11k Hands -> {out}", flush=True)
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="ryushinn/11k-Hands",
        repo_type="dataset",
        local_dir=str(out),
        local_dir_use_symlinks=False,
    )
    print(f"11k Hands done: {out}", flush=True)


def fetch_hagrid_512(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / "hagridv2_512.zip"
    ann_path = out / "annotations.zip"
    if not zip_path.exists() or zip_path.stat().st_size < 1_000_000_000:
        print(f"wget HaGRID 512px (~119GB) -> {zip_path}", flush=True)
        subprocess.check_call(
            [
                "wget",
                "-c",
                "--progress=dot:giga",
                "-O",
                str(zip_path),
                HAGRID_512_URL,
            ]
        )
    else:
        print(f"HaGRID zip already present: {zip_path}", flush=True)
    if not ann_path.exists():
        print(f"wget HaGRID annotations -> {ann_path}", flush=True)
        subprocess.check_call(
            ["wget", "-c", "--progress=dot:mega", "-O", str(ann_path), HAGRID_ANN_URL]
        )
    extract_dir = out / "images"
    if not extract_dir.exists() or not any(extract_dir.rglob("*.jpg")):
        print(f"unzipping HaGRID (this takes a while) -> {extract_dir}", flush=True)
        extract_dir.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["unzip", "-q", "-n", str(zip_path), "-d", str(extract_dir)])
    print(f"HaGRID 512 ready under {out}", flush=True)


def copy_bushon(src: Path, dest: Path, eval_dest: Path, eval_n: int = 4) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    eval_dest.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.jpg")) + sorted(src.glob("*.jpeg")) + sorted(src.glob("*.png"))
    if not files:
        print(f"no bushon photos in {src}", flush=True)
        return
    for path in files:
        target = dest / path.name
        if not target.exists():
            shutil.copy2(path, target)
    # Hold out last eval_n for hard-neg eval (not training mix).
    held = files[-eval_n:] if len(files) > eval_n else []
    for path in held:
        shutil.copy2(path, eval_dest / path.name)
    print(f"bushon: {len(files)} train-ish -> {dest}; {len(held)} eval -> {eval_dest}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/mnt/disk-e/gavel-data"))
    parser.add_argument("--wiki-target", type=int, default=3000)
    parser.add_argument("--per-query", type=int, default=80)
    parser.add_argument("--skip-wiki", action="store_true")
    parser.add_argument("--skip-11k", action="store_true")
    parser.add_argument("--hagrid-512", action="store_true")
    parser.add_argument("--skip-bushon", action="store_true")
    parser.add_argument(
        "--bushon-src",
        type=Path,
        default=Path("/home/emosensei/Downloads/бушон"),
    )
    args = parser.parse_args()

    hands = args.root / "real" / "hands"
    hands.mkdir(parents=True, exist_ok=True)

    if not args.skip_bushon:
        copy_bushon(args.bushon_src, hands / "bushon", hands / "bushon_eval")

    if not args.skip_wiki:
        fetch_wikimedia_hands(hands / "wikimedia", args.wiki_target, args.per_query)

    if not args.skip_11k:
        fetch_11k_hands(hands / "11k_hands")

    if args.hagrid_512:
        fetch_hagrid_512(hands / "hagrid_512")

    n = sum(1 for _ in hands.rglob("*") if _.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    print(f"DONE hands jpg count≈{n}; size:", flush=True)
    subprocess.call(["du", "-sh", str(hands)])


if __name__ == "__main__":
    main()
