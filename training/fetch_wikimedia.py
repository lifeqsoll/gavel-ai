"""Download freely licensed photographs from Wikimedia Commons as additional real photos.

Only files with CC0, public domain, CC BY or CC BY-SA licenses are kept. The
license and file title of every image are written to manifest.csv so that
the training data can be audited. The three sample photos under samples/real/
are excluded so that they stay usable as an untouched final check.

Usage:
    python training/fetch_wikimedia.py --out /mnt/disk-e/gavel-data/real/wikimedia --target 1500
    python training/fetch_wikimedia.py --out /mnt/disk-e/gavel-data/real/wikimedia_scenes \
        --queries scenes --target 600
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import requests

API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "GavelAI-dataset-builder/0.1 (research; contact via repository)"}
ALLOWED_LICENSES = {"CC0", "Public domain", "CC BY 2.0", "CC BY 3.0", "CC BY 4.0", "CC BY-SA 2.0",
                    "CC BY-SA 3.0", "CC BY-SA 4.0"}
EXCLUDED_TITLES = {
    "File:Close-up Hand holding dumbbell in gym.jpg",
    "File:Florence - Vespa Vanishing Point.jpg",
    "File:Perspective at Jama Masjid.jpg",
}
QUERIES = [
    "street photo", "city street people", "night city photo", "shop interior", "restaurant food photo",
    "kitchen interior", "living room interior", "portrait of a person", "outdoor portrait",
    "child playing", "dog photo", "cat photo", "bird photo", "wildlife photo", "forest landscape",
    "mountain landscape", "sea coast photo", "beach photo", "river landscape", "city skyline",
    "architecture building", "old church interior", "museum interior", "train station", "airport terminal",
    "market stall", "sports match photo", "football player", "hands holding object", "close-up of hands",
    "bicycle photo", "car on road", "crowd of people", "classroom photo", "office workspace",
    "vegetables market", "flowers close-up", "coffee cup photo", "rain on window", "snow landscape",
]


# Queries for the scene data: people, roads and streets, houses and buildings.
SCENE_QUERIES = [
    "people walking on a street", "portrait of a man", "portrait of a woman", "family photo outdoors",
    "people in a city", "crowd on the street", "pedestrians crossing a road", "group of people photo",
    "country road", "road with cars", "highway photo", "road through a village", "street with houses",
    "suburban house exterior", "house in a village", "residential street houses", "apartment building photo",
    "old house facade", "wooden house in forest", "farmhouse photo", "row of houses", "city street buildings",
]


def search_files(query: str, limit: int) -> list[dict]:
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {query}",
        "gsrnamespace": 6,
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|mime|size",
        "iiurlwidth": 1024,
        "format": "json",
    }
    response = requests.get(API, params=params, headers=HEADERS, timeout=30)
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", {})
    return list(pages.values())


def acceptable(page: dict) -> tuple[str, str] | None:
    title = page.get("title", "")
    if title in EXCLUDED_TITLES or not title.lower().endswith((".jpg", ".jpeg")):
        return None
    infos = page.get("imageinfo")
    if not infos:
        return None
    info = infos[0]
    if info.get("mime") != "image/jpeg" or min(info.get("width", 0), info.get("height", 0)) < 512:
        return None
    meta = info.get("extmetadata", {})
    license_name = meta.get("LicenseShortName", {}).get("value", "")
    if license_name not in ALLOWED_LICENSES:
        return None
    url = info.get("thumburl") or info.get("url")
    return url, license_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", type=int, default=1500)
    parser.add_argument("--per-query", type=int, default=50)
    parser.add_argument("--queries", choices=["default", "scenes"], default="default")
    args = parser.parse_args()
    queries = SCENE_QUERIES if args.queries == "scenes" else QUERIES

    image_dir = args.out / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.csv"
    seen: set[str] = set()
    rows: list[dict] = []

    for query in queries:
        if len(rows) >= args.target:
            break
        for page in search_files(query, args.per_query):
            if len(rows) >= args.target:
                break
            title = page.get("title", "")
            if title in seen:
                continue
            seen.add(title)
            picked = acceptable(page)
            if picked is None:
                continue
            url, license_name = picked
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
                time.sleep(0.3)
            rows.append({"file": name, "title": title, "license": license_name, "query": query})
            print(f"[{len(rows)}] {title} ({license_name})", flush=True)

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "title", "license", "query"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {len(rows)} photos, manifest at {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
