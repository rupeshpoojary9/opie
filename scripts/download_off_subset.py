"""Download a held-out subset of Open Food Facts products (image + ground truth).

Pages the public OFF search API for products that have a nutrition image, a completed
nutrition table, an ingredient list, and a published Nutri-Score grade, then downloads
each nutrition image locally. Writes data/off_subset/subset.jsonl.

Usage:
    python -m scripts.download_off_subset --n 60
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

from opie.config import IMAGES_DIR, OFF_SUBSET_DIR
from opie.data.off import OFFProduct, write_subset

SEARCH_URL = "https://world.openfoodfacts.org/api/v2/search"
FIELDS = ",".join([
    "code", "product_name", "nutriments", "ingredients", "ingredients_text",
    "nutriscore_grade", "image_nutrition_url", "image_front_url", "labels_tags",
])
HEADERS = {"User-Agent": "OPIE/0.1 (portfolio project; contact via github.com/rupeshpoojary9)"}


def _looks_complete(p: dict) -> bool:
    off = OFFProduct.from_json(p)
    return bool(off.image_url) and off.has_ground_truth() and bool(off.nutriscore_grade)


def _download_image(url: str, code: str, dest_dir: Path) -> Path | None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{code}.jpg"
    if dest.exists():
        return dest
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return dest
    except Exception as exc:
        print(f"  ! image download failed for {code}: {exc}", file=sys.stderr)
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=60, help="Number of valid products to collect.")
    ap.add_argument("--page-size", type=int, default=50)
    ap.add_argument("--max-pages", type=int, default=20)
    ap.add_argument("--out", type=Path, default=OFF_SUBSET_DIR / "subset.jsonl")
    args = ap.parse_args()

    collected: list[OFFProduct] = []
    page = 1
    while len(collected) < args.n and page <= args.max_pages:
        params = {
            "fields": FIELDS,
            "page_size": args.page_size,
            "page": page,
            "sort_by": "unique_scans_n",
            "countries_tags_en": "united-kingdom",
        }
        try:
            resp = requests.get(SEARCH_URL, params=params, headers=HEADERS, timeout=60)
            resp.raise_for_status()
        except Exception as exc:
            print(f"search page {page} failed: {exc}", file=sys.stderr)
            break
        products = resp.json().get("products", [])
        if not products:
            break
        for p in products:
            if len(collected) >= args.n:
                break
            if not _looks_complete(p):
                continue
            off = OFFProduct.from_json(p)
            local = _download_image(off.image_url, off.code, IMAGES_DIR)
            if local is None:
                continue
            off.local_image = local
            collected.append(off)
            print(f"[{len(collected)}/{args.n}] {off.code} {off.product_name[:40]}")
        page += 1
        time.sleep(1.0)  # be polite to the OFF API

    if not collected:
        print("No products collected. Check network / OFF availability.", file=sys.stderr)
        return 1

    write_subset(collected, args.out)
    print(f"\nWrote {len(collected)} products -> {args.out}")
    print(f"Images in {IMAGES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
