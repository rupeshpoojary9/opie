"""Run the full pipeline on a single product image and print the intelligence record.

Usage:
    python -m scripts.demo_one --image path/to/label.jpg --profiles diabetic,hypertension
    python -m scripts.demo_one --code 5000159407236   # pull image from the OFF subset
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from opie.config import OFF_SUBSET_DIR
from opie.data.off import read_subset
from opie.pipeline.graph import Pipeline


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", type=Path, help="Path to a label image.")
    ap.add_argument("--code", help="OFF product code present in the downloaded subset.")
    ap.add_argument("--profiles", default="general,diabetic")
    args = ap.parse_args()

    if args.image:
        product_id = args.image.stem
        image_bytes = args.image.read_bytes()
    elif args.code:
        match = next((p for p in read_subset(OFF_SUBSET_DIR / "subset.jsonl") if p.code == args.code), None)
        if not match or not match.local_image or not Path(match.local_image).exists():
            print(f"Code {args.code} not found in subset or image missing.", file=sys.stderr)
            return 1
        product_id = match.code
        image_bytes = Path(match.local_image).read_bytes()
    else:
        print("Provide --image or --code.", file=sys.stderr)
        return 1

    pipeline = Pipeline()
    print(f"# backend={pipeline.extractor.backend_name} engine={pipeline.engine}", file=sys.stderr)
    result = pipeline.run(product_id, image_bytes, profiles=args.profiles.split(","))
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
