"""Run the eval harness over the downloaded OFF subset and write a report.

Usage:
    python -m scripts.run_eval                       # uses data/off_subset/subset.jsonl
    python -m scripts.run_eval --subset path.jsonl --out results/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from opie.config import OFF_SUBSET_DIR, RESULTS_DIR, SETTINGS
from opie.data.off import read_subset
from opie.eval.harness import run_eval
from opie.pipeline.graph import Pipeline


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", type=Path, default=OFF_SUBSET_DIR / "subset.jsonl")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR)
    ap.add_argument("--profiles", default="general,diabetic,hypertension")
    args = ap.parse_args()

    if not args.subset.exists():
        print(f"Subset not found: {args.subset}\nRun: python -m scripts.download_off_subset --n 60",
              file=sys.stderr)
        return 1

    products = list(read_subset(args.subset))
    pipeline = Pipeline()
    print(f"Backend: {pipeline.extractor.backend_name} | engine: {pipeline.engine} "
          f"| model: {SETTINGS.model if pipeline.extractor.backend_name == 'anthropic' else 'n/a'}")
    if pipeline.extractor.backend_name == "offline":
        print("NOTE: offline backend — these numbers reflect the heuristic parser, NOT the "
              "Claude-vision pipeline. Add a key + OPIE_BACKEND=anthropic for real metrics.")

    report = run_eval(products, pipeline=pipeline, profiles=args.profiles.split(","))

    args.out.mkdir(parents=True, exist_ok=True)
    report.to_json(args.out / "eval.json")
    md = report.to_markdown()
    (args.out / "RESULTS.md").write_text(md)
    print("\n" + md)
    print(f"\nWrote {args.out / 'eval.json'} and {args.out / 'RESULTS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
