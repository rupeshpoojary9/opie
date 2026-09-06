"""Unified `opie` CLI.

    opie feedback-eval --rounds 5      # run the data flywheel, print rising-F1/shrinking-queue table
    opie eval                          # attribute eval over the OFF subset
    opie download --n 60               # pull an OFF held-out subset
    opie demo --code <code>            # run the pipeline on one product
    opie serve                         # FastAPI server
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_feedback_eval(args: argparse.Namespace) -> int:
    from opie.config import RESULTS_DIR
    from opie.feedback.corpus import load_corpus
    from opie.feedback.loop import FeedbackLoop
    from opie.feedback.report import FeedbackReport
    from opie.feedback.store import FeedbackStore

    products = load_corpus(n=args.products, seed=args.seed)
    is_real = getattr(products[0], "off", None) is not None
    print(f"Corpus: {len(products)} products "
          f"({'real OFF subset' if is_real else 'deterministic synthetic'}) | "
          f"backend: {args.backend} | rounds: {args.rounds} | seed: {args.seed}")

    emit_fn = None
    if args.backend != "simulated":
        from opie.config import Settings
        from opie.feedback.adapter import pipeline_emit_fn
        from opie.pipeline.graph import Pipeline
        settings = Settings(backend=args.backend)
        emit_fn = pipeline_emit_fn(Pipeline(settings=settings))

    out_dir = Path(args.out or (RESULTS_DIR / "feedback"))
    store = FeedbackStore(db_path=out_dir / "feedback.sqlite") if args.db else None
    if store is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    loop = FeedbackLoop(products, rounds=args.rounds, seed=args.seed, emit_fn=emit_fn, store=store)
    rounds = loop.run()
    report = FeedbackReport(rounds, backend=args.backend)

    md = report.to_markdown()
    print("\n" + md)
    out_dir.mkdir(parents=True, exist_ok=True)
    report.to_json(out_dir / "feedback_eval.json")
    (out_dir / "FEEDBACK.md").write_text(md)
    print(f"\nWrote {out_dir / 'feedback_eval.json'} and {out_dir / 'FEEDBACK.md'}")
    return 0 if report.passed() else 2


def _cmd_eval(args: argparse.Namespace) -> int:
    from scripts.run_eval import main as run_eval_main
    sys.argv = ["opie-eval"]
    return run_eval_main()


def _cmd_download(args: argparse.Namespace) -> int:
    from scripts.download_off_subset import main as dl_main
    sys.argv = ["opie-download", "--n", str(args.n)]
    return dl_main()


def _cmd_demo(args: argparse.Namespace) -> int:
    from scripts.demo_one import main as demo_main
    sys.argv = ["opie-demo"] + (["--code", args.code] if args.code else []) + \
               (["--image", args.image] if args.image else [])
    return demo_main()


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn
    uvicorn.run("opie.api.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="opie", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    fe = sub.add_parser("feedback-eval", help="Run the data flywheel across N rounds.")
    fe.add_argument("--rounds", type=int, default=5)
    fe.add_argument("--products", type=int, default=400)
    fe.add_argument("--seed", type=int, default=13)
    fe.add_argument("--backend", default="simulated", choices=["simulated", "offline", "anthropic"])
    fe.add_argument("--out", default=None)
    fe.add_argument("--db", action="store_true", help="Persist runs/observations/corrections to SQLite.")
    fe.set_defaults(func=_cmd_feedback_eval)

    ev = sub.add_parser("eval", help="Attribute eval over the OFF subset.")
    ev.set_defaults(func=_cmd_eval)

    dl = sub.add_parser("download", help="Download an OFF held-out subset.")
    dl.add_argument("--n", type=int, default=60)
    dl.set_defaults(func=_cmd_download)

    dm = sub.add_parser("demo", help="Run the pipeline on one product.")
    dm.add_argument("--code", default=None)
    dm.add_argument("--image", default=None)
    dm.set_defaults(func=_cmd_demo)

    sv = sub.add_parser("serve", help="Run the FastAPI server.")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--reload", action="store_true")
    sv.set_defaults(func=_cmd_serve)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
