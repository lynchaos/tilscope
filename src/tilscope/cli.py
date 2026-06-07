"""Command-line interface.

    tilscope run --out report.html [--seed 0] [--llm] [--input data.h5ad]

Runs the full pipeline and writes a self-contained HTML report + results JSON.
"""
from __future__ import annotations

import argparse
import sys
import time

from . import __version__
from .data import simulate_tme, load_h5ad
from .pipeline import run_pipeline
from .narrative import generate_narrative
from .report import write_report


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tilscope", description=__doc__)
    p.add_argument("--version", action="version", version=f"tilscope {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the pipeline and write a report")
    run.add_argument("--out", default="report.html", help="output HTML path")
    run.add_argument("--input", default=None, help="optional .h5ad of real data")
    run.add_argument("--seed", type=int, default=0, help="random seed (reproducibility)")
    run.add_argument("--n-cells", type=int, default=2000, help="cells to simulate")
    run.add_argument("--llm", action="store_true",
                     help="use Claude for the narrative (needs ANTHROPIC_API_KEY)")
    run.add_argument("--model", default=None, help="override Claude model id")
    run.add_argument("--min-genes", type=int, default=None, help="override QC gene floor")
    run.add_argument("--max-pct-mito", type=float, default=None, help="override QC mito ceiling")
    run.set_defaults(func=_cmd_run)
    return p


def _cmd_run(args: argparse.Namespace) -> int:
    t0 = time.time()
    if args.input:
        print(f"[tilscope] loading {args.input}")
        cd = load_h5ad(args.input)
    else:
        print(f"[tilscope] simulating {args.n_cells} TME cells (seed={args.seed})")
        cd = simulate_tme(n_cells=args.n_cells, seed=args.seed)

    print(f"[tilscope] running pipeline on {cd.n_obs} cells x {cd.n_vars} genes")
    results = run_pipeline(cd, seed=args.seed,
                           min_genes=args.min_genes, max_pct_mito=args.max_pct_mito)

    print(f"[tilscope] generating narrative (llm={args.llm})")
    narrative, source = generate_narrative(results, use_llm=args.llm, model=args.model)

    paths = write_report(results, narrative, source, args.out)
    dt = time.time() - t0
    print(f"[tilscope] done in {dt:.1f}s")
    print(f"[tilscope]   report : {paths['html']}")
    print(f"[tilscope]   json   : {paths['json']}")
    print(f"[tilscope]   method : {results.diagnostics['clustering_method']}"
          f"  ARI={results.diagnostics.get('adjusted_rand_index', 'n/a')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
