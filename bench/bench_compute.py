"""Time Calculator.compute() across (M, T, n_jobs, config) cells.

Reproducible replacement for the per-SPI walltime sweep currently done in the
sister-repo benchmark.ipynb. Outputs a tidy CSV (or parquet, if available)
with per-cell wall time plus per-SPI timings.

Usage:
    python -m bench.bench_compute \\
        --m 8,16 --t 200,800 --n-jobs 1,2,4 \\
        --config fast --repeats 2 \\
        --output bench/results/timings.csv

Designed for two distinct studies:

1. **Re-cutting amortized configs.** Fix --n-jobs 1 and vary --m/--t. The
   per-SPI timings (``spi_seconds`` column) are what feeds the amortized
   walltime calculation. n_jobs > 1 doesn't change per-SPI cost under the
   cache-aware scheduler — each cache group still runs sequentially within
   one worker — but it does change measurement noise via timer granularity,
   so n_jobs=1 is the clean measurement regime.

2. **Validating parallel speedup.** Fix --m, --t, --config, and sweep
   --n-jobs to characterise scaling. Compare ``cell_wall_seconds`` across
   n_jobs values; ideal speedup is linear up to the bucket count.

Determinism: each cell uses ``np.random.default_rng(seed + cell_index)`` so
re-running with the same flags produces the same data, and per-cell timings
are not cross-correlated.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pyspi.calculator import Calculator


def _resolve_config(arg: str) -> str:
    """Accept either a bundled subset name or a yaml path."""
    if arg in {"all", "fast", "sonnet", "fabfour"}:
        return arg
    p = Path(arg).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Config not found: {p}")
    return str(p)


def _make_calc(config: str, dataset: np.ndarray) -> Calculator:
    """Instantiate a Calculator from either a subset name or a yaml path."""
    if config in {"all", "fast", "sonnet", "fabfour"}:
        return Calculator(dataset=dataset, subset=config, normalise=False, verbose=False)
    return Calculator(dataset=dataset, configfile=config, normalise=False, verbose=False)


def run_cell(
    M: int,
    T: int,
    n_jobs: int,
    config: str,
    mp_context: str,
    repeats: int,
    seed: int,
) -> list[dict]:
    """Run one (M, T, n_jobs, config) cell ``repeats`` times. Return one row per repeat × SPI."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for r in range(repeats):
        arr = rng.standard_normal((M, T)).astype(np.float64)
        calc = _make_calc(config, arr)
        t0 = time.perf_counter()
        calc.compute(n_jobs=n_jobs, mp_context=mp_context, progress=False)
        cell_wall = time.perf_counter() - t0
        for spi_key, spi_seconds in calc.timings.items():
            rows.append({
                "M": M,
                "T": T,
                "n_jobs": n_jobs,
                "config": config,
                "mp_context": mp_context,
                "repeat": r,
                "cell_wall_seconds": cell_wall,
                "spi": spi_key,
                "spi_seconds": spi_seconds,
                "n_spis": len(calc.spis),
            })
    return rows


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="bench.bench_compute", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--m", type=_parse_int_list, default=[8],
                        help="Comma-separated process counts (default: 8)")
    parser.add_argument("--t", type=_parse_int_list, default=[200],
                        help="Comma-separated observation counts (default: 200)")
    parser.add_argument("--n-jobs", type=_parse_int_list, default=[1],
                        help="Comma-separated worker counts (default: 1)")
    parser.add_argument("--config", default="fabfour",
                        help="Bundled subset name (all/fast/sonnet/fabfour) or path to a yaml. Default: fabfour")
    parser.add_argument("--mp-context", choices=["spawn", "fork", "forkserver"], default="spawn")
    parser.add_argument("--repeats", type=int, default=2,
                        help="Repeats per cell (default: 2)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base RNG seed; cell index is added (default: 0)")
    parser.add_argument("--output", type=Path,
                        default=Path("bench/results") / f"timings_{datetime.now():%Y%m%d_%H%M%S}.csv",
                        help="Output CSV (parquet if .parquet suffix and pyarrow available).")
    args = parser.parse_args(argv)

    config = _resolve_config(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    cells = [(M, T, n) for M in args.m for T in args.t for n in args.n_jobs]
    print(f"[bench] {len(cells)} cells × {args.repeats} repeats = {len(cells) * args.repeats} runs",
          file=sys.stderr)
    print(f"[bench] config: {config}    mp_context: {args.mp_context}    output: {args.output}",
          file=sys.stderr)

    all_rows: list[dict] = []
    t_total = time.perf_counter()
    for idx, (M, T, n_jobs) in enumerate(cells):
        cell_start = time.perf_counter()
        rows = run_cell(M, T, n_jobs, config, args.mp_context, args.repeats, args.seed + idx)
        all_rows.extend(rows)
        wall = time.perf_counter() - cell_start
        print(f"[bench] [{idx+1}/{len(cells)}] M={M} T={T} n_jobs={n_jobs} "
              f"({args.repeats} repeats) -> {wall:.2f}s", file=sys.stderr)

    df = pd.DataFrame(all_rows)
    if args.output.suffix == ".parquet":
        try:
            df.to_parquet(args.output, index=False)
        except Exception as e:
            csv_fallback = args.output.with_suffix(".csv")
            print(f"[bench] parquet write failed ({e}); writing CSV: {csv_fallback}", file=sys.stderr)
            df.to_csv(csv_fallback, index=False)
    else:
        df.to_csv(args.output, index=False)

    # Companion metadata for reproducibility.
    meta = {
        "datetime": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "argv": sys.argv,
        "config": config,
        "mp_context": args.mp_context,
        "repeats": args.repeats,
        "seed": args.seed,
        "M": args.m,
        "T": args.t,
        "n_jobs": args.n_jobs,
        "total_wall_seconds": time.perf_counter() - t_total,
        "env": {k: os.environ.get(k, "") for k in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "PYSPI_N_JOBS")},
    }
    args.output.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))

    # Quick summary to stdout.
    print("\nCell summary (mean cell wall time across repeats):", file=sys.stderr)
    summary = (df.groupby(["M", "T", "n_jobs"])["cell_wall_seconds"]
                 .agg(["mean", "min", "max", "count"])
                 .reset_index())
    summary["count"] = summary["count"] // df["spi"].nunique()  # repeats, not row count
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
