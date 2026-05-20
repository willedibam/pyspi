"""Thin CLI for pyspi: compute all SPIs on a saved dataset.

    python -m pyspi compute \
        --data ts.npy \
        --config pyspi/benchmarked90_amortized_config.yaml \
        --output table.parquet \
        --n-jobs 4 \
        --checkpoint-dir results/

If ``--config`` is omitted, the bundled ``config.yaml`` is used. If
``--output`` is omitted, the result table is written next to the data file as
``<data-stem>.spi.parquet``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .calculator import Calculator


def _load_array(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".csv":
        return np.genfromtxt(path, delimiter=",")
    if path.suffix == ".txt":
        return np.genfromtxt(path)
    raise ValueError(f"Unsupported data extension: {path.suffix} (use .npy, .csv, or .txt)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="pyspi", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    cp = sub.add_parser("compute", help="Compute SPIs on a saved dataset.")
    cp.add_argument("--data", type=Path, required=True,
                    help="Path to time series array (.npy/.csv/.txt). Shape (processes, observations).")
    cp.add_argument("--config", type=Path, default=None,
                    help="YAML config selecting which SPIs to compute (default: bundled config.yaml).")
    cp.add_argument("--output", type=Path, default=None,
                    help="Where to write the result parquet (default: <data>.spi.parquet).")
    cp.add_argument("--n-jobs", type=int, default=1,
                    help="Worker process count. 1 = serial (default).")
    cp.add_argument("--checkpoint-dir", type=Path, default=None,
                    help="Directory for per-SPI .npy checkpoints. Enables resume.")
    cp.add_argument("--no-resume", action="store_true",
                    help="Ignore existing checkpoints; recompute every SPI.")
    cp.add_argument("--mp-context", choices=["spawn", "fork", "forkserver"], default=None,
                    help="Multiprocessing start method (default: spawn).")
    cp.add_argument("--normalise", action="store_true",
                    help="z-score each time series before computing.")
    cp.add_argument("--quiet", action="store_true",
                    help="Suppress INFO logging; show warnings/errors only.")

    args = parser.parse_args(argv)

    arr = _load_array(args.data)
    if arr.ndim != 2:
        raise SystemExit(f"Data must be 2D (processes x observations); got shape {arr.shape}")

    calc = Calculator(
        dataset=arr,
        configfile=str(args.config) if args.config else None,
        normalise=args.normalise,
        verbose=not args.quiet,
    )
    calc.compute(
        n_jobs=args.n_jobs,
        checkpoint_dir=args.checkpoint_dir,
        resume=not args.no_resume,
        mp_context=args.mp_context,
    )

    out = args.output or args.data.with_suffix(".spi.parquet")
    try:
        calc.table.to_parquet(out)
        print(f"Wrote results table -> {out}")
    except Exception as e:
        fallback = out.with_suffix(".pkl")
        calc.table.to_pickle(fallback)
        print(f"Parquet write failed ({e}); wrote pickle -> {fallback}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
