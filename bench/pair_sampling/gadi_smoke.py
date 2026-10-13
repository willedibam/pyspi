"""One p90 extraction on a cluster node, dense or on sampled dyads, with its resource record.

    python gadi_smoke.py --archive large.npz --member meg-...-M243 \\
        --n-pairs 1000 --n-jobs 4 --out results/m243-b1000-j4

``--n-pairs 0`` runs the dense ``compute()``. The archive is a named ``.npz``
whose members are ``(observations, processes)`` arrays. Writes ``values.npy``
(``(n_spis, M, M)`` or ``(n_spis, b, 2)``), ``pairs.npy`` for a sampled run,
and ``report.json``: wall time, per-SPI timings, errors, the sampling record,
bound cores, thread settings, start method and peak memory.
"""
import argparse
import json
import multiprocessing as mp
import os
import platform
import resource
import sys
import time
import warnings
from pathlib import Path

import numpy as np

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--member", required=True)
    parser.add_argument("--n-pairs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--config", default="benchmarked_p90")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import pyspi
    from pyspi import _parallel
    from pyspi.calculator import Calculator

    warnings.simplefilter("ignore")
    with np.load(args.archive, allow_pickle=False) as archive:
        series = np.asarray(archive[args.member], dtype=np.float64)
    T, M = series.shape
    cores = _parallel.available_cores()
    if args.n_jobs > cores:
        raise SystemExit(f"n_jobs={args.n_jobs} exceeds the {cores} bound core(s)")
    threads = {name: os.environ.get(name) for name in THREAD_VARIABLES}

    calc = Calculator(dataset=series.T, config=args.config, zscore=False, verbose=False)
    start = time.perf_counter()
    if args.n_pairs:
        values = calc.compute_pairs(n_pairs=args.n_pairs, seed=args.seed,
                                    n_jobs=args.n_jobs, progress=False)
    else:
        calc.compute(n_jobs=args.n_jobs, progress=False)
        values = np.stack([calc.table[key].to_numpy(dtype=float) for key in calc.spis])
    wall = time.perf_counter() - start

    unit = 1 if sys.platform == "darwin" else 1024      # ru_maxrss: bytes or KiB
    peak = lambda who: resource.getrusage(who).ru_maxrss * unit / 2**30
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "values.npy", values)
    if args.n_pairs:
        np.save(args.out / "pairs.npy", calc.pairs)
    report = {
        "member": args.member, "M": M, "T": T, "config": args.config,
        "mode": "sampled" if args.n_pairs else "dense",
        "n_dyads": M * (M - 1) // 2, "n_pairs": args.n_pairs or None, "seed": args.seed,
        "n_jobs": args.n_jobs, "wall_seconds": wall,
        "spi_seconds": sum(calc.timings.values()),
        "timings": calc.timings, "errors": calc.errors, "spis": list(calc.spis),
        "sampling": calc.sampling,
        "bound_cores": cores, "thread_variables": threads,
        "affinity": (sorted(os.sched_getaffinity(0))
                     if hasattr(os, "sched_getaffinity") else None),
        "start_method": _parallel.default_mp_context() if args.n_jobs > 1 else None,
        "peak_rss_gib": {"parent": peak(resource.RUSAGE_SELF),
                         "largest_worker": peak(resource.RUSAGE_CHILDREN)},
        "pyspi": {"file": pyspi.__file__, "computation": _parallel.COMPUTATION_VERSION},
        "host": platform.node(), "job": os.environ.get("PBS_JOBID"),
        "python": platform.python_version(), "numpy": np.__version__,
        "mp_start_methods": mp.get_all_start_methods(),
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: report[k] for k in ("member", "mode", "n_pairs", "n_jobs",
                                             "wall_seconds", "peak_rss_gib")}))


if __name__ == "__main__":
    main()
