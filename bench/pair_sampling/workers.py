"""Wall time and peak memory of dense and sampled p90 runs on 1, 2 and 4 workers.

One fresh interpreter per configuration, so peak memory is that run's own:

    python bench/pair_sampling/workers.py            # M=20, T=1000

Writes ``results/workers.csv``. ``wall_s`` includes starting the pool.
``parent_rss_mib`` and ``worker_rss_mib`` are the peak resident sizes of the
calling process and of its largest worker; workers overlap in time, so
``parent + n_jobs * worker`` bounds the total from above. ``values_sha256``
hashes the results and must not vary with ``n_jobs``.
"""
import argparse
import csv
import hashlib
import json
import resource
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).parent / "results"
# ru_maxrss is bytes on macOS and kilobytes on Linux.
RSS_UNIT = 1 if sys.platform == "darwin" else 1024


def one(M, T, n_pairs, n_jobs, seed):
    from pyspi.calculator import Calculator
    from validate import var_series

    warnings.simplefilter("ignore")
    calc = Calculator(dataset=var_series(M, T, seed), config="benchmarked_p90",
                      verbose=False)
    t0 = time.perf_counter()
    if n_pairs:
        values = calc.compute_pairs(n_pairs=n_pairs, seed=seed, n_jobs=n_jobs,
                                    progress=False)
    else:
        calc.compute(n_jobs=n_jobs, progress=False)
        values = calc.table.to_numpy(dtype=float)
    wall = time.perf_counter() - t0
    rss = lambda who: resource.getrusage(who).ru_maxrss * RSS_UNIT / 2**20
    return dict(
        M=M, T=T, n_dyads=M * (M - 1) // 2, n_pairs=n_pairs or "all", n_jobs=n_jobs,
        wall_s=round(wall, 1), spi_seconds=round(sum(calc.timings.values()), 1),
        parent_rss_mib=round(rss(resource.RUSAGE_SELF)),
        worker_rss_mib=round(rss(resource.RUSAGE_CHILDREN)) if n_jobs > 1 else 0,
        failed=len(calc.errors),
        values_sha256=hashlib.sha256(
            np.nan_to_num(values, nan=-1.0).tobytes()).hexdigest()[:16],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--M", type=int, default=20)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--one", nargs=2, type=int, metavar=("N_PAIRS", "N_JOBS"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.one:
        print(json.dumps(one(args.M, args.T, args.one[0], args.one[1], args.seed)))
        return

    n_dyads = args.M * (args.M - 1) // 2
    rows = []
    for n_pairs in (0, n_dyads // 2, n_dyads // 10):
        for n_jobs in (1, 2, 4):
            out = subprocess.run(
                [sys.executable, __file__, "--M", str(args.M), "--T", str(args.T),
                 "--seed", str(args.seed), "--one", str(n_pairs), str(n_jobs)],
                capture_output=True, text=True, check=True, cwd=Path(__file__).parent)
            rows.append(json.loads(out.stdout.strip().splitlines()[-1]))
            print(rows[-1], flush=True)
        assert len({r["values_sha256"] for r in rows[-3:]}) == 1, "worker counts disagree"
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "workers.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
