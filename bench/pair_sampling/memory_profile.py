"""Where one recording's memory goes, by size: the figures a memory request is built from.

    python memory_profile.py --M 64 --T 1024 --out results/memory/m64-t1024.json

A ``benchmarked_p90`` recording on a seeded VAR(1) series of ``M`` processes and
``T`` observations, in one process, in two parts:

* every SPI that one worker computes whole (``is_pair_loop`` false), in full, in
  the order a worker meets them: the high-water mark of resident memory, and
  the SPIs that raised it;
* the pair loops on 16 and then 64 dyads: the resident memory they leave behind
  per dyad (per-pair caches live for the whole run).

A full table on one worker then needs about the first figure plus the second
times ``M(M-1)/2``; with ``k`` workers the second is spread between them.
Linux only (``/proc/self/statm``).
"""
import argparse
import json
import os
import resource
import time
import warnings
from pathlib import Path

import numpy as np

GIB = 2**30


def resident():
    return int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / GIB


def high_water():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / GIB


def series(M, T, seed=0):
    rng = np.random.default_rng(seed)
    A = 0.6 * np.eye(M) + 0.3 * np.roll(np.eye(M), 1, axis=1)
    x = np.zeros((M, T + 200))
    for t in range(1, T + 200):
        x[:, t] = A @ x[:, t - 1] + rng.standard_normal(M)
    return x[:, 200:]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--config", default="benchmarked_p90")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    warnings.simplefilter("ignore")

    from pyspi import _parallel
    from pyspi.calculator import load_spis_from_yaml, resolve_config
    from pyspi.data import Data

    spis = load_spis_from_yaml(resolve_config(args.config), quiet=True)
    data = Data(data=series(args.M, args.T), dim_order="ps", zscore=True)
    M = data.n_processes
    tasks = _parallel.build_tasks(list(spis), spis)
    loops = [k for task in tasks for k in task if _parallel.is_pair_loop(spis[k])]
    whole = [k for task in tasks for k in task if not _parallel.is_pair_loop(spis[k])]
    report = {"M": M, "T": args.T, "config": args.config, "n_whole": len(whole),
              "n_pair_loops": len(loops), "after_setup_gib": high_water(),
              "computation": _parallel.COMPUTATION_VERSION, "job": os.environ.get("PBS_JOBID")}

    start, mark, raised, failed = time.perf_counter(), high_water(), {}, 0
    for key in whole:
        _, err, _, _ = _parallel.run_spi(spis[key], data, key, M)
        failed += err is not None
        if high_water() > mark + 0.01:
            raised[key] = high_water() - mark
            mark = high_water()
    report["whole"] = {"peak_gib": mark, "seconds": time.perf_counter() - start, "failed": failed,
                       "raised_by": dict(sorted(raised.items(), key=lambda kv: -kv[1])[:8])}

    pairs = _parallel.all_dyads(M)
    pairs = pairs[np.random.default_rng(0).permutation(len(pairs))[:64]]
    readings = [resident()]
    start = time.perf_counter()
    for chunk in (pairs[:16], pairs[16:64]):
        for key in loops:
            _parallel.run_spi_pairs(spis[key], data, key, chunk, complete=False)
        readings.append(resident())
    kept = (readings[2] - readings[1]) / max(1, len(pairs) - 16)
    report["pair_loops"] = {"resident_gib": readings, "dyads": [0, 16, len(pairs)],
                            "kept_gib_per_dyad": kept, "all_dyads_gib": kept * M * (M - 1) / 2,
                            "seconds_per_dyad": (time.perf_counter() - start) / len(pairs),
                            "peak_gib": high_water()}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("M", "T")} | {
        "whole_peak_gib": round(mark, 2), "kept_gib_per_dyad": round(kept, 5),
        "all_dyads_gib": round(report["pair_loops"]["all_dyads_gib"], 2)}))


if __name__ == "__main__":
    main()
