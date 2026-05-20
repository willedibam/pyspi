#!/usr/bin/env python
"""Benchmark Calculator.compute() across an (M, T, n_jobs) grid.

Reproducible, non-notebook timing suite. Each grid cell is run ``--repeats``
times on freshly generated synthetic data; per-SPI and total wall times are
reported as mean/std. Environment metadata (pyspi git sha, dependency
versions + fingerprint, platform) is captured so results pin to an exact
environment. Output is a single JSON file, written incrementally after each
cell so the run is interrupt-safe (``--resume`` skips completed cells).

Usage:
    python -m bench.bench_compute --preset parallel --config benchmarked90_config.yaml
    python -m bench.bench_compute --m 8,16 --t 200,800 --n-jobs 1,4 --config fast
    python -m bench.bench_compute --preset scaling --resume
    python -m bench.bench_compute --preset parallel --array-index $PBS_ARRAY_INDEX

Presets (each fixes an M/T/n_jobs grid; --config still applies):
    headline   reference points (M=10,T=500), (M=20,T=1000) at n_jobs=1.
    scaling    M={4,8,16,32} x T={200,400,800,1600} at n_jobs=1 — feeds the
               per-SPI amortized-walltime model used to cut benchmarked_*.yaml.
    parallel   M=16, T=800, n_jobs={1,2,4,8,16} — parallel speedup curve.
    amortized  M={8,16}, T=800, n_jobs=1 — per-SPI walltime for config cutting.

Note on n_jobs and the amortized configs: per-SPI cost is invariant to n_jobs
under the cache-aware scheduler (each cache group runs sequentially within one
worker), so n_jobs=1 is the clean measurement regime for config cutting.
n_jobs only changes makespan — that is what the 'parallel' preset measures.

Two-axis parallelism: this script benchmarks INNER parallelism
(Calculator.compute(n_jobs=)). OUTER parallelism (many datasets at once)
belongs to the job scheduler — e.g. a PBS array over --array-index.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as im
import json
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from pyspi.calculator import Calculator

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLED_CONFIG_DIR = REPO_ROOT / "pyspi"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "bench" / "results"

PRESETS = {
    "headline": {"points": [(10, 500), (20, 1000)], "n_jobs": [1]},
    "scaling": {"M": [4, 8, 16, 32], "T": [200, 400, 800, 1600], "n_jobs": [1]},
    "parallel": {"M": [16], "T": [800], "n_jobs": [1, 2, 4, 8, 16]},
    "amortized": {"M": [8, 16], "T": [800], "n_jobs": [1]},
}

TRACKED_DEPS = (
    "pyspi", "numpy", "scipy", "pandas", "scikit-learn", "statsmodels",
    "mne", "mne-connectivity", "spectral-connectivity", "nitime",
    "hyppo", "cdt", "torch", "tslearn", "dtaidistance", "pyEDM",
    "h5py", "pyyaml", "tqdm",
)


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--m", type=_parse_int_list, default=[8],
                   help="Comma-separated process counts (ignored if --preset is set).")
    p.add_argument("--t", type=_parse_int_list, default=[200],
                   help="Comma-separated observation counts (ignored if --preset is set).")
    p.add_argument("--n-jobs", dest="n_jobs", type=_parse_int_list, default=[1],
                   help="Comma-separated worker counts (ignored if --preset is set).")
    p.add_argument("--preset", choices=list(PRESETS), default=None,
                   help="Predefined M/T/n_jobs grid; overrides --m/--t/--n-jobs.")
    p.add_argument("--config", default="fabfour",
                   help="Bundled subset name (all/fast/sonnet/fabfour), a bundled "
                        "config filename (e.g. benchmarked90_config.yaml), or a path.")
    p.add_argument("--mp-context", choices=["spawn", "fork", "forkserver"], default="spawn",
                   help="Multiprocessing start method for n_jobs>1 (default: spawn).")
    p.add_argument("--repeats", type=int, default=2, help="Repeats per cell (default: 2).")
    p.add_argument("--seed", type=int, default=0, help="Base RNG seed (default: 0).")
    p.add_argument("--output", type=Path, default=None,
                   help="Output JSON path (default: bench/results/timings_<config>_<ts>.json).")
    p.add_argument("--resume", action="store_true",
                   help="Reuse an existing --output JSON; skip cells already at >= --repeats.")
    p.add_argument("--array-index", type=int, default=None,
                   help="Run only the Nth (1-indexed) cell of the resolved grid. For PBS arrays.")
    return p.parse_args(argv)


def resolve_config(arg: str) -> str:
    """Map a subset name / bundled filename / path to a value Calculator accepts."""
    if arg in {"all", "fast", "sonnet", "fabfour"}:
        return arg
    p = Path(arg).expanduser()
    if p.is_file():
        return str(p.resolve())
    bundled = BUNDLED_CONFIG_DIR / arg
    if bundled.is_file():
        return str(bundled.resolve())
    raise FileNotFoundError(
        f"--config '{arg}' is not a subset name, a bundled config, or a file path.")


def make_calculator(config: str, dataset: np.ndarray) -> Calculator:
    if config in {"all", "fast", "sonnet", "fabfour"}:
        return Calculator(dataset=dataset, subset=config, normalise=False, verbose=False)
    return Calculator(dataset=dataset, configfile=config, normalise=False, verbose=False)


def peak_rss_mb() -> float:
    """Process peak RSS in MB. macOS reports bytes; Linux reports KB."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1.0 if sys.platform == "darwin" else 1024.0
    return (r * scale) / (1024.0 * 1024.0)


def git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5)
        return out.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


def dep_versions() -> dict[str, str]:
    out = {}
    for name in TRACKED_DEPS:
        try:
            out[name] = im.version(name)
        except im.PackageNotFoundError:
            continue
    return out


def build_environment() -> dict:
    versions = dep_versions()
    payload = ";".join(f"{k}=={v}" for k, v in sorted(versions.items()))
    return {
        "datetime": datetime.now().isoformat(timespec="seconds"),
        "pyspi_git_sha": git_sha(),
        "python_version": platform.python_version(),
        "platform": f"{platform.system()}-{platform.release()}-{platform.machine()}",
        "dep_versions": versions,
        "dep_fingerprint": "sha256:" + hashlib.sha256(payload.encode()).hexdigest()[:16],
        "env": {k: os.environ.get(k, "") for k in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "PYSPI_N_JOBS")},
    }


def summarise(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": round(float(arr.mean()), 6),
        "std": round(float(arr.std(ddof=0)), 6),
        "values": [round(float(v), 6) for v in arr],
    }


def resolve_grid(args) -> list[tuple[int, int, int]]:
    """Return the list of (M, T, n_jobs) cells."""
    if args.preset is not None:
        spec = PRESETS[args.preset]
        n_jobs = spec["n_jobs"]
        if "points" in spec:
            mt = list(spec["points"])
        else:
            mt = [(m, t) for m in spec["M"] for t in spec["T"]]
    else:
        mt = [(m, t) for m in args.m for t in args.t]
        n_jobs = args.n_jobs
    return [(m, t, nj) for (m, t) in mt for nj in n_jobs]


def run_cell(M, T, n_jobs, config, mp_context, repeats, seed) -> dict:
    """Run one (M, T, n_jobs) cell ``repeats`` times. Returns a result entry dict."""
    rss_before = peak_rss_mb()
    rng = np.random.default_rng(seed)
    totals: list[float] = []
    per_spi: dict[str, list[float]] = {}
    n_spis = 0
    n_failed = 0
    error = None

    for _ in range(repeats):
        arr = rng.standard_normal((M, T)).astype(np.float64)
        try:
            calc = make_calculator(config, arr)
            t0 = time.perf_counter()
            calc.compute(n_jobs=n_jobs, mp_context=mp_context, progress=False)
            totals.append(time.perf_counter() - t0)
            n_spis = len(calc.spis)
            for k, v in calc.timings.items():
                per_spi.setdefault(k, []).append(float(v))
            tbl = calc.table
            n_failed = sum(
                bool(np.all(np.isnan(np.asarray(tbl[s])[~np.eye(M, dtype=bool)])))
                for s in calc.spis
            )
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            break

    entry = {
        "M": M, "T": T, "n_jobs": n_jobs, "repeats": len(totals),
        "cell_wall_seconds": summarise(totals) if totals else None,
        "n_spis": n_spis, "n_spis_failed": n_failed,
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "rss_delta_mb": round(peak_rss_mb() - rss_before, 1),
        "spi_seconds": {k: summarise(v) for k, v in per_spi.items()},
    }
    if error is not None:
        entry["error"] = error
    return entry


def main(argv=None) -> int:
    args = parse_args(argv)
    config = resolve_config(args.config)
    cfg_label = args.config

    cells = resolve_grid(args)
    if args.array_index is not None:
        if not 1 <= args.array_index <= len(cells):
            raise SystemExit(
                f"--array-index {args.array_index} out of range [1, {len(cells)}].")
        cells = [cells[args.array_index - 1]]

    output = args.output or (
        DEFAULT_OUTPUT_DIR / f"timings_{Path(cfg_label).stem}_{datetime.now():%Y%m%d_%H%M%S}.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.resume and output.exists():
        data = json.loads(output.read_text())
        done = {(r["M"], r["T"], r["n_jobs"])
                for r in data.get("results", [])
                if r.get("repeats", 0) >= args.repeats and "error" not in r}
    else:
        data = {
            "config": cfg_label, "mp_context": args.mp_context,
            "repeats": args.repeats, "seed": args.seed,
            "environment": build_environment(), "results": [],
        }
        done = set()

    print(f"[bench] config={cfg_label}  cells={len(cells)}  repeats={args.repeats}  "
          f"mp={args.mp_context}", file=sys.stderr)
    print(f"[bench] output={output}", file=sys.stderr)

    t_total = time.perf_counter()
    for i, (M, T, n_jobs) in enumerate(cells, 1):
        if (M, T, n_jobs) in done:
            print(f"[bench] [{i}/{len(cells)}] M={M} T={T} n_jobs={n_jobs} — skipped (resume)",
                  file=sys.stderr)
            continue
        print(f"[bench] [{i}/{len(cells)}] M={M} T={T} n_jobs={n_jobs} x{args.repeats} ...",
              file=sys.stderr, flush=True)
        t0 = time.perf_counter()
        entry = run_cell(M, T, n_jobs, config, args.mp_context, args.repeats, args.seed + i)
        wall = time.perf_counter() - t0
        if "error" in entry:
            print(f"[bench]   ERROR after {wall:.1f}s: {entry['error']}", file=sys.stderr)
        else:
            cw = entry["cell_wall_seconds"]
            print(f"[bench]   {wall:.1f}s wall (cell mean {cw['mean']:.2f}s "
                  f"+/- {cw['std']:.2f}s, {entry['n_spis']} SPIs, "
                  f"{entry['n_spis_failed']} failed, {entry['peak_rss_mb']:.0f} MB)",
                  file=sys.stderr)
        # Replace any prior entry for this cell, then incremental save.
        data["results"] = [r for r in data["results"]
                            if (r["M"], r["T"], r["n_jobs"]) != (M, T, n_jobs)]
        data["results"].append(entry)
        tmp = output.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, output)

    print(f"[bench] done in {time.perf_counter() - t_total:.1f}s -> {output}", file=sys.stderr)
    _print_speedup_summary(data)
    return 0


def _print_speedup_summary(data: dict) -> None:
    """If multiple n_jobs were run at the same (M,T), print the speedup curve."""
    rows = [r for r in data["results"] if "error" not in r and r.get("cell_wall_seconds")]
    by_mt: dict[tuple[int, int], list] = {}
    for r in rows:
        by_mt.setdefault((r["M"], r["T"]), []).append(r)
    printed = False
    for (M, T), group in sorted(by_mt.items()):
        if len(group) < 2:
            continue
        group.sort(key=lambda r: r["n_jobs"])
        base = next((r for r in group if r["n_jobs"] == 1), group[0])
        base_t = base["cell_wall_seconds"]["mean"]
        if not printed:
            print("\nSpeedup vs n_jobs=1:")
            printed = True
        print(f"  M={M} T={T}:")
        for r in group:
            t = r["cell_wall_seconds"]["mean"]
            print(f"    n_jobs={r['n_jobs']:>2}  {t:8.2f}s  {base_t / t:5.2f}x")


if __name__ == "__main__":
    sys.exit(main())
