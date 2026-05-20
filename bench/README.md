# pyspi benchmark suite

Reproducible timing for `Calculator.compute()`. Replaces the old notebook-based
sweeps with a single non-notebook script.

## Run

```bash
# Parallel speedup curve at M=16, T=800 (n_jobs = 1,2,4,8,16)
python -m bench.bench_compute --preset parallel --config benchmarked90_config.yaml

# Per-SPI walltime sweep over an M x T grid (n_jobs=1) — feeds amortized configs
python -m bench.bench_compute --preset scaling --config config.yaml

# Custom grid
python -m bench.bench_compute --m 8,16,32 --t 200,800 --n-jobs 1,4,8 --config fast
```

Run from the repo root so `bench.bench_compute` is importable. `--config`
accepts a bundled subset name (`all`/`fast`/`sonnet`/`fabfour`), a bundled
config filename (e.g. `benchmarked90_config.yaml`), or a path.

## Presets

| preset      | grid                                          | purpose |
|-------------|-----------------------------------------------|---------|
| `headline`  | (M=10,T=500), (M=20,T=1000), n_jobs=1         | quick reference points |
| `scaling`   | M={4,8,16,32} x T={200,400,800,1600}, n_jobs=1| per-SPI walltime for cutting `benchmarked_*.yaml` |
| `parallel`  | M=16, T=800, n_jobs={1,2,4,8,16}              | parallel speedup curve |
| `amortized` | M={8,16}, T=800, n_jobs=1                     | per-SPI walltime, minimal grid |

## Output

One JSON file (`bench/results/timings_*.json`), written incrementally after
each cell — interrupt-safe, `--resume` skips completed cells. Carries an
`environment` block (pyspi git sha, dependency versions + fingerprint,
platform) so results pin to an exact environment. Each result entry has
per-cell wall time, per-SPI timings (mean/std), peak RSS, and failed-SPI count.

Load for analysis:

```python
import json, pandas as pd
d = json.load(open("bench/results/timings_xxx.json"))
cells = pd.json_normalize(d["results"])              # one row per (M,T,n_jobs)
```

## Cut a benchmarked config

`cut_config.py` turns a timing JSON into a `benchmarked<N>_config.yaml` — the
former notebook step, now self-contained.

```bash
# Measure per-SPI walltime, then keep the fastest 90%
python -m bench.bench_compute --preset amortized --config config.yaml
python -m bench.cut_config --bench-json bench/results/timings_config_*.json --keep 90
```

Cost model (`--mode`):

- `amortized` (default) — SPIs sharing a `_cache_namespace` (Covariance/Precision,
  the multitaper spectral pairs, Cointegration, Barycenter, CCM, ...) split the
  group's total cost evenly: `cost = sum(group walltimes) / group size`. This is
  the true per-variant budget impact — the shared computation is built once.
- `raw` — each SPI's own measured walltime.

Output goes to `pyspi/benchmarked<N>[_amortized]_config.yaml`; dropped SPIs are
listed as trailing comments for auditability. `--m`/`--t` pick the bench cell
(default: largest).

## Cluster (PBS)

`run_benchmark.pbs` (Gadi) and `run_benchmark_physics.pbs` (USYD Physics) run
the suite on the cluster. Submit with `-v` overrides:

```bash
# config-cutting grid as a PBS array — one (M,T) cell per task, n_jobs=1:
qsub -J 1-4 -v M=32,64,T=1000,4000,CONFIG=config.yaml bench/run_benchmark.pbs

# a single (M,T) cell:
qsub -v M=64,T=2000,CONFIG=config.yaml bench/run_benchmark.pbs

# a bundled preset:
qsub -v PRESET=parallel,CONFIG=benchmarked90_config.yaml bench/run_benchmark.pbs
```

Set `M` and `T` (comma-separated) to benchmark your real data sizes — the
bundled presets only reach M=32. With `-J 1-N` each array task runs one grid
cell to its own JSON; cut a config from the cell that matches your target size.

## Notes

- **n_jobs and amortized configs**: per-SPI cost is invariant to `n_jobs` under
  the cache-aware scheduler (each cache group runs sequentially within one
  worker). Use `n_jobs=1` for config-cutting measurements; `n_jobs` only
  changes makespan, which the `parallel` preset measures.
- **Start method**: `Calculator.compute()` defaults to `fork` on Linux,
  `spawn` on macOS/Windows. fork is ~2x faster (workers inherit imported
  state via copy-on-write). Override with `--mp-context`.
- This suite measures **inner** parallelism (SPIs within one dataset). Outer
  parallelism (many datasets) belongs to the job scheduler — e.g. a PBS array.
