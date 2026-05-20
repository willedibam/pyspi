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

## Cluster (PBS / Gadi)

`run_benchmark.pbs` runs the suite on Gadi. Submit with `-v` overrides:

```bash
qsub -v PRESET=parallel,CONFIG=benchmarked90_config.yaml bench/run_benchmark.pbs
```

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
