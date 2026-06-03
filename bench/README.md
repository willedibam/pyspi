# pyspi benchmark suite

Reproducible timing for `Calculator.compute()`. Replaces the old notebook-based
sweeps with a single non-notebook script.

## Run

```bash
# Parallel speedup curve at M=16, T=800 (n_jobs = 1,2,4,8,16)
python -m bench.bench_compute --preset parallel --config benchmarked90_amortized_config.yaml

# Per-SPI walltime sweep over an M x T grid (n_jobs=1) — feeds amortized configs
python -m bench.bench_compute --preset scaling --config config.yaml

# Custom grid
python -m bench.bench_compute --m 8,16,32 --t 200,800 --n-jobs 1,4,8 --config fast
```

Run from the repo root so `bench.bench_compute` is importable. `--config`
accepts a bundled subset name (`all`/`fast`/`sonnet`/`fabfour`), a bundled
config filename (e.g. `benchmarked90_amortized_config.yaml`), or a path.

## Presets

| preset      | grid                                          | purpose |
|-------------|-----------------------------------------------|---------|
| `headline`  | (M=10,T=500), (M=20,T=1000), n_jobs=1         | quick reference points |
| `scaling`   | M={4,8,16,32} x T={200,400,800,1600}, n_jobs=1| per-SPI walltime for cutting `benchmarked_*.yaml` |
| `parallel`  | M=16, T=800, n_jobs={1,2,4,8,16}              | parallel speedup curve |
| `amortized` | M={8,16}, T=800, n_jobs=1                     | per-SPI walltime, minimal grid |

## Output

One JSON file per cell, written atomically to `bench/results/cells/` as
`<label>_M<M>_T<T>_n<n_jobs>.json`. `--resume` skips cells whose JSON exists
with `repeats >= --repeats`. Each file is self-contained, with an
`environment` block (pyspi git sha, dependency versions + fingerprint,
platform) pinning results to an exact environment. Per-cell fields:

- `cell_wall_seconds {mean, std, values}`, `n_spis`, `n_spis_failed`, `failed_spis: [...]`
- `rss_mb_end`, `rss_mb_delta` (per-cell, via `psutil.Process().memory_info().rss`)
- `spi_seconds: {identifier: {mean, std, values, category, labels}}`
  - `category` is one of `basic | distance | causal | infotheory | spectral | wavelet | misc`
  - `labels` is the SPI's merged label list (includes `Mxx` size tags + stat-type tags)

## Cut a benchmarked config

`cut_config.py` turns a single per-cell JSON into a `benchmarked<N>_config.yaml`.

```bash
# Measure per-SPI walltime, then keep the fastest 90%
python -m bench.bench_compute --preset amortized --config config.yaml
python -m bench.cut_config --bench-json bench/results/cells/<file>.json --keep 90
```

Cost model (`--mode`):

- `amortized` (default) — SPIs sharing a `_cache_namespace` (Covariance/Precision,
  the multitaper spectral pairs, Cointegration, Barycenter, CCM, ...) split the
  group's total cost evenly: `cost = sum(group walltimes) / group size`. This is
  the true per-variant budget impact — the shared computation is built once.
- `raw` — each SPI's own measured walltime.

Output goes to `pyspi/benchmarked<N>[_amortized]_config.yaml`; by default
dropped SPIs are commented out (not deleted) so the YAML carries the full
provenance of the cut. Pass `--no-preserve-dropped` to delete instead.

## Analyse + forecast

```bash
# Cross-cell analysis (anchor stability, scaling fits, cumulative cost)
python -m bench.analyse_cells --results-glob 'bench/results/cells/<pattern>.json' \
    --config pyspi/config.yaml --percentiles 80,90,95,99 \
    --output-dir bench/results/analysis

# Predict cell wall time at a target (M, T) from the scaling fits
python -m bench.forecast_cell --config pyspi/benchmarked90_amortized_config.yaml \
    --M 64 --T 3200
```

## Cluster (PBS)

`run_benchmark.pbs` (Gadi) and `run_benchmark_physics.pbs` (USYD Physics) run
the suite on the cluster. Submit with `-v` overrides:

```bash
# config-cutting grid as a PBS array — one (M,T) cell per task, n_jobs=1:
qsub -J 1-4 -v M=32,64,T=1000,4000,CONFIG=config.yaml bench/run_benchmark.pbs

# a single (M,T) cell:
qsub -v M=64,T=2000,CONFIG=config.yaml bench/run_benchmark.pbs

# a bundled preset:
qsub -v PRESET=parallel,CONFIG=benchmarked90_amortized_config.yaml bench/run_benchmark.pbs
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
