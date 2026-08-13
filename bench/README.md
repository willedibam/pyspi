# pyspi benchmark suite

Reproducible timing for `Calculator.compute()`, and the cost model behind the
shipped `pyspi/configs/benchmarked_p{80,90,95,99}.yaml` subsets.

## Install

```bash
pip install -e '.[bench]'
```

The `bench` extra adds `psutil` (per-cell RSS), `matplotlib`/`seaborn`/`plotly`
(analysis plots) and `nbformat` — none of which are runtime dependencies of
pyspi itself. Run everything from the repo root so `bench.*` is importable.

## Run

```bash
# Per-SPI walltime sweep over an M x T grid (n_jobs=1) — feeds the cost model
python -m bench.bench_compute --preset scaling --config full

# Parallel speedup curve at M=16, T=800 (n_jobs = 1,2,4,8,16)
python -m bench.bench_compute --preset parallel --config benchmarked_p90

# Custom grid
python -m bench.bench_compute --m 8,16,32 --t 200,800 --n-jobs 1,4,8 --config fast
```

`--config` takes a bundled config name (`full`, `fast`, `sonnet`, `fabfour`,
`benchmarked_p80/p90/p95/p99`) or a path to your own YAML — it is passed
straight to `pyspi.calculator.resolve_config`, so the two forms behave exactly
as they do for `Calculator(config=...)`.

## Presets

| preset      | grid                                          | purpose |
|-------------|-----------------------------------------------|---------|
| `headline`  | (M=10,T=500), (M=20,T=1000), n_jobs=1         | quick reference points |
| `scaling`   | M={4,8,16,32} x T={200,400,800,1600}, n_jobs=1| per-SPI walltime for cutting `benchmarked_p*.yaml` |
| `parallel`  | M=16, T=800, n_jobs={1,2,4,8,16}              | parallel speedup curve |
| `amortized` | M={8,16}, T=800, n_jobs=1                     | per-SPI walltime, minimal grid |

## Output

One JSON file per cell, written atomically to `bench/results/cells/` (the
default `--output-dir`) as `<label>_M<M>_T<T>_n<n_jobs>.json`. `--resume` skips
cells whose JSON exists with `repeats >= --repeats`. Each file is
self-contained, with an `environment` block (pyspi git sha, dependency versions
+ fingerprint, platform) pinning results to an exact environment. Per-cell
fields:

- `cell_wall_seconds {mean, std, values}`, `n_spis`, `n_spis_failed`, `failed_spis: [...]`
- `rss_mb_end`, `rss_mb_delta` (per-cell, via `psutil.Process().memory_info().rss`)
- `spi_seconds: {identifier: {mean, std, values, category, labels}}`
  - `category` is one of `basic | distance | causal | infotheory | spectral | wavelet | misc`
  - `labels` is the SPI's merged label list (includes `Mxx` size tags + stat-type tags)

`--repeats` defaults to **2**. The 22 committed cells in `bench/results/cells/`
were all measured with `--repeats 1` — at M=64, T=3200 a single repeat of the
full config is already a multi-day job — so their `cell_wall_seconds.std` is 0
by construction, and cross-cell consistency (see `report.md`) is what stands in
for a within-cell error bar.

## Cut a benchmarked config

`cut_config.py` turns a single per-cell JSON into `pyspi/configs/benchmarked_p<N>.yaml`.

```bash
python -m bench.bench_compute --preset amortized --config full
python -m bench.cut_config --bench-json bench/results/cells/<file>.json --keep 90
```

Cost model (`--mode`):

- `amortized` (default) — SPIs sharing a `_cache_namespace` (Covariance/Precision,
  the multitaper spectral pairs, Cointegration, Barycenter, CCM, ...) split the
  group's total cost evenly: `cost = sum(group walltimes) / group size`. This is
  the true per-variant budget impact — the shared computation is built once.
- `raw` — each SPI's own measured walltime. Written to
  `benchmarked_p<N>_raw.yaml` so it never overwrites the shipped amortized cut.

Output goes to `pyspi/configs/benchmarked_p<N>.yaml`; by default dropped SPIs
are commented out (not deleted) so the YAML carries the full provenance of the
cut. Pass `--no-preserve-dropped` to delete instead.

**`benchmarked_p90.yaml` carries a hand edit**: the two `te_kraskov_..._DCE_k-{1,2}`
variants were added back after the cut for methodological reasons. Re-running
`cut_config` overwrites it. The rationale lives in the config's own header
(`pyspi/configs/benchmarked_p90.yaml`, lines 6-16) — read it before regenerating.

## Analyse + forecast

```bash
# Cross-cell analysis (anchor stability, scaling fits, cumulative cost).
# With no arguments this analyses every committed cell against the full config.
python -m bench.analyse_cells

# Explicit form
python -m bench.analyse_cells \
    --results-glob 'bench/results/cells/physics_config_M*_T*_n1.json' \
    --config full --percentiles 80,90,95,99 \
    --output-dir bench/results/analysis

# Predict cell wall time at a target (M, T) from the scaling fits
python -m bench.forecast_cell --config benchmarked_p90 --M 64 --T 3200
```

`analyse_cells` writes into `bench/results/analysis/`. Only the small
human-readable summaries are committed — `report.md`, `scaling.csv`,
`cell_summary.csv` (and `dropped_spi_comparison.md`). The bulk artefacts
(`long_costs.csv`, `extrapolations_M*_T*.csv`, `jaccard_p*.csv`, `plot_*.png`)
are gitignored and regenerate in seconds; the Jaccard matrices are also
embedded verbatim in `report.md`. `forecast_cell` and `benchmark.ipynb` both
read `long_costs.csv`, so run `analyse_cells` once before either.

`benchmark.ipynb` is committed **without outputs** (its Plotly payloads were
2.4 MB and do not render on GitHub). Run it locally to regenerate the figures.

## Cluster (PBS)

Five PBS Pro scripts. None sets `#PBS -M`/`#PBS -m` — qsub does not expand
shell variables inside `#PBS` directives, so pass mail options at submit time
(`qsub -m bea -M you@example.org ...`). The venv is `${VENV:-<repo>/.venv}` in
every script; override with `-v VENV=/path/to/venv`.

**Generic (`bench/`)** — parameterised via `-v`:

```bash
# config-cutting grid as a PBS array — one (M,T) cell per task, n_jobs=1:
M=32,64 T=1000,4000 CONFIG=full qsub -J 1-4 -v M,T,CONFIG bench/run_benchmark.pbs

# a single (M,T) cell:
M=64 T=2000 CONFIG=full qsub -v M,T,CONFIG bench/run_benchmark.pbs

# a bundled preset:
qsub -v PRESET=parallel,CONFIG=benchmarked_p90 bench/run_benchmark.pbs
```

`run_benchmark.pbs` targets NCI Gadi (`#PBS -P`, `-l storage`, `module load
python3`); `run_benchmark_physics.pbs` targets a plain PBS Pro queue. Note
`qsub -v` splits on commas, so comma-containing values (`M`, `T`) must be
exported in the shell and passed by name, as above.

**Fixed grids (`bench/physics/`)** — the three scripts that actually produced
the committed cells, hardcoding the exact grid so the run is reproducible with
no arguments:

| script                | grid                                | form |
|-----------------------|-------------------------------------|------|
| `run_bench_light.pbs` | M={4,8,16} x T={200..3200}, 15 cells | one sequential job |
| `run_bench_m32.pbs`   | M=32 x T={200..3200}, 5 cells        | array `-J 1-5` |
| `run_bench_m64.pbs`   | M=64 x T={200..3200}, 5 cells        | array `-J 1-5`, 168 h walltime |

All three use `--config full --repeats 1 --output-dir bench/results/cells
--label physics_config --resume`, which is where the `physics_config_M*_T*_n1.json`
filenames come from. Submit from the repo root: `qsub bench/physics/run_bench_m64.pbs`.

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
- **JIDT parity harness (removed)**: `bench/jidt_parity/` validated the pure-NumPy
  information-theory estimators against JIDT 1.6.1 before the Java dependency was
  dropped. Gaussian/kernel MI and symbolic TE matched to machine precision
  (~1e-16); Kozachenko entropy to ~1e-5; the KSG (kraskov) MI/TE estimators to
  ~2e-3–9e-3, shrinking with T at the O(1/sqrt(N)) estimator noise floor. It
  needs `jpype` and the `infodynamics.jar` that was deleted with the Java code,
  so it is preserved at tag `jidt-parity-final`:
  `git checkout jidt-parity-final -- bench/jidt_parity/`.
