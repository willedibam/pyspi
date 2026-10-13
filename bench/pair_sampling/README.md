# Pair-sampling validation

Checks `Calculator.compute_pairs()` against dense extraction and against the
retrospective pair-sampling study in `mts-spi-study-cluster`
(`notebooks/optimization/channel-pair-sampling.ipynb`). Everything uses
`benchmarked_p90` (289 SPIs).

```bash
python bench/pair_sampling/validate.py extract          # ~4 min, M=20, T=1000
python bench/pair_sampling/validate.py retrospective \
    --study-results <mts-spi-study-cluster>/results/pyspi-optimisation/large_m_pair_baselines_260917
python bench/pair_sampling/validate.py figures          # needs latex + dvipng
python bench/pair_sampling/workers.py                   # ~5 min, M=20, T=1000
```

Outputs are written to `bench/pair_sampling/results/`.

## What each step establishes

**`extract` — the sampled entries are the dense entries.** One VAR(1) series,
M=20, T=1000 (190 dyads). Dense `compute()` once, then `compute_pairs()` at 95
and 19 dyads for three pair seeds each, with the global RNG seeded identically
before every run.

- Every one of the 289 SPIs matches the dense table on every run: 271-272 are
  bit-identical and the remaining 17-18, all from the four rounding-level
  pair-local classes, agree to 9e-16.
- The SPI-SPI vector built from the sampled extraction equals the one obtained
  by masking the dense matrices to 4e-15 on all 41,616 coordinates.
- Wall time: dense 60 s. Sampled 33-34 s at 95 dyads (1.8x) and 8 s at 19
  dyads (7.8-7.9x).

| tier | SPIs | dense | 95 dyads | 19 dyads |
|---|---:|---:|---:|---:|
| pair loop (dense matrix is a loop over `bivariate`; replayed exactly) | 100 | 53.5 s | 27.0 s | 5.4 s |
| pair-local (vectorised dense path; `bivariate` on sampled pairs) | 53 | 5.6 s | 5.3 s | 1.0 s |
| indexed (computed in full, entries read off) | 136 | 1.2 s | 1.2 s | 1.2 s |

Sampled columns are means over the three pair seeds. The pair-loop tier costs
the fraction of dyads visited (26.7 s and 5.3 s predicted). The pair-local tier
costs about 1.9x its dense cost per dyad, so it only pays below roughly half
of the census. Distance correlation and the six nonparametric spectral-Granger
SPIs count as pair loop: they are evaluated on the sampled dyads only and are
bit-identical.

An SPI can be recorded as failed on a sample and not on the full table: of the
three multitaper group-delay SPIs, two have no finite entry at all and the
third has none on some 19-dyad samples, where it is reported as
`no finite value on the sampled pairs`.

**`retrospective` — the sampler and the convergence match the study.** No
extraction. With the study's seed sequences, `sample_dyads` reproduces its draws
exactly: all 17,920 per-draw RMSE, 95th-percentile error and coverage values for
the twelve MEG records (M=64/100/243, 128 draws each) agree with its
`all-draws.csv` to 2e-6. Median whole-vector RMSE:

| dyads | M=64 | M=100 | M=243 |
|---:|---:|---:|---:|
| 100 | 0.07970 | 0.08540 | 0.08817 |
| 400 | 0.03653 | 0.04118 | 0.04386 |
| 800 | 0.02249 | 0.02738 | 0.03144 |

Together these mean a sampled extraction at large M yields the SPI-SPI vector
the study obtained by masking.

**`workers.py` -- sampling and workers together.** The same series, dense and
at 95 and 19 dyads, on 1, 2 and 4 spawned workers; one fresh interpreter per
run (Apple M5, 10 cores, macOS).

| run | 1 worker | 2 workers | 4 workers | peak memory, 1 worker | largest worker of 4 |
|---|---:|---:|---:|---:|---:|
| dense (190 dyads) | 62.7 s | 38.5 s (1.6x) | 26.9 s (2.3x) | 673 MiB | 624 MiB |
| 95 dyads | 35.3 s | 25.0 s (1.4x) | 18.0 s (2.0x) | 743 MiB | 641 MiB |
| 19 dyads | 9.9 s | 11.4 s (0.9x) | 9.9 s (1.0x) | 744 MiB | 671 MiB |

- The result hash is the same for every worker count in each row.
- Workers do not scale linearly here. The summed per-SPI time rises from 63 s
  to 85 s at four workers (each SPI runs slower alongside others; the cause
  was not isolated), and starting a spawned pool costs several seconds. At 19
  dyads the start-up is the larger part, so workers add nothing.
- A parallel run's parent holds about 300 MiB; `parent + n_jobs x largest
  worker` bounds the total (about 2.7 GiB for the dense run on four workers).
  These sizes are for M=20, T=1000 and say nothing about larger recordings.

## Measured on Gadi

`gadi_smoke.py` / `gadi_smoke.pbs`, one PBS job per configuration on a Cascade
Lake node, forked workers, each job bound to its cores; reduced by
`gadi_summarise.py` to `results/gadi-smoke/`. One MEG recording, T=1000,
computation r8. These runs predate two changes that lower the times below
without changing any value: the directed spectral SPIs now share one
factorisation per pair (they were 44% of the M=64 run), and nonparametric
spectral Granger causality is evaluated on the sampled pairs only (it was the
18-minute task at M=243). Neither has been re-timed on Gadi. The earlier full extractions (r7, one core) took 1.35 h at
M=64 and 19.3 h at M=243.

| run | workers | wall | speed-up | job peak memory |
|---|---:|---:|---:|---:|
| M=64, full table | 1 | 65.0 min | 1.0x | 3.5 GiB |
| | 4 | 18.0 min | 3.6x | 8.0 GiB |
| | 8 | 9.9 min | 6.6x | 8.3 GiB |
| M=243, 1,000 of 29,403 dyads | 1 | 61.2 min | 1.0x (19x the full r7 run) | 20.3 GiB |
| | 4 | 20.8 min | 2.9x (55x) | 35.5 GiB |
| | 8 | 18.3 min | 3.3x (63x) | 39.0 GiB |
| M=243, 2,500 dyads | 8 | 18.2 min | | 34.8 GiB |

- Results are identical for every worker count, and the 2,500-dyad run begins
  with the 1,000-dyad run's dyads and values.
- Against the r7 full extractions: at M=64 all 274 SPIs not changed in r8 are
  bit-identical; at M=243, 249 are bit-identical on the sampled entries, 17
  agree to rounding, and the same 8 fail. None differs.
- The SPI-SPI vector from the M=243 runs has RMSE 0.032 at 1,000 dyads and
  0.016 at 2,500 against the full extraction (r8-changed SPIs excluded).
- At M=243 the wall time stops at 18 min from four workers on, for either
  budget: that is one task, the 24 nonparametric spectral-Granger SPIs, which
  share a cache and are computed on every channel. Summed SPI time is 61 min at
  1,000 dyads and 119 min at 2,500.
- At M=64, eight workers are limited by the longest task (10 min, one group of
  directed spectral SPIs); the summed SPI time is 10% higher with workers than
  without.

## Projected large-M cost (superseded where measured)

Written before the Gadi runs above. At M=243 it predicted 1.1 h for 1,000
dyads and 2.0 h for 2,500 on one core; measured, 1.0 h and (summed over eight
workers) 2.0 h. The table applies a cost model to the per-SPI dense timings recorded with the cached extractions on
Gadi (`projected-speedup.csv`): indexed SPIs keep their dense cost, pair loops
scale with the fraction of dyads, and pair-local SPIs use the cost per dyad
measured at M=20, converted between machines by the ratio of their pair-loop
cost per dyad. That conversion is an assumption; both runs are at T=1000.

| M | dense | 500 dyads | 1,000 | 2,500 | 5,000 |
|---:|---:|---:|---:|---:|---:|
| 100 (MEG, mean of 4) | 3.4 h | 0.44 h (7.7x) | 0.80 h (4.2x) | 1.9 h (1.8x) | census |
| 243 (MEG, mean of 4) | 17.8 h | 0.74 h (24x) | 1.1 h (17x) | 2.0 h (8.8x) | 3.6 h (4.9x) |
| 256 (sparse VAR) | 20.1 h | 0.74 h (27x) | 1.1 h (19x) | 2.1 h (9.8x) | 3.7 h (5.5x) |

The 142 indexed SPIs are about 2% of a dense run, which caps the speedup near
45x however few pairs are sampled.

The Gadi timings behind this table predate the faster distance-correlation,
Kraskov and directed spectral kernels. Those lower the dense cost of
pair-evaluated SPIs, so the ratios above overstate what sampling adds on
current code; the table has not been recomputed.

## Figures

- `local-full-vector-convergence`: M=20 series. Lines are medians over 128
  draws from the dense matrices, bands the 10-90% spread of draws, red circles
  the actual `compute_pairs` extractions at their nested prefixes.
- `meg-full-vector-convergence`: the study's MEG figure, regenerated with the
  pyspi sampler. Bands pool four recordings and 128 draws each; they are not
  confidence intervals.
- `gallery-*`: 24 SPI-SPI coordinates chosen by what the statistics mean. Dashed
  line exact value, line and markers the median of 128 draws, band 2.5-97.5% of
  draws, faint lines every nested path. Vertical scales differ between panels.
