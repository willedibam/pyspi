# Pair-sampling validation

Checks `Calculator.compute_pairs()` against dense extraction and against the
retrospective pair-sampling study in `mts-spi-study-cluster`
(`notebooks/optimization/channel-pair-sampling.ipynb`). Everything uses
`benchmarked_p90` as it stood when they were run: 289 SPIs. Six `dspli`/`dswpli`
`_max` variants, constant on every input, have since been removed (283).

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
`gadi_summarise.py`. One MEG recording, T=1000, computation r8. The earlier
full extractions (r7, one core) took 1.35 h at M=64 and 19.3 h at M=243.
`results/gadi-smoke-2/` is the current code; `results/gadi-smoke/` the same
runs a day earlier, before the directed spectral SPIs shared a factorisation
and before nonparametric spectral Granger causality was sampled.

| run | workers | wall | before | job peak memory |
|---|---:|---:|---:|---:|
| M=64, full table | 1 | 40.3 min | 65.0 min | 2.7 GiB |
| | 4 | 10.8 min (3.8x) | 18.0 min | 5.1 GiB |
| | 8 | 5.9 min (6.9x) | 9.9 min | 5.9 GiB |
| M=243, 1,000 of 29,403 dyads | 1 | 52.0 min | 61.2 min | 20.0 GiB |
| | 4 | 9.4 min | 20.8 min | 41.7 GiB |
| | 8 | 4.8 min | 18.3 min | 39.0 GiB |
| M=243, 2,305 dyads (`tolerance=0.03`) | 4 | 15.7 min | | 27.9 GiB |

- Results are identical for every worker count and to the earlier runs, and
  the 2,305-dyad run begins with the 1,000-dyad run's dyads and values.
- Against the r7 full extractions: at M=64 all 274 SPIs not changed in r8 are
  bit-identical; at M=243, 249 are bit-identical on the sampled entries, 17
  agree to rounding, and the same 8 fail. None differs.
- The SPI-SPI vector from the M=243 runs has RMSE 0.032 at 1,000 dyads and
  0.017 at 2,305 against the full extraction (r8-changed SPIs excluded).
- The M=243 one-worker job ran each SPI about 40% slower than the same SPIs
  ran inside the worker jobs (summed SPI time 52 min against 36-37 min),
  probably a busier node; its wall time overstates the serial cost, and the
  apparent speed-ups over it (5.5x on 4, 10.9x on 8) are inflated accordingly.
- Memory at M=243 is set by the SPIs computed on every channel, about 16 GiB
  in the largest worker; it does not fall with the sample.
- `gadi_binding.pbs`: two M=64 recordings side by side under `nci-parallel`
  with four cores each were bound to four distinct cores apiece and took
  12.4 min each, against 10.8 min for one alone on another node.

## Pairs shared between workers

`results/gadi-split/`, the same recordings and scripts on computation r9 with
`split_pairs` (the default): an SPI that is a loop over pairs has its pairs cut
into one piece per worker. One PBS job per row, Cascade Lake, forked workers.
"Cold" is a compile cache that starts empty in every job (`gadi_smoke.pbs`'s
default); "warm" is `NUMBA_CACHE=<dir>` filled by an earlier job.

| M=64, full table | wall, cold | wall, warm | job memory, warm | summed SPI time, warm |
|---|---:|---:|---:|---:|
| 1 worker | 40.6 min | | 4.1 GiB | 2,436 s |
| 2 | 19.2 min | | 4.5 GiB | 2,302 s |
| 4 | 10.1 min | 10.0 min (4.1x) | 4.0 GiB | 2,385 s |
| 8 | 5.8 min | 5.0 min (8.1x) | 5.0 GiB | 2,385 s |
| 16 | 3.5 min | 2.6 min (15.6x) | 7.3 GiB | 2,488 s |
| 16, `split_pairs=False` | 5.1 min | | 6.2 GiB | 2,450 s |
| 32 | 1.9 min | 1.6 min (25.7x) | 11.3 GiB | 3,007 s |
| 48 | 1.5 min | 1.3 min (31.4x) | 13.6 GiB | 3,151 s |

- All fourteen M=64 tables, the serial one included, are the same file
  (`values_sha256` in `summary.json`, which also holds each run's wall time,
  summed SPI time and PBS memory).
- Single runs on different nodes: a speed-up above the worker count (8.1x on
  8) is a difference between nodes.
- Unsplit, the run stops gaining at eight workers (5.9 min in the runs above,
  5.1 on 16): the DTW barycenter task alone is five minutes. Split, the longest task
  left is the spectral Granger group at 76 s, which is what 48 workers reach.
- The cold runs spend about 30 s per worker compiling: summed over workers,
  `dcorr` rises from 109 s to 1,077 s on 48 and `bary_dtw_mean` from 308 s to
  859 s, and no other SPI moves. hyppo's kernels are cached on disk, so a warm
  cache removes the first; tslearn's are not, so each worker still compiles
  them. This is a fixed cost per worker and matters less the longer the run.
- M=243, 1,000 dyads: 3.6 min on 16 workers and on 48 (4.8 on 8, unsplit),
  72 and 81 GiB. The bound is the spectral Granger group, 218 s, of which 182 s
  is three parametric SPIs fitted on every channel; sampled runs have little
  pair-loop work left to share.

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
