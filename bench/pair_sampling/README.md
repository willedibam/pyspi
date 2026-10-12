# Pair-sampling validation

Checks `Calculator.compute_pairs()` against dense extraction and against the
retrospective pair-sampling study in `mts-spi-study-cluster`
(`notebooks/optimization/channel-pair-sampling.ipynb`). Everything uses
`benchmarked_p90` (289 SPIs).

```bash
python bench/pair_sampling/validate.py extract          # ~8 min, M=20, T=1000
python bench/pair_sampling/validate.py retrospective \
    --study-results <mts-spi-study-cluster>/results/pyspi-optimisation/large_m_pair_baselines_260917
python bench/pair_sampling/validate.py figures          # needs latex + dvipng
```

Outputs are written to `bench/pair_sampling/results/`.

## What each step establishes

**`extract` — the sampled entries are the dense entries.** One VAR(1) series,
M=20, T=1000 (190 dyads). Dense `compute()` once, then `compute_pairs()` at 95
and 19 dyads for three pair seeds each, with the global RNG seeded identically
before every run.

- Every one of the 289 SPIs matches the dense table on every run: 271-272 are
  bit-identical and the remaining 17-18, all from the four pair-local classes,
  agree to 9e-16.
- The SPI-SPI vector built from the sampled extraction equals the one obtained
  by masking the dense matrices to 4e-15 on all 41,616 coordinates.
- Wall time: dense 125 s. Sampled 67-72 s at 95 dyads (1.8x) and 16-17 s at 19
  dyads (7.5x).

| tier | SPIs | dense | 95 dyads | 19 dyads |
|---|---:|---:|---:|---:|
| pair loop (dense matrix is a loop over `bivariate`; replayed exactly) | 94 | 107.4 s | 54.9 s | 11.3 s |
| pair-local (vectorised dense path; `bivariate` on sampled pairs) | 53 | 14.9 s | 11.1 s | 2.3 s |
| indexed (computed in full, entries read off) | 142 | 2.8 s | 2.8 s | 3.0 s |

The pair-loop tier costs the fraction of dyads visited (53.7 s and 10.7 s
predicted). The pair-local tier costs about 1.5x its dense cost per dyad, so
it only pays below roughly two thirds of the census.

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

## Projected large-M cost (not measured)

End-to-end sampled extraction has only been timed at M=20. The table applies a
cost model to the per-SPI dense timings recorded with the cached extractions on
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
