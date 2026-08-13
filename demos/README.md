# Demo notebooks

Two notebooks, meant to be read in order.

**`01_getting_started.ipynb`** -- the first thing to open. Loads the bundled `forex`
dataset, runs `Calculator(config="fabfour")`, and explains how to read `calc.table`:
its `(spi, process)` column MultiIndex, how to pull out one SPI's `M x M` matrix, how
to get a single pair, and the row-is-source / column-is-target orientation that matters
for directed SPIs. Then it compares `fabfour` against `sonnet` using measured
`calc.timings` to show where the compute cost actually goes, plots one SPI as a
heatmap, measures `compute(n_jobs=...)` across several worker counts to show what
in-dataset parallelism does and does not buy, and gives the CLI equivalent.

**`02_comparing_spis.ipynb`** -- why the library exists. Builds a six-process synthetic
dataset with known couplings (linear, nonlinear, lagged-directed, independent), runs a
hand-written six-SPI config, and shows two concrete failure modes: covariance and
Spearman miss a `y = x^2` coupling that distance correlation and mutual information
find, and undirected/contemporaneous statistics cannot recover the direction of a lag-1
coupling that transfer entropy and time-lagged mutual information get right.

## Running them

Outputs are stripped from both notebooks, so nothing is visible until you run them
locally. Both execute top to bottom with no manual steps; `01` takes roughly 85 s
(dominated by the `sonnet` config and the `n_jobs` timing sweep) and `02` roughly 5 s.

Plotting cells need matplotlib, which is not a core pyspi dependency:
`pip install matplotlib`, or `pip install "pyspi[bench]"`.
