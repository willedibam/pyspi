# Changelog

## 3.0.0

A major overhaul. The Java/JIDT dependency is gone, the `Calculator` API is
simplified, and several long-standing correctness bugs are fixed. **This
release contains breaking changes** — see [Migrating from 2.x](#migrating-from-2x).

### Removed: Java and JIDT

Every information-theoretic estimator is now pure NumPy. `infodynamics.jar`,
`jpype` and the JVM startup path are gone, so installing pyspi no longer
requires a Java runtime.

The port was validated against JIDT 1.6.1 before the dependency was dropped.
That work caught four bugs in the port (kernel-entropy normalisation,
Theiler-windowed KSG neighbour counting, Gaussian auto-embed bias, and
KSG auto-embed estimator consistency), all fixed. Mean absolute error against
JIDT at T=1600:

| estimator | MI | TE | entropy |
|:----------|---:|---:|--------:|
| gaussian | 5.9e-17 | 4.7e-16 | 5.0e-09 |
| kernel | 6.8e-16 | 5.0e-05 | 3.1e-15 |
| symbolic | — | 2.4e-16 | — |
| kozachenko | — | — | 4.2e-05 |
| kraskov | 1.8e-03 | 2.5e-03 | — |

Gaussian/kernel MI, kernel entropy and symbolic TE agree to machine precision.
The gaussian-entropy offset is a deterministic ridge term (the analogue of
JIDT's stochastic `NOISE_LEVEL_TO_ADD`). The k-NN estimators differ at their
finite-sample noise floor, shrinking as O(1/sqrt(N)).

The harness itself is preserved at tag `jidt-parity-final`:
`git checkout jidt-parity-final -- bench/jidt_parity/`.

### Fixed

- **Directed spectral SPIs were transposed.** `SpectralGrangerCausality` (both
  methods), `DirectedCoherence`, `PartialDirectedCoherence`,
  `GeneralizedPartialDirectedCoherence`, `DirectedTransferFunction` and
  `DirectDirectedTransferFunction` reported `A[i, j]` as the influence *of j on
  i*, the opposite of every other directed SPI in the library. The spectral
  backends follow the DTF/PDC literature convention; their output was passed
  through unchanged. pyspi's convention is **row = source, column = target**,
  as set by `base.Directed.multivariate`, and all directed SPIs now follow it.
  **Results computed with 2.x for these six SPIs need transposing.**
  Undirected spectral SPIs are unaffected and bit-identical.
- **`MutualInfo`, `TimeLaggedMutualInfo` and `TransferEntropy` silently
  returned NaN** when given `estimator="kozachenko"`; there is no
  Kozachenko-Leonenko path for these measures. They now raise
  `NotImplementedError` at construction. Use `estimator="kraskov"` instead.
- **Kozachenko SPIs were labelled `linear`.** They are k-nearest-neighbour
  estimators and are now labelled `nonlinear`, so `filter_spis(["linear"])` no
  longer returns them.
- **`Data(procnames=...)` was silently discarded.** The length was validated
  and then never assigned, so custom process names never reached the results
  table.
- **`CalculatorFrame.compute()` accepted no arguments**, raising `TypeError`
  for any keyword. It now forwards to `Calculator.compute`.
- The default config, and every bundled config, was **missing from built
  wheels** — an installed pyspi could not construct a `Calculator`. Package
  data now uses globs.
- `pyspi/lib/ids/LICENSE.txt` was not shipped, despite MIT requiring it.
- `LICENSE.txt` had been corrupted by a global find-and-replace, altering the
  verbatim GPLv3 text ("technological *statistics*"). Restored.

### Changed

- **`Calculator(subset=..., configfile=...)` collapsed into `config=`**, which
  accepts either a bundled name or a path to your own YAML.
- **`normalise=` renamed to `zscore=`** on `Calculator` and `Data`. Behaviour is
  unchanged (per-process z-score along time); the old name collided with
  `utils.normalise`, which was min-max, and with the per-SPI `normalise`
  arguments in `statistics/distance.py`.
- **Configs renamed and moved to `pyspi/configs/`.** The filename stem is now
  the lookup key, so the cost-pruned sets are reachable by name for the first
  time.
- `load_dataset()` exposes only the three demo datasets (`forex`, `cml`,
  `standard_normal`); `available_datasets()` lists them. The regression
  fixtures moved to `tests/` and no longer ship in the wheel.
- Per-SPI timings are printed after `compute()` (total and slowest five).
  `calc.timings` was always populated but never surfaced.
- The CLI writes `.pkl` by default; output format follows the file extension
  (`.pkl`, `.csv`, `.parquet`). Parquet needs `pip install 'pyspi[parquet]'`.
- `JIDTBase` renamed to `InfoTheoryBase`. Config files are unaffected.

### Dependencies

- **Requires Python 3.10+.**
- Dropped five unused runtime dependencies: `h5py`, `seaborn`, `plotly`,
  `matplotlib`, `nbformat`. Plotting and notebook packages moved to a `bench`
  extra.
- Dropped the `setuptools>=68,<80` pin. It existed because pyEDM imported
  `pkg_resources`; pyEDM 2.5 no longer does, so the floor is now `pyEDM>=2.5`
  and modern setuptools is usable.
- `pandas>=2.1` for `DataFrame.stack(future_stack=True)`, the pandas 3
  semantics.
- New extras: `parquet`, `bench`. `testing` is unchanged.

### Testing

- The test suite previously **did not run at all** — collection aborted on an
  undeclared `dill` dependency. Fixed.
- New `tests/test_infotheory_analytic.py`: closed-form checks against
  `-0.5*ln(1-rho^2)`, `0.5*ln(2*pi*e*sigma^2)`, analytic Gaussian TE on a known
  AR(1), independence, and information-theoretic identities. Previously the
  suite contained three assertions comparing a computed value to an
  independently-known one, and five estimator classes were constructed but
  never computed.
- New `tests/test_directionality.py` pins the row=source convention for every
  directed SPI family.
- Frozen baselines regenerated from this fork as `.npz` (they were upstream
  2.0.1 pickles, the wrong oracle for deliberately-changed estimators), and the
  drift suite now fails hard on a NaN-pattern change or a baseline/current SPI
  set mismatch, with tolerances split by estimator family.

### Migrating from 2.x

| 2.x | 3.0 |
|:----|:----|
| `Calculator(subset="all")` | `Calculator(config="full")` |
| `Calculator(subset="fast")` | `Calculator(config="fast")` |
| `Calculator(configfile="my.yaml")` | `Calculator(config="my.yaml")` |
| `Calculator(normalise=False)` | `Calculator(zscore=False)` |
| `Data(..., normalise=False)` | `Data(..., zscore=False)` |
| `pyspi/config.yaml` | `pyspi/configs/full.yaml` |
| `pyspi/fast_config.yaml` | `pyspi/configs/fast.yaml` |
| `pyspi/sonnet_config.yaml` | `pyspi/configs/sonnet.yaml` |
| `pyspi/fabfour_config.yaml` | `pyspi/configs/fabfour.yaml` |
| `pyspi/benchmarked90_amortized_config.yaml` | `pyspi/configs/benchmarked_p90.yaml` |
| `load_dataset("cml7" \| "var1" \| "kuramoto")` | removed (test fixtures) |
| `utils.normalise` | removed (min-max; use `scipy.stats.zscore`) |
| `utils.standardise`, `utils.strshort` | removed (unused) |
| `utils.check_optional_deps` | removed (no optional runtime deps remain) |
| `JIDTBase` | `InfoTheoryBase` |
| `MutualInfo(estimator="kozachenko")` | raises; use `estimator="kraskov"` |

Values for the six directed spectral SPIs listed under **Fixed** are
transposed relative to 2.x. No other SPI values changed.
