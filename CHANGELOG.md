# Changelog

## 3.0.0

A major overhaul. The Java/JIDT dependency is gone, the `Calculator` API is simplified, and several long-standing correctness bugs are fixed. **This release contains breaking changes** — see [Migrating from 2.x](#migrating-from-2x).

### Removed: Java and JIDT

Every information-theoretic estimator is now pure NumPy. `infodynamics.jar`, `jpype` and the JVM startup path are gone, so installing pyspi no longer requires a Java runtime.

The port was validated against JIDT 1.6.1 before the dependency was dropped. That work caught four bugs in the port (kernel-entropy normalisation, Theiler-windowed KSG neighbour counting, Gaussian auto-embed bias, and KSG auto-embed estimator consistency), all fixed. Mean absolute error against JIDT at T=1600:

| estimator | MI | TE | entropy |
|:----------|---:|---:|--------:|
| gaussian | 5.9e-17 | 4.7e-16 | 5.0e-09 |
| kernel | 6.8e-16 | 5.0e-05 | 3.1e-15 |
| symbolic | — | 2.4e-16 | — |
| kozachenko | — | — | 4.2e-05 |
| kraskov | 1.8e-03 | 2.5e-03 | — |

Gaussian/kernel MI, kernel entropy and symbolic TE agree to machine precision. The gaussian-entropy offset is a deterministic ridge term (the analogue of JIDT's stochastic `NOISE_LEVEL_TO_ADD`). The k-NN estimators differ at their finite-sample noise floor, shrinking as $\mathcal{O}(1/\sqrt{N})$.

The harness itself is preserved at tag `jidt-parity-final`: `git checkout jidt-parity-final -- bench/jidt_parity/`.

### Fixed: state, identity, and estimator contracts

A second pass, driven by a suite of red tests, closed a set of defects that produced valid-looking but wrong results. Full details in each commit; the scientifically material ones:

- **Stale caches survived data mutation.** Statistics cache results directly on the `Data` instance keyed by their own parameters, never by the data. Nothing invalidated them, so `set_data`/`add_process`/`remove_process` left every cached statistic serving the *previous* dataset's numbers, silently. All 15 cache attributes are now registered and dropped on mutation, and `Data` copies and freezes its input so nothing can mutate the series behind a cache.

- **Checkpoints were not bound to a run.** Resume validated only the SPI identifier and an `(M, M)` shape, so a different dataset, config, preprocessing setting, or process order silently inherited the earlier run's results. Checkpoints now carry a `run.json` manifest bound to `Calculator.run_digest` (config *contents*, dataset bytes in native dtype, and a computation-version token). Failed or non-finite checkpoints are retried by default, workers load the parent's config snapshot rather than rereading a path that may have changed, and a directory belonging to another run is refused rather than emptied.

- **Spectral caches ignored `fs`,** and were written under a `str` key but read under a `tuple` key — so the first write was unreachable and staleness only appeared from the *third* call, which is why a two-call probe found nothing.

- **Six measures advertised `kraskov` and ran Gaussian.** Joint/conditional/ crossmap/causal entropy, directed info and stochastic interaction are composed from marginal entropies and have no KSG estimator; the argument is now rejected. No bundled config used it, so no shipped result changed.

- **Cointegration `aeg` was forcibly symmetric.** It is not symmetric in its arguments (~0.8 mean absolute difference between orientations, up to ~1.6), yet the cache wrote each value to both `(i,j)` and `(j,i)`, so the reported value depended on process order. `aeg` is now `directed`; `johansen`, which is symmetric to ~3e-14, is unchanged.

- **KSG accepted inputs it cannot estimate from.** `k=30` on `N=20` returned 0.414 and `k=100` returned 1.63; binary/tied series returned TE = -2.36, for a quantity bounded below by zero; negative Theiler windows were accepted. The effective-sample, tie/zero-radius and window checks now guard the MI, TE and auto-embedding paths, and the auto-embedding search skips candidates it cannot support instead of ranking them and failing on the winner.

- **Symbolic TE packed symbols into an integer that overflowed** at `k_history=10` (reaching `(k!)^3`). Now counts distinct rows directly, validated against a tuple-keyed reference. `k_history=1` is rejected: a length-1 ordinal pattern has one symbol, so TE is identically zero. Those were the only two symbolic variants shipped, so **symbolic TE now has no bundled representation** — reintroducing it needs a defensible `k` with benchmark support. `k_history=10` remains constructible for long series, where the undersampling argument does not apply.

- **Results tables no longer use pickle.** Names are stored as `dtype='U'` and loaded with `allow_pickle=False`; files carry a schema version, run spec, digest and errors, and are validated on load. Tables written by pyspi < 3.0.0 will not load — re-save them from a `Calculator`.

- **`DirectedInfo` did not implement directed information.** It summed `H(Y^i)/i` minus causal entropy, so with a source statistically independent of the target it returned 0.007 at target autocorrelation 0 and 1.53 at 0.95 -- it measured target self-predictability. It now implements Massey's `sum_i [H(Y_i|Y^{i-1}) - H(Y_i|Y^{i-1},X^i)]`, validated against the closed form `0.5*ln(1+c^2)`.

  The kernel and kozachenko variants are dropped. Composing DI from four separately-estimated entropies leaves each with its own dimension-dependent bias, and those do not cancel: on independent data kernel sat at 3.8-4.4 for every `T` from 100 to 8000 (a fixed bandwidth in ~11 dimensions does not improve with sample size), and kozachenko returned negatives. In their place `di_kraskov` estimates each `I(X^i; Y_i | Y^{i-1})` term *directly* with the KSG/Frenzel-Pompe conditional-MI estimator, which fixes one neighbour radius in the joint space and reuses it across marginals so the biases cancel by construction. It matches the closed form as closely as the Gaussian variant. `n` now reaches the identifier for `DirectedInfo` and `CausalEntropy`.

- **Wavelet phase-slope index lost its direction.** `mne_connectivity` returns a lower-triangular matrix and pyspi filled the upper triangle *without* negating, so `psi[i,j] == psi[j,i]` — the sign is PSI's entire lead/lag content. The fill must also happen per frequency, *before* the band statistic: only a statistic commuting with negation may be applied first, and `max_f(-v) = -min_f(v)`, not `-max_f(v)`. `mean` is antisymmetric, `max` asymmetric. `fmin=0` also asked for an unbounded period, giving an ~11.1-million-sample Morlet wavelet at `T=100`; `fmin` is now resolved against the data-supported floor *and* the cycle count capped so the wavelet always fits the signal.

- **Kozachenko entropy returned `-inf` on tied data.** A duplicated observation puts a nearest neighbour at distance zero, and `log(0)` sends the estimate to `-inf`. Quantised series do this readily -- the bundled `forex` dataset has a process with 24 distinct values in 250 samples -- so several kozachenko SPIs silently produced infinities there. They now fail with the cause named.

- **Conditional mutual information did not reduce to MI on an empty conditioning set.** `_ksg_cmi` faked the conditioning count as a constant `N-(2w+1)`, which coincides with the MI estimator only at `w=0` and drifted with the Theiler window (0.005 at `w=1`, 0.051 at `w=10`). It now delegates to the MI estimator. Reachable only from `DirectedInfo`'s first term, whose history is empty; transfer entropy always has at least one history column, so it never took this branch. No bundled SPI is affected — `di_kraskov` ships without a Theiler window — but a hand-configured `dyn_corr_excl` would have hit it.

- **`ConditionalEntropy` is `directed`.** An intermediate release note labelled it `undirected` on the grounds that the Gaussian form is symmetric under pyspi's default z-scoring. That was wrong: measured on `var1_M3_T100`, `max|A - Aᵀ|` is 0.115 (kozachenko) and 0.039 (kernel) *under* z-scoring, and the Gaussian form itself becomes asymmetric (0.73) with `zscore=False`. A structural label describes the measure, not one estimator under one preprocessing default.

- **Importing pyspi reseeded NumPy's global RNG.** `pyspi.lib.ids` called `np.random.seed(1717)` at import, silently overriding the caller's seed -- stochastic SPIs looked reproducible but ignored it. Removed.

- **`filter_spis` matched raw YAML family labels**, so per-variant traits set in `__init__` were invisible (`filter_spis(["antisymmetric"])` returned nothing despite 18 matching SPIs) and a matching family selected all of its configs. It now resolves each config and matches on the labels the SPI actually carries.

- **Gaussian joint/conditional entropy used two different regularisations.** The vectorised multivariate path clipped `r^2` while the scalar path applied a ridge, so `bivariate()` and `multivariate()` disagreed by 8.4 nats on singular data. Both now share one primitive.

New: `Calculator.errors`, `Calculator.run_spec`, `Calculator.run_digest`, `Calculator.to_frame()` (long-form results, one row per `(spi, source, target)`) and `Calculator.summary()`; an `antisymmetric` structural label for measures satisfying `A[i,j] == -A[j,i]`; and a useful `repr` — a computed `Calculator` previously displayed as `<pyspi.calculator.Calculator at 0x...>`.


Three group-delay SPIs that shipped as silent all-NaN columns are now recorded failures (values unchanged).

### Fixed

- **Directed spectral SPIs were transposed.** `SpectralGrangerCausality` (both methods), `DirectedCoherence`, `PartialDirectedCoherence`, `GeneralizedPartialDirectedCoherence`, `DirectedTransferFunction` and `DirectDirectedTransferFunction` reported `A[i, j]` as the influence *of j on i*, the opposite of every other directed SPI in the library. The spectral backends follow the DTF/PDC literature convention; their output was passed through unchanged. pyspi's convention is **row = source, column = target**, as set by `base.Directed.multivariate`, and all directed SPIs now follow it. **Results computed with 2.x for these six SPIs need transposing.** Undirected spectral SPIs are unaffected and bit-identical.
- **`MutualInfo`, `TimeLaggedMutualInfo` and `TransferEntropy` silently returned NaN** when given `estimator="kozachenko"`; there is no Kozachenko-Leonenko path for these measures. They now raise `NotImplementedError` at construction. Use `estimator="kraskov"` instead.
- **Kozachenko SPIs were labelled `linear`.** They are k-nearest-neighbour estimators and are now labelled `nonlinear`, so `filter_spis(["linear"])` no longer returns them.
- **`Data(procnames=...)` was silently discarded.** The length was validated and then never assigned, so custom process names never reached the results table.
- **`CalculatorFrame.compute()` accepted no arguments**, raising `TypeError` for any keyword. It now forwards to `Calculator.compute`.
- The default config, and every bundled config, was **missing from built wheels** — an installed pyspi could not construct a `Calculator`. Package data now uses globs.
- `pyspi/lib/ids/LICENSE.txt` was not shipped, despite MIT requiring it.
- `LICENSE.txt` had been corrupted by a global find-and-replace, altering the verbatim GPLv3 text ("technological *statistics*"). Restored.

### Changed

- **`Calculator(subset=..., configfile=...)` collapsed into `config=`**, which accepts either a bundled name or a path to your own YAML.
- **`normalise=` renamed to `zscore=`** on `Calculator` and `Data`. Behaviour is unchanged (per-process z-score along time); the old name collided with `utils.normalise`, which was min-max, and with the per-SPI `normalise` arguments in `statistics/distance.py`.
- **Configs renamed and moved to `pyspi/configs/`.** The filename stem is now the lookup key, so the cost-pruned sets are reachable by name for the first time.
- `load_dataset()` exposes only the three demo datasets (`forex`, `cml`, `standard_normal`); `available_datasets()` lists them. The regression fixtures moved to `tests/` and no longer ship in the wheel.
- Per-SPI timings are printed after `compute()` (total and slowest five). `calc.timings` was always populated but never surfaced.
- **New `Calculator.save()` and `pyspi.load_table()`.** Results had no documented persistence path from the Python API at all -- only the CLI wrote files. `.npz` is now the canonical format: it stores the results in their natural `(n_spis, M, M)` shape plus names, round-trips exactly, and needs nothing beyond numpy. `.csv` remains as a one-way human-readable export.
- **Dropped pickle and parquet output.** Pickle is version-fragile and executes arbitrary code on load, which is wrong for an archival scientific artifact. Parquet is columnar and built for heterogeneous tabular data; for a dense float tensor it bought nothing over `.npz` while costing a ~40 MB pyarrow dependency. The `parquet` extra is gone.
- A config that keeps only part of a shared-cache group now warns, since the cache is built regardless and the remaining members are close to free. Only applies to user-written configs and to caches expensive enough to matter.
- `JIDTBase` renamed to `InfoTheoryBase`. Config files are unaffected.

### Dependencies

- **Requires Python 3.10+.**
- Dropped five unused runtime dependencies: `h5py`, `seaborn`, `plotly`, `matplotlib`, `nbformat`. Plotting and notebook packages moved to a `bench` extra.
- Dropped the `setuptools>=68,<80` pin. It existed because pyEDM imported `pkg_resources`; pyEDM 2.5 no longer does, so the floor is now `pyEDM>=2.5` and modern setuptools is usable.
- `pandas>=2.1` for `DataFrame.stack(future_stack=True)`, the pandas 3 semantics.
- New extra: `bench`. `testing` is unchanged.

### Testing

- The test suite previously **did not run at all** — collection aborted on an undeclared `dill` dependency. Fixed.
- New `tests/test_infotheory_analytic.py`: closed-form checks against `-0.5*ln(1-rho^2)`, `0.5*ln(2*pi*e*sigma^2)`, analytic Gaussian TE on a known AR(1), independence, and information-theoretic identities. Previously the suite contained three assertions comparing a computed value to an independently-known one, and five estimator classes were constructed but never computed.
- New `tests/test_directionality.py` pins the row=source convention for every directed SPI family.
- Frozen baselines regenerated from this fork as `.npz` (they were upstream 2.0.1 pickles, the wrong oracle for deliberately-changed estimators), and the drift suite now fails hard on a NaN-pattern change or a baseline/current SPI set mismatch, with tolerances split by estimator family.

### References

Definitions the corrected measures are checked against:

- Massey, J. (1990). Causality, feedback and directed information. *Proc. ISITA*. — the `sum_i I(X^i; Y_i | Y^{i-1})` form now implemented by `DirectedInfo`.
- Frenzel, S. & Pompe, B. (2007). Partial mutual information for coupling analysis of multivariate time series. *Phys. Rev. Lett.* 99, 204101. — the conditional-MI estimator behind `di_kraskov` and kraskov transfer entropy.
- Kraskov, A., Stögbauer, H. & Grassberger, P. (2004). Estimating mutual information. *Phys. Rev. E* 69, 066138. — KSG estimator and its effective-sample conditions.
- Kozachenko, L. & Leonenko, N. (1987). Sample estimate of the entropy of a random vector. *Probl. Inf. Transm.* 23, 95–101. — the k-NN entropy that is undefined on tied data.
- Baccalá, L., Sameshima, K., Ballester, G., Do Valle, A. & Timo-Iaria, C. (1998). Studying the interaction between brain structures via directed coherence and Granger causality. *Appl. Sig. Process.* 5, 40–48. — `DC_ij = sqrt(σ_jj)|H_ij| / sqrt(Σ_k σ_kk|H_ik|²)`, the bounded form now computed.
- Kamiński, M. & Blinowska, K. (1991). A new method of the description of the information flow in the brain structures. *Biol. Cybern.* 65, 203–210. — DTF, and the `[target, source]` convention that pyspi transposes to `row = source`.
- Lizier, J. (2014). JIDT: an information-theoretic toolkit. *Front. Robot. AI* 1, 11. — the reference implementation the NumPy port was validated against.

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

### SPI set changes

`full` goes from **328 SPIs to 325**. Every change below is deliberate; nothing else moved on the frozen test fixtures.

Disabled variants are **commented out in the shipped configs rather than deleted**, each with the evidence for switching it off and the condition that would justify switching it back on.

**Flagged, not removed.** `dspli_multitaper_max_*` and `dswpli_multitaper_max_*` saturate: the band maximum reaches exactly 1 as soon as the sign of the imaginary coherency is consistent across tapers at any *one* frequency. On the frozen fixtures the share of pairs at exactly 1 runs **29-100%** depending on the data, with 1 to 16 distinct values; the `mean` variants are graded normally. This is empirical, not a law - saturation is **not** monotone in `T`, since changing `T` recomputes the tapers and Fourier coefficients rather than adding to them (measured non-monotone in 7 of 36 seed/band combinations). They stay **enabled**; the statistic is behaving as defined.

**Directed coherence corrected.** `spectral_connectivity.directed_coherence` puts `|H|²` in the numerator while its denominator stays on the magnitude scale, making the ratio unbounded: baselines reached 3.27 (VAR), 1.84 (CML) and **1139.47** (Kuramoto). pyspi now recomputes it from the same transfer function with `|H|`, per Baccalá et al. (1998). Verified two ways — bounded in [0,1] on all three fixtures, and under an identity noise covariance it reproduces `sqrt(directed_transfer_function())` to 4e-16, the identity DC must satisfy when noise variances are equal.

**Removed (4)**

| SPI | Why |
|:----|:----|
| `te_symbolic_k-1_kt-1_l-1_lt-1` | A length-1 ordinal pattern has one symbol, so TE is identically zero. |
| `te_symbolic_k-10_kt-1_l-1_lt-1` | Severely undersampled and unvalidated at `T=100`: `10!` symbols against ~91 usable samples. On var1 and cml every joint count is 1, so the value tracks sample size rather than dependence; that does not hold universally (kuramoto: 3 of 42 pairs). Still constructible, and defensible for long series. |
| `di_kernel_W-0.5` | ~3.8-4.4 on independent data at every `T` from 100 to 8000. |
| `di_kozachenko` | Negative values, for a nonnegative quantity. |

**Added (1)**

| SPI | Why |
|:----|:----|
| `di_kraskov_NN-4_n-5` | Direct KSG/Frenzel-Pompe conditional-MI estimate of directed information; the validated nonlinear replacement for the two dropped variants. Not in the `benchmarked_p*` sets until it has been timed. |

**Renamed (4)** — `n` changes the measure, so it now reaches the identifier. Values unchanged.

`cce_gaussian` → `cce_gaussian_n-5`, and likewise `cce_kernel_W-0.5`, `cce_kozachenko`, `di_gaussian`.

**Values changed (17)** — all from the corrections listed above.

| SPIs | Cause |
|:-----|:------|
| `coint_aeg_*` (3) | No longer forced symmetric; each orientation is reported as computed. |
| `psi_wavelet_*` (6) | Sign restored, negation moved before the band statistic, wavelet length bounded. |
| `di_gaussian_n-5` | Massey's definition instead of the old entropy-rate sum. |
| `je_gaussian`, `ce_gaussian` | One shared regularisation across both code paths (~1e-8). |
| `bary_sgddtw_*`, `bary-sq_sgddtw_*` (4) | Stochastic; they now honour the caller's seed instead of the import-time `seed(1717)`. |
| `te_kraskov_NN-4_DCE_k-max-10_tau-max-4` | Auto-embedding now skips embeddings the estimator cannot support. |

Values for the six directed spectral SPIs listed under **Fixed** are also transposed relative to 2.x.
