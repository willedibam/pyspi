The fork is directionally strong, but I would not release 3.0 yet. Several issues can silently produce valid-looking but wrong scientific results.

No files were edited. The existing `.gitignore` modification was untouched. The default non-slow suite reports `159 passed, 987 deselected, 5 xfailed`; I did not run the slow M5/M7 regression suite or benchmarks.

## Release blockers

1. **Checkpoints are not tied to an experiment.** Resume validates only SPI identifier and `(M, M)` shape ([calculator.py](/Users/wedi0306/Code/pyspi-fork/pyspi/calculator.py:520), [_parallel.py](/Users/wedi0306/Code/pyspi-fork/pyspi/_parallel.py:315)). Reusing a directory for another same-width dataset, preprocessing policy, config, or code version silently returns stale results. Failed `.error` checkpoints are also considered complete.

2. **Mutable `Data` objects retain stale caches.** Statistics cache covariance, spectra, entropy, CCM, barycentres, etc. directly on `Data` ([basic.py](/Users/wedi0306/Code/pyspi-fork/pyspi/statistics/basic.py:38)), but `set_data`, `add_process`, and `remove_process` do not invalidate them ([data.py](/Users/wedi0306/Code/pyspi-fork/pyspi/data.py:180)). Confirmed: after replacing a dataset, cached covariance remained `3.75`; a fresh computation was `-3.75`. `to_numpy()` also exposes mutable underlying storage.

3. **The direct-array SPI API is broken.** `_data=None` is now initialized eagerly, while `add_process()` checks only `hasattr` ([data.py](/Users/wedi0306/Code/pyspi-fork/pyspi/data.py:78), [data.py](/Users/wedi0306/Code/pyspi-fork/pyspi/data.py:242)). Consequently `Data().add_process(x)`, `spi.bivariate(x, y)`, and raw-array `multivariate()` all fail with a reshape-to-zero error. Existing tests instantiate `Data()` but never exercise the documented builder path.

4. **Estimator preconditions are insufficient.** KSG/KL paths do not reliably reject `k >= effective N`, oversized Theiler windows, invalid histories, ties, or degenerate samples ([infotheory.py](/Users/wedi0306/Code/pyspi-fork/pyspi/statistics/infotheory.py:520), [infotheory.py](/Users/wedi0306/Code/pyspi-fork/pyspi/statistics/infotheory.py:759)). A 20-sample probe with `k=30` returned finite MI instead of failing.

5. **Bundled symbolic TE contains degenerate variants.** `k_history=1` has one ordinal symbol and is identically zero; `k_history=10` has a `10!` state space and is severely undersampled for normal pyspi inputs ([implementation](/Users/wedi0306/Code/pyspi-fork/pyspi/statistics/infotheory.py:430), [config](/Users/wedi0306/Code/pyspi-fork/pyspi/configs/full.yaml:727)). Neither should ship in the default set without explicit experimental status.

6. **Canonical NPZ files still use pickle.** Names are stored as object arrays and loaded with `allow_pickle=True` ([save](/Users/wedi0306/Code/pyspi-fork/pyspi/calculator.py:461), [load](/Users/wedi0306/Code/pyspi-fork/pyspi/calculator.py:168)). This contradicts the security rationale for dropping pickle. Files also lack schema version, configuration, units, preprocessing, errors, seed, and input/code fingerprints.

7. **Scientific failures are too easy to miss.** Both execution paths catch every exception, replace the matrix with NaNs, and continue ([serial](/Users/wedi0306/Code/pyspi-fork/pyspi/calculator.py:567), [parallel](/Users/wedi0306/Code/pyspi-fork/pyspi/_parallel.py:241)). Failures are not retained in a structured public report or saved artifact. The expensive “full calculator computes” test has no assertion ([test_calculator.py](/Users/wedi0306/Code/pyspi-fork/tests/test_calculator.py:14)).

## High-value correctness work

- Gaussian joint and conditional entropy use different regularization in `bivariate()` and optimized `multivariate()` paths ([infotheory.py](/Users/wedi0306/Code/pyspi-fork/pyspi/statistics/infotheory.py:1055)). On duplicated inputs I observed joint entropy `0.71` versus `-7.70`. Both APIs must share one numerical primitive.

- Native phi needs stronger PSD, optimizer-success, convergence, and normalization checks ([phi_native.py](/Users/wedi0306/Code/pyspi-fork/pyspi/lib/phi_native.py:21)). Current tests mainly prove finiteness, not agreement with PhiToolbox or known limiting cases.

- Information-theory outputs mix bits and nats, documented only in [tests/README.md](/Users/wedi0306/Code/pyspi-fork/tests/README.md:27). Default z-scoring also changes differential-entropy SPIs into entropies of standardized variables. Both belong in result metadata and public documentation.

- `ConditionalEntropy` is implemented as directed but bundled configs label it undirected ([implementation](/Users/wedi0306/Code/pyspi-fork/pyspi/statistics/infotheory.py:1077), [config](/Users/wedi0306/Code/pyspi-fork/pyspi/configs/full.yaml:530)).

- Importing IDS resets NumPy’s global RNG ([numpy_dependence.py](/Users/wedi0306/Code/pyspi-fork/pyspi/lib/ids/numpy_dependence.py:11)), contaminating caller reproducibility.

- `Data` accepts duplicate/unknown `dim_order` symbols and infinities when z-scoring is disabled ([data.py](/Users/wedi0306/Code/pyspi-fork/pyspi/data.py:191)). Probes accepted `"xx"`, `"pp"`, and `inf`, producing invalid 4-D/5-D internal states.

## Architecture and release readiness

- Duplicate SPI detection cannot work because dictionary insertion has already overwritten collisions before counting ([calculator.py](/Users/wedi0306/Code/pyspi-fork/pyspi/calculator.py:202)).

- `filter_spis()` filters raw family YAML labels rather than the final merged per-variant labels ([utils.py](/Users/wedi0306/Code/pyspi-fork/pyspi/utils.py:92)). It can omit valid matches or retain incorrectly labelled variants.

- Configs dynamically import and invoke arbitrary module attributes. That is acceptable only for explicitly trusted configs; currently the trust boundary is undocumented. A safe YAML loader alone would not fix this.

- Numerical baseline drift is logged but never fails CI ([test_baseline_drift.py](/Users/wedi0306/Code/pyspi-fork/tests/test_baseline_drift.py:217)); directionality tests skip when implementations return NaN ([test_directionality.py](/Users/wedi0306/Code/pyspi-fork/tests/test_directionality.py:71)).

- CI tests editable source rather than the wheel, despite package-data omissions having already caused a release defect. It also ignores `uv.lock`, conflating reproducible testing with testing latest compatible dependencies.

- All specialist dependencies are mandatory, including `torch` and the old `cdt` stack ([pyproject.toml](/Users/wedi0306/Code/pyspi-fork/pyproject.toml:31)). The local Torch installation alone is roughly 370 MB. This weakens the ease-of-install argument gained by removing Java.

- The external documentation is not v3-ready: the current [Calculator API](https://time-series-features.gitbook.io/pyspi/information-about-pyspi/api-reference/pyspi.calculator.calculator) still documents `subset`, `configfile`, and `normalise`; the [installation page](https://time-series-features.gitbook.io/pyspi/installing-and-using-pyspi/installation) recommends Python 3.9.

## Recommended implementation sequence

1. **State and artifact integrity:** make `Data` immutable or revisioned; invalidate caches; repair raw-array APIs; introduce a versioned result schema and content-addressed checkpoint manifest.

2. **Estimator contracts:** centralize effective-sample validation, tie policy, units, embedding constraints, and numerical regularization. Remove or quarantine degenerate symbolic TE variants.

3. **Failure semantics:** add `errors="raise" | "collect" | "warn"`, `calc.errors`, output-shape/finite validation shared by serial and parallel paths, and nonzero CLI status for unapproved failures.

4. **Configuration:** introduce a validated SPI registry/specification model used by loading, filtering, benchmarking, persistence, and cache grouping. Reject duplicate identifiers during insertion.

5. **Scientific validation:** hard-fail deterministic drift; remove NaN skips and stale xfails; add independent/metamorphic tests for short/tied inputs, near-singular entropy, symbolic TE, and PhiToolbox parity.

6. **Release pipeline:** locked primary CI plus latest-compatible CI; build/install the wheel in a clean environment; test package data and CLI; update docs before tagging 3.0.

7. **Performance and installation:** cache Kraskov target auto-embedding, schedule cache buckets by measured cost and subkey, then split the package into a small core plus family extras and a `full` meta-extra.

For each area, the bad approach is additional warnings and local special cases; the good approach is targeted validation and manifests; the optimal approach is one immutable, versioned run specification connecting data, preprocessing, SPI parameters, units, errors, checkpoints, and saved results.

I would preserve the pure-Python direction, analytic tests, explicit source→target convention, shared-memory execution, package-data globs, and benchmark-derived configs. I would not restore Java, discard pandas/baselines, impose arbitrary dependency ceilings, or rewrite `calculator.py` wholesale before the correctness contracts are fixed.