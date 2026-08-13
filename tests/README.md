# pyspi test suite

## Running

```bash
pytest                      # fast suite (default: -m 'not slow'), ~2 min
pytest -m slow              # baseline drift suite only, ~3.5 min
pytest -m ''                # everything
```

`addopts = "-m 'not slow'"` is set in `pyproject.toml`, so the slow marker is
opt-in. Nothing outside `tests/tools/` performs work at import time: a missing or
corrupt baseline can never break collection of unrelated tests.

## What each file covers

| File | Covers |
| --- | --- |
| `test_infotheory_analytic.py` | **Closed-form correctness** of the pure-NumPy info-theory estimators: bivariate-Gaussian MI vs `-0.5*ln(1-rho^2)` for all four estimators, differential entropy of `N(0, sigma^2)`, Gaussian TE on an analytic AR(1), independence collapsing to ~0, and the entropy identities. The only place a `return np.zeros(...)` stub would be caught. |
| `test_calculator.py` | `Calculator` / `Data` API: `config=` resolution, `zscore=`, labels, grouping, dataset loading. |
| `test_utils.py` | Utility helpers. |
| `test_smoke.py` | Cheap shape / finiteness / sign checks across SPI families. Sanity, not correctness. |
| `test_parallel.py` | `Calculator.compute()` parallel path, checkpointing, per-SPI failure isolation. |
| `test_phi_native.py` | Native (non-JIDT) integrated-information implementation. |
| `test_baseline_drift.py` | `slow`. Every SPI on three frozen fixtures (M=3/5/7) vs a stored baseline. |

Log base matters when reading these: `gaussian`, `kraskov` and `kozachenko`
report **nats**; `kernel` and `symbolic` report **bits** (inherited from JIDT).

## Baselines

`tests/data/baselines/{var1_M3_T100,cml_M5_T100,kuramoto_M7_T100}.npz` — one
`MxM` matrix per SPI identifier, plus `__dataset__` / `__config__` / `__seed__`
provenance entries. Regenerate with:

```bash
python tests/tools/generate_benchmark_tables.py                 # all three
python tests/tools/generate_benchmark_tables.py -d cml_M5_T100
```

The frozen `.npy` fixtures themselves live in `tests/data/fixtures/` and come
from `tests/tools/generate_fixtures.py`. They are **test inputs, not shipped
data**: they are deliberately outside `pyspi/data/`, so they are not in the
wheel and not reachable via `pyspi.data.load_dataset` (which now exposes only
the three demo datasets `forex`, `cml`, `standard_normal`). Three generating
processes at three widths — VAR(1) at `M=3`, coupled map lattice at `M=5`,
Kuramoto at `M=7`, all `T=100` — so the SPI set is exercised across a range of
`M` rather than at a single width.

These baselines are generated from **this fork's current code**, not from
upstream pyspi 2.0.1. The fork deliberately rewrote every information-theoretic
estimator, so upstream values are the wrong oracle for exactly the code that
most needs one. The baselines are therefore a *forward-looking change detector*:
they tell you that something moved, not that it was right before. Independent
correctness lives in `test_infotheory_analytic.py`.

A single seeded pass is stored per dataset — not a mean over trials. The
datasets are frozen fixtures and nearly every SPI is a deterministic function of
them, so an exact oracle is more useful than an average that no individual run
reproduces.

## Drift is reported, not enforced

`test_baseline_drift.py` **fails hard** on exactly three things:

1. the baseline SPI set differing from the current `Calculator`'s SPI set, so a
   newly-broken or renamed SPI cannot escape by having no baseline;
2. a matrix shape change;
3. a change in the **NaN pattern** — a SPI going from finite to all-NaN (or
   back) is a categorical regression, not drift.

Everything else — numerical differences in the finite entries — is *reported*
to a session-end summary table (see `conftest.py`) and does **not** fail the
run, so library and BLAS version bumps stay visible without blocking CI.
Tolerances are split by the SPI's module: `1e-9` for the deterministic families
and `1e-2` for `causal` and `misc`, whose estimators use randomly-initialised
optimisers and permutation tests and are not bit-reproducible.

## Known gap

`MutualInfo` and `TimeLaggedMutualInfo` accept `estimator="kozachenko"` but have
no code path for it: `bivariate()` logs a warning and returns `NaN`. No shipped
config uses that combination, so no SPI is affected, but the constructor should
raise `NotImplementedError` the way the `symbolic` guard does. The analytic
suite pins this with `strict=True` xfails so they flip to failures the moment
the gap is closed.
