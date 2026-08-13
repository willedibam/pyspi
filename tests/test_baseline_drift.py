"""Baseline drift detector for the full SPI set.

For each (dataset, SPI) pair this recomputes the SPI on a frozen dataset and
compares it element-wise against a stored baseline matrix.

What is ENFORCED (hard assertion failure):
  * the baseline SPI set and the current Calculator's SPI set are identical,
    so a newly-broken or renamed SPI cannot disappear by having no baseline;
  * the shape of each matrix;
  * the NaN *pattern*. A SPI going from finite to all-NaN (or back) is a
    categorical regression, not drift — this fork changed NaN-on-failure
    semantics, so that is precisely the signal worth failing on.

What is REPORTED but NOT enforced:
  * numerical drift in the finite entries. Exceedances are routed to a
    session-end summary table via ``spi_warning_logger`` (see conftest.py) so
    library/BLAS version bumps are visible without blocking CI.

Baselines live in ``tests/data/baselines/<dataset>.npz`` (one MxM array per
SPI identifier) and are regenerated from the *current* fork by
``tests/tools/generate_benchmark_tables.py``. They are a forward-looking
change detector, not an independent oracle: the fork deliberately rewrote the
information-theoretic estimators, so upstream pyspi 2.0.1 values are the wrong
reference for exactly the code that most needs one. Independent correctness
lives in ``test_infotheory_analytic.py``.

Frozen fixtures live in ``tests/data/fixtures/`` (not in ``pyspi/data/``: they
are test inputs, not shipped demo data) and are built by
``tests/tools/generate_fixtures.py``. Three generating processes at three widths
— VAR(1) at M=3, coupled map lattice at M=5, Kuramoto at M=7, all T=100 — so the
SPI set is exercised across a range of M.
"""
import os

import numpy as np
import pytest

from pyspi.calculator import Calculator
from pyspi.data import Data

# Whole-file marker: this suite takes ~3.5 minutes. Skipped by default; run with
#   pytest -m slow tests/test_baseline_drift.py
# or
#   pytest -m '' tests/
pytestmark = pytest.mark.slow

DATASETS = ("var1_M3_T100", "cml_M5_T100", "kuramoto_M7_T100")

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
BASELINE_DIR = os.path.join(_DATA_DIR, "baselines")
FIXTURE_DIR = os.path.join(_DATA_DIR, "fixtures")

# Seed used when the baselines were generated; must match the generator's
# default so the RNG-consuming SPIs land in the same place.
SEED = 42

# Drift thresholds, per family. A value is OK if EITHER the absolute or the
# relative test passes (the absolute one protects near-zero references).
#
# TIGHT is the default: the benchmark datasets are frozen and almost every SPI
# is a deterministic function of them, so re-running the same code on the same
# machine reproduces the baseline to float round-off. 1e-9 sits well above that
# (~1e-12) while still catching genuine sub-percent regressions that the old
# blanket RTOL=1e-2 would have hidden.
TIGHT = (1e-12, 1e-9)   # (atol, rtol)
# LOOSE applies to families whose values are not a pure function of the data:
#   causal -- cdt estimators run randomly-initialised optimisers (and torch),
#             so they are not bit-reproducible across runs or thread counts;
#   misc   -- GP fitting with random restarts and permutation/randomised
#             independence tests (hyppo, IDS).
# Deriving this from the SPI's module keeps the split declarative rather than a
# hand-maintained list of SPI identifiers.
LOOSE = (1e-6, 1e-2)
LOOSE_MODULES = {"causal", "misc"}


def _baseline_path(dataset_name):
    return os.path.join(BASELINE_DIR, f"{dataset_name}.npz")


def _load_fixture(dataset_name):
    """Load a frozen fixture; stored (observations, processes) -> 'sp'.

    Must stay in step with ``tests/tools/generate_benchmark_tables.load_fixture``,
    or the baselines and the test would be reading different data.
    """
    return Data(data=os.path.join(FIXTURE_DIR, f"{dataset_name}.npy"),
                dim_order="sp", name=dataset_name)


def _baseline_keys(dataset_name):
    """SPI identifiers stored in a baseline archive, or None if unreadable.

    Reads only the zip central directory, so this is cheap enough to run at
    collection time. Returning None instead of raising is deliberate: a missing
    or corrupt baseline must never break collection of unrelated tests.
    """
    try:
        with np.load(_baseline_path(dataset_name)) as archive:
            return sorted(k for k in archive.files if not k.startswith("__"))
    except Exception:
        return None


def pytest_generate_tests(metafunc):
    """Parametrise over (dataset, SPI) using only the baseline archives.

    Nothing here constructs a Calculator or decompresses a matrix; the actual
    tables come from session-scoped fixtures, so importing this module costs
    nothing.
    """
    if "spi_key" not in metafunc.fixturenames:
        return
    params = []
    for dataset_name in DATASETS:
        keys = _baseline_keys(dataset_name)
        if keys is None:
            params.append(pytest.param(
                dataset_name, None,
                marks=pytest.mark.skip(
                    reason=f"missing/unreadable baseline {_baseline_path(dataset_name)}"
                ),
                id=f"{dataset_name}:<no-baseline>",
            ))
            continue
        params.extend(pytest.param(dataset_name, k, id=f"{dataset_name}:{k}")
                      for k in keys)
    metafunc.parametrize("dataset_name, spi_key", params)


@pytest.fixture(scope="session")
def baseline_tables():
    """dataset -> {spi_key: matrix}, loaded once per session on first use."""
    cache = {}

    def get(dataset_name):
        if dataset_name not in cache:
            with np.load(_baseline_path(dataset_name)) as archive:
                cache[dataset_name] = {
                    k: archive[k] for k in archive.files if not k.startswith("__")
                }
        return cache[dataset_name]

    return get


@pytest.fixture(scope="session")
def current_tables():
    """dataset -> (tables, spi_objects); one full Calculator run per dataset."""
    cache = {}

    def get(dataset_name):
        if dataset_name not in cache:
            np.random.seed(SEED)
            calc = Calculator(dataset=_load_fixture(dataset_name))
            calc.compute()
            cache[dataset_name] = (
                {spi: calc.table[spi].to_numpy() for spi in calc.spis},
                dict(calc.spis),
            )
        return cache[dataset_name]

    return get


@pytest.mark.parametrize("dataset_name", DATASETS)
def test_baseline_covers_every_spi(dataset_name, baseline_tables, current_tables):
    """The baseline and the current Calculator must expose the same SPI set.

    Without this, an SPI that is renamed or newly added is silently untested,
    which is how ~45-50 SPIs per dataset escaped the old suite.
    """
    baseline = baseline_tables(dataset_name)
    _, spis = current_tables(dataset_name)
    missing_from_baseline = sorted(set(spis) - set(baseline))
    missing_from_current = sorted(set(baseline) - set(spis))
    assert not missing_from_baseline and not missing_from_current, (
        f"[{dataset_name}] SPI set mismatch. "
        f"No baseline for: {missing_from_baseline}. "
        f"Baseline-only: {missing_from_current}. "
        f"Regenerate with tests/tools/generate_benchmark_tables.py."
    )


def test_baseline_drift(dataset_name, spi_key, baseline_tables, current_tables,
                        spi_warning_logger):
    """Hard-fail on shape or NaN-pattern change; report numerical drift."""
    ref = baseline_tables(dataset_name)[spi_key]
    tables, spis = current_tables(dataset_name)
    assert spi_key in tables, (
        f"[{dataset_name}] {spi_key}: present in baseline but not in the current "
        f"Calculator (see test_baseline_covers_every_spi)."
    )
    new = tables[spi_key]

    assert ref.shape == new.shape, (
        f"[{dataset_name}] {spi_key}: shape mismatch "
        f"baseline={ref.shape} new={new.shape}"
    )

    # --- Enforced: NaN pattern -------------------------------------------
    # Compared as masks, never coerced to 0. Folding NaN into 0 on both sides
    # (the old behaviour) turns "this SPI now fails everywhere" into a small
    # numeric drift entry, hiding the one regression class that matters most.
    ref_nan = ~np.isfinite(ref)
    new_nan = ~np.isfinite(new)
    if not np.array_equal(ref_nan, new_nan):
        gained = int(np.sum(new_nan & ~ref_nan))
        lost = int(np.sum(ref_nan & ~new_nan))
        pytest.fail(
            f"[{dataset_name}] {spi_key}: non-finite pattern changed "
            f"({gained} entries became NaN/inf, {lost} became finite). "
            f"baseline non-finite={int(ref_nan.sum())}/{ref.size}, "
            f"current non-finite={int(new_nan.sum())}/{new.size}."
        )

    # --- Reported: numerical drift on the finite entries ------------------
    finite = ~ref_nan
    if not finite.any():
        return

    module_name = spis[spi_key].__module__.split(".")[-1]
    atol, rtol = LOOSE if module_name in LOOSE_MODULES else TIGHT

    abs_diff = np.zeros_like(ref, dtype=np.float64)
    abs_diff[finite] = np.abs(new[finite] - ref[finite])
    ok = ~finite | (abs_diff <= atol) | (abs_diff <= rtol * np.abs(np.nan_to_num(ref)))
    if np.all(ok):
        return

    max_abs = float(abs_diff[~ok].max())
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(np.abs(ref) > 0, abs_diff / np.abs(ref), np.nan)
    bad_rel = rel[~ok]
    bad_rel = bad_rel[np.isfinite(bad_rel)]
    max_rel = float(bad_rel.max()) if bad_rel.size else float("nan")

    num_interactions = new.size - new.shape[0]
    num_exceed = int(np.count_nonzero(~ok))
    if "undirected" in spis[spi_key].labels:
        num_exceed //= 2
        num_interactions //= 2

    spi_warning_logger(
        f"{dataset_name}:{spi_key}",
        module_name,
        max_abs,
        max_rel,
        num_exceed,
        num_interactions,
    )
