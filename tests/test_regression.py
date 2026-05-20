"""SPI correctness / drift regression test.

For each (dataset, SPI) pair that appears in both the frozen-upstream
baseline pickle and the current Calculator, compare element-wise against
the baseline mean. A pair is "close enough" if at least one of:

  abs_diff(new, ref) <= ATOL                 (protects near-zero refs)
  abs_diff(new, ref) <= RTOL * |ref|         (protects large values)

Exceedances do not fail the test — they are logged to a session-end
summary table via the ``spi_warning_logger`` fixture so drift is visible
without blocking CI on version bumps.

The three frozen datasets are CML7 (coupled map lattice, 7 proc),
VAR1 (linear autoregressive, 7 proc), and Kuramoto (phase oscillators,
7 proc). Baselines were built with upstream pyspi 2.0.1 using 10 trials
and numpy seed 42 per trial (see tests/generate_benchmark_datasets.py
and the build_baselines.py helper in the ephemeral uv project).
"""
import dill
import numpy as np
import pytest

from pyspi.calculator import Calculator

# Whole-file marker: this suite takes ~12 minutes. Skipped by default; run with
#   pytest -m slow tests/test_regression.py
# or
#   pytest -m '' tests/
pytestmark = pytest.mark.slow


DATASETS = {
    "CML7": "pyspi/data/cml7.npy",
    "VAR1": "pyspi/data/var1_7.npy",
    "Kuramoto": "pyspi/data/kuramoto_7.npy",
}
BASELINES = {
    "CML7": "tests/CML7_benchmark_tables.pkl",
    "VAR1": "tests/VAR1_benchmark_tables.pkl",
    "Kuramoto": "tests/Kuramoto_benchmark_tables.pkl",
}

# Drift thresholds. ATOL catches tiny absolute changes near zero; RTOL
# catches proportional drift on larger values. A value is OK if EITHER
# test passes. Tuned to surface ~1% or larger drift while ignoring the
# BLAS / RNG / library-version noise floor.
ATOL = 1e-6
RTOL = 1e-2


def _load_dataset(name):
    return np.load(DATASETS[name]).T


def _load_baseline(name):
    with open(BASELINES[name], "rb") as f:
        return dill.load(f)


_tables_cache = {}


def _compute_current(name):
    """Compute the fork's current SPI tables on the named dataset. Cached."""
    if name in _tables_cache:
        return _tables_cache[name]
    np.random.seed(42)
    calc = Calculator(dataset=_load_dataset(name))
    calc.compute()
    out = {spi: calc.table[spi].to_numpy() for spi in calc.spis}
    _tables_cache[name] = out
    return out


def _build_params():
    """Cross-product of (dataset, SPI) restricted to SPIs in both sides.

    SPIs added or renamed in the fork (no baseline entry) and baseline
    entries removed from the fork (no current SPI) are skipped with a
    summary line per dataset.
    """
    calc = Calculator()
    current_spis = dict(calc.spis)
    current_keys = set(current_spis)

    params = []
    for ds in DATASETS:
        baseline = _load_baseline(ds)
        baseline_keys = set(baseline)
        only_current = sorted(current_keys - baseline_keys)
        only_baseline = sorted(baseline_keys - current_keys)
        shared = sorted(current_keys & baseline_keys)
        if only_current:
            preview = ", ".join(only_current[:3])
            more = f" (+{len(only_current) - 3} more)" if len(only_current) > 3 else ""
            print(f"[{ds}] skipped {len(only_current)} new/renamed SPIs: {preview}{more}")
        if only_baseline:
            preview = ", ".join(only_baseline[:3])
            more = f" (+{len(only_baseline) - 3} more)" if len(only_baseline) > 3 else ""
            print(f"[{ds}] skipped {len(only_baseline)} baseline-only SPIs: {preview}{more}")
        for spi_key in shared:
            params.append((ds, spi_key, current_spis[spi_key], baseline[spi_key]))
    return params


params = _build_params()


def pytest_generate_tests(metafunc):
    if "spi_key" in metafunc.fixturenames:
        metafunc.parametrize(
            "dataset_name, spi_key, spi_ob, baseline_entry",
            params,
            ids=[f"{p[0]}:{p[1]}" for p in params],
        )


def test_spi_tolerance(dataset_name, spi_key, spi_ob, baseline_entry, spi_warning_logger):
    """Compare current SPI table to frozen baseline mean; log drift."""
    ref_mean = baseline_entry["mean"]
    mpi_new = _compute_current(dataset_name)[spi_key]

    assert ref_mean.shape == mpi_new.shape, (
        f"[{dataset_name}] {spi_key}: shape mismatch "
        f"baseline={ref_mean.shape} new={mpi_new.shape}"
    )

    # Diagonal and any failure-NaNs are treated as 0 on both sides.
    ref = np.nan_to_num(ref_mean, copy=True)
    new = np.nan_to_num(mpi_new, copy=True)

    abs_diff = np.abs(new - ref)
    within_abs = abs_diff <= ATOL
    within_rel = abs_diff <= RTOL * np.abs(ref)
    ok = within_abs | within_rel

    if np.all(ok):
        return

    bad_idx = np.argwhere(~ok)
    max_abs = float(abs_diff[~ok].max())
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(np.abs(ref) > 0, abs_diff / np.abs(ref), np.nan)
    bad_rel = rel[~ok]
    bad_rel = bad_rel[np.isfinite(bad_rel)]
    max_rel = float(bad_rel.max()) if bad_rel.size else float("nan")

    num_interactions = new.size - new.shape[0]
    num_exceed = bad_idx.shape[0]
    if "undirected" in spi_ob.labels:
        num_exceed //= 2
        num_interactions //= 2

    module_name = spi_ob.__module__.split(".")[-1]
    spi_warning_logger(
        f"{dataset_name}:{spi_key}",
        module_name,
        max_abs,
        max_rel,
        int(num_exceed),
        int(num_interactions),
    )
