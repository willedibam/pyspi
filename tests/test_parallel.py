"""Tests for Calculator.compute() parallel path, checkpoint, and failure isolation.

Uses a small handcrafted config (parallel_test_config.yaml) covering:
  - covariance cache namespace (Covariance + Precision, multiple estimators)
  - spectral_mv cache namespace (CoherenceMagnitude, multiple freq bands)
  - ccm cache namespace (ConvergentCrossMapping) — exercises namespace bucketing
    and the pyEDM call-site pinning path under the worker pool
  - cacheless SPIs (SpearmanR, KendallTau, PowerEnvelopeCorrelation)
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyspi.calculator import Calculator

CONFIG = Path(__file__).parent / "parallel_test_config.yaml"

# fork is the Linux production default; spawn is the only safe method elsewhere.
MP_CONTEXTS = ["spawn"] + (["fork"] if sys.platform.startswith("linux") else [])


@pytest.fixture(scope="module")
def dataset():
    rng = np.random.default_rng(0)
    return rng.standard_normal((5, 300)).astype(np.float64)


@pytest.fixture(scope="module")
def serial_table(dataset):
    calc = Calculator(dataset=dataset, configfile=str(CONFIG), normalise=False)
    calc.compute(n_jobs=1, progress=False)
    return calc.table.copy()


@pytest.mark.parametrize("mp_context", MP_CONTEXTS)
@pytest.mark.parametrize("n_jobs", [2, 3])
def test_parallel_matches_serial(dataset, serial_table, n_jobs, mp_context):
    """Parallel n_jobs>1 must produce numerically identical tables to serial,
    for every start method and across all cache namespaces (incl. CCM)."""
    calc = Calculator(dataset=dataset, configfile=str(CONFIG), normalise=False)
    calc.compute(n_jobs=n_jobs, mp_context=mp_context, progress=False)
    parallel_table = calc.table

    assert list(parallel_table.columns) == list(serial_table.columns), \
        "Column order/identity diverged between serial and parallel."

    # Every SPI here is deterministic (CCM is seeded); tight tolerance.
    np.testing.assert_allclose(
        parallel_table.to_numpy(),
        serial_table.to_numpy(),
        rtol=1e-10, atol=1e-12, equal_nan=True,
        err_msg=f"Parallel (n_jobs={n_jobs}, mp={mp_context}) diverged from serial.",
    )


def test_checkpoint_resume_matches_full_run(dataset, serial_table, tmp_path):
    """Partial run -> delete some checkpoints -> resume; final table equals full serial."""
    cp_dir = tmp_path / "ckpt"

    # First pass: run serial with checkpoint_dir to populate .npy files.
    calc1 = Calculator(dataset=dataset, configfile=str(CONFIG), normalise=False)
    calc1.compute(n_jobs=1, checkpoint_dir=cp_dir, progress=False)
    saved = sorted(cp_dir.glob("*.npy"))
    assert len(saved) == len(calc1.spis), "Checkpoint dir missing files after first run."

    # Simulate partial failure by removing half the checkpoints.
    to_remove = saved[: len(saved) // 2]
    for f in to_remove:
        f.unlink()
    assert len(list(cp_dir.glob("*.npy"))) < len(calc1.spis)

    # Second pass: parallel resume. Should re-compute only the removed SPIs.
    calc2 = Calculator(dataset=dataset, configfile=str(CONFIG), normalise=False)
    calc2.compute(n_jobs=2, checkpoint_dir=cp_dir, resume=True,
                  mp_context="spawn", progress=False)

    assert list(calc2.table.columns) == list(serial_table.columns)
    np.testing.assert_allclose(
        calc2.table.to_numpy(),
        serial_table.to_numpy(),
        rtol=1e-10, atol=1e-12, equal_nan=True,
        err_msg="Resumed run diverged from full serial.",
    )


def test_failure_isolation(dataset):
    """A poisoned SPI must not break siblings: that SPI returns NaN, others OK.

    Run at n_jobs=1: an instance-level monkeypatch can't survive into a worker
    (workers re-instantiate SPIs from the config), so the failure must be raised
    in-process. The per-SPI try/except -> NaN-fill contract is the same in both
    paths (_compute_serial and _parallel._run_task); test_parallel_matches_serial
    covers that the parallel path completes the full table.
    """
    calc = Calculator(dataset=dataset, configfile=str(CONFIG), normalise=False)

    # Poison one Covariance instance's multivariate() at the instance level so
    # other Covariance/Precision siblings (which share the class method via the
    # Estimators base) are unaffected — tests that one failure inside a cache
    # bucket doesn't poison the rest of the bucket.
    victim_key = next(
        k for k, spi in calc.spis.items()
        if getattr(type(spi), "_cache_namespace", None) == "covariance"
    )
    siblings = [k for k in calc.spis.keys() if k != victim_key]

    def boom(*args, **kwargs):
        raise RuntimeError("intentional failure for test_failure_isolation")
    calc.spis[victim_key].multivariate = boom

    with pytest.warns(UserWarning):
        calc.compute(n_jobs=1, progress=False)

    M = calc.dataset.n_processes
    victim_mat = np.asarray(calc.table[victim_key])
    assert np.all(np.isnan(victim_mat)), "Poisoned SPI should be all-NaN."

    for sibling in siblings:
        mat = np.asarray(calc.table[sibling])
        # Diagonal is always NaN by convention; off-diagonal should be finite.
        offdiag = mat[~np.eye(M, dtype=bool)]
        assert np.isfinite(offdiag).all(), \
            f"Sibling SPI '{sibling}' has unexpected NaNs after isolated failure."


def test_pin_worker_thread_pools_sets_env():
    """Worker pinning must export PYSPI_PIN_BACKENDS — the flag statistics/causal.py
    reads to pass parallel=False to pyEDM."""
    from pyspi import _parallel
    os.environ.pop("PYSPI_PIN_BACKENDS", None)
    try:
        _parallel._pin_worker_thread_pools()
        assert os.environ.get("PYSPI_PIN_BACKENDS") == "1"
    finally:
        os.environ.pop("PYSPI_PIN_BACKENDS", None)


def test_cli_module_importable():
    """The CLI module must import cleanly; smoke test against argparse plumbing."""
    import importlib
    mod = importlib.import_module("pyspi.__main__")
    assert hasattr(mod, "main")
