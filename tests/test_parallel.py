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
    calc = Calculator(dataset=dataset, config=str(CONFIG), zscore=False)
    calc.compute(n_jobs=1, progress=False)
    return calc.table.copy()


@pytest.mark.parametrize("mp_context", MP_CONTEXTS)
@pytest.mark.parametrize("n_jobs", [2, 3])
def test_parallel_matches_serial(dataset, serial_table, n_jobs, mp_context):
    """Parallel n_jobs>1 must produce numerically identical tables to serial,
    for every start method and across all cache namespaces (incl. CCM)."""
    calc = Calculator(dataset=dataset, config=str(CONFIG), zscore=False)
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
    calc1 = Calculator(dataset=dataset, config=str(CONFIG), zscore=False)
    calc1.compute(n_jobs=1, checkpoint_dir=cp_dir, progress=False)
    saved = sorted(cp_dir.glob("*.npy"))
    assert len(saved) == len(calc1.spis), "Checkpoint dir missing files after first run."

    # Simulate partial failure by removing half the checkpoints.
    to_remove = saved[: len(saved) // 2]
    for f in to_remove:
        f.unlink()
    assert len(list(cp_dir.glob("*.npy"))) < len(calc1.spis)

    # Second pass: parallel resume. Should re-compute only the removed SPIs.
    calc2 = Calculator(dataset=dataset, config=str(CONFIG), zscore=False)
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
    calc = Calculator(dataset=dataset, config=str(CONFIG), zscore=False)

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


def test_pin_worker_thread_pools_pins_cdt():
    """Worker pinning must reach cdt, which autosets NJOBS=cpu_count() at import.

    Was a check on the PYSPI_PIN_BACKENDS env var, whose only consumer was
    pyEDM's nested pool in statistics/causal.py. That pool is now off
    unconditionally (it re-imports the caller's __main__), so the flag had no
    reader left; cdt is the pool this function can still actually pin.
    """
    import cdt

    from pyspi import _parallel
    before = cdt.SETTINGS.NJOBS
    try:
        _parallel._pin_worker_thread_pools()
        assert cdt.SETTINGS.NJOBS == 1
    finally:
        cdt.SETTINGS.NJOBS = before


def test_cli_module_importable():
    """The CLI module must import cleanly; smoke test against argparse plumbing."""
    import importlib
    mod = importlib.import_module("pyspi.__main__")
    assert hasattr(mod, "main")


# --------------------------------------------------------------------------
# CLI exit status
# --------------------------------------------------------------------------

def _cli_dataset(tmp_path):
    """A small VAR(1); enough for the SPIs in parity_failure_config to run."""
    rng = np.random.default_rng(0)
    A = np.array([[0.5, 0.0], [0.7, 0.4]])
    X = np.zeros((2, 120))
    for t in range(1, X.shape[1]):
        X[:, t] = A @ X[:, t - 1] + rng.standard_normal(2)
    path = tmp_path / "cli_data.npy"
    np.save(path, X)
    return path


@pytest.fixture
def cli_env(monkeypatch):
    """`failing_spis` importable by the CLI and by any worker it spawns."""
    tests_dir = str(Path(__file__).parent)
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH", tests_dir + (os.pathsep + existing if existing else "")
    )
    if tests_dir not in sys.path:
        monkeypatch.syspath_prepend(tests_dir)


def test_cli_reports_failures_and_exits_zero_by_default(tmp_path, capsys, cli_env):
    """`--quiet` must not turn a partly-failed run into a silent success.

    The computation summary is the only place a failed SPI was mentioned, and
    `--quiet` suppresses it -- so the CLI printed "Wrote results table" and
    exited 0 over a table whose columns had raised. Exit stays 0 (a few SPIs
    failing is normal on real data), but the failure is now on stderr.
    """
    from pyspi.__main__ import main

    data = _cli_dataset(tmp_path)
    out = tmp_path / "res.npz"
    code = main(["compute", "--data", str(data), "--quiet",
                 "--config", str(Path(__file__).parent / "parity_failure_config.yaml"),
                 "--output", str(out)])
    assert code == 0
    err = capsys.readouterr().err
    assert "always_raises" in err, f"failure not reported on stderr: {err!r}"
    assert out.exists()


def test_cli_fail_on_error_exits_nonzero(tmp_path, cli_env):
    """Opt-in hard failure for pipelines that want it."""
    from pyspi.__main__ import main

    data = _cli_dataset(tmp_path)
    code = main(["compute", "--data", str(data), "--quiet", "--fail-on-error",
                 "--config", str(Path(__file__).parent / "parity_failure_config.yaml"),
                 "--output", str(tmp_path / "res.npz")])
    assert code == 1


def test_cli_exits_nonzero_when_every_spi_is_empty(tmp_path, cli_env):
    """A table with no finite value anywhere is a failed run, not a result."""
    from pyspi.__main__ import main

    config = tmp_path / "all_failing.yaml"
    config.write_text(
        "failing_spis:\n"
        "  AlwaysRaises:\n"
        "    labels: [test]\n"
        "    configs:\n"
        "      - message: deliberate test failure\n"
    )
    data = _cli_dataset(tmp_path)
    code = main(["compute", "--data", str(data), "--quiet",
                 "--config", str(config), "--output", str(tmp_path / "res.npz")])
    assert code == 1


# --------------------------------------------------------------------------
# Task decomposition
# --------------------------------------------------------------------------

def test_build_tasks_splits_namespaces_into_the_caches_they_actually_share():
    """The longest task bounds the makespan, so it must not be a fiction.

    `build_tasks` bucketed on `_cache_namespace` alone, which serialises SPIs
    that share no cache at all: on `full` that produced one 84-member
    `spectral_mv` task covering 16 independent caches, and no amount of
    parallelism could split it. `_cache_subkey` is what separates them, and
    `calculator.warn_partial_cache_buckets` and `bench/cut_config.py` were
    already using it -- the scheduler was the odd one out.
    """
    from pyspi._parallel import build_tasks, cache_bucket
    from pyspi.calculator import load_spis_from_yaml, resolve_config

    spis = load_spis_from_yaml(resolve_config("full"), quiet=True)
    tasks = build_tasks(list(spis), spis)

    assert sorted(k for t in tasks for k in t) == sorted(spis), "keys lost or duplicated"

    # Every task is exactly one cache bucket (or one cacheless SPI).
    for task in tasks:
        buckets = {cache_bucket(spis[k]) for k in task}
        assert len(buckets) == 1, f"task mixes caches: {sorted(buckets)}"
        if buckets == {None}:
            assert len(task) == 1

    largest = max(len(t) for t in tasks)
    assert largest <= 30, (
        f"largest task has {largest} members; namespace-only bucketing gave 84"
    )


def test_build_tasks_starts_with_the_expensive_buckets():
    """Ordering is by estimated cost, not member count.

    A 3-member `ccm` bucket (292.7s amortized per SPI at the M=16, T=800 anchor)
    must be picked up before a 24-member `covariance` one (<0.3s). Scheduling
    only -- it cannot change a computed value.
    """
    from pyspi._parallel import build_tasks
    from pyspi.calculator import load_spis_from_yaml, resolve_config

    spis = load_spis_from_yaml(resolve_config("full"), quiet=True)
    tasks = build_tasks(list(spis), spis)
    assert tasks[0][0].startswith("ccm_"), f"first task is {tasks[0][0]}"
