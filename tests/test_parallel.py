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


def test_pin_worker_thread_pools_installs_the_blas_limiter():
    """Worker pinning must reach the BLAS/OpenMP pools.

    It used to also pin cdt's NJOBS and torch's thread counts; both
    dependencies are gone, and threadpoolctl is what is left to pin. n_jobs
    workers each fanning out to cpu_count() BLAS threads is the quadratic
    blow-up this exists to prevent.

    The limiter object is what is asserted, not `threadpool_info()`: that comes
    back empty where no threadpoolctl-visible BLAS is loaded (macOS Accelerate,
    for one), so keying the test on it would make it pass vacuously on some
    machines and fail on others.
    """
    from threadpoolctl import threadpool_info

    from pyspi import _parallel

    _parallel._pin_worker_thread_pools()
    try:
        assert _parallel._THREADPOOL_LIMITER is not None
        assert all(pool["num_threads"] == 1 for pool in threadpool_info())
    finally:
        _parallel._THREADPOOL_LIMITER.unregister()
        _parallel._THREADPOOL_LIMITER = None


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


def test_cli_exits_nonzero_on_a_partial_failure_by_default(tmp_path, capsys, cli_env):
    """A table with failed columns is one a pipeline must not ingest silently.

    `--quiet` suppresses the computation summary, which used to be the only
    place a failed SPI was named, and the exit status was 0 unconditionally --
    so `pyspi compute --quiet` printed "Wrote results table" and exited 0 over a
    table that could be entirely NaN. The default is now 1, with the failed
    identifiers on stderr, and the results file is still written so the partial
    table is available for inspection.
    """
    from pyspi.__main__ import main

    data = _cli_dataset(tmp_path)
    out = tmp_path / "res.npz"
    code = main(["compute", "--data", str(data), "--quiet",
                 "--config", str(Path(__file__).parent / "parity_failure_config.yaml"),
                 "--output", str(out)])
    assert code == 1
    err = capsys.readouterr().err
    assert "always_raises" in err, f"failure not reported on stderr: {err!r}"
    assert out.exists(), "the partial table must still be written"


def test_cli_allow_partial_exits_zero(tmp_path, capsys, cli_env):
    """The documented opt-out, for callers that expect some SPIs to fail."""
    from pyspi.__main__ import main

    data = _cli_dataset(tmp_path)
    code = main(["compute", "--data", str(data), "--quiet", "--allow-partial",
                 "--config", str(Path(__file__).parent / "parity_failure_config.yaml"),
                 "--output", str(tmp_path / "res.npz")])
    assert code == 0
    assert "always_raises" in capsys.readouterr().err


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


# --------------------------------------------------------------------------
# Pairs shared between workers
# --------------------------------------------------------------------------

SPLIT_CONFIG = Path(__file__).parent / "split_test_config.yaml"
FIXTURE = Path(__file__).parent / "data" / "fixtures" / "kuramoto_M7_T100.npy"


def _in_pieces(spis, keys, pieces, order, pairs, M, dense=True):
    """What ``run_parallel`` assembles for split SPIs, without a pool.

    Every piece gets a ``Data`` of its own, so no cache is carried from one
    piece to the next, as none is between two workers.
    """
    from pyspi._parallel import _SplitSPI, run_spi_pairs
    from pyspi.data import Data

    assembling = {key: _SplitSPI(len(pairs), len(pieces)) for key in keys}
    results = {}
    for piece in order:
        data = Data(data=str(FIXTURE), dim_order="sp")
        rows = slice(*pieces[piece])
        for key in keys:
            part = run_spi_pairs(spis[key], data, key, pairs[rows], complete=False)
            if assembling[key].add(piece, rows, *part):
                results[key] = assembling[key].result(pairs, M, dense)
    return results


def test_plan_pieces_shares_only_pair_loops():
    """A task is split only if every SPI in it is a loop over ``bivariate``."""
    from pyspi._parallel import all_dyads, build_tasks, is_pair_loop, plan_pieces
    from pyspi.calculator import load_spis_from_yaml, resolve_config

    spis = load_spis_from_yaml(resolve_config("benchmarked_p90"), quiet=True)
    tasks = build_tasks(list(spis), spis)
    n_pairs = len(all_dyads(7))
    assert n_pairs == 21

    whole, split, pieces = plan_pieces(tasks, spis, n_jobs=4, n_pairs=n_pairs)
    assert sorted(k for t in whole + split for k in t) == sorted(spis)
    assert all(is_pair_loop(spis[k]) for t in split for k in t)
    assert not any(all(is_pair_loop(spis[k]) for k in t) for t in whole)
    in_split = {k for t in split for k in t}
    assert {"bary_dtw_mean", "dcorr", "cds"} <= in_split
    # Vectorised, and fitted on every process at once: never split.
    assert not {"dtw", "spearmanr", "cov_EmpiricalCovariance"} & in_split

    # The pieces are one contiguous, non-empty range each and cover every pair.
    assert len(pieces) == 4
    assert pieces[0][0] == 0 and pieces[-1][1] == n_pairs
    assert all(a < b for a, b in pieces)
    assert all(pieces[i][1] == pieces[i + 1][0] for i in range(3))

    # More workers than pairs: one pair each. One worker, or one pair: no split.
    assert plan_pieces(tasks, spis, n_jobs=64, n_pairs=n_pairs)[2] == [
        (i, i + 1) for i in range(n_pairs)]
    assert plan_pieces(tasks, spis, n_jobs=1, n_pairs=n_pairs) == (tasks, [], [])
    assert plan_pieces(tasks, spis, n_jobs=4, n_pairs=1) == (tasks, [], [])


@pytest.mark.parametrize("scheme", ["one pair each, backwards", "four pieces, shuffled"])
def test_every_pair_loop_spi_is_the_same_in_pieces(scheme):
    """Each pair-loop SPI in ``full``: the matrix put together from pieces is the
    matrix one loop gives, bit for bit. (Failures: the next test.)

    This is what lets workers share an SPI's pairs. It fails for an SPI whose
    value on a pair depends on which pairs were computed before it.
    """
    from pyspi._parallel import all_dyads, is_pair_loop, plan_pieces, run_spi
    from pyspi.calculator import load_spis_from_yaml, resolve_config
    from pyspi.data import Data

    spis = load_spis_from_yaml(resolve_config("full"), quiet=True)
    keys = [key for key, spi in spis.items() if is_pair_loop(spi)]
    assert len(keys) > 100

    data = Data(data=str(FIXTURE), dim_order="sp")
    M = data.n_processes
    pairs = all_dyads(M)
    with np.errstate(all="ignore"):
        whole = {key: run_spi(spis[key], data, key, M) for key in keys}

        # A second set of instances, as the workers have their own.
        spis = load_spis_from_yaml(resolve_config("full"), quiet=True)
        if scheme.startswith("one pair"):
            pieces = [(i, i + 1) for i in range(len(pairs))]
            order = range(len(pieces))[::-1]
        else:
            pieces = plan_pieces([keys], spis, n_jobs=4, n_pairs=len(pairs))[2]
            order = np.random.default_rng(0).permutation(len(pieces))
        split = _in_pieces(spis, keys, pieces, order, pairs, M)

    for key in keys:
        assert split[key][1] == whole[key][1], key
        assert np.array_equal(split[key][0], whole[key][0], equal_nan=True), key


@pytest.mark.parametrize("name,error", [
    ("FiniteOnOnePair", None),
    ("InfiniteOnOnePair", "ValueError: SPI returned infinite value(s)"),
    ("RaisesOnOnePair", "RuntimeError: deliberate failure on one pair"),
    ("NeverFinite", "ValueError: SPI returned no finite off-diagonal values"),
    ("AlwaysRaises", "RuntimeError: deliberate test failure"),
])
def test_a_split_spi_is_judged_whole(name, error, monkeypatch):
    """Pieces carry values; success is decided once, on the assembled result.

    A piece with nothing finite in it is not a failure, an infinity in one
    piece fails the SPI, and so does one pair that raises: the outcomes of one
    loop over every pair.
    """
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    import failing_spis
    from pyspi._parallel import all_dyads, run_spi, run_spi_pairs
    from pyspi.data import Data

    spi = getattr(failing_spis, name)()
    data = Data(data=str(FIXTURE), dim_order="sp")
    M = data.n_processes
    pairs = all_dyads(M)
    S, err, _, _ = run_spi(spi, data, name, M)
    assert err == error

    for pieces in ([(i, i + 1) for i in range(len(pairs))], [(0, 5), (5, 6), (6, 21)]):
        out, split_err, _, _ = _in_pieces(
            {name: spi}, [name], pieces, range(len(pieces)), pairs, M)[name]
        assert split_err == error
        assert np.array_equal(out, S, equal_nan=True)

    # A sample is judged by the rule for samples.
    sample = pairs[[0, 3, 7, 11, 20]]
    V, err, _, _ = run_spi_pairs(spi, data, name, sample)
    out, split_err, _, _ = _in_pieces(
        {name: spi}, [name], [(0, 2), (2, 5)], [1, 0], sample, M, dense=False)[name]
    assert split_err == err
    assert np.array_equal(out, V, equal_nan=True)


@pytest.mark.parametrize("M,n_jobs", [(5, 2), (5, 4), (3, 3)])
def test_split_table_is_the_serial_table(M, n_jobs, tmp_path):
    """Through the pool: sharing pairs changes no value, on a full table or a
    sample, from checkpoints or not, and neither does switching it off."""
    x = np.random.default_rng(1).standard_normal((M, 200))

    def calculator():
        return Calculator(dataset=x, config=str(SPLIT_CONFIG), verbose=False)

    def run(**kwargs):
        calc = calculator()
        calc.compute(progress=False, **kwargs)
        return calc

    ref = run(n_jobs=1)
    assert not ref.errors
    expected = ref.table.to_numpy(float)

    split = run(n_jobs=n_jobs, mp_context="spawn", checkpoint_dir=tmp_path / "ckpt")
    assert list(split.table.columns) == list(ref.table.columns)
    assert np.array_equal(split.table.to_numpy(float), expected, equal_nan=True)
    assert not split.errors
    assert set(split.timings) == set(ref.spis)

    # A split SPI is checkpointed whole, in the format a serial run reads.
    resumed = run(n_jobs=1, checkpoint_dir=tmp_path / "ckpt")
    assert np.array_equal(resumed.table.to_numpy(float), expected, equal_nan=True)
    assert all(t == 0.0 for t in resumed.timings.values()), "recomputed, not resumed"

    unsplit = run(n_jobs=n_jobs, mp_context="spawn", split_pairs=False)
    assert np.array_equal(unsplit.table.to_numpy(float), expected, equal_nan=True)

    if M > 3:
        one, many = calculator(), calculator()
        values = one.compute_pairs(n_pairs=6, seed=0, n_jobs=1, progress=False)
        assert np.array_equal(
            many.compute_pairs(n_pairs=6, seed=0, n_jobs=n_jobs, mp_context="spawn",
                               progress=False),
            values, equal_nan=True)
        assert many.errors == one.errors


@pytest.mark.parametrize("name", ["KillsItsWorker", "KillsItsWorkerWhole"])
def test_a_dead_worker_fails_its_spi_and_not_the_run(name, tmp_path, monkeypatch):
    """A worker that dies takes its SPI with it, split between workers or not;
    ``compute`` still returns, with the loss recorded."""
    tests_dir = str(Path(__file__).parent)
    monkeypatch.syspath_prepend(tests_dir)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(
        filter(None, [tests_dir, os.environ.get("PYTHONPATH")])))
    config = tmp_path / "config.yaml"
    config.write_text(
        f"failing_spis:\n  {name}:\n    labels: [test]\n    configs:\n"
        ".statistics.basic:\n  Covariance:\n    labels: [undirected]\n"
        "    configs:\n      - estimator: EmpiricalCovariance\n")
    calc = Calculator(dataset=np.random.default_rng(0).standard_normal((4, 100)),
                      config=str(config), verbose=False)
    key = next(k for k in calc.spis if k.startswith("kills_its_worker"))

    with pytest.warns(UserWarning, match="worker died"):
        calc.compute(n_jobs=2, mp_context="spawn", progress=False)

    assert calc.errors[key].startswith("worker died")
    assert np.isnan(calc.table[key].to_numpy(float)).all()
    assert set(calc.table.columns.get_level_values(0)) == set(calc.spis)


def test_parallel_run_leaves_the_parent_environment_alone(monkeypatch):
    """The thread pins are for the workers and must not outlive the pool.

    Left in the parent, ``NUMBA_NUM_THREADS=1`` contradicted a numba runtime
    that had already launched its threads, and numba then refused every later
    compilation in the process: a serial run after a parallel one failed any SPI
    whose numba kernel had not been compiled yet.
    """
    monkeypatch.delenv("NUMBA_NUM_THREADS", raising=False)
    monkeypatch.setenv("NUMEXPR_NUM_THREADS", "1")
    calc = Calculator(dataset=np.random.default_rng(0).standard_normal((4, 120)),
                      config=CONFIG, verbose=False)
    calc.compute(n_jobs=2, mp_context="spawn", progress=False)
    assert "NUMBA_NUM_THREADS" not in os.environ
    assert os.environ["NUMEXPR_NUM_THREADS"] == "1"
    assert not calc.errors



@pytest.mark.slow
def test_p90_is_identical_for_any_worker_count_repeat_and_resume(tmp_path):
    """One table, bit for bit, however the work is divided.

    ``benchmarked_p90`` on a fixture where six SPIs fail, so values, missing
    entries, recorded errors and their order are all compared. The serial runs
    differ in the global NumPy seed; the parallel ones in worker count. The
    last run resumes from every other checkpoint with a different worker
    count, so cache groups are entered part-way through.
    """
    import warnings

    from pyspi.data import Data

    fixture = Path(__file__).parent / "data" / "fixtures" / "kuramoto_M7_T100.npy"

    def run(n_jobs, seed, **kwargs):
        np.random.seed(seed)
        calc = Calculator(dataset=Data(data=str(fixture), dim_order="sp"),
                          config="benchmarked_p90", verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            calc.compute(n_jobs=n_jobs, progress=False,
                         mp_context="spawn" if n_jobs > 1 else None, **kwargs)
        return calc

    def assert_same(calc, label):
        assert list(calc.table.columns) == list(ref.table.columns), label
        assert np.array_equal(calc.table.to_numpy(float), ref.table.to_numpy(float),
                              equal_nan=True), label
        assert list(calc.errors.items()) == list(ref.errors.items()), label

    ref = run(1, seed=0)
    assert len(ref.errors) == 6

    assert_same(run(1, seed=1), "serial, another global seed")
    assert_same(run(2, seed=2), "two workers")
    cp_dir = tmp_path / "ckpt"
    assert_same(run(4, seed=3, checkpoint_dir=cp_dir), "four workers")

    for key in list(ref.spis)[::2]:
        (cp_dir / f"{key}.npy").unlink()
    resumed = run(2, seed=4, checkpoint_dir=cp_dir)
    assert list(resumed.table.columns) == list(ref.table.columns)
    assert np.array_equal(resumed.table.to_numpy(float), ref.table.to_numpy(float),
                          equal_nan=True)
    assert resumed.errors == ref.errors


def test_computation_version_is_public():
    """Callers record it with their results; it is what checkpoints are keyed on."""
    import pyspi
    from pyspi import _parallel

    assert pyspi.COMPUTATION_VERSION == _parallel.COMPUTATION_VERSION
    assert "COMPUTATION_VERSION" in pyspi.__all__
