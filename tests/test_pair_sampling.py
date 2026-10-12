"""``Calculator.compute_pairs`` must return the entries ``compute`` would have.

Sampling process pairs is only an optimisation if it leaves the statistic
alone: a sampled entry has to be the number the dense table holds at that
position, for the same SPI, in the same direction. The reference here is the
dense path itself, which is the definition of "the same entry" and so is the
right oracle for *equivalence* -- but not for correctness of direction, which
is pinned separately against a closed-form transfer entropy.

Equality is exact wherever the sampled path makes the calls the dense path
makes: an SPI built by looping over pairs receives, for each sampled dyad, the
dense loop's calls in the dense loop's order, and an SPI computed in full is
simply indexed. Any nonzero difference there is a defect. The four classes
marked ``_pair_local`` are the exception. Their dense matrix comes from a
vectorised ``multivariate`` and the sampled path calls ``bivariate``, which adds
the same terms in a different order, so they are held to the baseline suite's
tight tolerance (1e-12 absolute or 1e-9 relative) rather than to equality.

Both paths are still seeded identically before each SPI. No p90 SPI reads the
global RNG any longer (``test_estimator_contracts`` pins that), so this is a
guard: if one starts to, it shows up there rather than as a sampling error here.
"""
import os
import random
import warnings
from pathlib import Path

import numpy as np
import pytest

from pyspi import _parallel
from pyspi.calculator import Calculator, sample_dyads
from pyspi.data import Data

TESTS_DIR = Path(__file__).parent
FIXTURE = str(TESTS_DIR / "data" / "fixtures" / "kuramoto_M7_T100.npy")
DETERMINISTIC_CONFIG = str(TESTS_DIR / "parallel_test_config.yaml")
FAILURE_CONFIG = str(TESTS_DIR / "parity_failure_config.yaml")
SEED = 42

TIGHT = dict(atol=1e-12, rtol=1e-9)
SGDDTW = ["bary_sgddtw_mean", "bary_sgddtw_max",
          "bary-sq_sgddtw_mean", "bary-sq_sgddtw_max"]


def _p90():
    data = Data(data=FIXTURE, dim_order="sp", name="kuramoto_M7_T100")
    return Calculator(dataset=data, config="benchmarked_p90", verbose=False)


def _seeded(run, calc, *args):
    """Run one primitive over every SPI, seeding the global RNGs before each."""
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for key, spi in calc.spis.items():
            np.random.seed(SEED)
            random.seed(SEED)
            values, err, _, _ = run(spi, calc.dataset, key, *args)
            out[key] = (values, err)
    return out


@pytest.fixture(scope="module")
def dense():
    calc = _p90()
    return _seeded(_parallel.run_spi, calc, calc.dataset.n_processes)


def _sampled(pairs):
    # A fresh Calculator: the dense run must not have warmed any cache.
    return _seeded(_parallel.run_spi_pairs, _p90(), pairs)


def _mismatches(dense, sampled, pairs, spis):
    i, j = pairs.T
    bad = []
    for key, (S, dense_err) in dense.items():
        V, err = sampled[key]
        expected = np.stack([S[i, j], S[j, i]], axis=1)
        if getattr(type(spis[key]), "_pair_local", None) == "directed":
            same = np.allclose(V, expected, equal_nan=True, **TIGHT)
        else:
            same = np.array_equal(V, expected, equal_nan=True)
        # Whether it failed, not the message: "no finite value" is worded for
        # the matrix in one path and for the sample in the other.
        if (err is None) != (dense_err is None) or not same:
            bad.append((key, err, dense_err, float(np.nanmax(np.abs(V - expected)))))
    return bad


# --------------------------------------------------------------------------
# Equivalence with the dense table
# --------------------------------------------------------------------------

def test_full_census_reproduces_the_dense_table(dense):
    """All dyads, shuffled: every p90 SPI, both directions, same failures."""
    pairs = sample_dyads(7, 21, seed=1)
    assert not _mismatches(dense, _sampled(pairs), pairs, _p90().spis)


def test_a_subset_holds_the_dense_entries(dense):
    pairs = sample_dyads(7, 8, seed=2)
    assert not _mismatches(dense, _sampled(pairs), pairs, _p90().spis)


def test_pair_local_classes_are_the_expensive_vectorised_ones():
    """The marker is an opt-in per class; pin which p90 SPIs carry it.

    ``"directed"`` classes match the dense table to rounding, the
    ``"retained"`` one exactly.
    """
    marked = {type(s).__name__: type(s)._pair_local for s in _p90().spis.values()
              if getattr(type(s), "_pair_local", None)}
    assert marked == {"LaggedCorrelation": "directed",
                      "DynamicTimeWarping": "directed",
                      "MutualInfo": "directed",
                      "TimeLaggedMutualInfo": "directed",
                      "DistanceCorrelation": "retained"}


def test_distance_correlation_makes_one_call_per_sampled_pair(monkeypatch):
    from pyspi.statistics.distance import DistanceCorrelation

    calls = []
    inner = DistanceCorrelation.bivariate
    monkeypatch.setattr(
        DistanceCorrelation, "bivariate",
        lambda self, data, i=None, j=None: calls.append((i, j)) or inner(self, data, i=i, j=j))
    calc = _p90()
    pairs = np.array([[2, 5], [0, 1], [1, 5]])
    V, err, _, _ = _parallel.run_spi_pairs(calc.spis["dcorr"], calc.dataset, "dcorr", pairs)
    assert err is None and np.array_equal(V[:, 0], V[:, 1])
    assert calls == [(0, 1), (1, 5), (2, 5)]      # dense order: by column, then row


def test_sgddtw_barycenter_ignores_and_preserves_the_global_rng():
    """Its value is a function of the pair alone, in any order of evaluation.

    It used to draw from the global NumPy stream, so an entry depended on how
    many pairs had been fitted before it and no subset could match the table.
    """
    calc = _p90()
    forward, backward = np.array([[0, 1], [2, 5], [3, 4]]), np.array([[3, 4], [0, 1]])
    for key in SGDDTW:
        np.random.seed(1)
        before = np.random.get_state()[1].copy()
        a, err, _, _ = _parallel.run_spi_pairs(calc.spis[key], _p90().dataset, key, forward)
        assert err is None and np.array_equal(before, np.random.get_state()[1])
        np.random.seed(2)
        b, _, _, _ = _parallel.run_spi_pairs(calc.spis[key], _p90().dataset, key, backward)
        assert np.array_equal(a[[2, 0]], b), key


def test_undirected_pair_loops_fill_both_directions_alike():
    pairs = sample_dyads(7, 8, seed=2)
    calc = _p90()
    undirected = {k: s for k, s in calc.spis.items()
                  if _parallel._dense_loop(s) == "undirected"}
    assert undirected
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for key, spi in undirected.items():
            V, err, _, _ = _parallel.run_spi_pairs(spi, calc.dataset, key, pairs)
            assert err is None, key
            assert np.array_equal(V[:, 0], V[:, 1], equal_nan=True), key


def test_direction_matches_a_closed_form_transfer_entropy():
    """Slot 0 is i -> j. Process 0 drives process 1 at lag one and nothing else.

    With x white and y[t] = a*x[t-1] + s*e[t], y is white too, so its own past
    explains nothing and TE(x -> y) = 0.5*ln(1 + a^2/s^2) nats, TE(y -> x) = 0.
    """
    a, s, T = 0.9, 0.3, 4000
    rng = np.random.default_rng(7)
    x, e = rng.standard_normal((2, T + 1))
    series = np.stack([x[1:], a * x[:-1] + s * e[1:], rng.standard_normal(T)])
    calc = Calculator(dataset=series, config="benchmarked_p90", verbose=False)
    key = "gc_gaussian_k-1_kt-1_l-1_lt-1"
    assert _parallel._dense_loop(calc.spis[key]) == "directed"

    V, err, _, _ = _parallel.run_spi_pairs(
        calc.spis[key], calc.dataset, key, np.array([[0, 1], [0, 2]]))
    assert err is None
    assert V[0, 0] == pytest.approx(0.5 * np.log(1 + a**2 / s**2), abs=0.03)
    assert abs(V[0, 1]) < 0.01
    assert np.abs(V[1]).max() < 0.01


# --------------------------------------------------------------------------
# The sampler
# --------------------------------------------------------------------------

def test_sampled_dyads_are_ordered_distinct_and_seeded():
    pairs = sample_dyads(12, 20, seed=5)
    assert pairs.shape == (20, 2)
    assert (pairs[:, 0] < pairs[:, 1]).all() and pairs.min() >= 0 and pairs.max() < 12
    assert len(np.unique(pairs, axis=0)) == 20
    assert np.array_equal(pairs, sample_dyads(12, 20, seed=5))
    assert not np.array_equal(pairs, sample_dyads(12, 20, seed=6))


def test_budgets_are_nested_and_the_largest_is_the_census():
    census = sample_dyads(9, 36, seed=3)
    assert np.array_equal(sample_dyads(9, 10, seed=3), census[:10])
    i, j = np.triu_indices(9, 1)
    assert {tuple(p) for p in census} == set(zip(i.tolist(), j.tolist()))


def test_every_dyad_is_equally_likely():
    M, b, draws = 6, 5, 3000
    counts = np.zeros((M, M))
    for seed in range(draws):
        i, j = sample_dyads(M, b, seed=seed).T
        counts[i, j] += 1
    freq = counts[np.triu_indices(M, 1)] / draws
    p = b / 15
    assert np.abs(freq - p).max() < 5 * np.sqrt(p * (1 - p) / draws)


@pytest.mark.parametrize("n_pairs,exc", [(16, ValueError), (0, ValueError),
                                         (2.0, TypeError), (True, TypeError)])
def test_sampler_rejects_an_impossible_budget(n_pairs, exc):
    with pytest.raises(exc):
        sample_dyads(6, n_pairs, seed=0)


# --------------------------------------------------------------------------
# Calculator.compute_pairs
# --------------------------------------------------------------------------

def _series(M=6, T=150):
    rng = np.random.default_rng(0)
    return rng.standard_normal((M, T)).cumsum(axis=1)


def _deterministic(**kwargs):
    calc = Calculator(dataset=_series(), config=DETERMINISTIC_CONFIG, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calc.compute_pairs(progress=False, **kwargs)
    return calc


def test_compute_pairs_matches_compute_and_leaves_the_table_alone():
    dense = Calculator(dataset=_series(), config=DETERMINISTIC_CONFIG, verbose=False)
    dense.compute(progress=False)
    calc = _deterministic(n_pairs=7, seed=11)

    assert not calc.errors and set(calc.timings) == set(calc.spis)
    assert calc.pair_values.shape == (calc.n_spis, 7, 2)
    assert np.isfinite(calc.pair_values).all()
    assert calc.table.isna().all().all()
    i, j = calc.pairs.T
    for row, key in zip(calc.pair_values, calc.spis):
        S = dense.table[key].to_numpy(dtype=float)
        assert np.array_equal(row, np.stack([S[i, j], S[j, i]], axis=1)), key


def test_result_does_not_depend_on_the_order_pairs_are_given_in():
    pairs = sample_dyads(6, 9, seed=4)
    forward = _deterministic(pairs=pairs)
    # Reversed rows, and each dyad written larger index first.
    backward = _deterministic(pairs=pairs[::-1, ::-1])
    assert np.array_equal(backward.pairs, pairs[::-1])
    assert np.array_equal(backward.pair_values, forward.pair_values[:, ::-1])


def test_a_smaller_budget_is_a_prefix_of_a_larger_one():
    small, large = _deterministic(n_pairs=4, seed=8), _deterministic(n_pairs=12, seed=8)
    assert np.array_equal(small.pairs, large.pairs[:4])
    assert np.array_equal(small.pair_values, large.pair_values[:, :4])


def test_serial_and_parallel_agree_exactly():
    serial = _deterministic(n_pairs=7, seed=11)
    parallel = _deterministic(n_pairs=7, seed=11, n_jobs=2, mp_context="spawn")
    assert np.array_equal(serial.pairs, parallel.pairs)
    assert np.array_equal(serial.pair_values, parallel.pair_values)
    assert serial.errors == parallel.errors == {}


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_failures_are_isolated_and_recorded(n_jobs, monkeypatch):
    monkeypatch.syspath_prepend(str(TESTS_DIR))
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH", str(TESTS_DIR) + (os.pathsep + existing if existing else ""))
    calc = Calculator(dataset=np.random.default_rng(0).standard_normal((4, 120)),
                      config=FAILURE_CONFIG, zscore=False, verbose=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calc.compute_pairs(n_pairs=3, seed=0, n_jobs=n_jobs, mp_context="spawn",
                           progress=False)

    assert set(calc.errors) == {"always_raises", "wrong_shape", "non_finite"}
    assert "deliberate test failure" in calc.errors["always_raises"]
    keys = list(calc.spis)
    for key in calc.errors:
        assert np.isnan(calc.pair_values[keys.index(key)]).all()
    assert np.isfinite(calc.pair_values[keys.index("cov_EmpiricalCovariance")]).all()
    assert (calc.pair_values[keys.index("always_warns")] == 0.5).all()
    assert any("deliberate test warning" in str(w.message) for w in caught)


@pytest.mark.parametrize("kwargs,exc", [
    ({}, ValueError),
    ({"n_pairs": 3, "pairs": [[0, 1]]}, ValueError),
    ({"pairs": [[0, 1]], "seed": 0}, ValueError),
    ({"n_pairs": 16}, ValueError),
    ({"pairs": [0, 1]}, ValueError),
    ({"pairs": [[0.0, 1.0]]}, ValueError),
    ({"pairs": [[0, 0]]}, ValueError),
    ({"pairs": [[0, 1], [1, 0]]}, ValueError),
    ({"pairs": [[0, 6]]}, IndexError),
    # Sampled results have no checkpoint format; resuming them is refused
    # rather than written into dense-shaped files.
    ({"n_pairs": 3, "checkpoint_dir": "unused"}, TypeError),
])
def test_invalid_requests_are_refused(kwargs, exc):
    calc = Calculator(dataset=_series(), config=DETERMINISTIC_CONFIG, verbose=False)
    with pytest.raises(exc):
        calc.compute_pairs(progress=False, **kwargs)
    assert calc.pairs is None and calc.pair_values is None


def test_loading_a_dataset_discards_sampled_results():
    calc = _deterministic(n_pairs=3, seed=0)
    calc.load_dataset(_series(M=5))
    assert calc.pairs is None and calc.pair_values is None and calc.sampling is None


def test_sampling_record_identifies_the_pairs_and_how_each_spi_was_evaluated():
    import hashlib

    calc = _deterministic(n_pairs=5, seed=3)
    record = calc.sampling
    assert {k: record[k] for k in ("n_processes", "n_dyads", "n_pairs", "source",
                                   "seed", "n_jobs")} == dict(
        n_processes=6, n_dyads=15, n_pairs=5, source="sampled", seed=3, n_jobs=1)
    assert record["computation_version"] == _parallel.COMPUTATION_VERSION
    assert record["pairs_sha256"] == hashlib.sha256(
        calc.pairs.astype("<i8").tobytes()).hexdigest()
    # Every SPI of this config has its own dense path (a covariance is fitted
    # on all processes at once); the p90 test below covers the mixture.
    assert list(record["evaluation"]) == list(calc.spis)
    assert set(record["evaluation"].values()) == {"dense"}

    explicit = _deterministic(pairs=calc.pairs)
    assert explicit.sampling["source"] == "explicit" and explicit.sampling["seed"] is None
    assert explicit.sampling["pairs_sha256"] == record["pairs_sha256"]
    # The identity is of the dyads in the order given.
    assert _deterministic(pairs=calc.pairs[::-1]).sampling["pairs_sha256"] != record["pairs_sha256"]
