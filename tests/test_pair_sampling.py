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
from pyspi.calculator import Calculator, pair_budget, sample_dyads
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
        if getattr(spis[key], "_pair_local", None) == "directed":
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

    ``"directed"`` classes match the dense table to rounding; ``"retained"``
    and ``"matrix"`` exactly.
    """
    marked = {(type(s).__name__, s._pair_local) for s in _p90().spis.values()
              if getattr(s, "_pair_local", None)}
    assert marked == {("LaggedCorrelation", "directed"),
                      ("DynamicTimeWarping", "directed"),
                      ("MutualInfo", "directed"),
                      ("TimeLaggedMutualInfo", "directed"),
                      ("DistanceCorrelation", "retained"),
                      ("SpectralGrangerCausality", "matrix")}


def test_nonparametric_spectral_granger_factorises_only_the_sampled_pairs(monkeypatch):
    """One factorisation per sampled dyad, and the dense entries exactly.

    The parametric variants fit every process and stay fully computed.
    """
    import spectral_connectivity.connectivity as backend

    calc = _p90()
    keys = [k for k in calc.spis if k.startswith("sgc_")]
    nonparametric = [k for k in keys if "nonparametric" in k]
    assert len(nonparametric) == 6 and len(keys) == 24
    assert {_parallel._dense_loop(calc.spis[k]) for k in nonparametric} == {"matrix"}
    assert {_parallel._dense_loop(calc.spis[k]) for k in keys if k not in nonparametric} == {None}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dense = {k: _parallel.run_spi(calc.spis[k], calc.dataset, k, 7)[0] for k in nonparametric}

    calls = []
    inner = backend.minimum_phase_decomposition
    monkeypatch.setattr(backend, "minimum_phase_decomposition",
                        lambda *a, **k: calls.append(1) or inner(*a, **k))
    fresh = _p90()
    pairs = np.array([[1, 4], [0, 6], [2, 3], [0, 1], [3, 5]])
    i, j = pairs.T
    for k in nonparametric:
        V, err, _, _ = _parallel.run_spi_pairs(fresh.spis[k], fresh.dataset, k, pairs)
        assert err is None, (k, err)
        assert np.array_equal(V, np.stack([dense[k][i, j], dense[k][j, i]], axis=1),
                              equal_nan=True), k
        assert np.isfinite(V).all()
    assert len(calls) == len(pairs)               # shared by all six variants


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


def test_compute_pairs_matches_compute_and_fills_the_table_at_the_sample():
    dense = Calculator(dataset=_series(), config=DETERMINISTIC_CONFIG, verbose=False)
    dense.compute(progress=False)
    calc = _deterministic(n_pairs=7, seed=11)

    assert not calc.errors and set(calc.timings) == set(calc.spis)
    assert calc.pair_values.shape == (calc.n_spis, 7, 2)
    assert np.isfinite(calc.pair_values).all()
    i, j = calc.pairs.T
    sampled = np.zeros((6, 6), dtype=bool)
    sampled[i, j] = sampled[j, i] = True
    for row, key in zip(calc.pair_values, calc.spis):
        S = dense.table[key].to_numpy(dtype=float)
        assert np.array_equal(row, np.stack([S[i, j], S[j, i]], axis=1)), key
        # The table is the full one with the unsampled entries missing.
        T = calc.table[key].to_numpy(dtype=float)
        assert np.array_equal(T[sampled], S[sampled]) and np.isnan(T[~sampled]).all(), key


def test_a_full_run_and_a_sampled_run_replace_each_other():
    calc = _deterministic(n_pairs=4, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calc.compute(progress=False)
    assert calc.pairs is None and calc.pair_values is None and calc.sampling is None
    off = ~np.eye(6, dtype=bool)
    assert np.isfinite(calc.table["spearmanr"].to_numpy(dtype=float)[off]).all()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calc.compute_pairs(n_pairs=4, seed=1, progress=False)
    assert np.isfinite(calc.table["spearmanr"].to_numpy(dtype=float)).sum() == 8


def test_a_sampled_table_saves_and_loads_with_its_pairs(tmp_path):
    from pyspi import load_table

    calc = _deterministic(tolerance=0.3, seed=2)
    table = load_table(calc.save(tmp_path / "sampled.npz"))
    assert np.array_equal(table.to_numpy(dtype=float), calc.table.to_numpy(dtype=float),
                          equal_nan=True)
    assert np.array_equal(table.attrs["pairs"], calc.pairs)
    record = table.attrs["sampling"]
    assert record["n_pairs"] == len(calc.pairs) and record["tolerance"] == 0.3
    assert record["pairs_sha256"] == calc.sampling["pairs_sha256"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calc.compute(progress=False)
    full = load_table(calc.save(tmp_path / "full.npz"))
    assert "pairs" not in full.attrs and "sampling" not in full.attrs


def test_command_line_computes_a_sampled_table(tmp_path):
    from pyspi import load_table
    from pyspi.__main__ import main

    data = tmp_path / "x.npy"
    np.save(data, _series())
    out = tmp_path / "out.npz"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        code = main(["compute", "--data", str(data), "--config", DETERMINISTIC_CONFIG,
                     "--output", str(out), "--n-pairs", "5", "--pair-seed", "3",
                     "--no-zscore", "--quiet"])
    assert code == 0
    table = load_table(out)
    assert np.array_equal(table.attrs["pairs"], sample_dyads(6, 5, seed=3))
    assert np.isfinite(table["spearmanr"].to_numpy(dtype=float)).sum() == 10


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
    ({"n_pairs": 3, "tolerance": 0.1}, ValueError),
    ({"tolerance": 0}, ValueError),
    ({"pairs": [[0, 1]], "seed": 0}, ValueError),
    ({"n_pairs": 16}, ValueError),
    ({"pairs": [0, 1]}, ValueError),
    ({"pairs": [[0.0, 1.0]]}, ValueError),
    ({"pairs": [[0, 0]]}, ValueError),
    ({"pairs": [[0, 1], [1, 0]]}, ValueError),
    ({"pairs": [[0, 6]]}, IndexError),
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


# --------------------------------------------------------------------------
# The budget rule
# --------------------------------------------------------------------------

def test_pair_budget_solves_the_error_model_and_never_exceeds_the_census():
    """b is the smallest budget whose modelled error is within tolerance."""
    for M in (2, 5, 20, 64, 243, 1000):
        D = M * (M - 1) // 2
        for tolerance, c in ((0.03, 1.5), (0.05, 1.0), (0.01, 1.8), (0.5, 1.5)):
            b = pair_budget(M, tolerance, c)
            assert 1 <= b <= D
            error = lambda n: c * np.sqrt((1 - n / D) / n)
            assert error(b) <= tolerance * (1 + 1e-9)
            assert b == 1 or error(b - 1) > tolerance

    assert [pair_budget(M) for M in (20, 64, 100, 243, 500)] == [177, 1117, 1662, 2305, 2451]
    assert pair_budget(10 ** 5) == 2500                # (c / tolerance)^2, whatever M
    assert pair_budget(8, tolerance=1e-6) == 28        # the census
    for bad in (dict(n_processes=1), dict(n_processes=5, tolerance=-0.1),
                dict(n_processes=5, c=0)):
        with pytest.raises(ValueError):
            pair_budget(**bad)


def test_compute_pairs_draws_the_budget_for_a_tolerance():
    calc = _deterministic(tolerance=0.2, seed=3)
    assert len(calc.pairs) == pair_budget(6, 0.2) == 12
    assert calc.sampling["tolerance"] == 0.2 and calc.sampling["n_pairs"] == 12
    assert np.array_equal(calc.pairs, sample_dyads(6, 12, seed=3))


# --------------------------------------------------------------------------
# Checkpoints, and sampling together with workers
# --------------------------------------------------------------------------

def test_a_checkpoint_directory_belongs_to_one_sample(tmp_path):
    """Same SPI names, different arrays: the dyads are part of the identity."""
    import json

    cp_dir = tmp_path / "sampled"
    first = _deterministic(n_pairs=5, seed=3, checkpoint_dir=cp_dir)
    manifest = json.loads((cp_dir / "run.json").read_text())
    assert manifest["computation"] == _parallel.COMPUTATION_VERSION
    assert manifest["spec"]["sampling"] == {
        "n_pairs": 5, "pairs_sha256": first.sampling["pairs_sha256"],
        "source": "sampled", "seed": 3}

    # The same dyads, however they were arrived at, resume without recomputing.
    again = _deterministic(pairs=first.pairs, checkpoint_dir=cp_dir)
    assert np.array_equal(again.pair_values, first.pair_values)
    assert set(again.timings.values()) == {0.0}

    calc = Calculator(dataset=_series(), config=DETERMINISTIC_CONFIG, verbose=False)
    for kwargs in ({"n_pairs": 5, "seed": 4},            # another sample
                   {"n_pairs": 6, "seed": 3},            # a longer prefix
                   {"pairs": first.pairs[::-1]}):        # another order
        with pytest.raises(ValueError, match="different run"):
            calc.compute_pairs(progress=False, checkpoint_dir=cp_dir, **kwargs)
    with pytest.raises(ValueError, match="different run"):
        calc.compute(progress=False, checkpoint_dir=cp_dir)

    dense_dir = tmp_path / "dense"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calc.compute(progress=False, checkpoint_dir=dense_dir)
    with pytest.raises(ValueError, match="different run"):
        calc.compute_pairs(n_pairs=5, seed=3, progress=False, checkpoint_dir=dense_dir)


@pytest.mark.slow
def test_p90_sampled_runs_agree_across_workers_census_and_resume(tmp_path):
    """Sampling and workers together, on the config that mixes every path.

    The dyads are fixed before any work is handed out, so a sampled run is the
    same array serially, on two or four workers, and resumed part-way; a full
    census is the dense table; and the seeded stochastic estimators (sgddtw
    barycenters, MinCovDet, EllipticEnvelope) give the dense entry exactly.
    """
    def run(**kwargs):
        calc = _p90()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if kwargs:
                calc.compute_pairs(progress=False, **kwargs)
            else:
                calc.compute(progress=False)
        return calc

    dense = run()
    keys = list(dense.spis)

    def assert_matches_dense(calc, label):
        i, j = calc.pairs.T
        assert set(calc.errors) == set(dense.errors), label
        for row, key in zip(calc.pair_values, keys):
            S = dense.table[key].to_numpy(dtype=float)
            expected = np.stack([S[i, j], S[j, i]], axis=1)
            if getattr(dense.spis[key], "_pair_local", None) == "directed":
                assert np.allclose(row, expected, equal_nan=True, **TIGHT), (label, key)
            else:
                assert np.array_equal(row, expected, equal_nan=True), (label, key)

    serial = run(n_pairs=8, seed=5)
    how = list(serial.sampling["evaluation"].values())
    assert (how.count("pairs"), how.count("dense")) == (153, 130)
    assert_matches_dense(serial, "serial sample")
    stochastic = [k for k in keys if "sgddtw" in k or "MinCovDet" in k or "EllipticEnvelope" in k]
    assert len(stochastic) == 12

    cp_dir = tmp_path / "ckpt"
    for label, kwargs in (("two workers", dict(n_jobs=2)),
                          ("four workers", dict(n_jobs=4, checkpoint_dir=cp_dir))):
        calc = run(n_pairs=8, seed=5, mp_context="spawn", **kwargs)
        assert np.array_equal(calc.pairs, serial.pairs), label
        assert np.array_equal(calc.pair_values, serial.pair_values, equal_nan=True), label
        assert calc.errors == serial.errors, label

    for key in keys[::2]:
        (cp_dir / f"{key}.npy").unlink()
    resumed = run(n_pairs=8, seed=5, n_jobs=2, mp_context="spawn", checkpoint_dir=cp_dir)
    assert np.array_equal(resumed.pair_values, serial.pair_values, equal_nan=True)
    assert resumed.errors == serial.errors

    assert_matches_dense(run(n_pairs=21, seed=6, n_jobs=4, mp_context="spawn"),
                         "census on four workers")
