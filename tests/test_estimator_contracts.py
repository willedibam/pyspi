"""An SPI must compute the estimator it advertises, or refuse.

Six classes accept ``estimator="kraskov"``, embed ``kraskov_NN-4`` in their
identifier, and then run the Gaussian estimator. Nothing in the result records
that substitution, so a table can report a k-NN estimate that was never
computed.

Scope note: none of these six appear in any bundled config, so no shipped
result or stored baseline is affected. This is a latent API defect — it bites
anyone hand-writing a config — not a corruption of the current numbers. The
tests are still blockers, because the failure is silent and scientific.

See tests/test_state_integrity.py for the xfail(strict=True) rationale.
"""
import numpy as np
import pytest

from pyspi.data import Data
from pyspi.statistics import infotheory as it

# Classes that advertise kraskov but dispatch to the Gaussian estimator.
FALSE_KRASKOV = [
    "JointEntropy",
    "ConditionalEntropy",
    "CrossmapEntropy",
    "CausalEntropy",
    "DirectedInfo",
    "StochasticInteraction",
]


def _data(seed=0, m=3, t=200):
    rng = np.random.default_rng(seed)
    return Data(data=rng.standard_normal((m, t)), dim_order="ps", zscore=True)


# --------------------------------------------------------------------------
# Estimator honesty
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cls_name", FALSE_KRASKOV)
def test_kraskov_is_not_silently_gaussian(cls_name):
    """Either compute a genuine k-NN estimate, or reject the argument."""
    cls = getattr(it, cls_name)
    data = _data()

    try:
        kraskov = cls(estimator="kraskov")
    except (ValueError, NotImplementedError):
        return  # Rejecting the unimplemented estimator is an acceptable fix.

    gaussian = cls(estimator="gaussian")
    kv = kraskov.multivariate(data)
    gv = gaussian.multivariate(data)

    assert not np.allclose(kv, gv, equal_nan=True), (
        f"{cls_name}(estimator='kraskov') returned exactly the Gaussian result "
        f"while advertising itself as {kraskov.identifier!r}."
    )


def test_invalid_auto_embed_method_is_rejected():
    with pytest.raises((ValueError, KeyError, NotImplementedError)):
        it.TransferEntropy(auto_embed_method="NOT_A_METHOD").multivariate(_data())


def test_unsupported_parameters_are_rejected():
    """A parameter that the chosen estimator ignores must not be accepted silently."""
    with pytest.raises((ValueError, TypeError)):
        # kernel_width is meaningless for the gaussian estimator.
        it.MutualInfo(estimator="gaussian", kernel_width=0.5)


# --------------------------------------------------------------------------
# Symbolic transfer entropy
# --------------------------------------------------------------------------

def test_symbolic_k_history_1_is_rejected():
    with pytest.raises(ValueError):
        it.TransferEntropy(estimator="symbolic", k_history=1)


def test_bundled_configs_exclude_degenerate_symbolic_variants():
    from pyspi.calculator import bundled_configs, load_spis_from_yaml, resolve_config

    offenders = []
    for name in bundled_configs():
        for ident in load_spis_from_yaml(resolve_config(name), quiet=True):
            if "symbolic" in ident and ("_k-1_" in ident or ident.endswith("_k-1")):
                offenders.append(f"{name}:{ident}")
            if "symbolic" in ident and "_k-10" in ident:
                offenders.append(f"{name}:{ident}")

    assert not offenders, (
        "Degenerate symbolic TE variants in bundled configs: " + ", ".join(offenders)
    )


def _reference_symbolic_te(src, targ, k):
    """Independent symbolic TE using Python tuples as histogram keys.

    Deliberately avoids any integer packing, so it cannot share the overflow
    failure mode of the implementation it checks.
    """
    from collections import Counter
    from math import log2

    from pyspi.statistics.infotheory import _series_to_ordinal_symbols

    s = _series_to_ordinal_symbols(np.asarray(src, dtype=float), k)
    t = _series_to_ordinal_symbols(np.asarray(targ, dtype=float), k)
    n = min(len(s), len(t)) - 1
    tn, tp, sc = t[1:n + 1], t[:n], s[:n]

    def H(*cols):
        counts = Counter(zip(*(list(map(int, c)) for c in cols)))
        total = sum(counts.values())
        return -sum((c / total) * log2(c / total) for c in counts.values())

    return H(tn, tp) - H(tp) - H(tn, tp, sc) + H(tp, sc)


@pytest.mark.parametrize("k", [2, 5, 10])
def test_symbolic_encoding_is_collision_free(k):
    """Joint symbol counting must be injective at every k.

    The old encoding multiplied by a multiplier that squared at each step, so
    the packed value reached (k!)^3 and exceeded int64 at k=10, wrapping
    silently. Wrapping is not the same as colliding -- no collisions occur on
    the shipped fixtures -- but the encoding gave no guarantee, so this checks
    the implementation against a reference that cannot overflow.
    """
    from pyspi.statistics.infotheory import SymbolicTECalculator

    rng = np.random.default_rng(0)
    src = rng.standard_normal(600)
    targ = np.roll(src, 1) + 0.5 * rng.standard_normal(600)

    calc = SymbolicTECalculator()
    calc.setProperty("k_HISTORY", str(k))
    calc.setObservations(src, targ)
    got = calc.computeAverageLocalOfObservations()

    expected = _reference_symbolic_te(src, targ, k)
    assert np.isclose(got, expected, rtol=1e-9, atol=1e-12), (
        f"Symbolic TE at k={k} disagrees with a tuple-keyed reference: "
        f"{got!r} vs {expected!r}."
    )


# --------------------------------------------------------------------------
# KSG preconditions
# --------------------------------------------------------------------------

# NOTE: parse_bivariate's signature is (self, data, data2=None, i=None, j=None),
# so bivariate(data, 0, 1) binds data2=0, i=1 and fails with an unrelated
# dimension error. Always pass i/j by keyword here, or these tests pass for the
# wrong reason. (The positional foot-gun is a usability issue in its own right.)

@pytest.mark.parametrize("k", [30, 100])
def test_ksg_rejects_k_at_or_above_sample_size(k):
    small = _data(m=2, t=20)
    spi = it.MutualInfo(estimator="kraskov", prop_k=k)
    with pytest.raises(ValueError):
        spi.bivariate(small, i=0, j=1)


def test_ksg_rejects_degenerate_samples():
    const = np.zeros((2, 200))
    const[1] = np.arange(200)
    data = Data(data=const, dim_order="ps", zscore=False)
    with pytest.raises(ValueError):
        it.MutualInfo(estimator="kraskov").bivariate(data, i=0, j=1)


# --------------------------------------------------------------------------
# Directed information
# --------------------------------------------------------------------------

def _di_system(phi, c, T=4000, seed=0):
    """y_t = phi*y_{t-1} + c*x_{t-1} + e_t, with x i.i.d."""
    r = np.random.default_rng(seed)
    x = r.standard_normal(T)
    e = r.standard_normal(T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = phi * y[t - 1] + c * x[t - 1] + e[t]
    return Data(data=np.vstack([x, y]), dim_order="ps", zscore=True)


@pytest.mark.parametrize("phi", [0.0, 0.6, 0.95])
def test_directed_info_is_zero_for_an_independent_source(phi):
    """DI(X->Y) must not grow with the target's own autocorrelation.

    The previous implementation summed H(Y^i)/i and subtracted causal entropy,
    which is not Massey's definition: with an independent source it returned
    0.007 at phi=0 and 1.53 at phi=0.95, i.e. it measured how predictable the
    target was from its own past.
    """
    di = it.DirectedInfo(estimator="gaussian").bivariate(_di_system(phi, 0.0), i=0, j=1)
    assert abs(di) < 0.02, (
        f"DI with an independent source is {di:.5f} at phi={phi}; it must be ~0 "
        f"regardless of the target's autocorrelation."
    )


def test_directed_info_matches_the_analytic_gaussian_value():
    """With phi=0 and lag-1 coupling, DI over horizon n=2 is 0.5*ln(1+c^2)."""
    for c in (0.5, 1.0):
        r = np.random.default_rng(1)
        T = 200_000
        x = r.standard_normal(T)
        e = r.standard_normal(T)
        y = np.zeros(T)
        y[1:] = c * x[:-1] + e[1:]
        d = Data(data=np.vstack([x, y]), dim_order="ps", zscore=True)
        got = it.DirectedInfo(estimator="gaussian", n=2).bivariate(d, i=0, j=1)
        expected = 0.5 * np.log(1 + c ** 2)
        assert abs(got - expected) < 5e-3, (
            f"DI={got:.6f} vs analytic {expected:.6f} for c={c}."
        )


def test_directed_info_is_directional():
    d = _di_system(0.5, 1.0)
    fwd = it.DirectedInfo(estimator="gaussian").bivariate(d, i=0, j=1)
    rev = it.DirectedInfo(estimator="gaussian").bivariate(d, i=1, j=0)
    assert fwd > 20 * max(rev, 1e-6), f"DI(X->Y)={fwd:.5f} not >> DI(Y->X)={rev:.5f}"


def test_ksg_validation_reaches_the_transfer_entropy_path():
    """The TE path embeds first, so its usable N is smaller than len(targ)."""
    small = _data(m=2, t=20)
    with pytest.raises(ValueError):
        it.TransferEntropy(estimator="kraskov", prop_k=30).bivariate(small, i=0, j=1)


def test_ksg_rejects_negative_theiler_window():
    from pyspi.statistics.infotheory import _validate_ksg_sample
    with pytest.raises(ValueError, match="Theiler"):
        _validate_ksg_sample(200, 4, -5)


def test_ksg_rejects_tied_inputs():
    """Quantised/constant inputs give a zero k-th radius and a bogus negative CMI.

    Binary series previously returned TE = -2.36, for a quantity bounded below
    by zero.
    """
    const = np.zeros((2, 200))
    const[1] = np.arange(200)
    data = Data(data=const, dim_order="ps", zscore=False)
    with pytest.raises(ValueError):
        it.TransferEntropy(estimator="kraskov").bivariate(data, i=0, j=1)


def test_directed_info_kraskov_matches_the_analytic_value():
    """The direct CMI estimator must hit the same closed form as Gaussian.

    DI composed from separate entropies cannot: kernel sat near +4 on
    independent data at every T tested (100 to 8000), because a fixed-bandwidth
    estimator's bias in ~11 dimensions does not shrink with sample size.
    """
    c = 1.0
    r = np.random.default_rng(1)
    T = 4000
    x = r.standard_normal(T)
    e = r.standard_normal(T)
    y = np.zeros(T)
    y[1:] = c * x[:-1] + e[1:]
    d = Data(data=np.vstack([x, y]), dim_order="ps", zscore=True)

    got = it.DirectedInfo(estimator="kraskov", n=2).bivariate(d, i=0, j=1)
    expected = 0.5 * np.log(1 + c ** 2)
    assert abs(got - expected) < 0.05, f"kraskov DI={got:.4f} vs analytic {expected:.4f}"


@pytest.mark.parametrize("T", [200, 1000])
def test_directed_info_kraskov_is_zero_for_independent_source(T):
    r = np.random.default_rng(0)
    d = Data(data=r.standard_normal((2, T)), dim_order="ps", zscore=True)
    di = it.DirectedInfo(estimator="kraskov").bivariate(d, i=0, j=1)
    assert abs(di) < 0.15, f"kraskov DI={di:.4f} on independent data at T={T}"
