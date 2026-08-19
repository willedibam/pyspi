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


@pytest.mark.parametrize("w", [0, 1, 3, 10])
def test_ksg_cmi_reduces_to_mi_when_conditioning_set_is_empty(w):
    """I(A;B|nothing) is I(A;B), at every Theiler window.

    The empty-C branch used to fake the conditioning count as a constant
    N-(2w+1). That matched the MI estimator only at w=0 and drifted with the
    window (0.005 at w=1, 0.051 at w=10). DirectedInfo's first term has an
    empty history, so this is on the shipped path whenever a Theiler window is
    configured.
    """
    from pyspi.statistics.infotheory import _ksg_cmi, _ksg_mi_general

    r = np.random.default_rng(0)
    n = 400
    a = r.standard_normal((n, 1))
    b = 0.6 * a + 0.8 * r.standard_normal((n, 1))
    empty = np.empty((n, 0))

    assert _ksg_cmi(a, b, empty, 4, w) == pytest.approx(
        _ksg_mi_general(a, b, 4, w), abs=1e-12
    )


def test_conditional_entropy_is_directed():
    """H(X|Y) != H(Y|X): the label describes the measure, not one estimator.

    The Gaussian form is symmetric under the default z-scoring only because
    equal marginal variances make it so; kozachenko and kernel are asymmetric
    even there, and Gaussian becomes asymmetric with zscore=False.
    """
    data = _data(m=3, t=200)
    asym = {}
    for est in ("gaussian", "kozachenko", "kernel"):
        A = np.asarray(it.ConditionalEntropy(estimator=est).multivariate(data))
        off = ~np.eye(3, dtype=bool)
        asym[est] = float(np.nanmax(np.abs(A - A.T)[off]))

    assert max(asym.values()) > 1e-6, f"no estimator is asymmetric: {asym}"
    for est in ("gaussian", "kozachenko", "kernel"):
        spi = it.ConditionalEntropy(estimator=est)
        assert "directed" in spi.labels, f"{est} lost the directed label"
        assert "undirected" not in spi.labels


def test_wilson_non_convergence_is_reported_not_swallowed():
    """A failed spectral factorisation must reach the caller's warnings.

    Wilson's algorithm is iterative and, on hitting its iteration cap, reports
    "Maximum iterations reached. N of M converged" through
    ``logging.Logger.warning`` and returns the unconverged factor anyway. Every
    Wilson-derived measure (DC, DTF, dDTF, PDC, gPDC, nonparametric spectral
    GC) is built from that factor.

    pyspi collects per-SPI diagnostics from the ``warnings`` channel only, so
    before the bridge in ``statistics/spectral.py`` those numbers reached the
    results table with nothing recorded against them. This is not hypothetical:
    it fires on a *bundled* fixture. Same class of defect as the six SPIs above
    -- a value that is quietly not what it claims to be.
    """
    import os
    import warnings

    from pyspi.data import Data
    from pyspi.statistics.spectral import DirectedCoherence

    fixture = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "data", "fixtures", "kuramoto_M7_T100.npy")
    data = Data(data=fixture, dim_order="sp")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        DirectedCoherence(statistic="mean", fmin=0, fmax=0.5).multivariate(data)

    messages = [str(w.message) for w in caught]
    assert any("Maximum iterations reached" in m for m in messages), (
        "the backend's factorisation-convergence warning was swallowed; "
        f"caught instead: {messages}"
    )
def test_ccm_auto_embedding_maximises_skill_rather_than_returning_max_e():
    """``E=None`` must select an embedding, not return the largest candidate.

    The call site read the winner as ``pyEDM.EmbedDimension(...).max()["E"]``.
    ``DataFrame.max()`` reduces column-wise, so that is the largest *candidate*
    E -- pyEDM's ``maxE`` default of 10 -- for every process on every dataset.
    The three shipped ``ccm_E-None_*`` SPIs were consequently bit-identical to
    ``ccm_E-10_*`` on all three frozen fixtures while advertising an inferred
    embedding: the identifier said one thing and the number was another.

    This pins the replacement against pyEDM's own per-E skill, and pins that
    the answer is data-dependent rather than the constant it used to be.
    """
    import os

    import pandas as pd
    import pyEDM

    from pyspi.data import Data
    from pyspi.statistics.causal import _optimal_embedding_dimension

    fixture = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "data", "fixtures", "var1_M3_T100.npy")
    z = Data(data=fixture, dim_order="sp").to_numpy(squeeze=True)
    M, N = z.shape
    df = pd.DataFrame(
        np.concatenate([np.atleast_2d(np.arange(N)), z]).T,
        columns=["index"] + [f"proc{p}" for p in range(M)],
    )
    lib_pred = f"10 {N - 10}"

    chosen = []
    for i in range(M):
        col = df.columns.values[i + 1]
        reference = pyEDM.EmbedDimension(dataFrame=df, lib=lib_pred, pred=lib_pred,
                                         columns=col, target=col, showPlot=False,
                                         numProcess=1)
        expected = int(reference.loc[reference["rho"].idxmax(), "E"])
        got = _optimal_embedding_dimension(df, col, lib_pred)
        assert got == expected, (
            f"{col}: chose E={got}, pyEDM's skill curve peaks at E={expected}"
        )
        chosen.append(got)

    assert any(E != 10 for E in chosen), (
        f"every process selected the maximum candidate E ({chosen}); that is "
        f"the symptom of reading max(E) instead of argmax(rho)"
    )


# ---------------------------------------------------------------------------
# KSG input conditioning: JIDT's NORMALISE and NOISE_LEVEL_TO_ADD
# ---------------------------------------------------------------------------

def _correlated_pair(n=2000, rho=0.8, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    return z, rho * z + np.sqrt(1 - rho ** 2) * rng.standard_normal(n)


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_ksg_mi_is_invariant_to_per_coordinate_rescaling(scale):
    """MI is invariant under any smooth invertible marginal transform.

    The KSG estimator's L-infinity neighbour radius is not, which is why JIDT
    normalises each column by default (``normalise = true`` on
    MutualInfoMultiVariateCommon). Without it, scaling one member of a
    correlated Gaussian pair by 1e-3 or 1e3 collapsed the estimate from 0.49 to
    0.05 and 0.06 against a true MI of 0.51.
    """
    from pyspi.statistics.infotheory import _ksg_mi_pair

    x, y = _correlated_pair()
    analytic = -0.5 * np.log(1 - np.corrcoef(x, y)[0, 1] ** 2)
    got = _ksg_mi_pair(x, scale * y, 4, 0)
    assert abs(got - analytic) < 0.05, (
        f"MI = {got:.4f} at scale {scale:g}; analytic {analytic:.4f}"
    )


@pytest.mark.parametrize("w", [0, 5])
@pytest.mark.parametrize("levels", [2, 4, None])
def test_ksg_mi_of_independent_quantised_marginals_is_near_zero(levels, w):
    """Ties must not be read as dependence.

    Every kth-nearest-neighbour radius is zero on quantised data, so the
    digamma counts saturate on the tie structure. Measured before the 1e-8
    dither JIDT adds by default: independent binary marginals (N=400, k=4)
    gave -3.35, four-level -1.96, and one-decimal-rounded Gaussians *+0.90* --
    a confident false positive on independent data.
    """
    from pyspi.statistics.infotheory import _ksg_mi_pair

    rng = np.random.default_rng(0)
    def draw():
        if levels is None:
            return np.round(rng.standard_normal(400), 1)
        return rng.integers(0, levels, 400).astype(float)

    mi = _ksg_mi_pair(draw(), draw(), 4, w)
    assert abs(mi) < 0.1, f"independent quantised marginals gave MI = {mi:+.4f}"


def test_ksg_mi_recovers_the_discrete_mutual_information_of_tied_data():
    """Dither is not just a tie-breaker; the limit is the right one.

    For independent additive noise, I(X + e*xi; Y + e*eta) -> I(X; Y) as
    e -> 0, so the dithered estimate targets the discrete MI rather than a
    quantisation artefact. Checked against the plug-in estimate on the same
    sample, so the comparison is not confounded by sampling error.
    """
    from pyspi.statistics.infotheory import _ksg_mi_pair

    rng = np.random.default_rng(1)
    x = (rng.random(4000) < 0.5)
    y = np.where(rng.random(4000) < 0.8, x, ~x)

    joint = np.array([[np.mean((x == a) & (y == b)) for b in (False, True)]
                      for a in (False, True)])
    px, py = joint.sum(1), joint.sum(0)
    plug_in = float(np.sum(joint * np.log(joint / np.outer(px, py))))

    got = _ksg_mi_pair(x.astype(float), y.astype(float), 4, 0)
    assert abs(got - plug_in) < 0.05, f"KSG {got:.4f} vs plug-in {plug_in:.4f}"


def test_ksg_dither_is_reproducible_and_independent_of_call_context():
    """The dither must be a pure function of the series.

    A per-call RNG would make ``bivariate(data, i, j)`` disagree with
    ``multivariate(data)[i, j]``, break serial/parallel equality, and put a
    stochastic term in every frozen baseline. The seed is derived from a digest
    of the (normalised) column instead, so the same series always draws the
    same noise however many processes it is passed alongside.
    """
    import pyspi.statistics.infotheory as it
    from pyspi.data import Data

    rng = np.random.default_rng(2)
    Z = rng.standard_normal((4, 300))
    data = Data(data=Z, dim_order="ps", zscore=False)

    for cls in (it.MutualInfo, it.TimeLaggedMutualInfo):
        spi = cls(estimator="kraskov")
        table = spi.multivariate(data)
        assert spi.bivariate(data, i=0, j=2) == table[0, 2]
        assert np.array_equal(spi.multivariate(data), table, equal_nan=True)


def test_kraskov_spis_are_finite_on_the_quantised_bundled_dataset():
    """`forex` has a process with 24 distinct values in 250 samples.

    That is exactly the regime where zero neighbour radii used to dominate, and
    it ships with the package, so it is a fixture rather than a hypothetical.
    """
    import pyspi.statistics.infotheory as it
    from pyspi.data import load_dataset

    data = load_dataset("forex")
    for spi in (it.MutualInfo(estimator="kraskov"),
                it.TimeLaggedMutualInfo(estimator="kraskov"),
                it.TransferEntropy(estimator="kraskov"),
                it.DirectedInfo(estimator="kraskov")):
        table = spi.multivariate(data)
        off = ~np.eye(table.shape[0], dtype=bool)
        assert np.isfinite(table[off]).all(), f"{spi.identifier} has non-finite values"
        assert np.abs(table[off]).max() < 10, f"{spi.identifier} is implausibly large"


def test_kozachenko_entropy_still_refuses_tied_data_rather_than_dithering():
    """A deliberate divergence from JIDT, and the reason is not stylistic.

    JIDT dithers its Kozachenko calculator with the same 1e-8 it uses for KSG.
    That is safe for mutual information, whose dithered limit is the discrete
    value, but not for differential entropy: H(X + e*xi) -> -inf as e -> 0 for
    discrete X, so a dithered estimate on quantised data reports the dither
    level. pyspi names the problem instead of returning a number set by an
    implementation constant.
    """
    import pyspi.statistics.infotheory as it
    from pyspi.data import load_dataset

    with pytest.raises(ValueError, match="tied observations"):
        it.JointEntropy(estimator="kozachenko").multivariate(load_dataset("forex"))
