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
    from math import log

    from pyspi.statistics.infotheory import _series_to_ordinal_symbols

    s = _series_to_ordinal_symbols(np.asarray(src, dtype=float), k)
    t = _series_to_ordinal_symbols(np.asarray(targ, dtype=float), k)
    n = min(len(s), len(t)) - 1
    tn, tp, sc = t[1:n + 1], t[:n], s[:n]

    def H(*cols):
        counts = Counter(zip(*(list(map(int, c)) for c in cols)))
        total = sum(counts.values())
        # nats, matching the module-wide convention (see InfoTheoryBase).
        return -sum((c / total) * log(c / total) for c in counts.values())

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


# ---------------------------------------------------------------------------
# TransferEntropy embedding parameters
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("estimator,extra", [
    ("symbolic", {"k_tau": 2}),
    ("symbolic", {"l_history": 2}),
    ("symbolic", {"l_tau": 2}),
    ("kernel", {"l_history": 1}),
    ("kernel", {"k_tau": 1}),
])
def test_transfer_entropy_refuses_embedding_parameters_it_does_not_implement(
        estimator, extra):
    """The identifier must not advertise an embedding that was never applied.

    `SymbolicTECalculator` reads only `k_HISTORY` and uses that one ordinal
    pattern length for source *and* destination, at unit delay -- which is how
    Staniek & Lehnertz (2008) define it. It nonetheless accepted `k_tau`,
    `l_history` and `l_tau`, stored them, ignored them, and wrote them into the
    identifier: `te_symbolic_k-3_kt-1_l-1_lt-1` claimed a destination history of
    3 against a source history of 1 while computing 3 for both. The kernel
    calculator has the same single-history contract and accepted them too,
    silently. Symbolic identifiers are now `te_symbolic_k-<k>`, as the kernel
    ones already were.
    """
    import pyspi.statistics.infotheory as it

    with pytest.raises(ValueError, match="not used by estimator"):
        it.TransferEntropy(estimator=estimator, k_history=2, **extra)


@pytest.mark.parametrize("bad", [
    {"k_history": 0}, {"k_history": -1},
    {"k_tau": 0}, {"l_history": 0}, {"l_tau": -2},
])
def test_transfer_entropy_rejects_non_positive_embedding_parameters(bad):
    """Validated at the API boundary, not discovered as an empty embedding."""
    import pyspi.statistics.infotheory as it

    with pytest.raises(ValueError, match=">= 1"):
        it.TransferEntropy(estimator="gaussian", **({"k_history": 1} | bad))


def test_symbolic_transfer_entropy_identifier_matches_what_is_computed():
    import pyspi.statistics.infotheory as it

    assert it.TransferEntropy(estimator="symbolic",
                              k_history=3).identifier == "te_symbolic_k-3"


# ---------------------------------------------------------------------------
# CrossCorrelation
# ---------------------------------------------------------------------------

def _xcorr_data(zscore=True, seed=0, T=500):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(T)
    b = np.r_[0.0, a[:-1]] + 0.1 * rng.standard_normal(T)   # b lags a by 1
    from pyspi.data import Data
    return Data(data=np.vstack([a, b]), dim_order="ps", zscore=zscore)


def test_cross_correlation_of_a_series_with_itself_is_one():
    """A correlation, so the self-pair must be exactly 1 and nothing exceeds it.

    `correlate(x, y) / x.std() / y.std() / (T - 1)` is neither the biased
    (divide by T) nor the unbiased (divide by T - |l|) normalisation, and it
    put the zero lag of a series against itself at T/(T-1): exactly 1.1111 for
    T = 10.
    """
    from pyspi.data import Data
    from pyspi.statistics.basic import CrossCorrelation

    x = np.arange(10.0)
    for zscore in (True, False):
        data = Data(data=np.vstack([x, x.copy()]), dim_order="ps", zscore=zscore)
        got = CrossCorrelation(statistic="max", sigonly=False).bivariate(
            data, i=0, j=1)
        assert got == pytest.approx(1.0, abs=1e-12), f"zscore={zscore}: {got}"


def test_cross_correlation_demeans_and_stays_bounded():
    """The correlate call used the raw series while the divisor demeaned.

    On `arange(10)` against itself with zscore=False that mismatch returned
    3.8384 -- for a quantity whose range is [-1, 1].
    """
    from pyspi.data import Data
    from pyspi.statistics.basic import CrossCorrelation

    rng = np.random.default_rng(1)
    u = rng.standard_normal(300) + 50.0        # large offset, small variance
    v = 0.5 * u + rng.standard_normal(300)
    data = Data(data=np.vstack([u, v]), dim_order="ps", zscore=False)

    spi = CrossCorrelation(statistic="max", sigonly=False)
    assert abs(spi.bivariate(data, i=0, j=1)) <= 1.0
    lags = data.xcorr[(0, 1)]
    assert np.abs(lags).max() <= 1.0
    # Zero lag is Pearson's r by construction.
    assert lags[len(lags) // 2] == pytest.approx(np.corrcoef(u, v)[0, 1], abs=1e-12)


@pytest.mark.parametrize("statistic", ["max", "mean"])
@pytest.mark.parametrize("sigonly", [True, False])
def test_cross_correlation_is_symmetric_in_its_arguments(statistic, sigonly):
    """It is declared undirected, so both orientations must agree.

    Two independent reasons they did not. The lag window
    `r_full[T - T//4 : T + T//4]` was centred on index T, but zero lag sits at
    T - 1, so the window was asymmetric by one lag; and the cached opposite
    orientation was `data.xcorr[(j,i)] = data.xcorr[(i,j)]` rather than its
    reverse, since r_yx(l) = r_xy(-l). The `sigonly` truncation then walked
    outwards from the centre, which on a pair where i leads j by one sample
    (r(0) already insignificant) extended one way and not the other: measured
    0.9957 against -0.0202.
    """
    from pyspi.statistics.basic import CrossCorrelation

    data = _xcorr_data()
    spi = CrossCorrelation(statistic=statistic, sigonly=sigonly)
    assert spi.bivariate(data, i=0, j=1) == pytest.approx(
        spi.bivariate(data, i=1, j=0), abs=1e-12)


def test_cross_correlation_cache_stores_the_reversed_lag_profile():
    from pyspi.statistics.basic import CrossCorrelation

    data = _xcorr_data()
    CrossCorrelation(sigonly=False).bivariate(data, i=0, j=1)
    assert np.array_equal(data.xcorr[(0, 1)], data.xcorr[(1, 0)][::-1])
    # Odd length, so the centre index is exactly the zero lag.
    assert len(data.xcorr[(0, 1)]) % 2 == 1


def test_cross_correlation_significance_band_scales_with_the_sample_size():
    """1.96/sqrt(T), not 1.96/sqrt(T//4).

    The old threshold was computed from the half-width of the lag *window*, so
    it was twice too wide and moved if the lag cut changed rather than if the
    record length did.
    """
    from pyspi.data import Data
    from pyspi.statistics.basic import CrossCorrelation

    rng = np.random.default_rng(2)
    T = 4000
    data = Data(data=rng.standard_normal((2, T)), dim_order="ps")
    spi = CrossCorrelation(statistic="max", sigonly=True)
    spi.bivariate(data, i=0, j=1)
    lags = data.xcorr[(0, 1)]
    # Under independence essentially nothing should clear 1.96/sqrt(T); the old
    # band (1.96/sqrt(T//4) = 2x wider) let through even fewer, masking that the
    # nominal 5% level was never being applied.
    assert (np.abs(lags) > 1.96 / np.sqrt(T)).mean() < 0.15


# ---------------------------------------------------------------------------
# Parameter/identifier/cache-key coverage
# ---------------------------------------------------------------------------

def test_dyn_corr_excl_value_reaches_the_identifier_and_the_cache_key():
    """`_DCE` alone named three different Theiler windows.

    dyn_corr_excl=5, =10 and ="AUTO" all produced `mi_kraskov_NN-4_DCE` and the
    same `_getkey()`, so a config setting two of them collided silently.
    """
    import pyspi.statistics.infotheory as it

    spis = [it.MutualInfo(estimator="kraskov", dyn_corr_excl=v)
            for v in (5, 10, "AUTO")]
    assert len({s.identifier for s in spis}) == 3, [s.identifier for s in spis]
    assert len({s._getkey() for s in spis}) == 3


def test_crossmap_entropy_embedding_dimension():
    """`history_length=k` gives k-1 source lags and a k-column joint space.

    Pinned rather than corrected: both readings of the parameter are internally
    consistent, cross-map entropy has no canonical published definition to
    arbitrate between them, and re-picking one would change every `xme_*` value
    on a guess about intent. See the class docstring.
    """
    import pyspi.statistics.infotheory as it
    from pyspi.data import Data

    k = 6
    rng = np.random.default_rng(0)
    data = Data(data=rng.standard_normal((2, 200)), dim_order="ps")

    seen = {}
    spi = it.CrossmapEntropy(history_length=k, estimator="gaussian")
    real_initialise = spi._entropy_calc.initialise

    def record(d):
        seen.setdefault("dims", []).append(d)
        return real_initialise(d)

    spi._entropy_calc.initialise = record
    spi.bivariate(data, i=0, j=1)
    assert seen["dims"] == [k, k - 1], seen["dims"]


def test_cointegration_aeg_tstat_is_signed_and_johansen_is_not():
    """The Engle-Granger t-statistic's sign is the finding, not noise.

    Reported as unsigned it went through `Calculator._rmmin`, which shifts the
    column by its minimum, and through `set_group`'s `abs()`. Johansen's trace
    and maximum-eigenvalue statistics are non-negative and stay unsigned.
    """
    from pyspi.statistics.misc import Cointegration

    assert Cointegration(method="aeg", statistic="tstat").issigned()
    assert not Cointegration(method="johansen", statistic="trace_stat").issigned()
    assert not Cointegration(method="johansen", statistic="max_eig_stat").issigned()


def test_itakura_dtw_normalisation_agrees_between_bivariate_and_multivariate():
    """The itakura branch of `multivariate` skipped the sqrt(T) division.

    The bivariate path and the dtaidistance path both apply it under
    `normalise=True`, so the two disagreed by a factor of sqrt(T) for that one
    constraint.
    """
    from pyspi.data import Data
    from pyspi.statistics.distance import DynamicTimeWarping

    rng = np.random.default_rng(0)
    data = Data(data=rng.standard_normal((3, 120)), dim_order="ps")
    spi = DynamicTimeWarping(global_constraint="itakura", normalise=True)
    assert spi.bivariate(data, i=0, j=1) == pytest.approx(
        spi.multivariate(data)[0, 1], rel=1e-12)


def test_rmse_naming_is_restricted_to_the_metric_it_describes():
    """`d/sqrt(T)` is a root *mean* square only for the Euclidean norm."""
    from pyspi.statistics.distance import PairwiseDistance

    assert PairwiseDistance(metric="euclidean",
                            normalise=True).identifier.endswith("_rmse")
    for metric in ("cityblock", "cosine", "canberra", "braycurtis"):
        identifier = PairwiseDistance(metric=metric, normalise=True).identifier
        assert not identifier.endswith("_rmse"), identifier


@pytest.mark.parametrize("kwargs,exc", [
    ({"i": 0}, ValueError),          # j omitted
    ({"j": 1}, ValueError),          # i omitted
    ({"i": 0, "j": 9}, IndexError),  # out of range
    ({"i": 0, "j": 1.5}, TypeError),
])
def test_bivariate_rejects_incomplete_or_invalid_indices(kwargs, exc):
    """`z[None]` is `np.newaxis`, not an error.

    A single index reached the SPI with the other left as None, and indexing
    the process array with None turned the "pair" into the whole (1, M, T)
    block -- so the SPI computed something with no relation to what was asked
    for, silently. The signature is `(data, data2, i, j)`, so
    `bivariate(data, 0, 3)` -- the obvious way to write it -- binds 0 to
    `data2` and 3 to `i`, and lands in exactly that state.
    """
    from pyspi.data import Data
    from pyspi.statistics.basic import CrossCorrelation

    rng = np.random.default_rng(0)
    data = Data(data=rng.standard_normal((4, 100)), dim_order="ps")
    with pytest.raises(exc):
        CrossCorrelation(sigonly=False).bivariate(data, **kwargs)


@pytest.mark.parametrize("kwargs", [
    {"estimator": "kernel", "kernel_width": 0},
    {"estimator": "kernel", "kernel_width": -1},
    {"estimator": "kraskov", "prop_k": 0},
    {"estimator": "kraskov", "dyn_corr_excl": -3},
])
def test_infotheory_parameters_are_validated_at_construction(kwargs):
    """A non-positive box-kernel half-width counts only the point itself.

    Every log ratio is then log(N) and the "estimate" is a constant; k < 1 has
    no kth neighbour at all. Both used to be discovered downstream, as a
    degenerate number rather than a rejected argument.
    """
    import pyspi.statistics.infotheory as it

    with pytest.raises(ValueError, match=">"):
        it.MutualInfo(**kwargs)


# ---------------------------------------------------------------------------
# KSG dither: one draw per coordinate occurrence
# ---------------------------------------------------------------------------

def _brute_force_ksg_mi(x, y, k):
    """O(N^2) transcription of KSG estimator 1, written from the paper.

    Independent of pyspi's cKDTree machinery: full pairwise L-infinity
    distances, the k-th smallest excluding self, strict marginal counts, and
    psi(k) - <psi(n_x+1) + psi(n_y+1)> + psi(N). Slow, so it is only used on
    small continuous fixtures -- but it shares no code with the implementation
    it checks.
    """
    from scipy.special import digamma

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    dx = np.abs(x[:, None] - x[None, :])
    dy = np.abs(y[:, None] - y[None, :])
    dz = np.maximum(dx, dy)
    np.fill_diagonal(dz, np.inf)
    eps = np.sort(dz, axis=1)[:, k - 1]
    n_x = (dx < eps[:, None]).sum(axis=1) - 1     # excludes self (dx == 0)
    n_y = (dy < eps[:, None]).sum(axis=1) - 1
    return float(digamma(k) - np.mean(digamma(n_x + 1) + digamma(n_y + 1))
                 + digamma(n))


def test_ksg_matches_an_independent_brute_force_reference():
    """Tie-free continuous data, where both implementations are unambiguous.

    The dither is 1e-8 standard deviations, so on tie-free data the two agree to
    far better than the estimator's own error; this pins the neighbour counting
    and the digamma assembly, not the tie policy.
    """
    from pyspi.statistics.infotheory import _knn_condition, _ksg_mi_pair

    rng = np.random.default_rng(7)
    n = 300
    z = rng.standard_normal(n)
    x, y = z, 0.7 * z + np.sqrt(1 - 0.49) * rng.standard_normal(n)

    for k in (1, 4, 10):
        conditioned = _knn_condition(np.column_stack([x, y]))
        expected = _brute_force_ksg_mi(conditioned[:, 0], conditioned[:, 1], k)
        assert _ksg_mi_pair(x, y, k, 0) == pytest.approx(expected, abs=1e-12), k


@pytest.mark.parametrize("n", [200, 400, 800, 1600])
def test_ksg_mi_of_a_variable_with_itself_is_its_discrete_entropy(n):
    """Duplicate columns must not share a dither.

    They did, so the dither never broke the degeneracy: the joint cloud
    collapsed onto the diagonal, every neighbour radius stayed at the dither
    scale, and the estimate reduced to psi(k) - 2*psi(k+1) + psi(N) -- a
    function of N and k with no dependence on the data. Measured 4.04, 4.73,
    5.43 and 6.12 nats at N = 200, 400, 800, 1600, growing by ln 2 per
    doubling, and *identical* (4.7341 at N=400) for binary, four-level and
    rounded-Gaussian duplicates.

    For a discrete X the finite oracle is exact: I(X + eps*a; X + eps*b) -> H(X)
    as eps -> 0 with independent a, b. No such oracle exists for identical
    *continuous* variables, whose differential mutual information is infinite,
    which is why this test is quantised.
    """
    from pyspi.statistics.infotheory import _ksg_mi_pair

    rng = np.random.default_rng(0)
    for levels in (2, 4):
        x = rng.integers(0, levels, n).astype(float)
        counts = np.bincount(x.astype(int), minlength=levels) / n
        entropy = float(-np.sum(counts[counts > 0] * np.log(counts[counts > 0])))
        got = _ksg_mi_pair(x, x.copy(), 4, 0)
        assert got == pytest.approx(entropy, abs=0.12), (
            f"{levels}-level duplicate at N={n}: {got:.4f} vs H(X)={entropy:.4f}")


def test_ksg_dither_is_symmetric_and_repeatable_including_duplicates():
    """Distinct columns are each replica 0, so the noise is content-determined.

    Identical columns take replicas 0 and 1 in slot order, and swapping the
    slots permutes identical values -- which the estimator cannot see. Both
    orders must therefore agree exactly, not approximately.
    """
    from pyspi.statistics.infotheory import _ksg_mi_pair

    rng = np.random.default_rng(3)
    z = rng.standard_normal(500)
    y = 0.6 * z + 0.8 * rng.standard_normal(500)
    b = (rng.random(400) < 0.5).astype(float)

    for u, v in ((z, y), (b, b.copy())):
        assert _ksg_mi_pair(u, v, 4, 0) == _ksg_mi_pair(v, u, 4, 0)
        assert _ksg_mi_pair(u, v, 4, 0) == _ksg_mi_pair(u, v, 4, 0)


def test_duplicated_processes_do_not_break_the_derived_ksg_measures():
    """TE, TLMI and DirectedInfo all route through the same conditioning.

    Each has an exactly-known answer when process 0 is duplicated at process 1
    and the series is i.i.d. binary: MI is H(X) = ln 2; TE(0->1) is 0, because
    the source adds nothing once the destination's own past is conditioned on;
    and directed information over n=5 terms is 5*H(X), since each
    I(X^i; Y_i | Y^{i-1}) reduces to H(Y_i) for an i.i.d. series.
    """
    import pyspi.statistics.infotheory as it
    from pyspi.data import Data

    rng = np.random.default_rng(0)
    x = (rng.random(400) < 0.5).astype(float)
    data = Data(data=np.vstack([x, x.copy(), rng.standard_normal(400)]),
                dim_order="ps", zscore=False)

    ln2 = np.log(2)
    assert it.MutualInfo(estimator="kraskov").multivariate(data)[0, 1] == \
        pytest.approx(ln2, abs=0.1)
    assert it.TransferEntropy(estimator="kraskov").multivariate(data)[0, 1] == \
        pytest.approx(0.0, abs=0.1)
    assert it.DirectedInfo(estimator="kraskov", n=5).multivariate(data)[0, 1] == \
        pytest.approx(5 * ln2, abs=0.5)


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_ksg_conditional_mi_is_invariant_to_per_coordinate_rescaling(scale):
    """The normalisation has to reach the conditioning set too, not just A and B."""
    from pyspi.statistics.infotheory import _ksg_cmi

    rng = np.random.default_rng(5)
    A = rng.standard_normal((500, 1))
    C = rng.standard_normal((500, 2))
    B = 0.6 * A + 0.4 * C[:, :1] + 0.5 * rng.standard_normal((500, 1))

    base = _ksg_cmi(A, B, C, 4, 0)
    assert _ksg_cmi(A * scale, B, C / scale, 4, 0) == pytest.approx(base, abs=1e-12)


def test_ksg_results_survive_the_parallel_boundary(tmp_path):
    """Serial and parallel must agree bit-for-bit; the dither uses no global RNG.

    Run on a dataset with a duplicated process, so the replica index -- the part
    of the dither key that is not simply the column's contents -- is exercised
    across the process boundary as well.
    """
    from pyspi.calculator import Calculator
    from pyspi.data import Data

    config = tmp_path / "ksg.yaml"
    config.write_text(
        ".statistics.infotheory:\n"
        "  MutualInfo:\n"
        "    labels: [infotheory]\n"
        "    configs:\n"
        "      - estimator: kraskov\n"
        "        prop_k: 4\n"
        "  TransferEntropy:\n"
        "    labels: [infotheory]\n"
        "    configs:\n"
        "      - estimator: kraskov\n"
        "        prop_k: 4\n"
        "  DirectedInfo:\n"
        "    labels: [infotheory]\n"
        "    configs:\n"
        "      - estimator: kraskov\n"
        "        prop_k: 4\n"
    )

    rng = np.random.default_rng(0)
    x = (rng.random(200) < 0.5).astype(float)
    dataset = np.vstack([x, x.copy(), rng.standard_normal(200)])

    tables = []
    for kwargs in ({"n_jobs": 1}, {"n_jobs": 2, "mp_context": "spawn"}):
        calc = Calculator(dataset=Data(data=dataset, dim_order="ps"),
                          config=str(config))
        calc.compute(**kwargs)
        tables.append({k: calc.table[k].to_numpy() for k in calc.spis})
    for key in tables[0]:
        assert np.array_equal(tables[0][key], tables[1][key], equal_nan=True), key


# ---------------------------------------------------------------------------
# Transfer-entropy auto-embedding: the support matrix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("estimator", ["kernel", "symbolic"])
@pytest.mark.parametrize("kwargs", [
    {"auto_embed_method": "MAX_CORR_AIS"},
    {"k_search_max": 5},
    {"tau_search_max": 3},
])
def test_auto_embedding_is_refused_by_the_estimators_that_cannot_do_it(
        estimator, kwargs):
    """No AIS criterion exists for the box-kernel or ordinal estimators.

    Both accepted `auto_embed_method` and the search bounds and then ran a fixed
    embedding, so the argument said an embedding had been selected and none had.
    """
    import pyspi.statistics.infotheory as it

    with pytest.raises(ValueError, match="not implemented for estimator"):
        it.TransferEntropy(estimator=estimator, k_history=2, **kwargs)


@pytest.mark.parametrize("fixed", ["k_history", "k_tau", "l_history", "l_tau"])
def test_full_max_corr_ais_refuses_a_fixed_embedding(fixed):
    """It selects all four, so accepting a fixed one would mean ignoring it."""
    import pyspi.statistics.infotheory as it

    with pytest.raises(ValueError, match="conflicts with auto_embed_method"):
        it.TransferEntropy(estimator="gaussian",
                           auto_embed_method="MAX_CORR_AIS", **{fixed: 2})


@pytest.mark.parametrize("fixed,allowed", [
    ("k_history", False), ("k_tau", False), ("l_history", True), ("l_tau", True),
])
def test_dest_only_refuses_a_fixed_destination_and_accepts_a_fixed_source(
        fixed, allowed):
    import pyspi.statistics.infotheory as it

    build = lambda: it.TransferEntropy(
        estimator="gaussian", auto_embed_method="MAX_CORR_AIS_DEST_ONLY",
        **{fixed: 2})
    if allowed:
        assert f"_{'l' if fixed == 'l_history' else 'lt'}-2" in build().identifier
    else:
        with pytest.raises(ValueError, match="conflicts with auto_embed_method"):
            build()


@pytest.mark.parametrize("bound", ["k_search_max", "tau_search_max"])
def test_search_bounds_are_refused_without_an_auto_method(bound):
    """They reached the identifier only under the auto branch, so with a fixed
    embedding they were accepted and silently discarded."""
    import pyspi.statistics.infotheory as it

    with pytest.raises(ValueError, match="requires auto_embed_method"):
        it.TransferEntropy(estimator="gaussian", **{bound: 5})


def test_auto_embedding_identifiers_name_the_method_and_resolved_bounds():
    """`MAX_CORR_AIS` changed meaning, so the identifier has to say which it is.

    It previously searched the destination only while carrying a name that
    denotes selection for both, and the identifier recorded neither. Defaults
    are resolved before the identifier is built, so an omitted bound still
    appears with the value actually used.
    """
    import pyspi.statistics.infotheory as it

    assert it.TransferEntropy(
        estimator="kraskov", auto_embed_method="MAX_CORR_AIS",
        k_search_max=10, tau_search_max=4
    ).identifier == "te_kraskov_NN-4_MAX-CORR-AIS_k-max-10_tau-max-4"
    # Omitted bounds resolve to 10 and 4 and are still named.
    assert it.TransferEntropy(
        estimator="kraskov", auto_embed_method="MAX_CORR_AIS"
    ).identifier == "te_kraskov_NN-4_MAX-CORR-AIS_k-max-10_tau-max-4"
    assert it.TransferEntropy(
        estimator="gaussian", auto_embed_method="MAX_CORR_AIS_DEST_ONLY",
        k_search_max=6, tau_search_max=2, l_history=3
    ).identifier == "gc_gaussian_MAX-CORR-AIS-DEST-ONLY_k-max-6_tau-max-2_l-3_lt-1"


@pytest.mark.parametrize("bad", [
    {"k_history": 2.0}, {"k_history": True}, {"k_tau": np.float64(1)},
    {"k_search_max": 1.5, "auto_embed_method": "MAX_CORR_AIS"},
])
def test_embedding_parameters_must_be_integral_and_not_boolean(bad):
    """`bool` subclasses `int`, so `k_history=True` would pass as 1, and
    `int(2.7)` silently truncates a parameter the caller meant otherwise."""
    import pyspi.statistics.infotheory as it

    with pytest.raises(TypeError, match="positive integer"):
        it.TransferEntropy(estimator="gaussian", **bad)


def test_auto_embedding_selection_is_cached_per_process():
    """One search per (process, estimator, bounds, Theiler window), not per pair."""
    import pyspi.statistics.infotheory as it
    from pyspi.data import Data

    rng = np.random.default_rng(0)
    data = Data(data=rng.standard_normal((4, 400)), dim_order="ps", zscore=False)
    spi = it.TransferEntropy(estimator="gaussian",
                             auto_embed_method="MAX_CORR_AIS",
                             k_search_max=4, tau_search_max=2)

    calls = []
    real = it._select_embedding
    it._select_embedding = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
    try:
        spi.multivariate(data)
    finally:
        it._select_embedding = real

    # 4 processes, 12 ordered pairs, 24 selections without caching.
    assert len(calls) == 4, calls
    assert len(data.ais_embedding) == 4
