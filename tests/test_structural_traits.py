"""Red tests: declared structural traits must match observed behaviour.

Two separate problems live here.

**AEG.** ``Cointegration`` is declared ``Undirected``, but the ``aeg`` method
computes ``statsmodels.tsa.stattools.coint(z[i], z[j])``, which is *not*
symmetric in its arguments. The cache then writes the single computed value to
both ``(i, j)`` and ``(j, i)``. So the reported value for a pair depends on
which orientation happened to be computed first, i.e. on process order. That is
a scientific-semantics question, not a rounding artifact: either AEG is
directed and must be labelled and stored as such, or a symmetric definition
(e.g. min/max over both orientations) must be chosen and documented.

**Trait/label agreement.** Labels are used for filtering and for grouping in
analyses, so a class declaring ``undirected`` while producing an asymmetric
matrix silently corrupts downstream selection. The audit below reads the
committed baseline matrices, so it costs no computation.

See tests/test_state_integrity.py for the xfail(strict=True) rationale.
"""
import os

import numpy as np
import pytest

from pyspi.calculator import load_spis_from_yaml, resolve_config
from pyspi.data import Data
from pyspi.statistics.misc import Cointegration

BASELINE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "baselines", "var1_M3_T100.npz"
)


def _offdiag(a):
    m = a.shape[0]
    return a[~np.eye(m, dtype=bool)]


# --------------------------------------------------------------------------
# AEG semantics
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="AEG is asymmetric but cached to both orientations")
def test_aeg_value_is_independent_of_process_order():
    """Permuting the input processes must not change a pair's AEG value."""
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((3, 200)).cumsum(axis=1)  # integrated series

    spi = Cointegration(method="aeg", statistic="tstat")

    forward = spi.multivariate(Data(data=arr, dim_order="ps", zscore=False))

    perm = [2, 1, 0]
    inv = np.argsort(perm)
    permuted = Cointegration(method="aeg", statistic="tstat").multivariate(
        Data(data=arr[perm], dim_order="ps", zscore=False)
    )
    restored = permuted[np.ix_(inv, inv)]

    assert np.allclose(_offdiag(forward), _offdiag(restored), equal_nan=True), (
        "AEG changed under a permutation of the processes: the cached value "
        "depends on which orientation was computed first."
    )


@pytest.mark.xfail(strict=True, reason="AEG declares Undirected but coint(x,y) != coint(y,x)")
def test_aeg_declared_symmetry_matches_the_statistic():
    """If AEG is labelled undirected, the underlying statistic must be symmetric."""
    from statsmodels.tsa import stattools

    rng = np.random.default_rng(1)
    x = rng.standard_normal(200).cumsum()
    y = rng.standard_normal(200).cumsum()

    fwd = stattools.coint(x, y, autolag="aic", maxlag=10, trend="c")[0]
    rev = stattools.coint(y, x, autolag="aic", maxlag=10, trend="c")[0]

    spi = Cointegration(method="aeg")
    declared_undirected = "undirected" in getattr(spi, "labels", [])

    if declared_undirected:
        assert np.isclose(fwd, rev), (
            f"Cointegration(method='aeg') is labelled undirected, but the AEG "
            f"t-statistic is asymmetric: coint(x,y)={fwd:.6g} vs coint(y,x)={rev:.6g}."
        )


# --------------------------------------------------------------------------
# Declared labels vs observed matrices
# --------------------------------------------------------------------------

def _classify(mat):
    """Classify an MxM matrix as symmetric / antisymmetric / asymmetric.

    Antisymmetry (``A[i,j] == -A[j,i]``) is a genuine third category, not a
    broken form of either other one: phase-based measures such as PLI, wPLI and
    PSI carry a sign that encodes lead/lag. The label vocabulary currently has
    no word for it, which is why they show up as "undirected but asymmetric".
    """
    finite = np.isfinite(mat)
    pairwise = finite & finite.T & ~np.eye(mat.shape[0], dtype=bool)
    if not pairwise.any():
        return "undetermined"
    a, at = mat[pairwise], mat.T[pairwise]
    if np.allclose(a, at, rtol=1e-9, atol=1e-12):
        # A constant (e.g. identically zero) matrix is vacuously symmetric;
        # calling it "symmetric" would mask a degenerate estimator.
        return "degenerate" if np.ptp(a) == 0 else "symmetric"
    if np.allclose(a, -at, rtol=1e-9, atol=1e-12):
        return "antisymmetric"
    return "asymmetric"


def _label_symmetry_audit():
    """Return {identifier: (declared, observed)} disagreements from baselines."""
    z = np.load(BASELINE, allow_pickle=False)
    spis = load_spis_from_yaml(resolve_config("full"), quiet=True)

    disagreements = {}
    for ident, spi in spis.items():
        if ident not in z.files:
            continue
        mat = np.asarray(z[ident], dtype=float)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            continue

        observed = _classify(mat)
        if observed in ("undetermined", "degenerate"):
            continue  # covered by the degeneracy tests, not by this one

        labels = set(getattr(spi, "labels", []) or [])
        if "undirected" in labels and observed == "asymmetric":
            disagreements[ident] = ("undirected", observed)
        elif "directed" in labels and observed == "symmetric":
            disagreements[ident] = ("directed", observed)
    return disagreements


@pytest.mark.xfail(strict=True, reason="several classes' declared symmetry contradicts their output")
def test_declared_symmetry_matches_observed_matrices():
    bad = _label_symmetry_audit()
    assert not bad, "Declared/observed symmetry disagreements:\n" + "\n".join(
        f"  {k}: declared {v[0]}, observed {v[1]}" for k, v in sorted(bad.items())
    )


@pytest.mark.xfail(strict=True, reason="no 'antisymmetric' label exists; these are labelled undirected")
def test_antisymmetric_measures_are_labelled_as_such():
    """PLI/wPLI/PSI encode lead-lag in their sign; 'undirected' misdescribes them."""
    z = np.load(BASELINE, allow_pickle=False)
    spis = load_spis_from_yaml(resolve_config("full"), quiet=True)

    mislabelled = []
    for ident, spi in spis.items():
        if ident not in z.files:
            continue
        if _classify(np.asarray(z[ident], dtype=float)) != "antisymmetric":
            continue
        labels = set(getattr(spi, "labels", []) or [])
        if "antisymmetric" not in labels:
            mislabelled.append(f"{ident} (labelled: {sorted(labels & {'directed', 'undirected'})})")

    assert not mislabelled, (
        "Antisymmetric SPIs carry no 'antisymmetric' label:\n  "
        + "\n  ".join(sorted(mislabelled))
    )


@pytest.mark.xfail(strict=True, reason="degenerate SPIs return a constant matrix and ship anyway")
def test_no_bundled_spi_returns_a_constant_matrix():
    """A constant matrix carries no pairwise information."""
    z = np.load(BASELINE, allow_pickle=False)
    spis = load_spis_from_yaml(resolve_config("full"), quiet=True)

    degenerate = [
        ident for ident in spis
        if ident in z.files
        and _classify(np.asarray(z[ident], dtype=float)) == "degenerate"
    ]
    assert not degenerate, (
        "SPIs returning a constant matrix on var1_M3_T100: " + ", ".join(sorted(degenerate))
    )


@pytest.mark.xfail(strict=True, reason="ConditionalEntropy is implemented directed, labelled undirected")
def test_conditional_entropy_label_matches_implementation():
    spis = load_spis_from_yaml(resolve_config("full"), quiet=True)
    ce = {k: v for k, v in spis.items() if k.startswith("ce_")}
    assert ce, "Precondition: full config must contain ConditionalEntropy variants."

    mislabelled = [
        k for k, v in ce.items() if "undirected" in (getattr(v, "labels", []) or [])
    ]
    assert not mislabelled, (
        "ConditionalEntropy computes H(X|Y), which is directed, but these "
        f"variants are labelled undirected: {sorted(mislabelled)}"
    )
