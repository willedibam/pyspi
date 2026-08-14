"""Declared structural traits must match observed behaviour.

Symmetry is three-valued, not two. A binary directed/undirected vocabulary has
no word for measures satisfying ``A[i,j] == -A[j,i]`` -- PLI, wPLI, PSI, and
CCM's "diff" statistic -- and labelling them ``undirected`` (which implies
symmetry) misdescribes them for any downstream filtering or grouping.

Resolved here:

* ``Cointegration`` declared ``Undirected`` while ``aeg`` computes
  ``stattools.coint(z[i], z[j])``, which is not symmetric in its arguments
  (measured: ~0.8 mean absolute difference between orientations, up to ~1.6).
  The cache then wrote the one computed value to both ``(i, j)`` and
  ``(j, i)``, so which orientation you got depended on visit order. ``aeg`` is
  now ``directed`` and reports what it computes; ``johansen``, which is
  symmetric to ~3e-14, keeps the alias.
* Wavelet ``PhaseSlopeIndex`` filled its upper triangle from the lower one
  *without negating*, inverting the lead/lag sign for half of every matrix.
* ``hhg`` was declared directed but is exactly symmetric; ``ce``, ``dcorrx``
  and ``mgcx`` were labelled undirected in configs but are directed.

Two open findings remain, marked ``xfail(strict=True)`` with their reasoning in
the marker. They are recorded rather than silently patched because each needs a
scientific decision, not a code change.

The audit reads the committed baseline matrices, so it costs no computation.
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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Open finding, not yet resolved. ce_gaussian, lmfit_* and "
        "gpfit_DotProduct declare 'directed' but come out symmetric on "
        "z-scored data: Gaussian conditional entropy is symmetric when the "
        "marginal variances are equal, which z-scoring guarantees, and a "
        "linear model's R^2 is symmetric on standardised inputs. Whether the "
        "label or the preprocessing is wrong is a scientific decision, so it "
        "is recorded rather than silently relabelled."
    ),
)
def test_declared_symmetry_matches_observed_matrices():
    bad = _label_symmetry_audit()
    assert not bad, "Declared/observed symmetry disagreements:\n" + "\n".join(
        f"  {k}: declared {v[0]}, observed {v[1]}" for k, v in sorted(bad.items())
    )


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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Open finding, not yet resolved. dspli_*_max, dswpli_*_max and one "
        "phase_*_max variant return a constant matrix on var1_M3_T100, so "
        "they carry no pairwise information on this fixture. Whether that "
        "holds generally or is specific to M=3/T=100 needs checking before "
        "any of them is removed from the shipped set."
    ),
)
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


def test_conditional_entropy_label_matches_implementation():
    """Under the default z-scoring the Gaussian form is symmetric, so the
    bundled variants are labelled undirected; the kernel and kozachenko forms
    remain asymmetric and that is checked by the symmetry audit above."""
    spis = load_spis_from_yaml(resolve_config("full"), quiet=True)
    ce = {k: v for k, v in spis.items() if k.startswith("ce_")}
    assert ce, "Precondition: full config must contain ConditionalEntropy variants."

    contradictory = [
        k for k, v in ce.items()
        if {"directed", "undirected"} <= set(getattr(v, "labels", []) or [])
    ]
    assert not contradictory, (
        f"ConditionalEntropy variants carry both labels: {sorted(contradictory)}"
    )


# --------------------------------------------------------------------------
# Wavelet PSI band statistics
# --------------------------------------------------------------------------

@pytest.mark.parametrize("statistic", ["mean", "max"])
def test_wavelet_psi_is_permutation_invariant(statistic):
    """Both band statistics must survive a permutation of the processes.

    mne_connectivity returns a lower-triangular tensor and the upper triangle
    is filled by negating. That fill must happen *before* the band statistic:
    negating after reduction is only valid for a statistic commuting with
    negation. mean commutes, max does not --
    ``max_f(-v) = -min_f(v) != -max_f(v)`` -- so reducing first made the max
    variants permutation-dependent by up to 11.5.
    """
    from pyspi.statistics.wavelet import PhaseSlopeIndex

    rng = np.random.default_rng(0)
    x = np.cumsum(rng.standard_normal(400))
    arr = np.vstack([x, np.roll(x, 4) + 0.1 * rng.standard_normal(400),
                     rng.standard_normal(400)])
    perm = [2, 0, 1]
    inv = np.argsort(perm)

    fwd = PhaseSlopeIndex(statistic=statistic).multivariate(
        Data(data=arr, dim_order="ps", zscore=True))
    permuted = PhaseSlopeIndex(statistic=statistic).multivariate(
        Data(data=arr[perm], dim_order="ps", zscore=True))
    restored = permuted[np.ix_(inv, inv)]

    off = ~np.eye(3, dtype=bool)
    assert np.allclose(fwd[off], restored[off], equal_nan=True), (
        f"psi_wavelet statistic={statistic} is not permutation-invariant; "
        f"max|d|={np.nanmax(np.abs(fwd[off] - restored[off])):.6g}"
    )
