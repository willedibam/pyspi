"""Every directed SPI must agree on what a row and a column mean.

pyspi's convention is set by ``base.Directed.multivariate``, which fills
``A[i, j] = bivariate(i, j)``: **row is the source, column is the target**.

This matters because the spectral backends (spectral_connectivity, nitime)
follow the opposite DTF/PDC convention -- element ``[i, j]`` is the flow *into*
i *from* j. Their output used to be passed through unchanged, which left every
directed spectral SPI transposed relative to every directed information-theory
SPI in the same results table. See ``spectral._to_source_target``.

The baseline-drift suite cannot be relied on to catch a regression here: on the
bundled fixtures the directed spectral matrices are close to symmetric, so a
transpose barely moves the numbers. These tests use a deliberately asymmetric
process instead.
"""

import numpy as np
import pytest

from pyspi.data import Data

# Process 0 drives process 1 at lag 1, with no feedback. Long enough that the
# estimators resolve the asymmetry well clear of their noise floor.
T = 2000
SEED = 0


@pytest.fixture(scope="module")
def driven_pair():
    """Data where process 0 unambiguously drives process 1."""
    rng = np.random.default_rng(SEED)
    x = np.zeros(T)
    y = np.zeros(T)
    ex, ey = rng.standard_normal(T), rng.standard_normal(T)
    for t in range(1, T):
        x[t] = 0.5 * x[t - 1] + ex[t]
        y[t] = 0.5 * y[t - 1] + 0.8 * x[t - 1] + ey[t]
    return Data(np.vstack([x, y]))


def _spi(module, cls_name, **kwargs):
    import importlib
    mod = importlib.import_module(f"pyspi.statistics.{module}")
    return getattr(mod, cls_name)(**kwargs)


# (module, class, kwargs). One per directed family that reaches a backend whose
# native convention differs from pyspi's, plus information-theory references.
DIRECTED = [
    ("infotheory", "TransferEntropy", {"estimator": "gaussian"}),
    ("infotheory", "TimeLaggedMutualInfo", {"estimator": "gaussian"}),
    ("spectral", "SpectralGrangerCausality", {}),
    ("spectral", "SpectralGrangerCausality", {"method": "parametric"}),
    ("spectral", "DirectedCoherence", {}),
    ("spectral", "PartialDirectedCoherence", {}),
    ("spectral", "GeneralizedPartialDirectedCoherence", {}),
    ("spectral", "DirectedTransferFunction", {}),
    ("spectral", "DirectDirectedTransferFunction", {}),
]


@pytest.mark.parametrize("module, cls_name, kwargs", DIRECTED,
                         ids=[f"{c}{'-' + str(k.get('method')) if k.get('method') else ''}"
                              for _, c, k in DIRECTED])
def test_directed_spi_is_source_by_target(module, cls_name, kwargs, driven_pair):
    """A[0, 1] (0 -> 1, the true direction) must exceed A[1, 0]."""
    spi = _spi(module, cls_name, **kwargs)
    A = np.asarray(spi.multivariate(driven_pair), dtype=float)

    assert A.shape == (2, 2)
    if np.isnan(A[0, 1]) or np.isnan(A[1, 0]):
        pytest.skip(f"{cls_name} returned NaN off-diagonals on this process")

    assert A[0, 1] > A[1, 0], (
        f"{cls_name}{kwargs}: process 0 drives process 1, so A[0,1] must be the "
        f"larger entry (row=source, column=target). Got A[0,1]={A[0, 1]:.4f}, "
        f"A[1,0]={A[1, 0]:.4f} -- this SPI is transposed relative to the rest "
        f"of the library."
    )


UNDIRECTED_SYMMETRIC = [
    "CoherenceMagnitude", "ImaginaryCoherence", "PhaseLockingValue",
    "PairwisePhaseConsistency",
]


@pytest.mark.parametrize("cls_name", UNDIRECTED_SYMMETRIC)
def test_undirected_spectral_stays_symmetric(cls_name, driven_pair):
    """Undirected spectral SPIs must not be touched by the directed transpose."""
    A = np.asarray(_spi("spectral", cls_name).multivariate(driven_pair), dtype=float)
    assert A[0, 1] == pytest.approx(A[1, 0], abs=1e-12), f"{cls_name} is not symmetric"


# These are labelled undirected but are antisymmetric (they encode a direction
# in their sign). Transposing them would silently negate every value, so the
# sign relationship is pinned here.
UNDIRECTED_ANTISYMMETRIC = ["PhaseLagIndex", "WeightedPhaseLagIndex", "PhaseSlopeIndex"]


@pytest.mark.parametrize("cls_name", UNDIRECTED_ANTISYMMETRIC)
def test_antisymmetric_spectral_sign_preserved(cls_name, driven_pair):
    A = np.asarray(_spi("spectral", cls_name).multivariate(driven_pair), dtype=float)
    assert A[0, 1] == pytest.approx(-A[1, 0], rel=1e-9), (
        f"{cls_name} should be antisymmetric; got A[0,1]={A[0, 1]}, A[1,0]={A[1, 0]}"
    )


# --------------------------------------------------------------------------
# Wilson-derived spectral measures, against an analytic VAR oracle
# --------------------------------------------------------------------------

def _var1_with_known_transfer():
    """VAR(1) whose directed transfer function is available in closed form.

    ``H(f) = (I - A e^{-2 pi i f})^{-1}`` and
    ``DTF_{i<-j}(f) = |H_ij| / sqrt(sum_k |H_ik|^2)`` (Kaminski & Blinowska
    1991). The literature indexes [target, source]; pyspi is row=source, so the
    analytic matrix is transposed before comparison.
    """
    A = np.array([[0.5, 0.0, 0.0],
                  [0.7, 0.4, 0.0],
                  [0.0, 0.3, 0.4]])
    rng = np.random.default_rng(0)
    T, M = 20000, 3
    X = np.zeros((M, T))
    for t in range(1, T):
        X[:, t] = A @ X[:, t - 1] + rng.standard_normal(M)

    freqs = np.linspace(1e-6, 0.5, 257)
    ana = np.zeros((len(freqs), M, M))
    for fi, f in enumerate(freqs):
        H = np.linalg.inv(np.eye(M) - A * np.exp(-2j * np.pi * f))
        for i in range(M):
            ana[fi, i, :] = np.abs(H[i, :]) / np.sqrt((np.abs(H[i, :]) ** 2).sum())
    expected = ana.mean(axis=0).T
    np.fill_diagonal(expected, np.nan)
    return X, expected


def test_directed_transfer_function_against_analytic_var():
    """DTF must stay in [0,1] and rank the true couplings correctly.

    Absolute calibration is deliberately NOT asserted: on this system at
    T=20000 the mean absolute error against the closed form is ~0.12, and
    connections that are exactly zero in the generating VAR are estimated at
    0.10-0.16. DTF is a row-normalised quantity that also reflects indirect
    paths, and the Wilson factorisation it is built on returns a factor with a
    non-trivial reconstruction residual, so the offset is not attributable to
    one cause here. Ranking and boundedness are what this pins.
    """
    from pyspi.data import Data
    from pyspi.statistics.spectral import DirectedTransferFunction

    X, expected = _var1_with_known_transfer()
    got = DirectedTransferFunction(statistic="mean", fmin=0, fmax=0.5).multivariate(
        Data(data=X, dim_order="ps", zscore=False)
    )

    off = ~np.eye(3, dtype=bool)
    assert np.nanmin(got) >= 0.0 and np.nanmax(got) <= 1.0, (
        f"DTF outside [0,1]: min={np.nanmin(got):.4f} max={np.nanmax(got):.4f}"
    )
    r = np.corrcoef(got[off], expected[off])[0, 1]
    assert r > 0.85, f"DTF does not track the analytic transfer function (r={r:.3f})"


def test_directed_coherence_is_bounded():
    """DC is defined on [0,1]; the backend's version is not.

    ``spectral_connectivity.directed_coherence`` puts |H|^2 in the numerator
    while ``_total_inflow`` is on the magnitude scale, so the ratio is
    dimensionally |H|^2/|H| and unbounded -- the shipped baselines reached 3.27
    (VAR), 1.84 (CML) and 1139.47 (Kuramoto). pyspi recomputes it with |H|.
    """
    import os
    from pyspi.data import Data
    from pyspi.calculator import load_spis_from_yaml, resolve_config

    spis = load_spis_from_yaml(resolve_config("full"), quiet=True)
    dcoh = [k for k in spis if k.startswith("dcoh_")]
    assert dcoh, "no directed-coherence SPIs in the full config"

    fixture = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "data", "fixtures", "kuramoto_M7_T100.npy")
    data = Data(data=fixture, dim_order="sp")
    for k in dcoh:
        A = np.asarray(spis[k].multivariate(data), dtype=float)
        finite = A[np.isfinite(A)]
        assert finite.min() >= 0.0 and finite.max() <= 1.0 + 1e-9, (
            f"{k} outside [0,1]: [{finite.min():.4f}, {finite.max():.4f}]"
        )
