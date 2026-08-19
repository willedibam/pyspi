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
# Wilson-derived spectral measures, against an exact analytic spectrum
# --------------------------------------------------------------------------

def test_wilson_factorisation_recovers_a_known_transfer_function():
    """Factorise an *exact* VAR(1) spectrum, bypassing sample estimation.

    For x_t = A x_{t-1} + e_t with noise covariance Sigma, the cross-spectral
    matrix is S(f) = H(f) Sigma H(f)^H with H(f) = (I - A e^{-2 pi i f})^{-1}.
    Feeding that exact S to the Wilson decomposition isolates the factorisation
    from every source of finite-sample error, so any discrepancy is the
    algorithm's own.

    An earlier version of this test compared pyspi's DTF against the *full*
    three-process transfer function. That was invalid: pyspi computes
    NonparametricSpectralBivariate measures on two-process subsystems
    (`z[[i, j]]`), and a subsystem of a larger VAR legitimately shows flow in
    both directions because the omitted processes induce correlation. The
    0.10-0.16 floor that comparison produced was the test's error, not the
    estimator's.
    """
    from spectral_connectivity.minimum_phase_decomposition import (
        minimum_phase_decomposition,
    )

    A = np.array([[0.5, 0.0], [0.7, 0.4]])
    M = A.shape[0]
    Sigma = np.eye(M)
    # The FULL two-sided grid over [0, 1): the algorithm takes an inverse FFT
    # internally to impose causality, so a half-spectrum silently gives a
    # factor unrelated to H even though S = G G^H still holds.
    n = 256
    freqs = np.arange(n) / n

    H = np.stack([np.linalg.inv(np.eye(M) - A * np.exp(-2j * np.pi * f)) for f in freqs])
    S = H @ Sigma @ np.conj(np.transpose(H, (0, 2, 1)))

    G = minimum_phase_decomposition(S[np.newaxis, ...])[0]

    # S = G G^H is the contract. The bound is the algorithm's own convergence
    # tolerance (default 1e-8), not machine precision -- this is an iterative
    # method, so ~1e-8 is the expected floor rather than a discrepancy.
    residual = np.abs(S - G @ np.conj(np.transpose(G, (0, 2, 1)))).max()
    assert residual < 1e-6, f"Wilson reconstruction residual {residual:.3g}"

    # G(f) = H(f) Sigma^{1/2}; the zeroth Fourier coefficient of G is Sigma^{1/2}.
    g0 = np.fft.ifft(G, axis=0)[0]
    H_hat = G @ np.linalg.inv(g0)
    err = np.abs(H_hat - H).max()
    assert err < 1e-9, f"recovered transfer function differs by {err:.3g}"

    num = np.abs(H_hat) ** 2
    dtf_hat = num / num.sum(axis=-1, keepdims=True)
    num = np.abs(H) ** 2
    dtf = num / num.sum(axis=-1, keepdims=True)
    assert np.abs(dtf_hat - dtf).max() < 1e-9


def test_directed_transfer_function_orientation_on_a_two_process_var():
    """On a 2-process VAR the subsystem *is* the system, so DTF is comparable.

    Bounded, and the driving direction dominates. Absolute calibration is not
    asserted: DTF as implemented is the squared form, and the multitaper
    estimate carries finite-sample bias at these lengths.
    """
    from pyspi.data import Data
    from pyspi.statistics.spectral import DirectedTransferFunction

    A = np.array([[0.5, 0.0], [0.7, 0.4]])   # 0 -> 1 only
    rng = np.random.default_rng(0)
    T = 500
    X = np.zeros((2, T))
    for t in range(1, T):
        X[:, t] = A @ X[:, t - 1] + rng.standard_normal(2)

    got = DirectedTransferFunction(statistic="mean", fmin=0, fmax=0.5).multivariate(
        Data(data=X, dim_order="ps", zscore=False)
    )
    finite = got[np.isfinite(got)]
    assert finite.min() >= 0.0 and finite.max() <= 1.0, (
        f"DTF outside [0,1]: [{finite.min():.4f}, {finite.max():.4f}]"
    )
    assert got[0, 1] > got[1, 0], (
        f"DTF did not favour the driving direction: 0->1={got[0,1]:.4f}, "
        f"1->0={got[1,0]:.4f}"
    )


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


def test_directed_coherence_matches_dtf_under_equal_noise_variances():
    """DC reduces to sqrt(DTF) when all noise variances are equal.

    A stronger check than boundedness: it pins the *form*, not just the range.
    The backend's version fails it by construction, since |H|^2 in the
    numerator is not sqrt of |H|^2/sum|H|^2.
    """
    import spectral_connectivity as sc
    from spectral_connectivity.connectivity import _get_noise_variance, _total_inflow
    from pyspi.statistics.spectral import _ensure_time_series_3d

    A = np.array([[0.5, 0.0, 0.0], [0.7, 0.4, 0.0], [0.0, 0.3, 0.4]])
    rng = np.random.default_rng(0)
    T = 4000
    X = np.zeros((3, T))
    for t in range(1, T):
        X[:, t] = A @ X[:, t - 1] + rng.standard_normal(3)

    m = sc.Multitaper(_ensure_time_series_3d(np.transpose(X)), sampling_frequency=1)
    conn = sc.Connectivity.from_multitaper(m)

    nv = _get_noise_variance(conn._noise_covariance)
    corrected = np.sqrt(nv) * np.abs(conn._transfer_function) / _total_inflow(
        conn._transfer_function, nv
    )
    err = np.nanmax(np.abs(corrected - np.sqrt(conn.directed_transfer_function())))
    assert err < 1e-12, f"DC != sqrt(DTF) under equal noise variances: {err:.3g}"
