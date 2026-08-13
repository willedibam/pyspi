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
