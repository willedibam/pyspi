"""Red tests: an SPI must compute the estimator it advertises, or refuse.

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
@pytest.mark.xfail(strict=True, reason="kraskov silently dispatches to the Gaussian estimator")
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


@pytest.mark.xfail(strict=True, reason="unknown auto_embed_method silently falls through to a default")
def test_invalid_auto_embed_method_is_rejected():
    with pytest.raises((ValueError, KeyError, NotImplementedError)):
        it.TransferEntropy(auto_embed_method="NOT_A_METHOD").multivariate(_data())


@pytest.mark.xfail(strict=True, reason="unsupported estimator-specific parameters are ignored")
def test_unsupported_parameters_are_rejected():
    """A parameter that the chosen estimator ignores must not be accepted silently."""
    with pytest.raises((ValueError, TypeError)):
        # kernel_width is meaningless for the gaussian estimator.
        it.MutualInfo(estimator="gaussian", kernel_width=0.5)


# --------------------------------------------------------------------------
# Symbolic transfer entropy
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="k_history=1 has a single ordinal symbol; TE is identically zero")
def test_symbolic_k_history_1_is_rejected():
    with pytest.raises(ValueError):
        it.TransferEntropy(estimator="symbolic", k_history=1)


@pytest.mark.xfail(strict=True, reason="degenerate symbolic variants still ship in bundled configs")
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


@pytest.mark.xfail(strict=True, reason="arithmetic symbol packing overflows int64 for large k")
def test_symbolic_encoding_is_collision_free():
    """Joint symbol encoding must be injective, whatever k is.

    The current encoding multiplies by a squaring multiplier, so the packed
    value exceeds int64 for k=10 and wraps. Wrapping is not by itself a
    collision -- no collisions occur on the shipped fixtures -- but the
    encoding offers no guarantee, and correctness must not rest on luck.
    """
    rng = np.random.default_rng(0)
    k = 10
    n_symbols = int(np.math.factorial(k))
    arrs = [rng.integers(0, n_symbols, size=500) for _ in range(3)]

    truth = {tuple(int(a[i]) for a in arrs) for i in range(arrs[0].size)}

    combined = arrs[0].copy()
    multiplier = n_symbols
    for arr in arrs[1:]:
        combined = combined * multiplier + arr
        multiplier *= n_symbols

    assert len(np.unique(combined)) == len(truth), (
        f"Symbol packing is not injective at k={k}: "
        f"{len(truth)} distinct tuples collapsed to {len(np.unique(combined))}."
    )


# --------------------------------------------------------------------------
# KSG preconditions
# --------------------------------------------------------------------------

# NOTE: parse_bivariate's signature is (self, data, data2=None, i=None, j=None),
# so bivariate(data, 0, 1) binds data2=0, i=1 and fails with an unrelated
# dimension error. Always pass i/j by keyword here, or these tests pass for the
# wrong reason. (The positional foot-gun is a usability issue in its own right.)

@pytest.mark.parametrize("k", [30, 100])
@pytest.mark.xfail(strict=True, reason="KSG accepts k >= effective N and returns a finite number")
def test_ksg_rejects_k_at_or_above_sample_size(k):
    small = _data(m=2, t=20)
    spi = it.MutualInfo(estimator="kraskov", prop_k=k)
    with pytest.raises(ValueError):
        spi.bivariate(small, i=0, j=1)


@pytest.mark.xfail(strict=True, reason="degenerate (zero-radius) samples are not detected")
def test_ksg_rejects_degenerate_samples():
    const = np.zeros((2, 200))
    const[1] = np.arange(200)
    data = Data(data=const, dim_order="ps", zscore=False)
    with pytest.raises(ValueError):
        it.MutualInfo(estimator="kraskov").bivariate(data, i=0, j=1)
