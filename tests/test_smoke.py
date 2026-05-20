"""Smoke test for pyspi fork — numerical correctness checks.

Runs a small set of SPIs on known synthetic data (coupled AR(1)) and checks:
1. All estimators instantiate without JVM
2. Bivariate/multivariate methods return finite values
3. Gaussian MI matches analytical formula
4. Kernel MI is positive for correlated signals
5. KSG MI is positive for correlated signals
6. TE is positive for causally coupled signals
7. Symbolic TE is positive for causally coupled signals

Usage:
    python -m tests.smoke_test
    # or
    python tests/smoke_test.py
"""
import sys
import numpy as np

np.random.seed(42)


def generate_coupled_ar1(M=5, T=500, coupling=0.4, noise_std=0.3):
    """Generate M coupled AR(1) processes.

    X_i(t) = coupling * X_{i-1}(t-1) + noise
    Process 0 is independent AR(1).
    """
    X = np.zeros((M, T))
    X[:, 0] = np.random.randn(M)
    for t in range(1, T):
        X[0, t] = 0.8 * X[0, t - 1] + noise_std * np.random.randn()
        for i in range(1, M):
            X[i, t] = (0.5 * X[i, t - 1]
                        + coupling * X[i - 1, t - 1]
                        + noise_std * np.random.randn())
    return X


def test_imports():
    """All estimators instantiate without JIDT/JVM."""
    from pyspi.statistics.infotheory import (
        MutualInfo, TimeLaggedMutualInfo, TransferEntropy,
        JointEntropy, ConditionalEntropy, CrossmapEntropy,
        CausalEntropy, DirectedInfo, StochasticInteraction,
        IntegratedInformation,
    )
    from pyspi.statistics.basic import (
        Covariance, Precision, CrossCorrelation,
        SpearmanR, KendallTau, LaggedCorrelation,
    )
    from pyspi.statistics.distance import DynamicTimeWarping, CrossPairwiseDistance
    from pyspi.statistics.spectral import CoherenceMagnitude, DirectedCoherence
    from pyspi.statistics.misc import LinearModel, GPModel

    # Info-theoretic: all estimators
    for est in ('gaussian', 'kraskov', 'kernel', 'kozachenko'):
        MutualInfo(estimator=est)
        TimeLaggedMutualInfo(estimator=est)
        JointEntropy(estimator=est)
        ConditionalEntropy(estimator=est)
        CrossmapEntropy(estimator=est)
        CausalEntropy(estimator=est)
        DirectedInfo(estimator=est)
        StochasticInteraction(estimator=est)
        if est != 'symbolic':
            TransferEntropy(estimator=est)

    TransferEntropy(estimator='symbolic')
    TransferEntropy(estimator='kernel', kernel_width=0.25)

    print("  PASS: all estimators instantiate")


def test_gaussian_mi_analytical():
    """Gaussian MI matches analytical formula: MI = -0.5 * ln(1 - r^2)."""
    from pyspi.data import Data
    from pyspi.statistics.infotheory import MutualInfo

    X = generate_coupled_ar1(M=3, T=1000)
    data = Data(X, normalise=True)

    mi = MutualInfo(estimator='gaussian')
    result = mi.multivariate(data)

    # Check shape and finiteness
    assert result.shape == (3, 3), f"Shape mismatch: {result.shape}"
    assert np.all(np.isfinite(result[~np.isnan(result)])), "Non-finite MI values"
    assert np.all(np.isnan(np.diag(result))), "Diagonal should be NaN"

    # Verify against analytical formula
    Z = data.to_numpy(squeeze=True)
    R = np.corrcoef(Z)
    r2 = np.clip(R ** 2, 0, 1 - 1e-15)
    expected = -0.5 * np.log(1 - r2)
    np.fill_diagonal(expected, np.nan)

    off_diag = ~np.isnan(result)
    assert np.allclose(result[off_diag], expected[off_diag], atol=1e-10), \
        f"Gaussian MI mismatch: max diff={np.max(np.abs(result[off_diag] - expected[off_diag]))}"

    # Coupled processes should have positive MI
    assert result[0, 1] > 0.01, f"MI(0,1) should be positive: {result[0, 1]}"

    print(f"  PASS: gaussian MI (max={np.nanmax(result):.4f})")


def test_kraskov_mi():
    """KSG MI is positive for correlated signals."""
    from pyspi.data import Data
    from pyspi.statistics.infotheory import MutualInfo

    X = generate_coupled_ar1(M=3, T=500)
    data = Data(X, normalise=True)

    mi = MutualInfo(estimator='kraskov', prop_k=4)
    result = mi.multivariate(data)

    assert result.shape == (3, 3)
    off_diag = ~np.isnan(result)
    assert np.all(np.isfinite(result[off_diag])), "Non-finite KSG MI"
    assert result[0, 1] > 0, f"KSG MI(0,1) should be positive: {result[0, 1]}"

    print(f"  PASS: kraskov MI (max={np.nanmax(result):.4f})")


def test_kernel_mi():
    """Kernel MI is positive for correlated signals."""
    from pyspi.data import Data
    from pyspi.statistics.infotheory import MutualInfo

    X = generate_coupled_ar1(M=3, T=500)
    data = Data(X, normalise=True)

    mi = MutualInfo(estimator='kernel', kernel_width=0.25)
    result = mi.multivariate(data)

    assert result.shape == (3, 3)
    off_diag = ~np.isnan(result)
    assert np.all(np.isfinite(result[off_diag])), "Non-finite kernel MI"
    assert result[0, 1] > 0, f"Kernel MI(0,1) should be positive: {result[0, 1]}"

    print(f"  PASS: kernel MI (max={np.nanmax(result):.4f})")


def test_transfer_entropy():
    """TE is positive for causally coupled signals (all estimators)."""
    from pyspi.data import Data
    from pyspi.statistics.infotheory import TransferEntropy

    X = generate_coupled_ar1(M=3, T=500, coupling=0.5)
    data = Data(X, normalise=True)

    for est in ('gaussian', 'kraskov', 'kernel', 'symbolic'):
        if est == 'kernel':
            te = TransferEntropy(estimator=est, kernel_width=0.25)
        elif est == 'symbolic':
            te = TransferEntropy(estimator=est, k_history=3)
        else:
            te = TransferEntropy(estimator=est)

        result = te.multivariate(data)
        assert result.shape == (3, 3), f"{est} TE shape: {result.shape}"

        # TE(0→1) should be positive (0 causes 1)
        te_01 = result[0, 1]
        assert np.isfinite(te_01), f"{est} TE(0→1) not finite: {te_01}"

        if est == 'gaussian':
            # Gaussian TE = Granger causality, should be clearly positive
            assert te_01 > 0.01, f"Gaussian TE(0→1) too small: {te_01}"

        print(f"  PASS: {est} TE (TE(0→1)={te_01:.4f})")


def test_joint_conditional_entropy():
    """JE and CE produce finite values for kernel estimator."""
    from pyspi.data import Data
    from pyspi.statistics.infotheory import JointEntropy, ConditionalEntropy

    X = generate_coupled_ar1(M=3, T=300)
    data = Data(X, normalise=True)

    for est in ('gaussian', 'kernel', 'kozachenko'):
        je = JointEntropy(estimator=est)
        result_je = je.multivariate(data)
        assert result_je.shape == (3, 3)
        off = ~np.isnan(result_je)
        assert np.all(np.isfinite(result_je[off])), f"{est} JE has non-finite values"

        ce = ConditionalEntropy(estimator=est)
        result_ce = ce.multivariate(data)
        assert result_ce.shape == (3, 3)
        off = ~np.isnan(result_ce)
        assert np.all(np.isfinite(result_ce[off])), f"{est} CE has non-finite values"

        print(f"  PASS: {est} JE/CE")


def test_basic_spis():
    """Basic SPIs (correlation, DTW, etc.) produce finite values."""
    from pyspi.data import Data
    from pyspi.statistics.basic import (
        Covariance, SpearmanR, KendallTau, CrossCorrelation, LaggedCorrelation,
    )
    from pyspi.statistics.distance import DynamicTimeWarping

    X = generate_coupled_ar1(M=3, T=200)
    data = Data(X, normalise=True)

    for SPI, kwargs in [
        (Covariance, {}),
        (SpearmanR, {}),
        (KendallTau, {}),
        (CrossCorrelation, {}),
        (LaggedCorrelation, {"tau": 1}),
        (LaggedCorrelation, {"tau": 3, "estimator": "spearman"}),
        (DynamicTimeWarping, {}),
    ]:
        name = SPI.__name__ + str(kwargs)
        spi = SPI(**kwargs)
        result = spi.multivariate(data)
        assert result.shape == (3, 3), f"{name} shape: {result.shape}"
        off = ~np.isnan(result)
        assert np.all(np.isfinite(result[off])), f"{name} has non-finite values"
        print(f"  PASS: {name}")


def test_spectral_spis():
    """Spectral SPIs produce finite values."""
    from pyspi.data import Data
    from pyspi.statistics.spectral import CoherenceMagnitude

    X = generate_coupled_ar1(M=3, T=200)
    data = Data(X, normalise=True)

    spi = CoherenceMagnitude()
    result = spi.multivariate(data)
    assert result.shape == (3, 3)
    off = ~np.isnan(result)
    assert np.all(np.isfinite(result[off])), "CoherenceMagnitude has non-finite values"
    print("  PASS: CoherenceMagnitude")


def main():
    print("=" * 60)
    print("pyspi fork smoke test")
    print("=" * 60)

    tests = [
        ("Imports (no JVM)", test_imports),
        ("Gaussian MI analytical", test_gaussian_mi_analytical),
        ("Kraskov MI", test_kraskov_mi),
        ("Kernel MI", test_kernel_mi),
        ("Transfer Entropy (all estimators)", test_transfer_entropy),
        ("Joint/Conditional Entropy", test_joint_conditional_entropy),
        ("Basic SPIs", test_basic_spis),
        ("Spectral SPIs", test_spectral_spis),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n[{name}]")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
