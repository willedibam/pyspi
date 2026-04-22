"""Generate frozen synthetic MTS datasets for correctness testing.

Two datasets are built with fixed seeds so the tables computed from them
are reproducible across runs:

  - VAR(1): 7-process linear vector autoregression with sparse coupling.
  - Kuramoto: 7 coupled phase oscillators, sin(phase) observed.

Both use M=7 / T=100 to match cml7.npy dimensions so the benchmark framework
stays uniform. Run this script to regenerate the .npy files if either
generator is changed.
"""
import numpy as np
import os


def generate_var1(M=7, T=100, seed=0):
    """VAR(1) with sparse coupling and stable spectral radius."""
    rng = np.random.default_rng(seed)

    # Sparse coupling: ~30% non-zero, small magnitudes, rescaled below target SR.
    A = rng.uniform(-0.4, 0.4, size=(M, M))
    mask = rng.random((M, M)) < 0.3
    A = A * mask
    # Guarantee stability: rescale so spectral radius < 0.9.
    sr = np.max(np.abs(np.linalg.eigvals(A)))
    if sr > 0.85:
        A = A * (0.85 / sr)

    Y = np.zeros((M, T))
    Y[:, 0] = rng.standard_normal(M)
    for t in range(1, T):
        Y[:, t] = A @ Y[:, t - 1] + rng.standard_normal(M) * 0.5
    return Y


def generate_kuramoto(M=7, T=100, dt=0.1, K=2.0, seed=0):
    """Kuramoto model with quasi-uniform coupling; observed = sin(phase)."""
    rng = np.random.default_rng(seed)
    omega = rng.uniform(-1.0, 1.0, size=M)
    theta = rng.uniform(0, 2 * np.pi, size=M)

    Y = np.zeros((M, T))
    for t in range(T):
        diff = theta[None, :] - theta[:, None]
        dtheta = omega + (K / M) * np.sum(np.sin(diff), axis=1)
        theta = theta + dt * dtheta
        Y[:, t] = np.sin(theta)
    return Y


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(here), "pyspi", "data")

    var_data = generate_var1(M=7, T=100, seed=0)
    kur_data = generate_kuramoto(M=7, T=100, dt=0.1, K=2.0, seed=0)

    np.save(os.path.join(data_dir, "var1_7.npy"), var_data.T)
    np.save(os.path.join(data_dir, "kuramoto_7.npy"), kur_data.T)

    print(f"VAR(1):    shape {var_data.shape}, range [{var_data.min():.3f}, {var_data.max():.3f}]")
    print(f"Kuramoto:  shape {kur_data.shape}, range [{kur_data.min():.3f}, {kur_data.max():.3f}]")
