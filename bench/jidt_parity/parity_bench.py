"""Numerical parity benchmark: NumPy infotheory port vs JIDT (Java reference).

Compares each estimator/measure in pyspi/statistics/infotheory.py against the
JIDT class it replaced. Reports the per-cell signed difference (numpy - jidt)
and the JIDT magnitude so finite-sample error can be separated from
implementation drift.

Test signal: bivariate AR(1) with unidirectional coupling x -> y.
    x_t = a * x_{t-1} + e_x
    y_t = a * y_{t-1} + c * x_{t-1} + e_y
This has known nonzero TE(x->y), ~zero TE(y->x), and finite MI / entropies.

Run:  uv run python bench/jidt_parity/parity_bench.py
Out:  bench/jidt_parity/parity_results.csv
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import jpype as jp

# pyspi NumPy calculators (the things we're testing)
from pyspi.statistics.infotheory import (
    GaussianEntropyCalculator,
    KLEntropyCalculator,
    KernelEntropyCalculator,
    KernelMICalculator,
    KernelTECalculator,
    SymbolicTECalculator,
    _gaussian_entropy_from_data,
    _ksg_mi_pair,
    _gaussian_te_bivariate,
    _kraskov_te_bivariate,
)

# ---------------------------------------------------------------------------
# JVM
# ---------------------------------------------------------------------------
JAR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "pyspi", "lib", "jidt",
                 "infodynamics.jar")
)
jp.startJVM(jp.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + JAR)

cont = jp.JPackage("infodynamics.measures.continuous")
sym = jp.JPackage("infodynamics.measures.continuous.symbolic")

# ---------------------------------------------------------------------------
# Data generator
# ---------------------------------------------------------------------------
def gen_ar1_coupled(T, a=0.5, c=0.4, seed=0, burn=200):
    rng = np.random.default_rng(seed)
    n = T + burn
    x = np.zeros(n)
    y = np.zeros(n)
    ex = rng.standard_normal(n)
    ey = rng.standard_normal(n)
    for t in range(1, n):
        x[t] = a * x[t-1] + ex[t]
        y[t] = a * y[t-1] + c * x[t-1] + ey[t]
    return x[burn:], y[burn:]

def to_jarr(x):
    return jp.JArray(jp.JDouble, 1)(np.ascontiguousarray(x, dtype=np.float64))

def to_jarr2d(X):
    return jp.JArray(jp.JDouble, 2)(np.ascontiguousarray(X, dtype=np.float64))

# ---------------------------------------------------------------------------
# Per-cell comparisons. Each returns (numpy_val, jidt_val) for a measure.
# JIDT setup mirrors what pyspi/statistics/infotheory.py used historically:
# BIAS_CORRECTION=false, NOISE_SEED=42; NORMALISE left at JIDT defaults.
# ---------------------------------------------------------------------------
def _setup_jidt(calc, kernel_width=None, prop_k=None):
    if kernel_width is not None:
        calc.setProperty("KERNEL_WIDTH", str(kernel_width))
    if prop_k is not None:
        calc.setProperty("k", str(prop_k))
    calc.setProperty("BIAS_CORRECTION", "false")
    calc.setProperty("NOISE_SEED", "42")
    return calc

# --- Entropy (univariate, on x) ---
def H_gaussian(x):
    n_val = _gaussian_entropy_from_data(x.reshape(-1, 1))
    calc = _setup_jidt(cont.gaussian.EntropyCalculatorMultiVariateGaussian())
    calc.initialise(1)
    calc.setObservations(to_jarr(x))
    j_val = float(calc.computeAverageLocalOfObservations())
    return n_val, j_val

def H_kozachenko(x):
    n_calc = KLEntropyCalculator(); n_calc.initialise(1); n_calc.setObservations(x)
    n_val = n_calc.computeAverageLocalOfObservations()
    calc = _setup_jidt(cont.kozachenko.EntropyCalculatorMultiVariateKozachenko())
    calc.initialise(1)
    calc.setObservations(to_jarr(x))
    j_val = float(calc.computeAverageLocalOfObservations())
    return n_val, j_val

def H_kernel(x, w=0.5):
    n_calc = KernelEntropyCalculator()
    n_calc.setProperty("KERNEL_WIDTH", str(w))
    n_calc.initialise(1); n_calc.setObservations(x)
    n_val = n_calc.computeAverageLocalOfObservations()
    calc = _setup_jidt(cont.kernel.EntropyCalculatorMultiVariateKernel(),
                       kernel_width=w)
    calc.initialise(1)
    calc.setObservations(to_jarr(x))
    j_val = float(calc.computeAverageLocalOfObservations())
    return n_val, j_val

# --- MI(x, y) ---
def MI_gaussian(x, y):
    # NumPy: closed-form via correlation coefficient
    r = np.corrcoef(x, y)[0, 1]
    r2 = np.clip(r ** 2, 0, 1 - 1e-15)
    n_val = -0.5 * np.log(1 - r2)
    calc = _setup_jidt(cont.gaussian.MutualInfoCalculatorMultiVariateGaussian())
    calc.initialise(1, 1)
    calc.setObservations(to_jarr(x), to_jarr(y))
    j_val = float(calc.computeAverageLocalOfObservations())
    return float(n_val), j_val

def MI_kraskov(x, y, k=4):
    tx = cKDTree(x.reshape(-1, 1)); ty = cKDTree(y.reshape(-1, 1))
    n_val = _ksg_mi_pair(x, y, k, w=0, tree_x=tx, tree_y=ty)
    calc = _setup_jidt(cont.kraskov.MutualInfoCalculatorMultiVariateKraskov1(),
                       prop_k=k)
    calc.initialise(1, 1)
    calc.setObservations(to_jarr(x), to_jarr(y))
    j_val = float(calc.computeAverageLocalOfObservations())
    return n_val, j_val

def MI_kernel(x, y, w=0.5):
    n_calc = KernelMICalculator()
    n_calc.setProperty("KERNEL_WIDTH", str(w))
    n_calc.initialise(1, 1); n_calc.setObservations(x, y)
    n_val = n_calc.computeAverageLocalOfObservations()
    calc = _setup_jidt(cont.kernel.MutualInfoCalculatorMultiVariateKernel(),
                       kernel_width=w)
    calc.initialise(1, 1)
    calc.setObservations(to_jarr(x), to_jarr(y))
    j_val = float(calc.computeAverageLocalOfObservations())
    return n_val, j_val

# --- TE(x -> y), k_history=1, l_history=1, k_tau=l_tau=1 ---
def TE_gaussian(x, y):
    n_val = _gaussian_te_bivariate(x, y, 1, 1, 1, 1)
    calc = _setup_jidt(cont.gaussian.TransferEntropyCalculatorGaussian())
    calc.setProperty("k_HISTORY", "1")
    calc.setProperty("k_TAU", "1")
    calc.setProperty("l_HISTORY", "1")
    calc.setProperty("l_TAU", "1")
    calc.initialise()
    calc.setObservations(to_jarr(x), to_jarr(y))
    j_val = float(calc.computeAverageLocalOfObservations())
    return n_val, j_val

def TE_kraskov(x, y, k=4):
    n_val = _kraskov_te_bivariate(x, y, 1, 1, 1, 1, k_nn=k, w=0)
    calc = _setup_jidt(cont.kraskov.TransferEntropyCalculatorKraskov(),
                       prop_k=k)
    calc.setProperty("k_HISTORY", "1")
    calc.setProperty("k_TAU", "1")
    calc.setProperty("l_HISTORY", "1")
    calc.setProperty("l_TAU", "1")
    calc.initialise()
    calc.setObservations(to_jarr(x), to_jarr(y))
    j_val = float(calc.computeAverageLocalOfObservations())
    return n_val, j_val

def TE_kernel(x, y, w=0.5):
    n_calc = KernelTECalculator()
    n_calc.setProperty("KERNEL_WIDTH", str(w))
    n_calc.setProperty("k_HISTORY", "1")
    n_calc.initialise(); n_calc.setObservations(x, y)
    n_val = n_calc.computeAverageLocalOfObservations()
    calc = _setup_jidt(cont.kernel.TransferEntropyCalculatorKernel(),
                       kernel_width=w)
    calc.setProperty("k_HISTORY", "1")
    calc.initialise()
    calc.setObservations(to_jarr(x), to_jarr(y))
    j_val = float(calc.computeAverageLocalOfObservations())
    return n_val, j_val

def TE_symbolic(x, y, k=2):
    # k=1 is degenerate (1 ordinal symbol); use k=2 so the comparison
    # is non-trivial. JIDT signature: initialise(int destEmbeddingLength).
    n_calc = SymbolicTECalculator()
    n_calc.setProperty("k_HISTORY", str(k))
    n_calc.initialise(); n_calc.setObservations(x, y)
    n_val = n_calc.computeAverageLocalOfObservations()
    calc = sym.TransferEntropyCalculatorSymbolic()
    calc.initialise(k)
    calc.setObservations(to_jarr(x), to_jarr(y))
    j_val = float(calc.computeAverageLocalOfObservations())
    return n_val, j_val

# ---------------------------------------------------------------------------
# Cell dispatch table
# ---------------------------------------------------------------------------
CELLS = [
    # (estimator,   measure,    fn)
    ("gaussian",    "entropy",  H_gaussian),
    ("kozachenko",  "entropy",  H_kozachenko),
    ("kernel",      "entropy",  H_kernel),
    ("gaussian",    "MI",       MI_gaussian),
    ("kraskov",     "MI",       MI_kraskov),
    ("kernel",      "MI",       MI_kernel),
    ("gaussian",    "TE",       TE_gaussian),
    ("kraskov",     "TE",       TE_kraskov),
    ("kernel",      "TE",       TE_kernel),
    ("symbolic",    "TE",       TE_symbolic),
]

def run():
    Ts = [200, 800, 1600]
    seeds = list(range(10))
    rows = []
    for T in Ts:
        for seed in seeds:
            x, y = gen_ar1_coupled(T, seed=seed)
            print(f"  T={T} seed={seed}", flush=True)
            for est, meas, fn in CELLS:
                try:
                    n_val, j_val = fn(x, y) if meas != "entropy" else fn(x)
                except Exception as e:
                    print(f"    !! {est}/{meas}: {e}", flush=True)
                    n_val, j_val = np.nan, np.nan
                rows.append(dict(T=T, seed=seed, estimator=est, measure=meas,
                                 numpy_val=n_val, jidt_val=j_val))
    df = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(__file__), "parity_results.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows to {out}")
    return df

if __name__ == "__main__":
    df = run()
    # Quick summary print to stdout
    df["abs_err"] = (df.numpy_val - df.jidt_val).abs()
    df["rel_err"] = df["abs_err"] / df.jidt_val.abs().clip(lower=1e-12)
    summary = (df.groupby(["estimator", "measure", "T"])
                 .agg(jidt_mean=("jidt_val", "mean"),
                      numpy_mean=("numpy_val", "mean"),
                      abs_err_mean=("abs_err", "mean"),
                      abs_err_std=("abs_err", "std"),
                      rel_err_mean=("rel_err", "mean"))
                 .reset_index())
    print("\n=== Summary (mean over 10 seeds) ===")
    pd.set_option("display.float_format", lambda v: f"{v:+.4e}")
    print(summary.to_string(index=False))
    jp.shutdownJVM()
