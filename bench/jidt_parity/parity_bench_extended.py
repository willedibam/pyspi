"""Extended parity benchmark: the parameterised paths the core bench skipped.

parity_bench.py covered defaults (fixed k=1 embedding, no Theiler window).
The shipped configs (pyspi/benchmarked*_config.yaml) also use:
  - dyn_corr_excl (Theiler window)  -> kraskov MI/TE, kernel TE
  - auto_embed_method=MAX_CORR_AIS  -> gaussian/kraskov TE
  - higher k_history               -> kraskov TE (k=2), symbolic TE (k=10)

This script benchmarks each against JIDT so we know whether the ported path
reproduces the Java reference, converges to it, or diverges by construction.

For the Theiler cells we use a FIXED integer window (DCE=5) on both sides so
the comparison isolates the windowed-counting math. (The AUTO window is pure
NumPy from autocorrelation; see findings re: MI/TLMI dropping it.)

Run:  uv run python bench/jidt_parity/parity_bench_extended.py
Out:  bench/jidt_parity/parity_results_extended.csv
"""
import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import jpype as jp

from pyspi.statistics.infotheory import (
    KernelTECalculator,
    SymbolicTECalculator,
    _ksg_mi_pair,
    _kraskov_te_bivariate,
    _gaussian_te_bivariate,
    _gaussian_ais,
    _auto_embed_gaussian_te,
)

JAR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                   "pyspi", "lib", "jidt", "infodynamics.jar"))
jp.startJVM(jp.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + JAR)
cont = jp.JPackage("infodynamics.measures.continuous")
sym = jp.JPackage("infodynamics.measures.continuous.symbolic")

DCE = 5          # fixed Theiler window for the windowed cells
K_NN = 4         # kraskov neighbours
W = 0.5          # kernel width


def gen_ar1_coupled(T, a=0.5, c=0.4, seed=0, burn=200):
    rng = np.random.default_rng(seed)
    n = T + burn
    x = np.zeros(n); y = np.zeros(n)
    ex = rng.standard_normal(n); ey = rng.standard_normal(n)
    for t in range(1, n):
        x[t] = a * x[t-1] + ex[t]
        y[t] = a * y[t-1] + c * x[t-1] + ey[t]
    return x[burn:], y[burn:]


def J(x):
    return jp.JArray(jp.JDouble, 1)(np.ascontiguousarray(x, dtype=np.float64))


# --------------------------------------------------------------------------
# Cells: each returns (numpy_val, jidt_val). Extra diagnostics via globals.
# --------------------------------------------------------------------------
def MI_kraskov_dce(x, y):
    tx = cKDTree(x.reshape(-1, 1)); ty = cKDTree(y.reshape(-1, 1))
    n_val = _ksg_mi_pair(x, y, K_NN, w=DCE, tree_x=tx, tree_y=ty)
    calc = cont.kraskov.MutualInfoCalculatorMultiVariateKraskov1()
    calc.setProperty("k", str(K_NN)); calc.setProperty("DYN_CORR_EXCL", str(DCE))
    calc.setProperty("NOISE_SEED", "42")
    calc.initialise(1, 1); calc.setObservations(J(x), J(y))
    return n_val, float(calc.computeAverageLocalOfObservations())


def TE_kraskov_dce(x, y):
    n_val = _kraskov_te_bivariate(x, y, 1, 1, 1, 1, k_nn=K_NN, w=DCE)
    calc = cont.kraskov.TransferEntropyCalculatorKraskov()
    calc.setProperty("k", str(K_NN)); calc.setProperty("k_HISTORY", "1")
    calc.setProperty("DYN_CORR_EXCL", str(DCE)); calc.setProperty("NOISE_SEED", "42")
    calc.initialise(); calc.setObservations(J(x), J(y))
    return n_val, float(calc.computeAverageLocalOfObservations())


def TE_kernel_dce(x, y):
    n = KernelTECalculator()
    n.setProperty("KERNEL_WIDTH", str(W)); n.setProperty("k_HISTORY", "1")
    n.setProperty("DYN_CORR_EXCL", str(DCE))
    n.initialise(); n.setObservations(x, y)
    n_val = n.computeAverageLocalOfObservations()
    calc = cont.kernel.TransferEntropyCalculatorKernel()
    calc.setProperty("KERNEL_WIDTH", str(W)); calc.setProperty("k_HISTORY", "1")
    calc.setProperty("DYN_CORR_EXCL", str(DCE))
    calc.initialise(); calc.setObservations(J(x), J(y))
    return n_val, float(calc.computeAverageLocalOfObservations())


def TE_gaussian_k2(x, y):
    n_val = _gaussian_te_bivariate(x, y, 2, 1, 1, 1)
    calc = cont.gaussian.TransferEntropyCalculatorGaussian()
    calc.setProperty("k_HISTORY", "2"); calc.setProperty("k_TAU", "1")
    calc.setProperty("l_HISTORY", "1"); calc.setProperty("l_TAU", "1")
    calc.initialise(); calc.setObservations(J(x), J(y))
    return n_val, float(calc.computeAverageLocalOfObservations())


def TE_kraskov_k2(x, y):
    n_val = _kraskov_te_bivariate(x, y, 2, 1, 1, 1, k_nn=K_NN, w=0)
    calc = cont.kraskov.TransferEntropyCalculatorKraskov()
    calc.setProperty("k", str(K_NN)); calc.setProperty("k_HISTORY", "2")
    calc.setProperty("k_TAU", "1"); calc.setProperty("l_HISTORY", "1")
    calc.setProperty("l_TAU", "1"); calc.setProperty("NOISE_SEED", "42")
    calc.initialise(); calc.setObservations(J(x), J(y))
    return n_val, float(calc.computeAverageLocalOfObservations())


# Auto-embed. The NumPy port selects (k,tau) by maximising *Gaussian* AIS,
# then runs the chosen estimator. JIDT MAX_CORR_AIS uses its own criterion.
# We record the selected embeddings so divergence is explained, not just measured.
_picks = []  # list of dicts


def _numpy_pick_embedding(targ, k_max, tau_max):
    best_k, best_tau, best_ais = 1, 1, -np.inf
    for k in range(1, k_max + 1):
        for tau in range(1, tau_max + 1):
            ais = _gaussian_ais(targ, k, tau)
            if ais > best_ais:
                best_ais, best_k, best_tau = ais, k, tau
    return best_k, best_tau


def TE_gaussian_auto(x, y):
    k_max, tau_max = 10, 2
    nk, ntau = _numpy_pick_embedding(y, k_max, tau_max)
    n_val = _auto_embed_gaussian_te(x, y, k_max, tau_max)
    calc = cont.gaussian.TransferEntropyCalculatorGaussian()
    calc.setProperty("AUTO_EMBED_METHOD", "MAX_CORR_AIS")
    calc.setProperty("AUTO_EMBED_K_SEARCH_MAX", str(k_max))
    calc.setProperty("AUTO_EMBED_TAU_SEARCH_MAX", str(tau_max))
    calc.initialise(); calc.setObservations(J(x), J(y))
    j_val = float(calc.computeAverageLocalOfObservations())
    _picks.append(dict(cell="TE_gaussian_auto",
                       numpy_k=nk, numpy_tau=ntau,
                       jidt_k=int(str(calc.getProperty("k_HISTORY"))),
                       jidt_tau=int(str(calc.getProperty("k_TAU")))))
    return n_val, j_val


def TE_kraskov_auto(x, y):
    k_max, tau_max = 10, 2
    nk, ntau = _numpy_pick_embedding(y, k_max, tau_max)   # gaussian AIS (port's choice)
    n_val = _kraskov_te_bivariate(x, y, nk, ntau, 1, 1, k_nn=K_NN, w=0)
    calc = cont.kraskov.TransferEntropyCalculatorKraskov()
    calc.setProperty("k", str(K_NN))
    calc.setProperty("AUTO_EMBED_METHOD", "MAX_CORR_AIS")
    calc.setProperty("AUTO_EMBED_K_SEARCH_MAX", str(k_max))
    calc.setProperty("AUTO_EMBED_TAU_SEARCH_MAX", str(tau_max))
    calc.setProperty("NOISE_SEED", "42")
    calc.initialise(); calc.setObservations(J(x), J(y))
    j_val = float(calc.computeAverageLocalOfObservations())
    _picks.append(dict(cell="TE_kraskov_auto",
                       numpy_k=nk, numpy_tau=ntau,
                       jidt_k=int(str(calc.getProperty("k_HISTORY"))),
                       jidt_tau=int(str(calc.getProperty("k_TAU")))))
    return n_val, j_val


def TE_symbolic_k10(x, y):
    # JIDT overflows its joint histogram at k=10 (10! symbols). Record NaN for
    # JIDT, and the port's finite value (np.unique over observed symbols only).
    n = SymbolicTECalculator(); n.setProperty("k_HISTORY", "10")
    n.initialise(); n.setObservations(x, y)
    n_val = n.computeAverageLocalOfObservations()
    try:
        calc = sym.TransferEntropyCalculatorSymbolic(); calc.initialise(10)
        calc.setObservations(J(x), J(y))
        j_val = float(calc.computeAverageLocalOfObservations())
    except Exception:
        j_val = np.nan
    return n_val, j_val


CELLS = [
    ("kraskov", "MI_DCE5",        MI_kraskov_dce),
    ("kraskov", "TE_DCE5",        TE_kraskov_dce),
    ("kernel",  "TE_DCE5",        TE_kernel_dce),
    ("gaussian","TE_k2",          TE_gaussian_k2),
    ("kraskov", "TE_k2",          TE_kraskov_k2),
    ("gaussian","TE_autoembed",   TE_gaussian_auto),
    ("kraskov", "TE_autoembed",   TE_kraskov_auto),
    ("symbolic","TE_k10",         TE_symbolic_k10),
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
                    nv, jv = fn(x, y)
                except Exception as e:
                    print(f"    !! {est}/{meas}: {e}", flush=True)
                    nv, jv = np.nan, np.nan
                rows.append(dict(T=T, seed=seed, estimator=est, measure=meas,
                                 numpy_val=nv, jidt_val=jv))
    df = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(__file__), "parity_results_extended.csv")
    df.to_csv(out, index=False)

    if _picks:
        picks = pd.DataFrame(_picks)
        pick_summary = (picks.groupby("cell")
                        .agg(numpy_k=("numpy_k", "median"), numpy_tau=("numpy_tau", "median"),
                             jidt_k=("jidt_k", "median"), jidt_tau=("jidt_tau", "median"),
                             agree=("cell", "size")).reset_index())
        # fraction where (k,tau) match
        picks["match"] = (picks.numpy_k == picks.jidt_k) & (picks.numpy_tau == picks.jidt_tau)
        frac = picks.groupby("cell").match.mean()
        print("\n=== Auto-embed (k,tau) selection: NumPy(Gaussian-AIS) vs JIDT(MAX_CORR_AIS) ===")
        for _, r in pick_summary.iterrows():
            print(f"  {r.cell}: numpy median (k={int(r.numpy_k)},tau={int(r.numpy_tau)})  "
                  f"jidt median (k={int(r.jidt_k)},tau={int(r.jidt_tau)})  "
                  f"exact-match across seeds/T = {frac[r.cell]*100:.0f}%")

    df["abs_err"] = (df.numpy_val - df.jidt_val).abs()
    df["rel_err"] = df.abs_err / df.jidt_val.abs().clip(lower=1e-12)
    summary = (df.groupby(["estimator", "measure", "T"])
                 .agg(jidt_mean=("jidt_val", "mean"), numpy_mean=("numpy_val", "mean"),
                      abs_err_mean=("abs_err", "mean"), abs_err_std=("abs_err", "std"),
                      rel_err_mean=("rel_err", "mean")).reset_index())
    summary.to_csv(os.path.join(os.path.dirname(__file__),
                                "parity_summary_extended.csv"), index=False)
    print("\n=== Extended summary (mean over 10 seeds) ===")
    pd.set_option("display.float_format", lambda v: f"{v:+.4e}")
    print(summary.to_string(index=False))
    return df


if __name__ == "__main__":
    run()
    jp.shutdownJVM()
