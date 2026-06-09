"""Build the parity analysis notebook + summary plot.

Reads parity_results.csv (produced by parity_bench.py) and writes:
  - parity_summary.csv  : per-cell summary table
  - parity_plot.png     : abs_err vs T per (estimator, measure)
  - parity_notebook.ipynb : Jupyter notebook of the analysis
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# 1) Load + summarise
# ---------------------------------------------------------------------------
df = pd.read_csv(os.path.join(HERE, "parity_results.csv"))
df["signed_err"] = df.numpy_val - df.jidt_val
df["abs_err"] = df.signed_err.abs()
df["rel_err"] = df.abs_err / df.jidt_val.abs().clip(lower=1e-12)

summary = (df.groupby(["estimator", "measure", "T"])
             .agg(jidt_mean=("jidt_val", "mean"),
                  numpy_mean=("numpy_val", "mean"),
                  signed_err_mean=("signed_err", "mean"),
                  abs_err_mean=("abs_err", "mean"),
                  abs_err_std=("abs_err", "std"),
                  rel_err_mean=("rel_err", "mean"))
             .reset_index())
summary.to_csv(os.path.join(HERE, "parity_summary.csv"), index=False)

# ---------------------------------------------------------------------------
# 2) Plot abs_err vs T per cell
# ---------------------------------------------------------------------------
cells = list(summary[["estimator", "measure"]].drop_duplicates().itertuples(index=False, name=None))
fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
markers = {"entropy": "o", "MI": "s", "TE": "^"}
colors = {"gaussian": "C0", "kraskov": "C1", "kozachenko": "C2",
          "kernel": "C3", "symbolic": "C4"}

for est, meas in cells:
    sub = summary[(summary.estimator == est) & (summary.measure == meas)]
    err = sub.abs_err_mean.clip(lower=1e-17)  # so log scale doesn't break
    label = f"{est}/{meas}"
    ax.errorbar(sub["T"], err, yerr=sub.abs_err_std.fillna(0),
                marker=markers[meas], color=colors[est],
                label=label, linewidth=1.5, capsize=3)

ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xticks([200, 800, 1600]); ax.set_xticklabels([200, 800, 1600])
ax.set_xlabel("T (samples)")
ax.set_ylabel("|NumPy - JIDT|  (mean +/- std over 10 seeds)")
ax.set_title("Per-estimator parity: NumPy port vs JIDT reference\nbivariate AR(1), 10 seeds")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
ax.axhline(1e-14, color="gray", linestyle=":", alpha=0.6)
ax.text(2000, 1.4e-14, "machine precision", color="gray", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(HERE, "parity_plot.png"), dpi=130, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 3) Build notebook
# ---------------------------------------------------------------------------
nb = nbf.v4.new_notebook()
nb.cells = []

nb.cells.append(nbf.v4.new_markdown_cell(
    "# Infotheory parity benchmark: NumPy port vs JIDT\n"
    "\n"
    "Tests every estimator/measure in `pyspi/statistics/infotheory.py` against the JIDT class it replaced.\n"
    "\n"
    "**Test signal**: bivariate AR(1), unidirectional coupling x->y, 10 seeds per cell.\n"
    "\n"
    "**T**: {200, 800, 1600}.\n"
    "\n"
    "**JIDT setup**: matches the historical pyspi `_setup()` (`BIAS_CORRECTION=false`, `NOISE_SEED=42`), JIDT defaults otherwise.\n"
))

nb.cells.append(nbf.v4.new_code_cell(
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "from IPython.display import Image\n"
    "pd.set_option('display.float_format', lambda v: f'{v:+.3e}')\n"
    "df = pd.read_csv('parity_results.csv')\n"
    "summary = pd.read_csv('parity_summary.csv')\n"
    "print(f'{len(df)} cells (estimator x measure x T x seed)')\n"
))

nb.cells.append(nbf.v4.new_markdown_cell("## Summary table\n"))
nb.cells.append(nbf.v4.new_code_cell(
    "summary.sort_values(['measure', 'estimator', 'T'])\n"
))

nb.cells.append(nbf.v4.new_markdown_cell("## Error vs T\n"))
nb.cells.append(nbf.v4.new_code_cell("Image('parity_plot.png')\n"))

nb.cells.append(nbf.v4.new_markdown_cell(
    "## Findings\n"
    "\n"
    "### Machine-precision parity (rel_err ~1e-14)\n"
    "- `gaussian/MI`, `gaussian/TE`, `kernel/MI`, `kernel/entropy`, `symbolic/TE` agree with JIDT to floating-point precision.\n"
    "\n"
    "### Tiny systematic bias (5e-9 nats, expected)\n"
    "- `gaussian/entropy`: constant offset of `0.5 * log(1 + 1e-8) = 5e-9` nats from the NumPy port's ridge regularisation `Sigma + 1e-8 * mean(diag) * I`. Deterministic analogue of JIDT's stochastic `NOISE_LEVEL_TO_ADD=1e-8`, documented in `_gaussian_log_det`.\n"
    "\n"
    "### Convergent finite-sample agreement\n"
    "- `kraskov/MI`, `kraskov/TE`: KSG-family. Absolute error ~2e-3 nats at T=1600, decreasing with T. Both implementations are unbiased KSG estimators; the residual gap is driven by (i) JIDT's tiny additive observation noise vs NumPy's `eps * (1 - 1e-10)` strict-inequality trick, (ii) `count - 1` vs `count` self-exclusion semantics. Convergence is the expected `O(1/sqrt(N))`.\n"
    "- `kozachenko/entropy`: <1e-4 nats at T=1600.\n"
    "- `kernel/TE`: 5e-5 nats at T=1600.\n"
    "\n"
    "### Bug found + fixed during this benchmark\n"
    "Initial run showed `kernel/entropy` with a constant ~0.21 nats gap independent of T. Root cause: `KernelEntropyCalculator` standardised the data when `NORMALISE=true` but forgot to add the scale-correction term `d * log2(prod(std))` to the entropy. JIDT instead leaves the data raw and rescales the bandwidth (`kernelWidthsInUse = w * std`), which gives the full entropy with the std term built into the volume `(2*w*std)^d`. The two approaches are equivalent only with that correction.\n"
    "\n"
    "Fix applied to `pyspi/statistics/infotheory.py::KernelEntropyCalculator.computeAverageLocalOfObservations` — add `+ sum_d log2(std_d)` when normalise=true. Gap drops from +0.21 to 0.00 +/- 0.00 across all T. MI/TE were unaffected because the std factor cancels in the count ratios.\n"
    "\n"
    "Reference: Kantz & Schreiber, *Nonlinear Time Series Analysis* (1997); Schreiber (2000); Lizier 2014 JIDT paper.\n"
    "\n"
    "## Convergence expectation\n"
    "\n"
    "At T=1600, the typical AR(1) MI/TE we're estimating is ~0.03-0.10 nats with KSG estimator standard error of order `1/sqrt(k * N) ~ 0.012`. The implementation-vs-implementation absolute gap we measure (~2e-3 nats) is one order of magnitude below the inherent estimator noise -- the two implementations agree to within ~10% of one estimator standard deviation. T=200 already resolves the patterns; longer T not needed for this question.\n"
))

out_nb = os.path.join(HERE, "parity_notebook.ipynb")
with open(out_nb, "w") as f:
    nbf.write(nb, f)
print(f"Wrote {out_nb}")
print(f"Wrote {os.path.join(HERE, 'parity_plot.png')}")
print(f"Wrote {os.path.join(HERE, 'parity_summary.csv')}")
