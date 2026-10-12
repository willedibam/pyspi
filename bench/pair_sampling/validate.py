"""Validate ``Calculator.compute_pairs`` against dense extraction and cached MPIs.

Three steps, each writing into ``--out`` (default ``bench/pair_sampling/results``):

``extract``
    One small series (VAR(1), M <= 20). Dense ``compute`` once, then
    ``compute_pairs`` for a few pair seeds and budgets. Checks that every sampled entry is
    the dense entry, and that the SPI-SPI vector built from the sampled
    extraction is the one obtained by masking the dense matrices with the same
    dyads. Records wall time by execution tier.

``retrospective``
    No extraction. Applies the same sampler and SPI-SPI helper to (a) the local
    dense matrices from ``extract`` and (b) the cached large-M MPIs of the
    mts-spi-study-cluster pair-sampling panel (``--study-results``), where the
    per-draw errors must reproduce that study's ``all-draws.csv``. Also projects
    the large-M speedup from the per-SPI timings stored with those MPIs.

``figures``
    Convergence figures in the study notebook's style.

The SPI-SPI vector z holds, for every pair of SPIs, the Pearson correlation of
their values over the ordered process pairs. It is defined here, not in pyspi.
"""
import argparse
import hashlib
import json
import random
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from pyspi import _parallel
from pyspi.calculator import Calculator, sample_dyads

OUT = Path(__file__).parent / "results"
CONFIG = "benchmarked_p90"
REPEATS = 128
ESTIMATOR_SEED = 0
TIERS = ("pair loop", "pair-local", "indexed")
TIGHT = dict(atol=1e-12, rtol=1e-9)

PEARSON = "cov_EmpiricalCovariance"  # Equal to Pearson on z-scored processes.
BAND = "multitaper_mean_fs-1_fmin-0_fmax-0-5"
PLV, COH, PLI = f"plv_{BAND}", f"cohmag_{BAND}", f"pli_{BAND}"
GC = "gc_gaussian_k-1_kt-1_l-1_lt-1"
TE = "te_kraskov_NN-4_k-1_kt-1_l-1_lt-1"
# The study's twelve coordinates, then twelve more. All chosen by what the
# statistics mean, before looking at how well any of them is recovered.
EXAMPLES = [
    ("Spearman × Kendall", "spearmanr", "kendalltau"),
    ("Pearson × Spearman", PEARSON, "spearmanr"),
    ("Pearson × Kendall", PEARSON, "kendalltau"),
    ("Pearson × lag-1 Pearson", PEARSON, "corr_pearson_tau-1"),
    ("Pearson × distance corr.", PEARSON, "dcorr"),
    ("Pearson × Gaussian MI", PEARSON, "mi_gaussian"),
    ("Gaussian MI × KSG MI", "mi_gaussian", "mi_kraskov_NN-4"),
    ("Pearson × PLV", PEARSON, PLV),
    ("Coherence × PLV", COH, PLV),
    ("Pearson × DTW", PEARSON, "dtw"),
    ("Granger × transfer entropy", GC, TE),
    ("Lag-1 Pearson × Granger", "corr_pearson_tau-1", GC),
    ("Spearman × distance corr.", "spearmanr", "dcorr"),
    ("Pearson × max cross-corr.", PEARSON, "xcorr_max_sig-True"),
    ("Pearson × precision", PEARSON, "prec_EmpiricalCovariance"),
    ("Pearson × Euclidean dist.", PEARSON, "pdist_euclidean"),
    ("Coherence × imag. coherence", COH, f"icoh_{BAND}"),
    # Not PLV x PLI: PLI is antisymmetric and PLV symmetric, so over both
    # directions of any set of dyads that coordinate is identically zero.
    ("PLV × phase consistency", PLV, f"ppc_{BAND}"),
    ("PLI × weighted PLI", PLI, f"wpli_{BAND}"),
    ("Granger × directed info.", GC, "di_gaussian_n-5"),
    ("Transfer entropy × lagged MI", TE, "tlmi_kraskov_NN-4"),
    ("DTF × partial dir. coherence", f"dtf_{BAND}", f"pdcoh_{BAND}"),
    ("Phase-slope index × Granger", f"psi_{BAND}", GC),
    ("Envelope corr. × Pearson", "pec", PEARSON),
]


# --------------------------------------------------------------------------
# SPI-SPI vector and its error
# --------------------------------------------------------------------------

def pair_values(matrices, pairs):
    """``(K, M, M)`` dense matrices -> ``(K, b, 2)`` entries at the dyads."""
    i, j = pairs.T
    return np.stack([matrices[:, i, j], matrices[:, j, i]], axis=-1)


def spi_spi_features(values):
    """Pearson correlation between SPIs over the ``2b`` sampled entries.

    Returns the upper triangle, K(K-1)/2 coordinates. A coordinate is NaN when
    either SPI is non-finite anywhere on the sample or constant on it.
    """
    vectors = values.reshape(len(values), -1)
    centred = vectors - vectors.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centred, axis=1)
    valid = np.isfinite(centred).all(axis=1) & (norms >= 1e-12)
    unit = np.zeros_like(centred)
    unit[valid] = centred[valid] / norms[valid, None]
    z = unit @ unit.T
    z[~valid, :] = np.nan
    z[:, ~valid] = np.nan
    return z[np.triu_indices(len(values), 1)]


def errors(estimate, exact):
    """Error on coordinates defined in both, and coverage of the exact ones."""
    reference = np.isfinite(exact)
    shared = reference & np.isfinite(estimate)
    error = np.abs(estimate[shared] - exact[shared])
    return dict(rmse=float(np.sqrt(np.mean(error ** 2))),
                p95_absolute_error=float(np.quantile(error, .95)),
                retained=float(shared.sum() / reference.sum()))


def budgets_for(n_dyads):
    ladder = [8, 16, 32, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]
    return sorted({min(n_dyads, b) for b in ladder + [n_dyads]})


def convergence(matrices, seed_for_draw, repeats=REPEATS, select=None):
    """Nested-budget draws from dense matrices.

    Returns the budgets, the exact vector, and estimates of shape
    ``(budgets, repeats, coordinates)``. ``select`` keeps a subset of
    coordinates to bound memory.
    """
    M = matrices.shape[1]
    n_dyads = M * (M - 1) // 2
    budgets = budgets_for(n_dyads)
    keep = slice(None) if select is None else select
    exact = spi_spi_features(pair_values(matrices, sample_dyads(M, n_dyads, 0)))[keep]
    estimates = np.empty((len(budgets), repeats, len(exact)), dtype=np.float32)
    for r in range(repeats):
        values = pair_values(matrices, sample_dyads(M, n_dyads, seed_for_draw(r)))
        for k, b in enumerate(budgets):
            estimates[k, r] = exact if b == n_dyads else spi_spi_features(values[:, :b])[keep]
    return np.array(budgets), exact, estimates


def draw_table(budgets, exact, estimates, **labels):
    return pd.DataFrame([dict(labels, draw=r, dyads=int(b), **errors(estimates[k, r], exact))
                         for k, b in enumerate(budgets) for r in range(estimates.shape[1])])


# --------------------------------------------------------------------------
# extract
# --------------------------------------------------------------------------

def var_series(M, T, seed):
    """Stable VAR(1) on a randomly labelled directed ring, spectral radius 0.85.

    The sparse VAR design of the study panel: x[t] = A x[t-1] + e[t] with
    A = 0.55 I + 0.30 W, W a cyclic permutation, e iid N(0, I).
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(M)
    A = .55 * np.eye(M)
    A[order, np.roll(order, 1)] += .30
    x = rng.standard_normal(M)
    for _ in range(500):
        x = A @ x + rng.standard_normal(M)
    out = np.empty((M, T))
    for t in range(T):
        x = A @ x + rng.standard_normal(M)
        out[:, t] = x
    return out


def _seed_estimators():
    np.random.seed(ESTIMATOR_SEED)
    random.seed(ESTIMATOR_SEED)


def tier_of(spi):
    """How compute_pairs evaluates an SPI.

    ``pair loop``: the dense matrix is itself a loop over ``bivariate``, replayed
    on the sampled dyads. ``pair-local``: a vectorised dense path, with
    ``bivariate`` called on the sampled dyads. ``indexed``: computed in full.
    """
    if _parallel._dense_loop(spi) is None:
        return "indexed"
    return ("pair-local" if getattr(type(spi), "_pair_local", None) == "directed"
            else "pair loop")


def extract(out, M, T, pair_seeds, budgets):
    if M > 20:
        raise ValueError("Local extraction is limited to M <= 20.")
    series = var_series(M, T, seed=261012)

    def calculator():
        return Calculator(dataset=series, config=CONFIG, verbose=False)

    dense = calculator()
    keys = list(dense.spis)
    tier = {k: tier_of(s) for k, s in dense.spis.items()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Pay import and JIT-compilation costs before anything is timed;
        # otherwise they land on whichever run happens to go first.
        Calculator(dataset=series[:3], config=CONFIG, verbose=False).compute(progress=False)
        _seed_estimators()
        t0 = time.perf_counter()
        dense.compute(progress=False)
        dense_wall = time.perf_counter() - t0
        matrices = np.stack([dense.table[k].to_numpy(dtype=float) for k in keys])

        runs = []
        for n_pairs in budgets:
            for seed in pair_seeds:
                calc = calculator()
                _seed_estimators()
                t0 = time.perf_counter()
                calc.compute_pairs(n_pairs=n_pairs, seed=seed, progress=False)
                runs.append(dict(seed=seed, n_pairs=n_pairs, wall=time.perf_counter() - t0,
                                 pairs=calc.pairs, values=calc.pair_values,
                                 errors=calc.errors, timings=calc.timings))

    def by_tier(timings):
        return {name: float(sum(v for k, v in timings.items() if tier[k] == name))
                for name in TIERS}

    n_dyads = M * (M - 1) // 2
    report = dict(M=M, T=T, config=CONFIG, n_spis=len(keys), n_dyads=n_dyads,
                  estimator_seed=ESTIMATOR_SEED,
                  tiers={name: sum(t == name for t in tier.values()) for name in TIERS},
                  dense=dict(wall=dense_wall, errors=dense.errors,
                             by_tier=by_tier(dense.timings)), sampled=[])
    index = {k: n for n, k in enumerate(keys)}
    dense_tier = report["dense"]["by_tier"]
    for run in runs:
        masked = pair_values(matrices, run["pairs"])
        gaps = {}
        for k in keys:
            got, want = run["values"][index[k]], masked[index[k]]
            if np.array_equal(got, want, equal_nan=True):
                continue
            # Only a pair-local SPI may differ at all, and only by rounding.
            assert tier[k] == "pair-local" and np.allclose(got, want, equal_nan=True, **TIGHT), k
            gaps[k] = float(np.nanmax(np.abs(got - want)))
        # A sampled SPI fails exactly when the dense one did, or when the dense
        # matrix holds no finite value on these particular dyads.
        expected_failures = set(dense.errors) | {
            k for k in keys if not np.isfinite(masked[index[k]]).any()}
        assert set(run["errors"]) == expected_failures, (run["errors"], dense.errors)

        z_sampled, z_masked = spi_spi_features(run["values"]), spi_spi_features(masked)
        assert np.array_equal(np.isnan(z_sampled), np.isnan(z_masked))
        z_gap = float(np.nanmax(np.abs(z_sampled - z_masked)))
        assert z_gap < 1e-9, z_gap
        sampled_tier = by_tier(run["timings"])
        # Cost model for the two tiers that follow from dense timings alone:
        # indexed SPIs cost what they cost densely, and the replayed pair loops
        # scale with the fraction of dyads visited.
        fraction = run["n_pairs"] / n_dyads
        report["sampled"].append(dict(
            seed=run["seed"], n_pairs=run["n_pairs"], wall=run["wall"],
            speedup=dense_wall / run["wall"], by_tier=sampled_tier,
            predicted_pair_loop=fraction * dense_tier["pair loop"],
            identical_spis=len(keys) - len(gaps), rounding_spis=len(gaps),
            max_abs_entry_gap=max(gaps.values(), default=0.0), max_abs_z_gap=z_gap))

    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "extraction.npz", series=series, spis=np.array(keys, dtype="U"),
        matrices=matrices,
        **{f"values_b{r['n_pairs']}_seed{r['seed']}": r["values"] for r in runs})
    (out / "extraction.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("M", "T", "n_dyads", "tiers")}))
    print(f"dense {dense_wall:.0f}s by tier { {k: round(v, 1) for k, v in dense_tier.items()} }")
    for r in report["sampled"]:
        print(f"b={r['n_pairs']} seed {r['seed']}: {r['wall']:.0f}s ({r['speedup']:.2f}x) by tier "
              f"{ {k: round(v, 1) for k, v in r['by_tier'].items()} }, pair loop predicted "
              f"{r['predicted_pair_loop']:.1f}s; {r['identical_spis']} SPIs bit-identical, "
              f"{r['rounding_spis']} within rounding (max {r['max_abs_entry_gap']:.1e}); "
              f"max |z gap| {r['max_abs_z_gap']:.1e}")


# --------------------------------------------------------------------------
# retrospective
# --------------------------------------------------------------------------

def study_seed(name):
    """The study's per-record draw seeds, so its draws are reproduced exactly."""
    record = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
    return lambda r: np.random.SeedSequence([260917801, record, r])


def load_record(directory):
    meta = json.loads((directory / "meta.json").read_text())
    order = [s["name"] for s in meta["pyspi"]["spis"]]
    with np.load(directory / "spi_mpis.npz", allow_pickle=False) as bank:
        matrices = np.stack([bank[n] for n in order]).astype(float)
    return meta, order, matrices


def project_speedup(meta, n_pairs_list, local):
    """Sampled wall time implied by the per-SPI dense timings of one record.

    Indexed SPIs keep their dense cost and replayed pair loops scale with the
    fraction of dyads. A pair-local SPI has no dense timing to scale, so its
    cost per dyad is taken from the local extraction and converted to the
    record's machine by the ratio of the two machines' pair-loop cost per dyad.
    Both were run at T=1000; the conversion assumes that ratio carries over.
    """
    calc = Calculator(config=CONFIG, verbose=False)
    timings = meta["pyspi"]["timings"]
    dense = {name: sum(t for k, t in timings.items() if tier_of(calc.spis[k]) == name)
             for name in TIERS}
    total = sum(timings.values())
    n_dyads = meta["M"] * (meta["M"] - 1) // 2
    machine = (dense["pair loop"] / n_dyads) / (
        local["dense"]["by_tier"]["pair loop"] / local["n_dyads"])
    pair_local_per_dyad = machine * np.mean(
        [r["by_tier"]["pair-local"] / r["n_pairs"] for r in local["sampled"]])
    rows = []
    for b in n_pairs_list:
        b = min(b, n_dyads)
        sampled = (dense["indexed"] + dense["pair loop"] * b / n_dyads
                   + min(dense["pair-local"], pair_local_per_dyad * b))
        without = dense["indexed"] + dense["pair-local"] + dense["pair loop"] * b / n_dyads
        rows.append(dict(name=meta["name"], M=meta["M"], T=meta["T"], n_dyads=n_dyads,
                         n_pairs=b, dense_hours=total / 3600,
                         indexed_share=dense["indexed"] / total,
                         projected_hours=sampled / 3600, projected_speedup=total / sampled,
                         speedup_without_pair_local=total / without))
    return rows


def retrospective(out, study):
    report = json.loads((out / "extraction.json").read_text())
    with np.load(out / "extraction.npz") as local:
        keys = [str(k) for k in local["spis"]]
        matrices = local["matrices"]
        sampled = {name: local[name] for name in local.files if name.startswith("values_")}
    budgets, exact, estimates = convergence(matrices, seed_for_draw=int)
    M = matrices.shape[1]
    draw_table(budgets, exact, estimates, M=M).to_csv(out / "local-draws.csv", index=False)
    # The extracted values themselves, at the budgets they contain as prefixes.
    rows = [dict(run=name, dyads=int(b), **errors(spi_spi_features(values[:, :b]), exact))
            for name, values in sampled.items()
            for b in list(budgets) + [values.shape[1]] if b <= values.shape[1]]
    pd.DataFrame(rows).to_csv(out / "local-extracted.csv", index=False)
    lookup = {frozenset((keys[i], keys[j])): n
              for n, (i, j) in enumerate(zip(*np.triu_indices(len(keys), 1)))}
    chosen = [lookup[frozenset((a, b))] for _, a, b in EXAMPLES]
    np.savez_compressed(out / f"gallery-local-M{M}.npz", budgets=budgets, exact=exact[chosen],
                        estimates=estimates[:, :, chosen])
    if study is None:
        return

    cached = sorted((study / "cached-mpis").glob("*/*-meg-*"))
    published = pd.read_csv(study / "sampling-available" / "all-draws.csv")
    frames, projections = [], []
    for directory in cached:
        meta, order, matrices = load_record(directory)
        budgets, exact, estimates = convergence(matrices, study_seed(meta["dataset_name"]))
        frame = draw_table(budgets, exact, estimates, name=meta["dataset_name"], M=meta["M"])
        theirs = published[published.name == meta["dataset_name"]].set_index(["dyads", "draw"])
        ours = frame.set_index(["dyads", "draw"]).loc[theirs.index]
        for column in ("rmse", "p95_absolute_error", "retained"):
            np.testing.assert_allclose(ours[column], theirs[column], rtol=0, atol=2e-6)
        frames.append(frame)
        projections += project_speedup(meta, [500, 1000, 2500, 5000], report)
        print(f"{meta['dataset_name']}: {len(theirs)} draws reproduce the study to 2e-6")
        if meta["dataset_name"].startswith("meg-105923-run6-block01"):
            lookup = {frozenset((order[i], order[j])): n
                      for n, (i, j) in enumerate(zip(*np.triu_indices(len(order), 1)))}
            chosen = [lookup[frozenset((a, b))] for _, a, b in EXAMPLES]
            np.savez_compressed(out / f"gallery-meg-M{meta['M']}.npz", budgets=budgets,
                                exact=exact[chosen], estimates=estimates[:, :, chosen])
    for directory in sorted((study / "cached-mpis").glob("*-large-*/*-var-sparse-*"))[:1]:
        projections += project_speedup(load_record(directory)[0], [500, 1000, 2500, 5000],
                                       report)
    pd.concat(frames).to_csv(out / "meg-draws.csv", index=False)
    pd.DataFrame(projections).to_csv(out / "projected-speedup.csv", index=False)
    medians = pd.concat(frames).groupby(["M", "dyads"]).rmse.median().unstack(0)
    print(medians.loc[[100, 400, 800]].round(5).to_string())


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

COLORS = {20: "#555555", 64: "#2879a8", 100: "#d46a29", 243: "#3b9b78"}
X_LABEL = (r"$b$ sampled channel dyads $(i,j) = 2b$ directed channel-pairs: "
           r"$(i\to j)$ and $(j\to i)$")


def figure_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not (shutil.which("latex") and shutil.which("dvipng")):
        raise RuntimeError("These figures use LaTeX text: latex and dvipng are required.")
    plt.rcParams.update({
        "text.usetex": True, "font.family": "serif",
        "font.serif": ["Computer Modern Roman"], "mathtext.fontset": "cm",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out", "legend.frameon": False,
        "lines.linewidth": 1.7, "lines.markersize": 2.7,
        "figure.constrained_layout.use": True, "savefig.bbox": "tight"})
    return plt


def _save(fig, path):
    for ext in ("png", "svg"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=180)


def whole_vector_figure(plt, frame, path, title, extracted=None):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    panels = [("rmse", r"RMSE$(\hat{\mathbf{z}},\mathbf{z})$", "RMSE"),
              ("p95_absolute_error", "95th-percentile absolute feature error", "Absolute error"),
              ("retained", r"Reference features retained (\%)", r"Features retained (\%)")]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.5))
    for M, part in frame.groupby("M"):
        groups = part.groupby("dyads")
        n = part.name.nunique() if "name" in part else 1
        for ax, (key, _, _) in zip(axes, panels):
            scale = 100 if key == "retained" else 1
            med = groups[key].median() * scale
            ax.plot(med.index, med, "o-", color=COLORS[M], label=rf"$M={M}$ ($n={n}$)")
            ax.fill_between(med.index, groups[key].quantile(.1) * scale,
                            groups[key].quantile(.9) * scale, color=COLORS[M], alpha=.14)
            ax.plot(med.index[-1], med.iloc[-1], "D", color=COLORS[M], ms=4)
    if extracted is not None:
        for ax, (key, _, _) in zip(axes, panels):
            ax.plot(extracted.dyads, extracted[key] * (100 if key == "retained" else 1),
                    "o", mfc="none", mec="#c0392b", ms=5, mew=.8, zorder=4)
    for ax, (key, name, label) in zip(axes, panels):
        ax.set(title=name, ylabel=label, xscale="log")
        ax.grid(axis="y", alpha=.12)
    axes[0].axhline(.05, color=".55", ls=":", lw=.8)
    axes[2].set_ylim(100 * frame.retained.min() - 1, 100.3)
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [Patch(facecolor=".5", alpha=.14), Line2D([], [], color=".3", marker="D", ls="none"),
                Line2D([], [], color=".55", ls=":", lw=.8)]
    labels += [r"10--90\% instance/draw spread", "Full census", r"RMSE $=0.05$ (illustrative)"]
    if extracted is not None:
        handles.append(Line2D([], [], marker="o", mfc="none", mec="#c0392b", ls="none", ms=5))
        labels.append(r"\texttt{compute\_pairs} extraction")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, -.015),
               ncol=len(handles), fontsize=8)
    fig.supxlabel(X_LABEL)
    fig.suptitle(title, fontsize=10)
    _save(fig, path)
    plt.close(fig)


def gallery_figure(plt, source, path, color, title):
    from matplotlib.lines import Line2D
    with np.load(source) as saved:
        budgets, exact, estimates = saved["budgets"], saved["exact"], saved["estimates"]
    fig, axes = plt.subplots(4, 6, figsize=(13.0, 7.4), sharex=True)
    for k, (ax, (label, _, _)) in enumerate(zip(axes.flat, EXAMPLES)):
        v = estimates[:, :, k].astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            low, med, high = np.nanquantile(v, [.025, .5, .975], axis=1)
        ax.plot(budgets, v, color=".35", alpha=.05, lw=.45, zorder=1)
        ax.fill_between(budgets, low, high, color=color, alpha=.12, label=r"2.5--97.5\% of draws")
        ax.plot(budgets, med, "o-", color=color, ms=2.2, lw=1.4,
                label=f"Median of {v.shape[1]} draws")
        ax.axhline(exact[k], color=".15", ls="--", lw=.9, label="Exact all-pairs value")
        valid = np.isfinite(v).mean(axis=1).min()
        value = "undefined" if not np.isfinite(exact[k]) else rf"$z = {exact[k]:.3f}$"
        extra = "" if valid == 1 else rf"; min. valid {100 * valid:.0f}\%"
        ax.set_title(label.replace("×", r"$\times$") + "\n" + value + extra, fontsize=7.5)
        ax.set_xscale("log")
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", alpha=.12)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(Line2D([], [], color=".6", lw=.6))
    labels.append(f"{estimates.shape[1]} nested paths (faint)")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, -.015), ncol=4, fontsize=8)
    fig.supxlabel(X_LABEL)
    fig.supylabel(r"Estimated SPI-SPI feature value $\hat{z}$")
    fig.suptitle(title, fontsize=10)
    _save(fig, path)
    plt.close(fig)


def figures(out):
    plt = figure_style()
    report = json.loads((out / "extraction.json").read_text())
    M, T = report["M"], report["T"]
    local = pd.read_csv(out / "local-draws.csv")
    whole_vector_figure(
        plt, local, out / "local-full-vector-convergence",
        rf"data=VAR(1) ($M={M}$, $T={T}$, 1 instance) $\cdot$ {REPEATS} draws $\cdot$ "
        r"median curves", extracted=pd.read_csv(out / "local-extracted.csv"))
    gallery_figure(plt, out / f"gallery-local-M{M}.npz", out / f"gallery-local-M{M}",
                   COLORS[M], rf"select examples $\cdot$ data=VAR(1) ($M={M}$, $T={T}$, "
                   rf"1 instance) $\cdot$ {REPEATS} draws $\cdot$ vertical scales differ")
    if (out / "meg-draws.csv").exists():
        meg = pd.read_csv(out / "meg-draws.csv")
        sizes = ",".join(map(str, sorted(meg.M.unique())))
        whole_vector_figure(
            plt, meg, out / "meg-full-vector-convergence",
            rf"data=MEG ($M={sizes}$, $T=1000$, instances=4) $\cdot$ {REPEATS} draws per instance")
        for source in sorted(out.glob("gallery-meg-M*.npz")):
            size = int(source.stem.rsplit("M", 1)[1])
            gallery_figure(plt, source, out / source.stem, COLORS[size],
                           rf"select examples $\cdot$ data=MEG ($M={size}$, $T=1000$, "
                           rf"1 instance) $\cdot$ {REPEATS} draws $\cdot$ vertical scales differ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("step", choices=["extract", "retrospective", "figures"])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--M", type=int, default=20)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--n-pairs", type=int, nargs="+", default=[95, 19])
    parser.add_argument("--pair-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--study-results", type=Path, default=None,
                        help="mts-spi-study-cluster .../large_m_pair_baselines_260917")
    args = parser.parse_args()
    if args.step == "extract":
        extract(args.out, args.M, args.T, args.pair_seeds, args.n_pairs)
    elif args.step == "retrospective":
        retrospective(args.out, args.study_results)
    else:
        figures(args.out)
