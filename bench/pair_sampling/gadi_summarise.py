"""Reduce the Gadi smoke runs to what the notebooks and README use.

    python gadi_summarise.py --runs <copied results root> \\
        --dense <study results>/cached-mpis --pbs pbs.json

``--runs`` holds one folder per ``gadi_smoke.py`` run. ``--dense`` holds the
study's earlier full extractions of the same recordings (computation r7,
matrices stored with a zero diagonal and undirected SPIs symmetrised), against
which values are compared. ``--pbs`` maps run name to PBS's own usage record.
Writes ``results/gadi-smoke/``: each ``report.json``, ``summary.json`` and
``m243-examples.npz``.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from validate import errors, load_record, pair_values, spi_spi_features

OUT = Path(__file__).parent / "results" / "gadi-smoke"
# SPIs whose values changed between computation r7 and r8.
CHANGED = ("sgddtw", "MinCovDet", "EllipticEnvelope", "psi_multitaper")
EXAMPLES = {"Pearson": "cov_EmpiricalCovariance", "Distance correlation": "dcorr",
            "Mutual information (KSG)": "mi_kraskov_NN-4",
            "Transfer entropy (KSG)": "te_kraskov_NN-4_k-1_kt-1_l-1_lt-1"}


def stored_form(values, directed):
    """Sampled ``(K, b, 2)`` entries as the study stored them: undirected SPIs averaged."""
    out = values.copy()
    mean = values.mean(axis=2, keepdims=True)
    out[~directed] = np.broadcast_to(mean, values.shape)[~directed]
    return out


def compare(got, want, names, got_errors):
    """Per-SPI agreement of two ``(K, ...)`` arrays; r8-changed SPIs listed apart."""
    groups = {"identical": [], "rounding": [], "changed_in_r8": [], "different": [],
              "failed_in_either": []}
    for name, a, b in zip(names, got, want):
        if name in got_errors or not np.isfinite(b).any():
            groups["failed_in_either"].append(name)
        elif any(tag in name for tag in CHANGED):
            groups["changed_in_r8"].append(name)
        elif np.array_equal(a, b, equal_nan=True):
            groups["identical"].append(name)
        elif np.allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True):
            groups["rounding"].append(name)
        else:
            groups["different"].append(name)
    return groups


def describe(groups):
    counts = {k: len(v) for k, v in groups.items()}
    text = ", ".join(f"{n} {k.replace('_', ' ')}" for k, n in counts.items())
    return text + (f" ({', '.join(groups['different'])})" if groups["different"] else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--pbs", type=Path, required=True)
    args = parser.parse_args()
    pbs = json.loads(args.pbs.read_text())
    OUT.mkdir(parents=True, exist_ok=True)

    runs, values = [], {}
    for folder in sorted(p for p in args.runs.iterdir() if (p / "report.json").exists()):
        report = json.loads((folder / "report.json").read_text())
        values[folder.name] = np.load(folder / "values.npy")
        (OUT / folder.name).mkdir(exist_ok=True)
        shutil.copy(folder / "report.json", OUT / folder.name / "report.json")
        used = pbs.get(folder.name, {})
        runs.append({
            "run": folder.name, "M": report["M"], "T": report["T"], "mode": report["mode"],
            "n_pairs": report["n_pairs"], "n_jobs": report["n_jobs"],
            "bound_cores": report["bound_cores"], "start_method": report["start_method"],
            "wall_seconds": report["wall_seconds"], "wall_hours": report["wall_seconds"] / 3600,
            "spi_seconds": report["spi_seconds"],
            "parent_rss_gib": report["peak_rss_gib"]["parent"],
            "largest_worker_rss_gib": report["peak_rss_gib"]["largest_worker"],
            "pbs_mem_gib": used.get("mem_gib"), "pbs_walltime": used.get("walltime"),
            "pbs_cput": used.get("cput"), "job": report["job"],
            "failed": len(report["errors"]),
            "values_sha256": hashlib.sha256(
                np.nan_to_num(values[folder.name], nan=-1.0).tobytes()).hexdigest()[:16],
        })
    by_name = {r["run"]: r for r in runs}
    same = lambda names: len({by_name[n]["values_sha256"] for n in names if n in by_name}) == 1
    summary = {"runs": runs, "identical_across_workers": {
        "m64-dense": same([f"m64-dense-j{k}" for k in (1, 4, 8)]),
        "m243-b1000": same([f"m243-b1000-j{k}" for k in (1, 4, 8)])}}

    first = lambda stem: next((f"{stem}-j{k}" for k in (1, 4, 8) if f"{stem}-j{k}" in values), None)

    # M=64: the full table against the earlier full extraction.
    m64 = first("m64-dense")
    report = json.loads((args.runs / m64 / "report.json").read_text())
    folder = next(args.dense.glob(f"*-m64-*/*{report['member'].lower().rsplit('-m', 1)[0]}*"))
    meta, order, stored = load_record(folder)
    assert order == report["spis"]
    directed = np.array([s["directed"] for s in meta["pyspi"]["spis"]])
    table = values[m64].copy()
    table[~directed] = 0.5 * (table[~directed] + table[~directed].transpose(0, 2, 1))
    off = ~np.eye(report["M"], dtype=bool)
    summary["m64_vs_dense"] = ("against the r7 full extraction: "
                               + describe(compare(table[:, off], stored[:, off], order, report["errors"])))

    # M=243: sampled entries against the earlier full extraction, and z.
    m243 = first("m243-b1000")
    if m243:
        report = json.loads((args.runs / m243 / "report.json").read_text())
        folder = next(args.dense.glob(f"*-large-*/*{report['member'].lower().rsplit('-m', 1)[0]}*"))
        meta, order, stored = load_record(folder)
        assert order == report["spis"]
        directed = np.array([s["directed"] for s in meta["pyspi"]["spis"]])
        M = report["M"]
        summary["m243_dense_reference"] = (
            f"full extraction (r7, one core): {sum(meta['pyspi']['timings'].values()) / 3600:.1f} h, "
            f"{len(meta['pyspi']['errors'])} failed SPIs")
        keep = np.array([not any(tag in n for tag in CHANGED) for n in order])
        exact = spi_spi_features(pair_values(stored[keep], np.stack(np.triu_indices(M, 1), 1)))
        z_lines = []
        for name in (m243, "m243-b2500-j8"):
            if name not in values:
                continue
            pairs = np.load(args.runs / name / "pairs.npy")
            got = stored_form(values[name], directed)
            want = pair_values(stored, pairs)
            run_errors = json.loads((args.runs / name / "report.json").read_text())["errors"]
            if name == m243:
                summary["m243_vs_dense"] = ("sampled entries against the r7 full extraction: "
                                            + describe(compare(got, want, order, run_errors)))
                index = [order.index(k) for k in EXAMPLES.values()]
                np.savez_compressed(
                    OUT / "m243-examples.npz", names=np.array(list(EXAMPLES)), pairs=pairs,
                    dense=stored[index].astype(np.float32), sampled=got[index].astype(np.float32))
            b, D = len(pairs), M * (M - 1) // 2
            e = errors(spi_spi_features(got[keep]), exact)
            z_lines.append(f"b={b}: RMSE {e['rmse']:.4f} (model with c=0.90: "
                           f"{0.90 * np.sqrt((1 - b / D) / b):.4f}), "
                           f"{100 * e['retained']:.1f}% of coordinates retained")
        summary["m243_z_error"] = ("z from the sampled run against z from the full extraction, "
                                   "SPIs changed in r8 excluded -- " + "; ".join(z_lines))
        if "m243-b2500-j8" in values:
            small, big = (np.load(args.runs / n / "pairs.npy") for n in (m243, "m243-b2500-j8"))
            summary["nested_prefix_identical"] = bool(
                np.array_equal(big[:len(small)], small)
                and np.array_equal(values["m243-b2500-j8"][:, :len(small)],
                                   values[m243], equal_nan=True))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=1))


if __name__ == "__main__":
    main()
