#!/usr/bin/env python
"""Predict cell wall-time at a target (M, T) for a given config.

Uses the per-SPI log-linear scaling fits in
``bench/results/analysis/scaling.csv`` (produced by ``bench.analyse_cells``).
For each SPI in the chosen config, predicts amortized cost at the target cell
via ``log t = intercept + p_M * log M + q_T * log T``; sums them (since
sum(amortized) = total wall time at n_jobs=1, as verified by cell_summary.csv).

For SPIs not in the scaling fit (cheap noise-floor SPIs that were ~0 in every
cell), uses their median amortized cost across the analysed cells, or 0 if
absent everywhere.

Usage:
    python -m bench.forecast_cell --config pyspi/benchmarked90_amortized_config.yaml \
                                  --M 64 --T 3200
    python -m bench.forecast_cell --config pyspi/config.yaml --M 50 --T 3000 \
                                  --top 20  # also list 20 worst-offender SPIs
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pyspi.calculator import _expand_lagged_correlation_configs, _split_config_params

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCALING = REPO_ROOT / "bench" / "results" / "analysis" / "scaling.csv"
DEFAULT_LONG = REPO_ROOT / "bench" / "results" / "analysis" / "long_costs.csv"


def walk_identifiers(configfile: Path) -> list[str]:
    """Return list of SPI identifiers loaded by this config (mirrors load_spis_from_yaml)."""
    source = yaml.safe_load(configfile.read_text())
    ids = []
    for module_name, module_spis in source.items():
        module = importlib.import_module(module_name, "pyspi")
        for class_name, entry in (module_spis or {}).items():
            if entry is None:
                continue
            configs = entry.get("configs")
            if class_name == "LaggedCorrelation" and configs is not None:
                configs = _expand_lagged_correlation_configs(configs)
            cls = getattr(module, class_name)
            for params in [None] if configs is None else configs:
                if params is None:
                    spi = cls()
                else:
                    ctor_params, _ = _split_config_params(params)
                    spi = cls(**ctor_params)
                ids.append(spi.identifier)
    return ids


def predict_amortized(scaling: pd.DataFrame, long: pd.DataFrame,
                      ident: str, M: int, T: int) -> tuple[float, float]:
    """Predict amortized cost (seconds) and a 1-sigma log-residual envelope.

    Returns (point_estimate, log_sigma) — multiply by exp(+/- log_sigma) for hi/lo.
    Falls back to median across analysed cells for noise-floor SPIs (no fit row).
    """
    row = scaling[scaling["identifier"] == ident]
    if not row.empty:
        r = row.iloc[0]
        log_t = r["intercept"] + r["p_M"] * np.log(M) + r["q_T"] * np.log(T)
        return float(np.exp(log_t)), float(r["log_resid_std"])
    sub = long[long["identifier"] == ident]
    if sub.empty:
        return 0.0, 0.0
    return float(sub["amortized_s"].median()), 0.0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True,
                   help="Path to the config YAML to forecast (bundled or custom).")
    p.add_argument("--M", type=int, required=True)
    p.add_argument("--T", type=int, required=True)
    p.add_argument("--scaling", type=Path, default=DEFAULT_SCALING)
    p.add_argument("--long-costs", type=Path, default=DEFAULT_LONG,
                   help="Per-SPI per-cell amortized costs CSV.")
    p.add_argument("--top", type=int, default=10,
                   help="Show top-N SPIs by predicted cost in the breakdown.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.scaling.exists():
        raise SystemExit(f"missing {args.scaling}; run bench.analyse_cells first.")
    if not args.long_costs.exists():
        raise SystemExit(f"missing {args.long_costs}; run bench.analyse_cells first.")
    scaling = pd.read_csv(args.scaling)
    long = pd.read_csv(args.long_costs)

    idents = walk_identifiers(args.config)
    rows = []
    for ident in idents:
        pred, sigma = predict_amortized(scaling, long, ident, args.M, args.T)
        rows.append({"identifier": ident, "predicted_s": pred, "log_sigma": sigma})
    df = pd.DataFrame(rows).sort_values("predicted_s", ascending=False).reset_index(drop=True)

    total = float(df["predicted_s"].sum())
    fitted = df[df["log_sigma"] > 0]
    sigmas = fitted["log_sigma"].values
    weights = fitted["predicted_s"].values
    weighted_sigma = (float(np.sqrt(np.sum((weights * sigmas) ** 2)) / total)
                      if total > 0 else 0.0)

    print(f"\nForecast for config: {args.config}")
    print(f"  Target cell      : M={args.M} T={args.T}")
    print(f"  N SPIs in config : {len(idents)}")
    print(f"  Fitted / unfitted: {len(fitted)} / {len(idents) - len(fitted)}")
    print(f"\n  Predicted cell wall time (n_jobs=1):")
    print(f"    point  : {total:>12.1f} s   ({total/3600:.2f} h, {total/86400:.2f} d)")
    print(f"    ~lo    : {total * np.exp(-weighted_sigma):>12.1f} s   "
          f"(weighted log-sigma ~ {weighted_sigma:.3f})")
    print(f"    ~hi    : {total * np.exp(+weighted_sigma):>12.1f} s")
    print(f"\n  Top {args.top} SPIs by predicted cost:")
    print(f"    {'identifier':<48s} {'pred_s':>10s} {'log_sigma':>9s}")
    for _, r in df.head(args.top).iterrows():
        print(f"    {r['identifier']:<48s} {r['predicted_s']:>10.2f} "
              f"{r['log_sigma']:>9.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
