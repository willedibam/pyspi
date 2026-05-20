#!/usr/bin/env python
"""Cut a benchmarked SPI subset config from a bench_compute.py timing JSON.

Replaces the legacy notebook step: reads per-SPI wall times produced by
``bench_compute.py``, ranks SPIs by cost, and emits a config.yaml containing
only the fastest ``--keep`` percent.

Cost model — two modes:
  raw        each SPI's own measured wall time.
  amortized  (default) SPIs that share a within-class cache (``_cache_namespace``
             — Covariance/Precision, the multitaper spectral pairs, Cointegration,
             Barycenter, CCM, ...) split the group's total cost evenly:
                 cost(spi) = sum(group wall times) / (group size)
             This is the true per-variant budget impact: the expensive shared
             computation is built once and reused, so blaming its full cost to
             one variant overcounts. Ungrouped SPIs use their raw time.

Usage:
    python -m bench.cut_config --bench-json bench/results/timings_config_*.json --keep 90
    python -m bench.cut_config --bench-json <json> --keep 80 --mode raw
    python -m bench.cut_config --bench-json <json> --keep 85 --m 16 --t 800 -o out.yaml
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml

from pyspi.calculator import _expand_lagged_correlation_configs

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLED_CONFIG_DIR = REPO_ROOT / "pyspi"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench-json", type=Path, required=True,
                   help="Timing JSON from bench_compute.py.")
    p.add_argument("--config", type=Path, default=BUNDLED_CONFIG_DIR / "config.yaml",
                   help="Source config to cut from (default: bundled config.yaml).")
    p.add_argument("--keep", type=int, default=90,
                   help="Percent of SPIs to keep, fastest-first (default: 90).")
    p.add_argument("--mode", choices=["amortized", "raw"], default="amortized",
                   help="Cost model (default: amortized).")
    p.add_argument("--m", type=int, default=None,
                   help="Select the bench cell with this M (default: largest M).")
    p.add_argument("--t", type=int, default=None,
                   help="Select the bench cell with this T (default: largest T).")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output config path (default: pyspi/benchmarked<keep>[_amortized]_config.yaml).")
    return p.parse_args(argv)


def pick_cell(data: dict, m: int | None, t: int | None) -> dict:
    """Choose one (M, T) cell's results. Prefer n_jobs=1, then largest M, then T."""
    cells = [r for r in data.get("results", []) if "error" not in r and r.get("spi_seconds")]
    if m is not None:
        cells = [r for r in cells if r["M"] == m]
    if t is not None:
        cells = [r for r in cells if r["T"] == t]
    if not cells:
        raise SystemExit(f"No usable cell in bench JSON for M={m}, T={t}.")
    cells.sort(key=lambda r: (r["n_jobs"] != 1, -r["M"], -r["T"]))
    return cells[0]


def walk_spis(configfile: Path):
    """Yield (module_name, class_name, params, identifier, spi) for every SPI in
    the config — mirrors load_spis_from_yaml, including LaggedCorrelation expansion."""
    source = yaml.safe_load(configfile.read_text())
    for module_name, module_spis in source.items():
        module = importlib.import_module(module_name, "pyspi")
        for class_name, entry in (module_spis or {}).items():
            configs = entry.get("configs")
            if class_name == "LaggedCorrelation" and configs is not None:
                configs = _expand_lagged_correlation_configs(configs)
            for params in ([None] if configs is None else configs):
                spi = (getattr(module, class_name)() if params is None
                       else getattr(module, class_name)(**params))
                yield module_name, class_name, params, spi.identifier, spi
    return


def amortized_costs(records: list, raw: dict[str, float]) -> dict[str, float]:
    """Per-SPI cost where _cache_namespace groups split their total evenly."""
    groups: dict = defaultdict(list)
    for _module, _cls, _params, identifier, spi in records:
        groups[getattr(type(spi), "_cache_namespace", None)].append(identifier)
    cost: dict[str, float] = {}
    for namespace, ids in groups.items():
        if namespace is None:
            for i in ids:
                cost[i] = raw[i]
        else:
            share = sum(raw[i] for i in ids) / len(ids)
            for i in ids:
                cost[i] = share
    return cost


def emit_config(source_path: Path, records: list, kept: set[str], header: str) -> str:
    """Re-emit the config with only kept SPIs; classes with nothing kept are dropped."""
    source = yaml.safe_load(source_path.read_text())
    out: dict = {}
    for module_name, class_name, params, identifier, _spi in records:
        if identifier not in kept:
            continue
        module = out.setdefault(module_name, {})
        if class_name not in module:
            src_entry = source[module_name][class_name]
            module[class_name] = {
                "labels": src_entry.get("labels"),
                "dependencies": src_entry.get("dependencies"),
                "configs": [],
            }
        module[class_name]["configs"].append(params)
    # A class with a single default-args SPI has params=None -> emit configs: null.
    for module in out.values():
        for entry in module.values():
            if entry["configs"] == [None]:
                entry["configs"] = None
    body = yaml.dump(out, sort_keys=False, default_flow_style=False)
    return header + body


def main(argv=None) -> int:
    args = parse_args(argv)
    data = json.loads(args.bench_json.read_text())
    cell = pick_cell(data, args.m, args.t)
    raw_cell = {spi: v["mean"] for spi, v in cell["spi_seconds"].items()}

    records = list(walk_spis(args.config))
    ids = [r[3] for r in records]
    missing = sorted(i for i in ids if i not in raw_cell)
    if missing:
        print(f"[cut] WARNING: {len(missing)} SPI(s) in config but not in bench JSON "
              f"— kept unconditionally (cost 0): {', '.join(missing[:5])}"
              f"{' ...' if len(missing) > 5 else ''}", file=sys.stderr)
    raw = {i: raw_cell.get(i, 0.0) for i in ids}

    cost = raw if args.mode == "raw" else amortized_costs(records, raw)
    ranked = sorted(ids, key=lambda i: cost[i])
    n_keep = round(args.keep / 100 * len(ranked))
    kept = set(ranked[:n_keep])
    dropped = ranked[n_keep:]

    kept_max = cost[ranked[n_keep - 1]] if n_keep else 0.0
    drop_min = cost[ranked[n_keep]] if dropped else float("inf")

    sha = (data.get("environment") or {}).get("pyspi_git_sha") or "?"
    header = (
        f"# benchmarked{args.keep}{'_amortized' if args.mode == 'amortized' else ''}_config.yaml\n"
        f"# Generated by bench/cut_config.py on {datetime.now():%Y-%m-%d}.\n"
        f"# Source config : {args.config}\n"
        f"# Bench JSON    : {args.bench_json.name} (pyspi {sha[:12]}, "
        f"cell M={cell['M']} T={cell['T']} n_jobs={cell['n_jobs']})\n"
        f"# Cost model    : {args.mode}\n"
        f"# Keep {args.keep}% : kept {len(kept)} / {len(ranked)} SPIs, dropped {len(dropped)}.\n"
        f"# Cutoff        : fastest kept <= {kept_max:.3f}s ; slowest dropped >= "
        f"{drop_min:.3f}s.\n#\n"
    )
    text = emit_config(args.config, records, kept, header)
    if dropped:
        text += "\n# --- DROPPED (slowest %d, %s cost) ---\n" % (len(dropped), args.mode)
        text += "".join(f"#   {cost[i]:9.3f}s  {i}\n" for i in reversed(dropped))

    output = args.output or (
        BUNDLED_CONFIG_DIR
        / f"benchmarked{args.keep}{'_amortized' if args.mode == 'amortized' else ''}_config.yaml")
    output.write_text(text)
    print(f"[cut] {len(kept)}/{len(ranked)} SPIs kept ({args.mode}, M={cell['M']} "
          f"T={cell['T']}) -> {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
