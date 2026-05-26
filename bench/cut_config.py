#!/usr/bin/env python
"""Cut a benchmarked SPI subset config from a bench_compute.py per-cell JSON.

Reads ONE per-cell JSON (``<label>_M<M>_T<T>_n<n>.json``), ranks SPIs by cost,
and emits a ``benchmarked<keep>[_amortized]_config.yaml`` containing only the
fastest ``--keep`` percent. (M, T, n_jobs) are read from the JSON itself, not
parsed from the filename.

Cost model — two modes:
  raw        each SPI's own measured wall time at this (M, T).
  amortized  (default) SPIs sharing a within-class cache (``_cache_namespace``
             — Covariance/Precision, multitaper spectral pairs, CCM,
             Cointegration, Barycenter, ...) split the group's total cost
             evenly:
                 cost(spi) = sum(group wall times) / (group size)
             The shared computation is built once and reused, so blaming its
             full cost to one variant overcounts. Ungrouped SPIs use raw time.

Usage:
    python -m bench.cut_config --bench-json bench/results/cells/physics_config_M64_T3200_n1.json --keep 90
    python -m bench.cut_config --bench-json <path> --keep 80 --mode raw
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

from pyspi.calculator import _expand_lagged_correlation_configs, _split_config_params

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLED_CONFIG_DIR = REPO_ROOT / "pyspi"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench-json", type=Path, required=True,
                   help="Per-cell timing JSON from bench_compute.py.")
    p.add_argument("--config", type=Path, default=BUNDLED_CONFIG_DIR / "config.yaml",
                   help="Source config to cut from (default: bundled config.yaml).")
    p.add_argument("--keep", type=int, default=90,
                   help="Percent of SPIs to keep, fastest-first (default: 90).")
    p.add_argument("--mode", choices=["amortized", "raw"], default="amortized",
                   help="Cost model (default: amortized).")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output config path (default: pyspi/benchmarked<keep>[_amortized]_config.yaml).")
    p.add_argument("--no-preserve-dropped", action="store_true",
                   help="Delete dropped variants/classes from the output instead of "
                        "commenting them out (default: preserve as comments).")
    return p.parse_args(argv)


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
                if params is None:
                    spi = getattr(module, class_name)()
                    ctor_params = None
                else:
                    ctor_params, _ = _split_config_params(params)
                    spi = getattr(module, class_name)(**ctor_params)
                yield module_name, class_name, params, spi.identifier, spi


def amortized_costs(records: list, raw: dict[str, float]) -> dict[str, float]:
    """Amortize cost within each shared-cache bucket.

    A bucket is identified by ``(_cache_namespace, *_cache_subkey)``. The
    subkey (a per-instance tuple, default ``()``) lets a class declare which
    constructor params split its namespace into independent caches — e.g.
    Barycenter caches per ``mode``, so its variants amortize per mode rather
    than across all 16 of them.
    """
    groups: dict = defaultdict(list)
    for _, _, _, identifier, spi in records:
        ns = getattr(type(spi), "_cache_namespace", None)
        if ns is None:
            groups[None].append(identifier)
            continue
        subkey = tuple(getattr(spi, "_cache_subkey", ()))
        groups[(ns, *subkey)].append(identifier)
    cost: dict[str, float] = {}
    for key, ids in groups.items():
        if key is None:
            for i in ids:
                cost[i] = raw[i]
        else:
            share = sum(raw[i] for i in ids) / len(ids)
            for i in ids:
                cost[i] = share
    return cost


def _dump_variant(params: dict | None) -> list[str]:
    """Render one config variant (a param dict, or None for no-args) as YAML lines.

    Returns a list of lines like ``["- estimator: kraskov", "  prop_k: 4"]``. For
    None (no-args SPI), returns ``["- {}"]`` — but that case shouldn't reach here
    in normal use (it's handled at the class level via ``configs: null``).
    """
    if params is None or params == {}:
        return ["- {}"]
    text = yaml.dump([params], sort_keys=False, default_flow_style=False, indent=2).rstrip()
    return text.splitlines()


def _render_class_block(class_name: str, src_entry: dict,
                        kept: list, dropped: list) -> str:
    """Render a class block: labels, dependencies, configs (kept), then commented dropped.

    The class header is at the LEFT MARGIN (the caller indents under the module).
    Returns a string with NO trailing newline.

    Special cases:
      - single no-args SPI (configs: null in source):  kept=[None] -> configs: null
        kept; kept=[] -> the whole class is fully dropped (caller comments it out).
      - some kept, some dropped: kept variants emitted normally, dropped appended as
        ``  # - estimator: ...`` comments under the configs: list.
    """
    lines = [f"{class_name}:"]
    labels = src_entry.get("labels")
    if labels is not None:
        lines.append("  labels:")
        for lab in labels:
            lines.append(f"    - {lab}")
    if "dependencies" in src_entry:
        deps = src_entry["dependencies"]
        if deps is None:
            lines.append("  dependencies:")
        else:
            lines.append("  dependencies:")
            for d in deps:
                lines.append(f"    - {d}")

    # configs handling
    if kept == [None]:
        lines.append("  configs:")
    else:
        lines.append("  configs:")
        for p in kept:
            if p is None:
                continue
            for vl in _dump_variant(p):
                lines.append("  " + vl)
        for p in dropped:
            if p is None:
                # commented no-args means the class is fully dropped — handled by caller
                lines.append("  # (no-args variant dropped)")
                continue
            for vl in _dump_variant(p):
                lines.append("  # " + vl)
    return "\n".join(lines)


def emit_config(source_path: Path, records: list, kept_ids: set[str], header: str,
                preserve_dropped: bool = True) -> str:
    """Emit the cut YAML. If preserve_dropped, dropped variants/classes are kept
    as commented blocks instead of being removed."""
    source = yaml.safe_load(source_path.read_text())

    # Group records by (module, class), preserving source order.
    grouped: dict[tuple[str, str], list[tuple]] = {}
    order: list[tuple[str, str]] = []
    for module, cls, params, ident, _ in records:
        key = (module, cls)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append((params, ident))

    out_lines: list[str] = [header.rstrip()]
    current_module: str | None = None

    for (module, cls) in order:
        variants = grouped[(module, cls)]
        kept = [p for p, i in variants if i in kept_ids]
        dropped = [p for p, i in variants if i not in kept_ids]

        if not preserve_dropped and not kept:
            continue  # drop the class entirely

        if module != current_module:
            if current_module is not None:
                out_lines.append("")
            out_lines.append(f"{module}:")
            current_module = module

        src_entry = source[module][cls]
        if kept or not preserve_dropped:
            # Some variants kept: render normally with kept + commented dropped.
            block = _render_class_block(cls, src_entry, kept, dropped if preserve_dropped else [])
            out_lines.append("\n".join("  " + l for l in block.splitlines()))
        else:
            # All variants dropped: render the full class block and comment every line.
            block = _render_class_block(cls, src_entry, dropped, [])
            out_lines.append("\n".join("  # " + l for l in block.splitlines()))
        out_lines.append("")  # blank line between class blocks

    return "\n".join(out_lines) + "\n"


def main(argv=None) -> int:
    args = parse_args(argv)
    cell = json.loads(args.bench_json.read_text())
    if "spi_seconds" not in cell:
        raise SystemExit(
            f"{args.bench_json} is not a per-cell bench JSON "
            "(no 'spi_seconds' at top level).")
    if "error" in cell:
        raise SystemExit(f"{args.bench_json} has an error entry: {cell['error']}")
    raw_cell = {spi: v["mean"] for spi, v in cell["spi_seconds"].items()}
    M, T, n_jobs = cell.get("M"), cell.get("T"), cell.get("n_jobs")

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

    env = cell.get("environment") or {}
    sha = env.get("pyspi_git_sha") or "?"
    header = (
        f"# benchmarked{args.keep}{'_amortized' if args.mode == 'amortized' else ''}_config.yaml\n"
        f"# Generated by bench/cut_config.py on {datetime.now():%Y-%m-%d}.\n"
        f"# Source config : {args.config}\n"
        f"# Bench JSON    : {args.bench_json.name} (pyspi {sha[:12]}, "
        f"M={M} T={T} n_jobs={n_jobs})\n"
        f"# Cost model    : {args.mode}\n"
        f"# Keep {args.keep}% : kept {len(kept)} / {len(ranked)} SPIs, dropped {len(dropped)}.\n"
        f"# Cutoff        : fastest kept <= {kept_max:.3f}s ; slowest dropped >= "
        f"{drop_min:.3f}s.\n#\n"
    )
    text = emit_config(args.config, records, kept, header,
                       preserve_dropped=not args.no_preserve_dropped)
    if dropped:
        text += "\n# --- DROPPED (slowest %d, %s cost) ---\n" % (len(dropped), args.mode)
        text += "".join(f"#   {cost[i]:9.3f}s  {i}\n" for i in reversed(dropped))

    output = args.output or (
        BUNDLED_CONFIG_DIR
        / f"benchmarked{args.keep}{'_amortized' if args.mode == 'amortized' else ''}_config.yaml")
    output.write_text(text)
    print(f"[cut] {len(kept)}/{len(ranked)} SPIs kept ({args.mode}, M={M} T={T}) -> {output}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
