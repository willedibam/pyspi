#!/usr/bin/env python
"""Backfill per-SPI ``category`` and ``labels`` into older bench JSONs.

Older cell JSONs (pre-metadata change) only have ``{mean, std, values}`` per
SPI. This script looks up the category (python module suffix) and merged
``labels`` list from a source config and writes them into each spi_seconds
entry in place.

Idempotent: rewrites only entries missing ``category``. Atomic per-file write.

Caveat: backfill uses **current** pyspi labels. Each cell JSON's
``environment.pyspi_git_sha`` records the sha at bench time; if labels for an
SPI have changed since, the new labels will be written.

Usage:
    python -m bench.backfill_metadata --glob 'bench/results/cells/physics_config_M64_T*_n1.json'
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from bench.bench_compute import spi_metadata
from pyspi.calculator import Calculator

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--glob", required=True,
                   help="Glob (relative to repo root) matching cell JSONs to update.")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "pyspi" / "config.yaml",
                   help="Source config to read SPI category/labels from.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report changes without writing.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    paths = sorted(REPO_ROOT.glob(args.glob))
    if not paths:
        raise SystemExit(f"no JSONs match {args.glob}")

    print(f"[backfill] loading metadata from {args.config}", file=sys.stderr)
    calc = Calculator(dataset=np.random.randn(4, 50), configfile=str(args.config),
                      normalise=False, verbose=False)
    meta = spi_metadata(calc)
    print(f"[backfill] metadata for {len(meta)} SPIs", file=sys.stderr)

    for p in paths:
        d = json.loads(p.read_text())
        spi_seconds = d.get("spi_seconds")
        if not spi_seconds:
            print(f"[backfill] skip {p.name}: no spi_seconds", file=sys.stderr)
            continue
        n_added = 0
        n_missing = 0
        for ident, entry in spi_seconds.items():
            if "category" in entry and "labels" in entry:
                continue
            if ident not in meta:
                n_missing += 1
                continue
            entry["category"] = meta[ident]["category"]
            entry["labels"] = meta[ident]["labels"]
            n_added += 1
        if n_added == 0:
            print(f"[backfill] {p.name}: already up-to-date "
                  f"({n_missing} not in current config)", file=sys.stderr)
            continue
        msg = (f"[backfill] {p.name}: +{n_added} enriched"
               f"{f', {n_missing} not in current config' if n_missing else ''}")
        if args.dry_run:
            print(f"{msg}  (dry-run, not written)", file=sys.stderr)
            continue
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(d, indent=2))
        os.replace(tmp, p)
        print(msg, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
