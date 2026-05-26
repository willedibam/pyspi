#!/usr/bin/env python
"""One-shot: list dropped SPIs at each percentile, comparing the two candidate
anchors (M=16,T=800 and M=32,T=1600). Writes dropped_spi_comparison.md next to
this script."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ANALYSIS = Path(__file__).resolve().parent
CUTS = ANALYSIS / "proposed_cuts"
OUT = ANALYSIS / "dropped_spi_comparison.md"


def parse_dropped(yaml_path: Path) -> dict[str, float]:
    """Return {identifier: amortized_cost_s} from the trailing '--- DROPPED ---' block."""
    text = yaml_path.read_text()
    out = {}
    for line in text.splitlines():
        m = re.match(r"#\s+(\d+\.\d+)s\s+(\S+)$", line)
        if m:
            out[m.group(2)] = float(m.group(1))
    return out


def load_long_costs() -> pd.DataFrame:
    return pd.read_csv(ANALYSIS / "long_costs.csv")


def cost_at(df: pd.DataFrame, ident: str, M: int, T: int) -> float | None:
    sub = df[(df["identifier"] == ident) & (df["M"] == M) & (df["T"] == T)]
    if sub.empty:
        return None
    return float(sub.iloc[0]["amortized_s"])


def main() -> int:
    df = load_long_costs()
    pcts = [80, 90, 95, 99]
    anchors = [("M16_T800", 16, 800), ("M32_T1600", 32, 1600)]

    lines = ["# Dropped-SPI comparison across percentiles and anchors", "",
             "Cost = amortized walltime (group cache-amortized, per `cut_config.py --mode amortized`).",
             "", "Layout: for each percentile, three lists —",
             "(a) dropped at **both** anchors (= robust drop),",
             "(b) dropped **only at M=16,T=800** (cheap at M=32,T=1600),",
             "(c) dropped **only at M=32,T=1600** (cheap at M=16,T=800)."]
    lines.append("")

    for pct in pcts:
        a_path = CUTS / f"benchmarked{pct}_amortized_M16_T800.yaml"
        b_path = CUTS / f"benchmarked{pct}_amortized_M32_T1600.yaml"
        a_drops = parse_dropped(a_path)
        b_drops = parse_dropped(b_path)
        common = sorted(set(a_drops) & set(b_drops),
                        key=lambda i: -max(a_drops[i], b_drops[i]))
        only_a = sorted(set(a_drops) - set(b_drops), key=lambda i: -a_drops[i])
        only_b = sorted(set(b_drops) - set(a_drops), key=lambda i: -b_drops[i])

        lines.append(f"## p{pct}  (keep top {pct}% fastest — drop {len(a_drops)} SPIs)")
        lines.append("")
        lines.append(f"### dropped at BOTH anchors ({len(common)})")
        lines.append("")
        lines.append("| identifier | cost@M16T800 (s) | cost@M32T1600 (s) |")
        lines.append("|---|---:|---:|")
        for i in common:
            lines.append(f"| `{i}` | {a_drops[i]:.2f} | {b_drops[i]:.2f} |")
        lines.append("")
        if only_a:
            lines.append(f"### dropped ONLY at M=16,T=800 ({len(only_a)})")
            lines.append("")
            lines.append("| identifier | cost@M16T800 (s) | cost@M32T1600 (s) |")
            lines.append("|---|---:|---:|")
            for i in only_a:
                c2 = cost_at(df, i, 32, 1600)
                c2s = f"{c2:.2f}" if c2 is not None else "?"
                lines.append(f"| `{i}` | {a_drops[i]:.2f} | {c2s} |")
            lines.append("")
        if only_b:
            lines.append(f"### dropped ONLY at M=32,T=1600 ({len(only_b)})")
            lines.append("")
            lines.append("| identifier | cost@M16T800 (s) | cost@M32T1600 (s) |")
            lines.append("|---|---:|---:|")
            for i in only_b:
                c1 = cost_at(df, i, 16, 800)
                c1s = f"{c1:.2f}" if c1 is not None else "?"
                lines.append(f"| `{i}` | {c1s} | {b_drops[i]:.2f} |")
            lines.append("")

    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
