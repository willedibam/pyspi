"""Independent-pair null processes, and a drift report for any config.

Every pair is two *independent* series, so every dependence measure has a
population value of zero (or its no-coupling value). What changes across the
conditions is only the autocorrelation of each series. A measure whose null
distribution moves with that autocorrelation will report "coupling" that is
shared slowness.

``tests/test_autocorr_null.py`` imports the generator and the drift statistic
from here. Run as a script, this module computes every SPI of a config over the
whole grid and writes one row per (SPI, cell), so drift can be inspected for
SPIs the test does not assert on. That run is slow (``full`` takes about 12 s
per repetition at T=200 and about 64 s at T=1000), so it is a tool, not a test.

Usage
-----
    python tests/tools/autocorr_null.py full 200 30 null_T200.csv
    python tests/tools/autocorr_null.py path/to/subset.yaml 1000 12 out.csv
"""
import argparse
import logging
import time
import warnings

import numpy as np
from scipy.signal import lfilter

BURN = 2000  # 0.99**2000 ~ 2e-9: the start-up transient is gone.


def _ar2(r=0.98, f0=0.1):
    """AR(2) with complex poles at radius ``r`` and frequency ``f0`` cycles/sample."""
    return [2 * r * np.cos(2 * np.pi * f0), -r * r]


# Name -> (AR coefficients of process 0, AR coefficients of process 1).
CONDITIONS = {
    "phi0": ([0.0], [0.0]),
    "phi0.5": ([0.5], [0.5]),
    "phi0.9": ([0.9], [0.9]),
    "phi0.99": ([0.99], [0.99]),
    "slow_x_white": ([0.95], [0.0]),  # process 0 slow, process 1 white
    "ar2": (_ar2(), _ar2()),  # narrowband, alpha-like
}
BASELINE = "phi0"


def ar_series(coefs, T, rng):
    """One stationary AR(p) series of length ``T`` with unit innovations."""
    e = rng.standard_normal(T + BURN)
    return lfilter([1.0], np.r_[1.0, -np.asarray(coefs, float)], e)[BURN:]


def null_pair(cond, T, seed):
    """A 2 x T array of two independent series under condition ``cond``."""
    a, b = CONDITIONS[cond]
    rng = np.random.default_rng(seed)
    return np.vstack([ar_series(a, T, rng), ar_series(b, T, rng)])


def cell_seed(cond, T, rep):
    """Fixed integer seed per (condition, T, repetition); never ``hash()``."""
    return [list(CONDITIONS).index(cond), T, rep]


def cells(values, antisymmetric=False):
    """Split ``{cond: (R, 2) array of (v01, v10)}`` into one sample per cell.

    One direction per pair, so a symmetric SPI does not count each pair twice.
    ``slow_x_white`` is not exchangeable, so its two directions are separate
    cells: ``s>w`` is slow -> white (v01), ``w>s`` is white -> slow (v10).

    An antisymmetric SPI has mean zero on any exchangeable pair whatever its
    estimator does, so its sample is ``|v|``: the spread is what can drift.
    """
    out = {}
    for cond, v in values.items():
        v = np.abs(v) if antisymmetric else v
        if cond == "slow_x_white":
            out["slow_x_white_s>w"] = v[:, 0]
            out["slow_x_white_w>s"] = v[:, 1]
        else:
            out[cond] = v[:, 0]
    return out


def drift(sample, base):
    """``(mean shift in white-noise SDs, its standard error, SD ratio)``.

    Non-finite values are dropped; the caller checks the non-finite rate.
    """
    s, b = sample[np.isfinite(sample)], base[np.isfinite(base)]
    sd0 = b.std(ddof=1)
    if not (sd0 > 0) or s.size < 2:
        return np.nan, np.nan, np.nan
    d = (s.mean() - b.mean()) / sd0
    se = np.sqrt(s.var(ddof=1) / s.size + b.var(ddof=1) / b.size) / sd0
    return d, se, s.std(ddof=1) / sd0


def main():
    import pandas as pd

    from pyspi.calculator import Calculator
    from pyspi.data import Data

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config", help="bundled config name or path to a yaml")
    ap.add_argument("T", type=int)
    ap.add_argument("reps", type=int)
    ap.add_argument("out", help="output csv")
    args = ap.parse_args()
    warnings.simplefilter("ignore")
    logging.disable(logging.CRITICAL)

    values, errors = {}, {}
    for cond in CONDITIONS:
        t0 = time.time()
        for rep in range(args.reps):
            z = null_pair(cond, args.T, cell_seed(cond, args.T, rep))
            calc = Calculator(Data(z, dim_order="ps", zscore=True),
                              config=args.config, verbose=False)
            calc.compute(progress=False)
            for spi, msg in calc.errors.items():
                errors.setdefault(spi, msg)
            for spi in calc.spis:
                m = calc.table[spi].to_numpy()
                values.setdefault(spi, {}).setdefault(cond, []).append((m[0, 1], m[1, 0]))
        print(f"{cond} T={args.T}: {(time.time() - t0) / args.reps:.1f} s/rep", flush=True)

    rows = []
    for spi, by_cond in values.items():
        anti = "antisymmetric" in getattr(calc.spis[spi], "labels", [])
        c = cells({k: np.asarray(v, float) for k, v in by_cond.items()}, anti)
        base = c[BASELINE]
        for cell, sample in c.items():
            d, se, ratio = drift(sample, base)
            rows.append(dict(spi=spi, cell=cell, mean=np.nanmean(sample),
                             drift_sd=d, drift_se=se, sd_ratio=ratio,
                             nonfinite=float(np.mean(~np.isfinite(sample))),
                             error=errors.get(spi, "")))
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(values)} SPIs, {len(errors)} with errors")


if __name__ == "__main__":
    main()
