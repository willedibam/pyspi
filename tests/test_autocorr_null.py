"""Null contract: on independent pairs, autocorrelation alone must not look like coupling.

Two kinds of assertion, both on pairs of *independent* series whose
autocorrelation varies (white, AR(1) at phi 0.5/0.9/0.99, slow x white, and a
narrowband AR(2); see ``tests/tools/autocorr_null.py``):

1. **Invariance.** For SPIs whose null should not depend on the spectra of the
   two series (multitaper coherence-type measures, phase-lag measures, PSI,
   directed coherence, nonparametric spectral GC, auto-embedded Gaussian GC),
   the null mean under every condition stays within ``DELTA`` white-noise null
   SDs of the white-noise null mean. This is an effect-size bound, not a
   significance test: low-band multitaper estimates do carry a small, real
   leakage bias on steep spectra (up to ~0.75 SD at T=200 on the AR(2), under
   ~0.45 SD at T=1000), and a pure k-SE test would fail on it once the
   repetition count is large enough. Measures known to drift are not silently
   dropped: each is in ``NOT_ASSERTED`` with its measured reason, and two of
   them are positive controls that must *fail* the same bound, so a pass here
   is evidence of power, not of a vacuous criterion.

2. **Analytic anchors** for the Gaussian estimators, so a regenerated baseline is
   never the only oracle. For a regression of a series ``y`` that is iid
   Gaussian given its conditioning set, on ``p`` regressors independent of it,
   with ``q`` conditioning columns and ``N`` centred rows, the partial
   ``R^2 ~ Beta(p/2, (N-1-q-p)/2)``, so

       E[-1/2 ln(1 - R^2)] = 1/2 [psi((N-1-q)/2) - psi((N-1-q-p)/2)]

   whatever the autocorrelation of the regressors. For MI with one white series
   this is exact. For DI with a white target and GC with a correctly specified
   AR(1) target the design contains lagged values of ``y`` itself, so it holds
   to O(1/T); ``ANCHOR_REL`` bounds that. Each anchor also has a control cell
   that violates its premise and must fail it.

The full-config drift table, including every SPI not asserted here, is a tool:
``python tests/tools/autocorr_null.py full 200 30 out.csv``.
"""
import logging
import os
import re
import sys
import warnings

import numpy as np
import pytest
import yaml
from scipy.special import digamma

from pyspi.calculator import load_spis_from_yaml, resolve_config
from pyspi.data import Data
from pyspi.statistics.infotheory import DirectedInfo, MutualInfo, TransferEntropy

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
from autocorr_null import (  # noqa: E402
    BASELINE, CONDITIONS, ar_series, cell_seed, cells, drift, null_pair, _ar2,
)

# Bound on |null mean shift| in white-noise null SDs.
DELTA = 1.0

# full.yaml classes whose SPIs are candidates for the invariance contract.
_CLASSES = {
    ".statistics.spectral": [
        "CoherenceMagnitude", "ImaginaryCoherence", "PhaseSlopeIndex",
        "PhaseLockingValue", "PhaseLagIndex", "WeightedPhaseLagIndex",
        "DebiasedSquaredPhaseLagIndex", "DebiasedSquaredWeightedPhaseLagIndex",
        "PairwisePhaseConsistency", "DirectedCoherence",
        "GeneralizedPartialDirectedCoherence", "SpectralGrangerCausality",
    ],
    ".statistics.wavelet": ["PhaseSlopeIndex"],
    # Gaussian configs only: the auto-embedded GC (asserted) and the two
    # positive controls below.
    ".statistics.infotheory": ["MutualInfo", "TransferEntropy"],
}

# Loaded but not asserted, with the measured reason (300 reps at T=200, 100 at
# T=1000). Every pattern must match a loaded identifier, so this list cannot
# go stale silently.
NOT_ASSERTED = {
    r"^mi_gaussian$": "positive control: null mean scales with the Bartlett factor",
    r"^gc_gaussian_k-1_kt-1_l-1_lt-1$": "positive control: lag-1 history under-specifies the AR(2)",
    r"^sgc_parametric_.*_order-None$": "BIC usually keeps order 0 on white noise, so GC is exactly 0 and the white-noise SD is ~0",
    r"^sgc_parametric_.*_order-1$": "drifts up to ~100 SD on AR(1) pairs in the low band",
    r"^sgc_parametric_max_fs-1_fmin-1e-05_fmax-0-(25|5)_order-20$": "max over the low band drifts ~2.3 SD on the AR(2)",
    r"^sgc_nonparametric_max_fs-1_fmin-1e-05_fmax-0-(25|5)$": "max over the low band drifts ~1.0 SD on the AR(2) at T=200",
}

# (identifier, cell) pairs that must exceed DELTA, under the same statistic.
POSITIVE_CONTROLS = [
    ("mi_gaussian", "phi0.9"),
    ("gc_gaussian_k-1_kt-1_l-1_lt-1", "ar2"),
]


@pytest.fixture(scope="module")
def spis(tmp_path_factory):
    full = yaml.safe_load(open(resolve_config("full")))
    sub = {m: {c: dict(full[m][c]) for c in cs} for m, cs in _CLASSES.items()}
    for entry in sub[".statistics.infotheory"].values():
        entry["configs"] = [p for p in entry["configs"] if p.get("estimator") == "gaussian"]
    path = tmp_path_factory.mktemp("null") / "subset.yaml"
    path.write_text(yaml.safe_dump(sub))
    return load_spis_from_yaml(str(path), quiet=True)


def _compute(spis, T, reps):
    """``{identifier: {cond: (reps, 2) array of (v01, v10)}}``; failures are NaN."""
    out = {k: {c: np.full((reps, 2), np.nan) for c in CONDITIONS} for k in spis}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.disable(logging.CRITICAL)
        try:
            for cond in CONDITIONS:
                for rep in range(reps):
                    data = Data(null_pair(cond, T, cell_seed(cond, T, rep)),
                                dim_order="ps", zscore=True)
                    for key, spi in spis.items():
                        try:
                            m = spi.multivariate(data)
                        except Exception:
                            continue
                        out[key][cond][rep] = m[0, 1], m[1, 0]
        finally:
            logging.disable(logging.NOTSET)
    return out


def _excluded(identifier):
    return any(re.search(p, identifier) for p in NOT_ASSERTED)


def _check_invariance(spis, T, reps):
    for pattern in NOT_ASSERTED:
        assert any(re.search(pattern, k) for k in spis), f"stale NOT_ASSERTED entry {pattern!r}"

    values = _compute(spis, T, reps)
    failures = []
    stats = {}
    for key, by_cond in values.items():
        anti = "antisymmetric" in spis[key].labels
        c = cells(by_cond, anti)
        stats[key] = {cell: drift(s, c[BASELINE]) for cell, s in c.items() if cell != BASELINE}
        if _excluded(key):
            continue
        for cell, s in c.items():
            if not np.isfinite(s).all():
                failures.append(f"{key} [{cell}]: {np.mean(~np.isfinite(s)):.0%} non-finite")
        for cell, (d, se, _) in stats[key].items():
            if not abs(d) < DELTA:
                failures.append(f"{key} [{cell}]: drift {d:+.2f} SD (SE {se:.2f})")

    for key, cell in POSITIVE_CONTROLS:
        d = stats[key][cell][0]
        assert d > DELTA, f"positive control {key} [{cell}] drift {d:+.2f} SD <= {DELTA}: the bound has no power"

    asserted = [k for k in spis if not _excluded(k)]
    assert len(asserted) >= 60, f"only {len(asserted)} SPIs asserted"
    assert not failures, (f"{len(failures)} invariance failures at T={T}, reps={reps}:\n"
                          + "\n".join(failures))


def test_invariant_spis_ignore_autocorrelation(spis):
    # 120 reps: SE of the drift is ~0.13 SD, against a worst measured drift of
    # ~0.75 SD among the asserted SPIs. About 40 s.
    _check_invariance(spis, T=200, reps=120)


@pytest.mark.slow
def test_invariant_spis_ignore_autocorrelation_T1000(spis):
    _check_invariance(spis, T=1000, reps=100)


# --- analytic anchors -------------------------------------------------------

ANCHOR_REL = 0.05  # O(1/T) allowance where the design contains lags of y


def _beta_term(N, q, p):
    """E[-1/2 ln(1 - R^2)] for the partial R^2 of ``p`` regressors given ``q``."""
    return 0.5 * (digamma((N - 1 - q) / 2) - digamma((N - 1 - q - p) / 2))


def _null_mean(spi, src, tgt, T, reps, seed):
    """Mean and SE of ``spi`` from process 0 (``src`` AR coefs) to process 1."""
    v = np.empty(reps)
    for r in range(reps):
        rng = np.random.default_rng([seed, T, r])
        z = np.vstack([ar_series(src, T, rng), ar_series(tgt, T, rng)])
        v[r] = spi.multivariate(Data(z, dim_order="ps", zscore=True))[0, 1]
    return v.mean(), v.std(ddof=1) / np.sqrt(reps)


def _within(mean, se, expected, rel):
    return abs(mean - expected) <= 4 * se + rel * expected


def _bartlett(phi):
    """sum_k rho_x(k) rho_y(k) for two AR(1) series with the same ``phi``."""
    return (1 + phi**2) / (1 - phi**2)


MI = MutualInfo(estimator="gaussian")
DI = DirectedInfo(estimator="gaussian", n=5)
GC = TransferEntropy(estimator="gaussian", k_history=1, l_history=1)

# (spi, source AR, target AR, T, reps, expected, rel tolerance, should_hold)
ANCHORS = {
    # One white series: exact for any autocorrelation of the other.
    "mi_white_white": (MI, [0.0], [0.0], 200, 4000, _beta_term(200, 0, 1), 0.0, True),
    "mi_slow_white": (MI, [0.99], [0.0], 200, 4000, _beta_term(200, 0, 1), 0.0, True),
    "mi_control": (MI, [0.9], [0.9], 200, 1000, _beta_term(200, 0, 1), 0.0, False),
    # Both slow: E[r^2] ~ B/T, so E[MI] ~ B/(2T) at leading order.
    "mi_bartlett": (MI, [0.9], [0.9], 1000, 2000, _bartlett(0.9) / 2000, ANCHOR_REL, True),
    # DI, white target: term i has i source columns given i-1 target lags, on T-n rows.
    "di_white_target": (DI, [0.0], [0.0], 200, 800,
                        sum(_beta_term(195, i - 1, i) for i in range(1, 6)), ANCHOR_REL, True),
    "di_slow_source": (DI, [0.99], [0.0], 200, 800,
                       sum(_beta_term(195, i - 1, i) for i in range(1, 6)), ANCHOR_REL, True),
    "di_control": (DI, [0.9], [0.9], 200, 800,
                   sum(_beta_term(195, i - 1, i) for i in range(1, 6)), ANCHOR_REL, False),
    # GC with k=1 on an AR(1) target is correctly specified: flat in either phi.
    "gc_ar1_target": (GC, [0.0], [0.9], 1000, 2000, _beta_term(999, 1, 1), ANCHOR_REL, True),
    "gc_ar1_both": (GC, [0.9], [0.9], 1000, 2000, _beta_term(999, 1, 1), ANCHOR_REL, True),
    # Lag-1 history under-specifies an AR(2) target; the residual autocorrelation
    # only leaks into GC when the source is autocorrelated too.
    "gc_control": (GC, _ar2(), _ar2(), 1000, 500, _beta_term(999, 1, 1), ANCHOR_REL, False),
}


def test_anchor_identifiers_are_the_bundled_ones():
    assert (MI.identifier, DI.identifier, GC.identifier) == (
        "mi_gaussian", "di_gaussian_n-5", "gc_gaussian_k-1_kt-1_l-1_lt-1")


@pytest.mark.parametrize("name", list(ANCHORS))
def test_gaussian_null_anchor(name):
    spi, src, tgt, T, reps, expected, rel, should_hold = ANCHORS[name]
    mean, se = _null_mean(spi, src, tgt, T, reps, seed=list(ANCHORS).index(name))
    held = _within(mean, se, expected, rel)
    assert held == should_hold, (
        f"{name}: mean {mean:.5f} (SE {se:.5f}) vs expected {expected:.5f}, "
        f"ratio {mean / expected:.3f}; anchor {'should' if should_hold else 'should not'} hold")
