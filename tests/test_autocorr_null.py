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


# --- xcorr: the Bartlett-widened, lag-corrected threshold ---------------------

from pyspi.statistics.basic import CrossCorrelation  # noqa: E402

XCORR_REPS = 200
# The family-wise level is 5%; 0.10 is that plus ~3 binomial SEs at 200 reps.
XCORR_MAX_RATE = 0.10


def _nonzero_rate(spi, cond, T):
    hits = 0
    for rep in range(XCORR_REPS):
        data = Data(null_pair(cond, T, cell_seed(cond, T, rep)), dim_order="ps", zscore=True)
        hits += spi.multivariate(data)[0, 1] != 0
    return hits / XCORR_REPS


@pytest.mark.parametrize("cond", list(CONDITIONS))
def test_xcorr_bartlett_threshold_holds_its_level(cond):
    """On independent pairs `sig-bartlett` is nonzero on at most ~5% of pairs;
    `sig-True` (the unadjusted 1.96/sqrt(T) cut) is the positive control that
    must not, since its cut is cleared by chance at some lag of nearly every pair."""
    rate = _nonzero_rate(CrossCorrelation(statistic="max", sigonly="bartlett"), cond, 200)
    assert rate <= XCORR_MAX_RATE, f"{cond}: {rate:.3f} of null pairs nonzero"
    if cond != "slow_x_white":  # the old cut fires on ~60% there, still > 10%
        assert _nonzero_rate(CrossCorrelation(statistic="max", sigonly=True), cond, 200) > 0.5


def test_xcorr_bartlett_threshold_keeps_power_on_coupled_slow_series():
    """Two AR(1) series at phi=0.9 sharing half their innovation variance
    (lag-0 correlation ~0.5), T=1000: the coupling must clear the widened cut.

    Not T=200: there N/B ~ 21, r=0.5 is ~2.3 null SDs, and the corrected cut is
    ~3.5 SDs, so a valid test is expected to miss it most of the time.
    """
    from scipy.signal import lfilter

    spi = CrossCorrelation(statistic="max", sigonly="bartlett")
    hits = 0
    for rep in range(50):
        rng = np.random.default_rng([11, rep])
        e = rng.standard_normal((3, 3000))
        x = lfilter([1.0], [1.0, -0.9], e[0] + e[2])[2000:]
        y = lfilter([1.0], [1.0, -0.9], e[1] + e[2])[2000:]
        hits += spi.multivariate(Data(np.vstack([x, y]), dim_order="ps", zscore=True))[0, 1] > 0
    assert hits >= 45, f"detected {hits}/50"


def test_xcorr_sigonly_values():
    assert CrossCorrelation(statistic="mean", sigonly="bartlett").identifier == "xcorr_mean_sig-bartlett"
    assert CrossCorrelation(squared=True, sigonly="bartlett").identifier == "xcorr-sq_max_sig-bartlett"
    for bad in ("Bartlett", "true", 2):
        with pytest.raises(ValueError, match="sigonly"):
            CrossCorrelation(sigonly=bad)


# --- Bartlett-corrected correlation z -----------------------------------------

from pyspi.statistics.basic import BartlettCorrelation  # noqa: E402
from pyspi.utils import bartlett_factor  # noqa: E402


@pytest.mark.parametrize("cond", list(CONDITIONS))
def test_bartlett_z_holds_its_level(cond):
    """|z| > 1.96 on at most ~5% of independent pairs at T=200 (bound 10%, as
    for xcorr). Plain r * sqrt(T) is the positive control on the slow
    conditions, where it must exceed the same cut far more often."""
    spi, hits, naive = BartlettCorrelation(), 0, 0
    for rep in range(XCORR_REPS):
        z = null_pair(cond, 200, cell_seed(cond, 200, rep))
        hits += abs(spi.multivariate(Data(z, dim_order="ps", zscore=True))[0, 1]) > 1.96
        naive += abs(np.corrcoef(z)[0, 1]) * np.sqrt(200) > 1.96
    assert hits / XCORR_REPS <= XCORR_MAX_RATE, f"{cond}: {hits / XCORR_REPS:.3f}"
    if cond in ("phi0.9", "phi0.99", "ar2"):
        assert naive / XCORR_REPS > 0.3, f"{cond}: control {naive / XCORR_REPS:.3f}"


def test_bartlett_z_detects_coupled_slow_series():
    """AR(1) phi=0.9 pair with lag-0 correlation ~0.5 at T=1000 (N/B ~ 100):
    z > 1.96 on nearly every pair."""
    from scipy.signal import lfilter

    spi, hits = BartlettCorrelation(), 0
    for rep in range(50):
        e = np.random.default_rng([12, rep]).standard_normal((3, 3000))
        x = lfilter([1.0], [1.0, -0.9], e[0] + e[2])[2000:]
        y = lfilter([1.0], [1.0, -0.9], e[1] + e[2])[2000:]
        hits += spi.multivariate(Data(np.vstack([x, y]), dim_order="ps", zscore=True))[0, 1] > 1.96
    assert hits >= 45, f"detected {hits}/50"


def test_bartlett_z_matches_its_definition():
    """Vectorised multivariate == per-pair bivariate == r * sqrt(T / B)."""
    rng = np.random.default_rng(5)
    Z = np.vstack([ar_series([0.8], 300, rng), ar_series([0.3], 300, rng),
                   ar_series(_ar2(), 300, rng), rng.standard_normal(300)])
    data = Data(Z, dim_order="ps", zscore=True)
    z = data.to_numpy(squeeze=True)
    spi = BartlettCorrelation()
    mv = spi.multivariate(data)
    for i in range(4):
        for j in range(4):
            if i == j:
                assert np.isnan(mv[i, j])
                continue
            ref = np.corrcoef(z[i], z[j])[0, 1] * np.sqrt(300 / bartlett_factor(z[i], z[j]))
            assert abs(mv[i, j] - ref) < 1e-10
            assert abs(spi.bivariate(data, i=i, j=j) - ref) < 1e-10
    assert spi.identifier == "corr_pearson_bartlett-z"


# --- kernel TE: self-counts and temporal neighbours inflate the null ----------

@pytest.mark.slow
@pytest.mark.parametrize("cell", ["phi0", "phi0.9", "slow_x_white_w>s"])
def test_kernel_te_theiler_window_removes_most_of_the_null_bias(cell):
    """Box-kernel TE at T=1000 on independent pairs: ~0.35 nats on white noise
    with no window (self-counts and temporal neighbours), under 0.1 with
    dyn_corr_excl="AUTO". Measured: 0.06 / -0.005 / -0.015 nats for the three cells."""
    cond = cell.split("_w>s")[0] if cell.endswith("_w>s") else cell
    col = (1, 0) if cell.endswith("_w>s") else (0, 1)
    fixed = TransferEntropy(estimator="kernel", kernel_width=0.25, dyn_corr_excl="AUTO")
    plain = TransferEntropy(estimator="kernel", kernel_width=0.25)
    v_fixed, v_plain = [], []
    for rep in range(8):
        data = Data(null_pair(cond, 1000, cell_seed(cond, 1000, rep)), dim_order="ps", zscore=True)
        v_fixed.append(fixed.multivariate(data)[col])
        v_plain.append(plain.multivariate(data)[col])
    assert abs(np.mean(v_fixed)) < 0.1, np.mean(v_fixed)
    if cell == "phi0":  # control: without the window the bias is there
        assert np.mean(v_plain) > 0.25, np.mean(v_plain)


# --- KSG shift baseline: subtract the estimate on circularly shifted sources ---

from pyspi.statistics.infotheory import TimeLaggedMutualInfo  # noqa: E402


def _ksg_null(spi, phi, T, reps, col=(0, 1)):
    v = np.empty(reps)
    for rep in range(reps):
        rng = np.random.default_rng([31, T, rep, int(phi * 100)])
        z = np.vstack([ar_series([phi], T, rng), ar_series([phi], T, rng)])
        v[rep] = spi.multivariate(Data(z, dim_order="ps", zscore=True))[col]
    return v.mean(), v.std(ddof=1) / np.sqrt(reps)


def _shift_null_case(spi_factory, phi, T, reps):
    """Shift-corrected null mean within 4 SE of zero; the raw estimator, on the
    same draws, is the control that must be more than 4 SE above it."""
    m, se = _ksg_null(spi_factory(shift_baseline=5), phi, T, reps)
    assert abs(m) <= 4 * se, f"corrected null mean {m:+.4f} (SE {se:.4f})"
    m_raw, se_raw = _ksg_null(spi_factory(), phi, T, reps)
    assert m_raw > 4 * se_raw, f"control: raw null mean {m_raw:+.4f} (SE {se_raw:.4f})"


@pytest.mark.parametrize("name,factory,T", [
    ("mi", lambda **kw: MutualInfo(estimator="kraskov", **kw), 1000),
    ("te", lambda **kw: TransferEntropy(estimator="kraskov", k_history=1, l_history=1, **kw), 200),
])
def test_ksg_shift_baseline_centres_the_slow_null(name, factory, T):
    # Pilot, phi=0.99, 30 reps: MI at T=1000 raw +0.279, corrected +0.001
    # (SE 0.009); TE at T=200 raw +0.043, corrected +0.007 (SE 0.004). TE's raw
    # bias at T=1000 (+0.015) is too small to make a control.
    _shift_null_case(factory, 0.99, T, 30)


@pytest.mark.slow
@pytest.mark.parametrize("name,factory", [
    ("tlmi", lambda **kw: TimeLaggedMutualInfo(estimator="kraskov", **kw)),
    ("di", lambda **kw: DirectedInfo(estimator="kraskov", **kw)),
])
def test_ksg_shift_baseline_centres_the_slow_null_slow(name, factory):
    _shift_null_case(factory, 0.99, 1000, 20)


def test_ksg_shift_baseline_keeps_the_analytic_coupling():
    """Coupled white Gaussian pair, rho = 0.6: the shifted surrogates carry no
    coupling, so the corrected estimate still recovers -0.5 ln(1 - rho^2)."""
    rho, T = 0.6, 2000
    rng = np.random.default_rng(12)
    x = rng.standard_normal(T)
    y = rho * x + np.sqrt(1 - rho**2) * rng.standard_normal(T)
    data = Data(np.vstack([x, y]), dim_order="ps", zscore=True)
    corrected = MutualInfo(estimator="kraskov", shift_baseline=5).multivariate(data)[0, 1]
    assert abs(corrected - (-0.5 * np.log(1 - rho**2))) < 0.03
