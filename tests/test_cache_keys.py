"""Parameterised statistic caches must key on every parameter.

These began as red tests and are now green. The spectral cache had two
defects that compounded:

1. ``NonparametricSpectral.key`` omitted ``fs``, so two SPIs differing only in
   sampling frequency collided in the cache.
2. ``_get_cache`` *wrote* the first result under ``self.measure`` (a plain
   string) but *read* under ``self.key`` (a tuple). The first write was
   therefore unreachable, and staleness only surfaced from the third call
   onward, once a tuple-keyed entry finally existed.

Defect 2 is why a naive two-call probe reports no problem. The sequence below
(fs=1 -> 4 -> 1 -> 4) is the minimum that exposes it, and every value is
compared against a freshly-constructed Data rather than against its
predecessor.

Keep these as regression tests: both defects were invisible to the obvious
probe, and the second would return silently wrong numbers if reintroduced.
"""
import numpy as np
import pytest

from pyspi.data import Data
from pyspi.statistics.spectral import CoherenceMagnitude


def _fixture_data():
    rng = np.random.default_rng(0)
    return Data(data=rng.standard_normal((3, 128)), dim_order="ps", zscore=False)


# An explicit band is required for fs to matter: with the default fmax=fs/2 the
# whole spectrum is selected whatever fs is, so both settings legitimately agree
# and the test would be vacuous. With a fixed [0, 0.25] band, fs changes which
# frequency bins fall inside it.
BAND = dict(fmin=0.0, fmax=0.25)


def _spi(fs):
    return CoherenceMagnitude(fs=fs, **BAND)


def _fresh_value(fs):
    """Ground truth: a brand-new Data can never serve a stale cache entry."""
    return _spi(fs).multivariate(_fixture_data())[0, 1]


def test_spectral_cache_distinguishes_sampling_frequency():
    """Alternating fs on one Data must match a fresh Data at every step."""
    data = _fixture_data()
    truth = {1: _fresh_value(1), 4: _fresh_value(4)}

    assert not np.isclose(truth[1], truth[4]), (
        "Test is vacuous: fs=1 and fs=4 give the same value on this fixture."
    )

    observed = []
    for step, fs in enumerate((1, 4, 1, 4), start=1):
        got = _spi(fs).multivariate(data)[0, 1]
        observed.append((step, fs, got, truth[fs]))

    bad = [o for o in observed if not np.isclose(o[2], o[3], equal_nan=True)]
    assert not bad, "Stale cache hits at " + ", ".join(
        f"call {s} (fs={f}): got {g:.6g}, fresh Data gives {t:.6g}" for s, f, g, t in bad
    )


def test_spectral_cache_uses_one_key_type():
    """The written and read cache keys must be the same type."""
    data = _fixture_data()
    # Two calls: the first creates the dict with a *string* key, the second
    # misses on the tuple lookup and inserts a *tuple* key alongside it.
    _spi(1).multivariate(data)
    _spi(1).multivariate(data)

    keys = list(data.spectral_mv.keys())
    stat_keys = [k for k in keys if k != "freq"]
    kinds = {type(k).__name__ for k in stat_keys}

    assert len(kinds) == 1, (
        f"Cache holds mixed key types {kinds} ({stat_keys!r}); the first write is "
        "unreachable by the reader."
    )


def test_cache_key_covers_every_identifier_parameter():
    """Any parameter that changes the identifier must change the cache key."""
    a, b = _spi(1), _spi(4)
    assert a.identifier != b.identifier, "Precondition: fs must reach the identifier."
    assert a.key != b.key, (
        f"fs changes the identifier ({a.identifier!r} vs {b.identifier!r}) but not "
        f"the cache key ({a.key!r} == {b.key!r})."
    )
