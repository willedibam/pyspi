"""Deliberately misbehaving SPIs, used to test failure semantics.

Kept as a top-level module (not a package submodule) so a YAML config can name
it directly and a spawned worker can import it, provided ``tests/`` is on
``PYTHONPATH`` -- see ``tests/test_execution_parity.py``, which sets that.

These exist to pin how the two execution paths report failure. They are not
part of the shipped SPI set.
"""
import numpy as np

from pyspi.base import Directed, Undirected, Unsigned, parse_bivariate


class AlwaysRaises(Undirected, Unsigned):
    """Raises a distinctive exception from every pair."""

    name = "AlwaysRaises"
    identifier = "always_raises"
    labels = ["test"]

    def __init__(self, message="deliberate test failure"):
        self._message = message

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        raise RuntimeError(self._message)


class AlwaysWarns(Undirected, Unsigned):
    """Emits a distinctive warning, then returns a finite value."""

    name = "AlwaysWarns"
    identifier = "always_warns"
    labels = ["test"]

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        import warnings

        warnings.warn("deliberate test warning", UserWarning)
        return 0.5


class WrongShape(Undirected, Unsigned):
    """Returns a matrix of the wrong shape from multivariate()."""

    name = "WrongShape"
    identifier = "wrong_shape"
    labels = ["test"]

    def multivariate(self, data):
        return np.zeros((2, 5))


class NonFinite(Undirected, Unsigned):
    """Returns infinities rather than raising."""

    name = "NonFinite"
    identifier = "non_finite"
    labels = ["test"]

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        return np.inf


class FiniteOnOnePair(Undirected, Unsigned):
    """Finite for processes 0 and 1 only: a valid result with one finite entry."""

    name = "FiniteOnOnePair"
    identifier = "finite_on_one_pair"
    labels = ["test"]

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        return 0.5 if {i, j} == {0, 1} else np.nan


class InfiniteOnOnePair(Undirected, Unsigned):
    """Finite everywhere except one infinity, between processes 0 and 1."""

    name = "InfiniteOnOnePair"
    identifier = "infinite_on_one_pair"
    labels = ["test"]

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        return np.inf if {i, j} == {0, 1} else 0.5


class RaisesOnOnePair(Directed, Unsigned):
    """Raises for the ordered pair (2, 1) and is finite everywhere else."""

    name = "RaisesOnOnePair"
    identifier = "raises_on_one_pair"
    labels = ["test"]

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        if (i, j) == (2, 1):
            raise RuntimeError("deliberate failure on one pair")
        return 0.5


class NeverFinite(Undirected, Unsigned):
    """NaN for every pair, without raising."""

    name = "NeverFinite"
    identifier = "never_finite"
    labels = ["test"]

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        return np.nan


class KillsItsWorker(Undirected, Unsigned):
    """Ends the process that evaluates it, as a segfault or the OOM killer does.

    Only ever run with ``n_jobs > 1``: in a serial run that process is the caller.
    """

    name = "KillsItsWorker"
    identifier = "kills_its_worker"
    labels = ["test"]

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        import os

        os._exit(1)


class KillsItsWorkerWhole(KillsItsWorker):
    """The same from ``multivariate``, so it is one task that no worker shares."""

    name = "KillsItsWorkerWhole"
    identifier = "kills_its_worker_whole"

    def multivariate(self, data):
        return self.bivariate(data, i=0, j=1)
