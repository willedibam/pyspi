import warnings
import sklearn.covariance as cov
from scipy import stats, signal
import numpy as np
import pandas as pd

from pyspi.base import Undirected, Signed, parse_bivariate, parse_multivariate


class Estimators(Undirected, Signed):
    """Base class for (functional) connectivity-based statistics

    Information on covariance estimators at: https://scikit-learn.org/stable/modules/covariance.html
    """

    name = "Covariance"
    labels = ["basic", "unordered", "linear", "undirected"]

    def __init__(self, kind, estimator="EmpiricalCovariance", squared=False):
        paramstr = f"_{estimator}"
        if squared:
            paramstr = "-sq" + paramstr
            self.labels = Estimators.labels + ["unsigned"]
            self.issigned = lambda: False
        else:
            self.labels = Estimators.labels + ["signed"]
        self.identifier = self.identifier + paramstr
        self._squared = squared
        self._estimator = estimator
        self._kind = kind

    def _from_cache(self, data):
        try:
            mycov = data.covariance[self._estimator]
        except (AttributeError, KeyError):
            z = data.to_numpy(squeeze=True).T

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mycov = getattr(cov, self._estimator)().fit(z)
            try:
                data.covariance[self._estimator] = mycov
            except AttributeError:
                data.covariance = {self._estimator: mycov}
        return mycov

    @parse_multivariate
    def multivariate(self, data):
        mycov = self._from_cache(data)
        matrix = getattr(mycov, self._kind + "_")
        np.fill_diagonal(matrix, np.nan)
        if self._squared:
            return np.square(matrix)
        else:
            return matrix


class Covariance(Estimators):

    name = "Covariance"
    identifier = "cov"

    def __init__(self, estimator="EmpiricalCovariance", squared=False):
        super().__init__(kind="covariance", squared=squared, estimator=estimator)


class Precision(Estimators):

    name = "Precision"
    identifier = "prec"

    def __init__(self, estimator="EmpiricalCovariance", squared=False):
        super().__init__(kind="precision", squared=squared, estimator=estimator)


class CrossCorrelation(Undirected, Signed):

    name = "Cross correlation"
    labels = ["basic", "linear", "undirected", "temporal"]

    def __init__(self, squared=False, statistic="max", sigonly=True):
        self.identifier = "xcorr"
        self._squared = squared
        self._statistic = statistic
        self._sigonly = sigonly

        if self._squared:
            self.issigned = lambda: False
            self.identifier = self.identifier + "-sq"
            self.labels = CrossCorrelation.labels + ["unsigned"]
        else:
            self.labels = CrossCorrelation.labels + ["signed"]
        self.identifier += f"_{statistic}_sig-{sigonly}"

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        T = data.n_observations
        try:
            r_ij = data.xcorr[(i, j)]
        except (KeyError, AttributeError):
            x, y = data.to_numpy()[[i, j]]

            # Force FFT method: O(N log N) vs O(N^2) direct for short signals.
            r_ij = np.squeeze(signal.correlate(x, y, "full", method="fft"))
            r_ij = r_ij / x.std() / y.std() / (T - 1)

            r_ij = r_ij[T - T // 4 : T + T // 4]

            try:
                data.xcorr[(i, j)] = r_ij
            except AttributeError:
                data.xcorr = {(i, j): r_ij}
            data.xcorr[(j, i)] = data.xcorr[(i, j)]

        # Truncate at first statistically significant zero
        sigonly = getattr(self, "_sigonly", False)
        if sigonly:
            N = len(r_ij) // 2
            threshold = 1.96 / np.sqrt(N)
            try:
                fzf = np.where(np.abs(r_ij[len(r_ij) // 2 :]) <= threshold)[0][0]
                fzr = np.where(np.abs(r_ij[: len(r_ij) // 2]) <= threshold)[0][-1]
                r_ij = r_ij[N - fzr : N + fzf]
            except IndexError:
                # All values significant or none — use full truncated r_ij
                pass

        if self._statistic == "max":
            if self._squared:
                return np.max(r_ij**2)
            return np.max(r_ij)
        elif self._statistic == "mean":
            if self._squared:
                return np.mean(r_ij**2)
            return np.mean(r_ij)
        else:
            raise TypeError(f"Unknown statistic: {self._statistic}")


class SpearmanR(Undirected, Signed):

    name = "Spearman's correlation coefficient"
    identifier = "spearmanr"
    labels = ["basic", "unordered", "rank", "linear", "undirected"]

    def __init__(self, squared=False):
        self._squared = squared
        if squared:
            self.issigned = lambda: False
            self.identifier = self.identifier + "-sq"
            self.labels = self.labels + ["unsigned"]
        else:
            self.labels = self.labels + ["signed"]

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        x, y = data.to_numpy()[[i, j]]
        if self._squared:
            return stats.spearmanr(x, y).correlation ** 2
        else:
            return stats.spearmanr(x, y).correlation

    @parse_multivariate
    def multivariate(self, data):
        """Vectorized: scipy.stats.spearmanr on full (M, T) matrix at once."""
        Z = data.to_numpy(squeeze=True)  # (M, T)
        rho, _ = stats.spearmanr(Z, axis=1)
        if Z.shape[0] == 2:
            rho = np.array([[1.0, rho], [rho, 1.0]])
        if self._squared:
            rho = rho ** 2
        np.fill_diagonal(rho, np.nan)
        return rho


class KendallTau(Undirected, Signed):

    name = "Kendall's tau"
    identifier = "kendalltau"
    labels = ["basic", "unordered", "rank", "linear", "undirected"]

    def __init__(self, squared=False):
        self._squared = squared
        if squared:
            self.issigned = lambda: False
            self.identifier = self.identifier + "-sq"
            self.labels = self.labels + ["unsigned"]
        else:
            self.labels = self.labels + ["signed"]

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        x, y = data.to_numpy()[[i, j]]
        if self._squared:
            return stats.kendalltau(x, y).correlation ** 2
        else:
            return stats.kendalltau(x, y).correlation

    @parse_multivariate
    def multivariate(self, data):
        """Vectorized: pandas .corr(method='kendall') on (T, M) DataFrame."""
        Z = data.to_numpy(squeeze=True)  # (M, T)
        df = pd.DataFrame(Z.T)
        tau = df.corr(method="kendall").values
        if self._squared:
            tau = tau ** 2
        np.fill_diagonal(tau, np.nan)
        return tau


class LaggedCorrelation(Undirected, Signed):
    """Lagged correlation SPI.

    Computes symmetric lagged correlation: 0.5 * (corr(x[tau:], y[:-tau]) + corr(y[tau:], x[:-tau])).
    Supports Pearson, Spearman, and Kendall estimators.
    """
    name = "Lagged correlation"
    labels = ["basic", "linear", "undirected", "temporal"]

    def __init__(self, estimator="pearson", tau=None, max_tau=None, squared=False):
        est = str(estimator).lower()
        if est not in {"pearson", "spearman", "kendall"}:
            raise ValueError(f"Unknown estimator: {estimator}")
        if max_tau is not None:
            raise ValueError("max_tau is only supported in config expansion; use tau.")
        if tau is None:
            raise ValueError("LaggedCorrelation requires tau.")

        self._estimator = est
        self._squared = bool(squared)
        if self._squared:
            self.issigned = lambda: False
            self.labels = LaggedCorrelation.labels + ["unsigned"]
            suffix = "-sq"
        else:
            self.labels = LaggedCorrelation.labels + ["signed"]
            suffix = ""
        self._tau = int(tau)
        if self._tau < 0:
            raise ValueError("tau must be >= 0.")
        self.identifier = f"corr_{est}_tau-{self._tau}{suffix}"

    def _corr(self, x, y):
        x = np.asarray(x).reshape(-1)
        y = np.asarray(y).reshape(-1)
        if x.size < 2 or y.size < 2:
            return np.nan
        if self._estimator == "pearson":
            return stats.pearsonr(x, y).correlation
        if self._estimator == "spearman":
            return stats.spearmanr(x, y).correlation
        if self._estimator == "kendall":
            return stats.kendalltau(x, y).correlation
        raise ValueError(f"Unknown estimator: {self._estimator}")

    def _lagged_corr(self, x, y, tau):
        if tau == 0:
            return self._corr(x, y)
        if tau >= x.size:
            return np.nan
        return self._corr(x[tau:], y[:-tau])

    def _symmetric_lagged_corr(self, x, y, tau):
        forward = self._lagged_corr(x, y, tau)
        backward = self._lagged_corr(y, x, tau)
        if np.isnan(forward):
            return backward
        if np.isnan(backward):
            return forward
        return 0.5 * (forward + backward)

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        x, y = data.to_numpy()[[i, j]]
        value = self._symmetric_lagged_corr(x, y, self._tau)
        return value**2 if self._squared else value

    @parse_multivariate
    def multivariate(self, data):
        """Vectorized multivariate lagged correlation."""
        Z = data.to_numpy(squeeze=True)  # (M, T)
        M, T = Z.shape
        tau = self._tau

        if tau == 0 or tau >= T:
            if tau >= T:
                return np.full((M, M), np.nan)
            if self._estimator == "pearson":
                C = np.corrcoef(Z)
            elif self._estimator == "spearman":
                C, _ = stats.spearmanr(Z, axis=1)
                if M == 2:
                    C = np.array([[1.0, C], [C, 1.0]])
            elif self._estimator == "kendall":
                C = pd.DataFrame(Z.T).corr(method="kendall").values
            else:
                raise ValueError(f"Unknown estimator: {self._estimator}")
            if self._squared:
                C = C ** 2
            np.fill_diagonal(C, np.nan)
            return C

        Z_lead = Z[:, tau:]
        Z_lag = Z[:, :-tau]

        if self._estimator == "pearson":
            stacked = np.vstack([Z_lead, Z_lag])
            C_full = np.corrcoef(stacked)
            forward = C_full[:M, M:]
            backward = C_full[M:, :M]
        elif self._estimator == "spearman":
            stacked = np.vstack([Z_lead, Z_lag])
            rho, _ = stats.spearmanr(stacked, axis=1)
            if stacked.shape[0] == 2:
                rho = np.array([[1.0, rho], [rho, 1.0]])
            forward = rho[:M, M:]
            backward = rho[M:, :M]
        elif self._estimator == "kendall":
            stacked = np.vstack([Z_lead, Z_lag])
            df = pd.DataFrame(stacked.T)
            C_full = df.corr(method="kendall").values
            forward = C_full[:M, M:]
            backward = C_full[M:, :M]
        else:
            raise ValueError(f"Unknown estimator: {self._estimator}")

        C = 0.5 * (forward + backward)
        if self._squared:
            C = C ** 2
        np.fill_diagonal(C, np.nan)
        return C
