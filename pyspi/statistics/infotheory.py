"""Information-theoretic statistics, implemented in pure NumPy.

The estimators here were ported from JIDT (Lizier, 2014, "JIDT: An
information-theoretic toolkit for studying the dynamics of complex systems",
Frontiers in Robotics and AI), which served as the reference implementation the
port was validated against; the version used for validation was JIDT 1.6.1.
"""

import math
import numpy as np
from pyspi import utils
import copy
import logging
import warnings

from scipy.spatial import cKDTree
from scipy.special import digamma, gammaln

from pyspi.base import Undirected, Directed, Unsigned, parse_univariate, parse_bivariate, parse_multivariate

# ---------------------------------------------------------------------------
# Pure-numpy entropy calculators (drop-in replacements for JIDT)
# ---------------------------------------------------------------------------

def _gaussian_log_det(cov, ridge_rel=1e-8):
    """log|Σ + εI| with ε = ridge_rel * mean(diag(Σ)).

    Returns NaN on degenerate input (non-finite mean, non-positive mean,
    or still-singular after ridge). Never returns -inf, so that downstream
    entropy differences (H_xy - H_y) cannot cascade to catastrophic overflow
    in np.nan_to_num.

    Academic rationale: for rank-deficient Σ the unregularised log|Σ| is
    mathematically -inf (data lives on a lower-dim subspace). JIDT adds
    NOISE_LEVEL_TO_ADD=1e-8 gaussian noise to observations for the same
    reason; ridge-regularising Σ directly is equivalent in expectation
    (cov of x+η equals Σ + σ²I for iid additive noise η) and numerically
    stabler than perturbing samples.
    """
    if np.ndim(cov) == 0:
        var = float(cov)
        if not np.isfinite(var):
            return np.nan
        if var <= 0:
            return np.nan
        eps = ridge_rel * var
        return float(np.log(var + eps))
    cov = np.asarray(cov, dtype=np.float64)
    d = cov.shape[0]
    mean_diag = float(np.trace(cov)) / d
    if not np.isfinite(mean_diag) or mean_diag <= 0:
        return np.nan
    eps = ridge_rel * mean_diag
    try:
        sign, log_det = np.linalg.slogdet(cov + eps * np.eye(d))
    except np.linalg.LinAlgError:
        return np.nan
    if sign <= 0 or not np.isfinite(log_det):
        return np.nan
    return float(log_det)


class GaussianEntropyCalculator:
    """Drop-in for JIDT's EntropyCalculatorMultiVariateGaussian.

    H = 0.5 * d * log(2*pi*e) + 0.5 * log|Sigma|
    """
    def __init__(self):
        self._d = None
        self._obs = None

    def initialise(self, d):
        self._d = int(d)
        self._obs = None

    def setObservations(self, data):
        self._obs = np.asarray(data, dtype=np.float64)
        if self._obs.ndim == 1:
            self._obs = self._obs.reshape(-1, 1)

    def setProperty(self, key, value):
        pass

    def computeAverageLocalOfObservations(self):
        X = self._obs
        N, d = X.shape
        cov = np.cov(X, rowvar=False, ddof=1)
        log_det = _gaussian_log_det(cov)
        if not np.isfinite(log_det):
            return float('nan')
        return float(0.5 * d * np.log(2 * np.pi * np.e) + 0.5 * log_det)


class KLEntropyCalculator:
    """Drop-in for JIDT's EntropyCalculatorMultiVariateKozachenko.

    Uses L2 norm with k=1, matching JIDT.
    H = psi(N) - psi(1) + log(c_d) + (d/N) * sum(log(eps_i))
    """
    def __init__(self):
        self._d = None
        self._obs = None

    def initialise(self, d):
        self._d = int(d)
        self._obs = None

    def setObservations(self, data):
        self._obs = np.asarray(data, dtype=np.float64)
        if self._obs.ndim == 1:
            self._obs = self._obs.reshape(-1, 1)

    def setProperty(self, key, value):
        pass

    def computeAverageLocalOfObservations(self):
        X = self._obs
        N, d = X.shape
        tree = cKDTree(X)
        dists, _ = tree.query(X, k=2, p=2)  # k=1 NN (index 0 = self)
        eps = dists[:, 1]
        log_cd = (d / 2.0) * np.log(np.pi) - gammaln(d / 2.0 + 1)
        return float(
            digamma(N) - digamma(1) + log_cd + (d / N) * np.sum(np.log(eps))
        )


def _gaussian_entropy_from_data(data_2d):
    """Compute Gaussian entropy from (N, d) array.

    Returns NaN (not -inf) on singular covariance after ridge regularisation,
    so that downstream differences (e.g. H(X,Y)-H(Y)) never cascade to
    +/-inf and get clipped to float-max by np.nan_to_num.
    """
    N, d = data_2d.shape
    cov = np.cov(data_2d, rowvar=False, ddof=1)
    log_det = _gaussian_log_det(cov)
    if not np.isfinite(log_det):
        return float('nan')
    return 0.5 * d * np.log(2 * np.pi * np.e) + 0.5 * log_det


# ---------------------------------------------------------------------------
# Box-kernel KDE entropy/MI/TE calculators (replace JIDT kernel estimators)
# ---------------------------------------------------------------------------

class KernelEntropyCalculator:
    """Drop-in for JIDT's EntropyCalculatorMultiVariateKernel.

    Box kernel (Heaviside) with L-infinity norm, matching JIDT exactly:
    H = mean(log2(N) - log2(count_i))  [bits]
    where count_i = #{j : |x_j - x_i|_inf <= kernel_width} (includes self).

    When NORMALISE=true (JIDT default), data is normalised by std before
    counting; kernel_width is then in units of std dev.
    """

    def __init__(self):
        self._d = None
        self._obs = None
        self._kernel_width = 0.25
        self._normalise = True

    def initialise(self, d):
        self._d = int(d)
        self._obs = None

    def setProperty(self, key, value):
        if key == "KERNEL_WIDTH":
            self._kernel_width = float(value)
        elif key == "NORMALISE":
            self._normalise = str(value).lower() == "true"

    def setObservations(self, data):
        self._obs = np.asarray(data, dtype=np.float64)
        if self._obs.ndim == 1:
            self._obs = self._obs.reshape(-1, 1)

    def computeAverageLocalOfObservations(self):
        X = self._obs
        N, d = X.shape
        w = self._kernel_width
        # NORMALISE: JIDT scales the bandwidth by std (kernelWidthsInUse = w*std)
        # rather than standardising data. Equivalent operation here is to
        # standardise X *and* add log2(prod(std)) to the entropy — otherwise
        # we'd be reporting H(X/std) = H(X) - log2(std), missing the scale
        # term and giving a constant downward bias of d*log2(std) nats/bits.
        log_std_total = 0.0
        if self._normalise:
            stds = np.std(X, axis=0, ddof=1)
            stds = np.where(stds > 0, stds, 1.0)
            X = X / stds[None, :]
            log_std_total = float(np.sum(np.log2(stds)))

        tree = cKDTree(X)
        # JIDT half-width = kernel_width (not kernel_width/2)
        counts = tree.query_ball_point(X, r=w, p=np.inf,
                                        return_length=True)
        counts = np.asarray(counts, dtype=np.float64)
        # H = mean(log2(N) - log2(count)) + d*log2(2*w) + sum_d log2(std_d)  [bits]
        return float(np.mean(np.log2(N) - np.log2(counts))
                     + d * np.log2(2.0 * w) + log_std_total)


class KernelMICalculator:
    """Drop-in for JIDT's MutualInfoCalculatorMultiVariateKernel.

    MI = mean(log2(n_xy * N / (n_x * n_y)))  [bits]
    where n_x, n_y, n_xy are counts within L∞ ball of radius kernel_width.
    Matches JIDT exactly.
    """

    def __init__(self):
        self._kernel_width = 0.25
        self._normalise = True
        self._d1 = 1
        self._d2 = 1
        self._obs1 = None
        self._obs2 = None

    def initialise(self, d1, d2):
        self._d1 = int(d1)
        self._d2 = int(d2)

    def setProperty(self, key, value):
        if key == "KERNEL_WIDTH":
            self._kernel_width = float(value)
        elif key == "NORMALISE":
            self._normalise = str(value).lower() == "true"

    def setObservations(self, src, targ):
        self._obs1 = np.asarray(src, dtype=np.float64)
        self._obs2 = np.asarray(targ, dtype=np.float64)
        if self._obs1.ndim == 1:
            self._obs1 = self._obs1.reshape(-1, 1)
        if self._obs2.ndim == 1:
            self._obs2 = self._obs2.reshape(-1, 1)

    def computeAverageLocalOfObservations(self):
        X = self._obs1
        Y = self._obs2
        N = X.shape[0]
        w = self._kernel_width

        if self._normalise:
            def _norm(data):
                stds = np.std(data, axis=0, ddof=1)
                stds = np.where(stds > 0, stds, 1.0)
                return data / stds[None, :]
            X = _norm(X)
            Y = _norm(Y)

        XY = np.column_stack([X, Y])
        tree_x = cKDTree(X)
        tree_y = cKDTree(Y)
        tree_xy = cKDTree(XY)

        # JIDT half-width = kernel_width
        n_x = np.asarray(tree_x.query_ball_point(X, r=w, p=np.inf,
                                                   return_length=True), dtype=np.float64)
        n_y = np.asarray(tree_y.query_ball_point(Y, r=w, p=np.inf,
                                                   return_length=True), dtype=np.float64)
        n_xy = np.asarray(tree_xy.query_ball_point(XY, r=w, p=np.inf,
                                                     return_length=True), dtype=np.float64)

        # MI = mean(log2(n_xy * N / (n_x * n_y)))  [bits]
        mi = np.mean(np.log2(n_xy) + np.log2(N) - np.log2(n_x) - np.log2(n_y))
        return float(mi)


class KernelTECalculator:
    """Drop-in for JIDT's TransferEntropyCalculatorKernel.

    TE(X→Y) = mean(log2(n_yn_yp_x * n_yp / (n_yp_x * n_yn_yp)))  [bits]
    Uses box kernel with L∞ norm, half-width = kernel_width.
    Matches JIDT exactly.
    """

    def __init__(self):
        self._kernel_width = 0.25
        self._normalise = True
        self._k_history = 1
        self._dyn_corr_excl = None
        self._props = {}

    def initialise(self):
        pass

    def setProperty(self, key, value):
        self._props[key] = value
        if key == "KERNEL_WIDTH":
            self._kernel_width = float(value)
        elif key == "NORMALISE":
            self._normalise = str(value).lower() == "true"
        elif key == "k_HISTORY":
            self._k_history = int(value)
        elif key == "DYN_CORR_EXCL":
            self._dyn_corr_excl = int(value)

    def setObservations(self, src, targ):
        self._src = np.asarray(src, dtype=np.float64).ravel()
        self._targ = np.asarray(targ, dtype=np.float64).ravel()

    def computeAverageLocalOfObservations(self):
        src = self._src
        targ = self._targ
        k = self._k_history
        T = len(src)
        w = self._kernel_width

        if T <= k:
            return np.nan

        # Build vectors
        n_pts = T - k
        y_past = np.column_stack([targ[k - 1 - lag: T - 1 - lag] for lag in range(k)])
        y_next = targ[k:].reshape(-1, 1)
        x_t = src[k - 1: T - 1].reshape(-1, 1)

        if self._normalise:
            def _norm(data):
                stds = np.std(data, axis=0, ddof=1)
                stds = np.where(stds > 0, stds, 1.0)
                return data / stds[None, :]
            y_past = _norm(y_past)
            y_next = _norm(y_next)
            x_t = _norm(x_t)

        # Joint spaces
        yn_yp = np.column_stack([y_next, y_past])
        yp_x = np.column_stack([y_past, x_t])
        yn_yp_x = np.column_stack([y_next, y_past, x_t])

        tree_yp = cKDTree(y_past)
        tree_yn_yp = cKDTree(yn_yp)
        tree_yp_x = cKDTree(yp_x)
        tree_yn_yp_x = cKDTree(yn_yp_x)

        # JIDT half-width = kernel_width
        dce = self._dyn_corr_excl
        if dce is None or dce <= 0:
            n_yp = np.asarray(tree_yp.query_ball_point(y_past, r=w, p=np.inf,
                                                         return_length=True), dtype=np.float64)
            n_yn_yp = np.asarray(tree_yn_yp.query_ball_point(yn_yp, r=w, p=np.inf,
                                                               return_length=True), dtype=np.float64)
            n_yp_x = np.asarray(tree_yp_x.query_ball_point(yp_x, r=w, p=np.inf,
                                                             return_length=True), dtype=np.float64)
            n_yn_yp_x = np.asarray(tree_yn_yp_x.query_ball_point(yn_yp_x, r=w, p=np.inf,
                                                                   return_length=True), dtype=np.float64)
        else:
            # Theiler window: exclude neighbours j with |j - i| <= dce, matching
            # JIDT's DYN_CORR_EXCL convention (excludes 2*dce+1 points centered on i,
            # including self). We query index lists then filter by temporal separation.
            def _counts_with_theiler(tree, pts):
                idx_lists = tree.query_ball_point(pts, r=w, p=np.inf)
                out = np.empty(len(idx_lists), dtype=np.float64)
                for i, nbrs in enumerate(idx_lists):
                    nbrs_arr = np.asarray(nbrs, dtype=np.int64)
                    out[i] = np.sum(np.abs(nbrs_arr - i) > dce)
                return out
            n_yp = _counts_with_theiler(tree_yp, y_past)
            n_yn_yp = _counts_with_theiler(tree_yn_yp, yn_yp)
            n_yp_x = _counts_with_theiler(tree_yp_x, yp_x)
            n_yn_yp_x = _counts_with_theiler(tree_yn_yp_x, yn_yp_x)

        # Drop samples where any bin is empty (log2(0) = -inf; JIDT skips these).
        valid = (n_yp > 0) & (n_yn_yp > 0) & (n_yp_x > 0) & (n_yn_yp_x > 0)
        if not np.any(valid):
            return float('nan')
        n_yp, n_yn_yp = n_yp[valid], n_yn_yp[valid]
        n_yp_x, n_yn_yp_x = n_yp_x[valid], n_yn_yp_x[valid]

        # TE = mean(log2(n_yn_yp_x * n_yp / (n_yp_x * n_yn_yp)))  [bits]
        te = np.mean(np.log2(n_yn_yp_x) + np.log2(n_yp) - np.log2(n_yp_x) - np.log2(n_yn_yp))
        return float(te)


# ---------------------------------------------------------------------------
# Symbolic Transfer Entropy (ordinal patterns)
# ---------------------------------------------------------------------------

def _ordinal_pattern_id(vec):
    """Convert a vector to its ordinal pattern ID.

    The ordinal pattern is the rank ordering. Maps to an integer in [0, d!).
    Uses the factorial number system (Lehmer code).
    """
    d = len(vec)
    # Get the rank order (argsort of argsort)
    order = np.argsort(vec)
    # Lehmer code
    code = 0
    remaining = list(range(d))
    factorial = 1
    for i in range(1, d):
        factorial *= i
    for i in range(d - 1):
        pos = remaining.index(order[i])
        code += pos * factorial
        remaining.pop(pos)
        if i < d - 2:
            factorial //= (d - 1 - i)
    return code


def _series_to_ordinal_symbols(x, k):
    """Convert a 1D time series to a sequence of ordinal pattern symbols.

    For each t, the embedding vector is [x[t], x[t-1], ..., x[t-k+1]].
    Returns integer array of symbol IDs, length T-k+1.
    """
    x = np.asarray(x).ravel()
    T = len(x)
    if T < k:
        return np.array([], dtype=int)

    n_pts = T - k + 1
    symbols = np.empty(n_pts, dtype=int)

    # Build embedding matrix
    embedding = np.column_stack([x[k - 1 - lag: T - lag] for lag in range(k)])

    for t in range(n_pts):
        symbols[t] = _ordinal_pattern_id(embedding[t])

    return symbols


class SymbolicTECalculator:
    """Drop-in for JIDT's TransferEntropyCalculatorSymbolic.

    Converts source and target to ordinal patterns of length k,
    then computes TE from joint symbol histograms.

    TE = H(Y_next | Y_past) - H(Y_next | Y_past, X)
       = H(Y_next, Y_past) - H(Y_past) - H(Y_next, Y_past, X) + H(Y_past, X)
    where all entropies are discrete (histogram-based).
    """

    def __init__(self):
        self._k_history = 1
        self._props = {}

    def initialise(self):
        pass

    def setProperty(self, key, value):
        self._props[key] = value
        if key == "k_HISTORY":
            self._k_history = int(value)

    def setObservations(self, src, targ):
        self._src = np.asarray(src, dtype=np.float64).ravel()
        self._targ = np.asarray(targ, dtype=np.float64).ravel()

    def computeAverageLocalOfObservations(self):
        src = self._src
        targ = self._targ
        k = self._k_history
        T = len(src)

        if T <= k:
            return np.nan

        # Convert to ordinal patterns
        src_symbols = _series_to_ordinal_symbols(src, k)
        targ_symbols = _series_to_ordinal_symbols(targ, k)

        # Align: targ_next starts at index 1 of the symbol sequence
        # targ_past = targ_symbols[:-1], targ_next = targ_symbols[1:]
        # src_current = src_symbols[:-1] (concurrent with targ_past)
        n = min(len(src_symbols), len(targ_symbols)) - 1
        if n <= 0:
            return np.nan

        targ_next = targ_symbols[1:n + 1]
        targ_past = targ_symbols[:n]
        src_curr = src_symbols[:n]

        def _discrete_entropy(*arrs):
            """Joint entropy of integer-valued arrays using histograms.

            Counts distinct rows directly rather than packing the symbols into
            a single integer. The previous encoding multiplied by a multiplier
            that squared at each step, so the packed value reached (k!)^3 and
            exceeded int64 for k >= 10, wrapping silently. Wrapping is not the
            same as colliding -- no collisions occur on the shipped fixtures --
            but the encoding gave no guarantee, and correctness should not rest
            on the arithmetic happening to stay injective.
            """
            if len(arrs) == 1:
                _, counts = np.unique(arrs[0], return_counts=True)
            else:
                stacked = np.column_stack(arrs)
                _, counts = np.unique(stacked, axis=0, return_counts=True)
            probs = counts / counts.sum()
            return -np.sum(probs * np.log2(probs))

        # TE = H(yn, yp) - H(yp) - H(yn, yp, x) + H(yp, x)
        H_yn_yp = _discrete_entropy(targ_next, targ_past)
        H_yp = _discrete_entropy(targ_past)
        H_yn_yp_x = _discrete_entropy(targ_next, targ_past, src_curr)
        H_yp_x = _discrete_entropy(targ_past, src_curr)

        te = H_yn_yp - H_yp - H_yn_yp_x + H_yp_x
        return float(te)


def _numpy_delay_embedding(x, dim):
    """Numpy equivalent of JIDT's MatrixUtils.makeDelayEmbeddingVector."""
    x = np.asarray(x).ravel()
    T = len(x)
    if dim == 0:
        return np.empty((T + 1, 0))
    return np.column_stack([x[dim - 1 - lag: T - lag] for lag in range(dim)])


# ---------------------------------------------------------------------------
# KSG MI estimator
# ---------------------------------------------------------------------------

def _validate_ksg_sample(N, k, w, context=""):
    """Reject KSG settings that cannot produce a meaningful estimate.

    The estimator needs k neighbours drawn from the points that survive the
    Theiler exclusion. Without this check, k >= N silently returned a finite
    number that grows with k: k=30 on N=20 gave 0.414 and k=100 gave 1.63,
    neither of which is an estimate of anything.
    """
    effective = N - (2 * w + 1) if w else N - 1
    where = f" ({context})" if context else ""
    if N < 2:
        raise ValueError(f"KSG needs at least 2 observations, got {N}{where}.")
    if k < 1:
        raise ValueError(f"KSG needs k >= 1, got k={k}{where}.")
    if k > effective:
        raise ValueError(
            f"KSG k={k} exceeds the {effective} usable neighbour(s) for "
            f"N={N} observations with Theiler window w={w}{where}. Reduce k, "
            f"lengthen the series, or narrow the Theiler window."
        )


def _ksg_mi_pair(x, y, k, w, tree_x, tree_y):
    """KSG Estimator 1 MI for a single pair."""
    N = len(x)
    _validate_ksg_sample(N, k, w, context="mutual information")
    # A constant marginal has zero radius everywhere: every neighbour distance
    # collapses to 0, the digamma counts saturate, and the estimator returns a
    # number that reflects the tie structure rather than any dependence.
    for name, v in (("x", x), ("y", y)):
        if np.ptp(v) == 0:
            raise ValueError(
                f"KSG cannot estimate mutual information: marginal {name} is "
                f"constant, so all neighbour distances are zero."
            )
    xy = np.column_stack([x, y])
    tree_xy = cKDTree(xy)

    if w == 0:
        dists, _ = tree_xy.query(xy, k=k + 1, p=np.inf)
        eps = dists[:, k]
        eps_strict = eps * (1.0 - 1e-10)
        nx_lists = tree_x.query_ball_point(x.reshape(-1, 1), eps_strict, p=np.inf)
        ny_lists = tree_y.query_ball_point(y.reshape(-1, 1), eps_strict, p=np.inf)
        n_x = np.array([len(lst) - 1 for lst in nx_lists], dtype=np.float64)
        n_y = np.array([len(lst) - 1 for lst in ny_lists], dtype=np.float64)
    else:
        n_query = min(k + 2 * w + 2, N)
        dists_all, idx_all = tree_xy.query(xy, k=n_query, p=np.inf)

        eps = np.empty(N)
        n_x = np.empty(N)
        n_y = np.empty(N)

        for i in range(N):
            valid = np.abs(idx_all[i] - i) > w
            valid[0] = False
            d_valid = dists_all[i][valid]

            if len(d_valid) < k:
                all_dists = np.max(np.abs(xy - xy[i]), axis=1)
                all_dists[max(0, i - w): i + w + 1] = np.inf
                all_dists[i] = np.inf
                d_valid = np.sort(all_dists)
                d_valid = d_valid[np.isfinite(d_valid)]

            e = d_valid[k - 1] if len(d_valid) >= k else np.inf
            eps[i] = e

            # Count marginal neighbours STRICTLY within eps (< eps), matching
            # KSG1 and the w==0 branch above. Inclusive (<= eps) counting wrongly
            # admits the k-th neighbour at the boundary, inflating n_x/n_y and
            # flipping the sign of the Theiler-window effect on MI.
            e_strict = e * (1.0 - 1e-10)
            ix = tree_x.query_ball_point([[x[i]]], e_strict, p=np.inf)[0]
            iy = tree_y.query_ball_point([[y[i]]], e_strict, p=np.inf)[0]
            n_x[i] = sum(1 for j in ix if abs(j - i) > w and j != i)
            n_y[i] = sum(1 for j in iy if abs(j - i) > w and j != i)

    # psi(N) uses the full, unreduced N (KSG1 convention; only the neighbour
    # set is window-restricted, not the normalisation). See JIDT
    # MutualInfoCalculatorMultiVariateKraskov.
    mi = digamma(k) - np.mean(digamma(n_x + 1) + digamma(n_y + 1)) + digamma(N)
    return float(mi)


def _ksg_mi_general(A, B, k_nn, w=0):
    """KSG Estimator 1 MI(A; B) for multivariate A (N,dA), B (N,dB), L-inf norm.

    Generalisation of _ksg_mi_pair to arbitrary marginal dimensions (the pair
    version is kept for the 1-D/1-D MI grids where pre-built trees are reused).
    Marginal neighbours are counted strictly (< eps); psi(N) uses full N. An
    optional Theiler window w excludes |j-i| <= w from the neighbour set.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim == 1:
        A = A[:, None]
    if B.ndim == 1:
        B = B[:, None]
    N = A.shape[0]
    AB = np.column_stack([A, B])
    tree_ab = cKDTree(AB)
    tree_a = cKDTree(A)
    tree_b = cKDTree(B)

    if w == 0:
        dists, _ = tree_ab.query(AB, k=k_nn + 1, p=np.inf)
        eps = dists[:, k_nn] * (1.0 - 1e-10)
        n_a = np.array([len(lst) - 1 for lst in
                        tree_a.query_ball_point(A, eps, p=np.inf)], dtype=np.float64)
        n_b = np.array([len(lst) - 1 for lst in
                        tree_b.query_ball_point(B, eps, p=np.inf)], dtype=np.float64)
    else:
        n_query = min(k_nn + 2 * w + 2, N)
        dists_all, idx_all = tree_ab.query(AB, k=n_query, p=np.inf)
        n_a = np.empty(N)
        n_b = np.empty(N)
        for i in range(N):
            valid = np.abs(idx_all[i] - i) > w
            valid[0] = False
            d_valid = dists_all[i][valid]
            if len(d_valid) < k_nn:
                all_d = np.max(np.abs(AB - AB[i]), axis=1)
                all_d[max(0, i - w): i + w + 1] = np.inf
                all_d[i] = np.inf
                d_valid = np.sort(all_d)
                d_valid = d_valid[np.isfinite(d_valid)]
            e = d_valid[k_nn - 1] if len(d_valid) >= k_nn else np.inf
            e_strict = e * (1.0 - 1e-10)
            ia = tree_a.query_ball_point(A[i], e_strict, p=np.inf)
            ib = tree_b.query_ball_point(B[i], e_strict, p=np.inf)
            n_a[i] = sum(1 for j in ia if abs(j - i) > w and j != i)
            n_b[i] = sum(1 for j in ib if abs(j - i) > w and j != i)

    return float(digamma(k_nn) + digamma(N)
                 - np.mean(digamma(n_a + 1) + digamma(n_b + 1)))


# ---------------------------------------------------------------------------
# Transfer Entropy helpers
# ---------------------------------------------------------------------------

class _DummyTECalculator:
    """Dummy TE calculator for gaussian/kraskov fixed-embedding."""
    def __init__(self):
        self._props = {}
    def setProperty(self, key, value):
        self._props[key] = value
    def initialise(self):
        pass
    def setObservations(self, src, targ):
        pass
    def computeAverageLocalOfObservations(self):
        return np.nan


def _te_build_embeddings(src, targ, k_history, k_tau, l_history, l_tau):
    """Build delay embeddings for transfer entropy computation."""
    T = len(src)
    max_lookback = max((k_history - 1) * k_tau, (l_history - 1) * l_tau)
    start = max_lookback
    end = T - 1

    if start >= end:
        return None, None, None

    Y_future = targ[start + 1: end + 1].reshape(-1, 1)

    Y_past_cols = []
    for lag_idx in range(k_history):
        lag = lag_idx * k_tau
        Y_past_cols.append(targ[start - lag: end - lag])
    Y_past = np.column_stack(Y_past_cols)

    X_past_cols = []
    for lag_idx in range(l_history):
        lag = lag_idx * l_tau
        X_past_cols.append(src[start - lag: end - lag])
    X_past = np.column_stack(X_past_cols)

    return Y_future, Y_past, X_past


def _gaussian_ais(targ, k, tau):
    """Bias-corrected Gaussian AIS criterion: MI(Y_future; Y_past_embedding(k, tau)).

    Used for AIS-criterion auto-embedding: select (k, tau) = argmax AIS.

    The raw in-sample log-det multiinformation is biased upward by the mean of
    its chi-squared null, E[MI_null] = df / (2N) with df = dim(Y_f)*k = k (the
    future is 1-D). Without this correction the criterion increases monotonically
    in k and saturates at k_max. Subtracting k/(2N) gives an interior maximum and
    matches JIDT's ActiveInfoStorageCalculatorGaussian.computeAdditionalBiasToRemove.
    Ref: Ragwitz & Kantz (2002); Wibral et al. (2014); Lizier JIDT.
    """
    T = len(targ)
    start = (k - 1) * tau
    end = T - 1
    if start >= end:
        return -np.inf
    Y_f = targ[start + 1: end + 1].reshape(-1, 1)
    Y_p = np.column_stack([targ[start - i * tau: end - i * tau] for i in range(k)])
    YfYp = np.concatenate([Y_f, Y_p], axis=1)
    N = Y_f.shape[0]

    def _slogdet(arr):
        cov = np.cov(arr, rowvar=False, ddof=1)
        if arr.shape[1] == 1:
            return np.log(max(float(cov), 1e-300))
        sign, ld = np.linalg.slogdet(cov)
        return ld if sign > 0 else -np.inf

    ais_raw = 0.5 * (_slogdet(Y_f) + _slogdet(Y_p) - _slogdet(YfYp))
    return ais_raw - k / (2.0 * N)


def _ksg_ais(targ, k, tau, k_nn, w=0):
    """KSG (Kraskov) AIS = MI(Y_future; Y_past_embedding(k, tau)), for embedding
    selection of the *kraskov* TE estimator.

    Estimator-consistent counterpart of _gaussian_ais: the KSG estimator is
    approximately bias-free, so max-KSG-AIS over k has a genuine interior peak
    without an explicit bias term (cf. JIDT MAX_CORR_AIS, which returns 0 extra
    bias for KSG; Wibral et al. 2014). Selecting the embedding with the same
    estimator used for the final TE avoids the linear/nonlinear mismatch of
    using Gaussian AIS to embed a nonlinear estimator.
    """
    T = len(targ)
    start = (k - 1) * tau
    end = T - 1
    if start >= end:
        return -np.inf
    Y_f = targ[start + 1: end + 1].reshape(-1, 1)
    Y_p = np.column_stack([targ[start - i * tau: end - i * tau] for i in range(k)])
    return _ksg_mi_general(Y_f, Y_p, k_nn, w)


def _auto_embed_gaussian_te(src, targ, k_max, tau_max):
    """Gaussian TE with Ragwitz-style AIS auto-embedding on target."""
    best_k, best_tau, best_ais = 1, 1, -np.inf
    for k in range(1, k_max + 1):
        for tau in range(1, tau_max + 1):
            ais = _gaussian_ais(targ, k, tau)
            if ais > best_ais:
                best_ais, best_k, best_tau = ais, k, tau
    return _gaussian_te_bivariate(src, targ, best_k, best_tau, 1, 1)


def _gaussian_te_bivariate(src, targ, k_history, k_tau, l_history, l_tau):
    """Gaussian TE via log-determinant ratio."""
    Y_f, Y_p, X_p = _te_build_embeddings(src, targ, k_history, k_tau, l_history, l_tau)
    if Y_f is None:
        return np.nan

    def _slogdet(data):
        N, d = data.shape
        cov = np.cov(data, rowvar=False, ddof=1)
        if d == 1:
            return np.log(max(float(cov), 1e-300))
        sign, ld = np.linalg.slogdet(cov)
        return ld if sign > 0 else -np.inf

    YfYp = np.concatenate([Y_f, Y_p], axis=1)
    YfYpXp = np.concatenate([Y_f, Y_p, X_p], axis=1)
    YpXp = np.concatenate([Y_p, X_p], axis=1)

    te = 0.5 * (_slogdet(YfYp) - _slogdet(Y_p) - _slogdet(YfYpXp) + _slogdet(YpXp))
    return float(te)


def _kraskov_te_bivariate(src, targ, k_history, k_tau, l_history, l_tau, k_nn, w):
    """Kraskov TE via Frenzel-Pompe CMI estimator."""
    Y_f, Y_p, X_p = _te_build_embeddings(src, targ, k_history, k_tau, l_history, l_tau)
    # Same precondition as the MI path, applied to the *embedded* sample count
    # rather than the raw series length: embedding consumes the lookback, so
    # the usable N here is smaller than len(targ). Validating only inside
    # _ksg_mi_pair left this path unguarded, and TE with k=30 on T=20 returned
    # a finite 0.467.
    _validate_ksg_sample(Y_f.shape[0], k_nn, w, context="transfer entropy")
    if Y_f is None:
        return np.nan

    N = len(Y_f)

    joint = np.concatenate([Y_f, X_p, Y_p], axis=1)
    tree_joint = cKDTree(joint)

    YfYp = np.concatenate([Y_f, Y_p], axis=1)
    XpYp = np.concatenate([X_p, Y_p], axis=1)

    tree_YfYp = cKDTree(YfYp)
    tree_XpYp = cKDTree(XpYp)
    tree_Yp = cKDTree(Y_p)

    if w == 0:
        dists, _ = tree_joint.query(joint, k=k_nn + 1, p=np.inf)
        eps = dists[:, k_nn]
        eps_strict = eps * (1.0 - 1e-10)

        n_YfYp = np.array([len(lst) - 1 for lst in
                           tree_YfYp.query_ball_point(YfYp, eps_strict, p=np.inf)],
                          dtype=np.float64)
        n_XpYp = np.array([len(lst) - 1 for lst in
                           tree_XpYp.query_ball_point(XpYp, eps_strict, p=np.inf)],
                          dtype=np.float64)
        n_Yp = np.array([len(lst) - 1 for lst in
                         tree_Yp.query_ball_point(Y_p, eps_strict, p=np.inf)],
                        dtype=np.float64)
    else:
        n_query = min(k_nn + 2 * w + 2, N)
        dists_all, idx_all = tree_joint.query(joint, k=n_query, p=np.inf)

        n_YfYp = np.empty(N)
        n_XpYp = np.empty(N)
        n_Yp = np.empty(N)

        for i in range(N):
            valid = np.abs(idx_all[i] - i) > w
            valid[0] = False
            d_valid = dists_all[i][valid]

            if len(d_valid) < k_nn:
                all_dists = np.max(np.abs(joint - joint[i]), axis=1)
                all_dists[max(0, i - w): i + w + 1] = np.inf
                all_dists[i] = np.inf
                d_valid = np.sort(all_dists)
                d_valid = d_valid[np.isfinite(d_valid)]

            e = d_valid[k_nn - 1] if len(d_valid) >= k_nn else np.inf
            e_strict = e * (1.0 - 1e-10)

            lsts_YfYp = tree_YfYp.query_ball_point(YfYp[i], e_strict, p=np.inf)
            lsts_XpYp = tree_XpYp.query_ball_point(XpYp[i], e_strict, p=np.inf)
            lsts_Yp = tree_Yp.query_ball_point(Y_p[i], e_strict, p=np.inf)

            n_YfYp[i] = sum(1 for j in lsts_YfYp if abs(j - i) > w and j != i)
            n_XpYp[i] = sum(1 for j in lsts_XpYp if abs(j - i) > w and j != i)
            n_Yp[i] = sum(1 for j in lsts_Yp if abs(j - i) > w and j != i)

    te = float(digamma(k_nn) + np.mean(
        digamma(n_Yp + 1) - digamma(n_YfYp + 1) - digamma(n_XpYp + 1)
    ))
    return te


# ---------------------------------------------------------------------------
# Information-theory base class — with estimator dispatch
# ---------------------------------------------------------------------------

_ESTIMATORS = frozenset({"gaussian", "kraskov", "kernel", "kozachenko", "symbolic"})

# Auto-embedding selection criteria that are actually implemented. The search
# maximises active information storage under the destination's own estimator.
_AUTO_EMBED_METHODS = frozenset({"MAX_CORR_AIS"})


class InfoTheoryBase(Unsigned):

    _AUTO_EMBED_METHOD_PROP_NAME = "AUTO_EMBED_METHOD"
    _K_HISTORY_PROP_NAME = "k_HISTORY"
    _K_TAU_PROP_NAME = "k_TAU"
    _L_HISTORY_PROP_NAME = "l_HISTORY"
    _L_TAU_PROP_NAME = "l_TAU"
    _K_SEARCH_MAX_PROP_NAME = "AUTO_EMBED_K_SEARCH_MAX"
    _TAU_SEARCH_MAX_PROP_NAME = "AUTO_EMBED_TAU_SEARCH_MAX"

    # Which estimator each optional parameter belongs to. A parameter supplied
    # to an estimator that ignores it is rejected rather than silently dropped:
    # accepting kernel_width under estimator="gaussian" told the caller a
    # kernel width had been applied when nothing used it.
    _PARAM_OWNER = {
        "kernel_width": ("kernel",),
        "prop_k": ("kraskov",),
        "dyn_corr_excl": ("kraskov",),
    }

    def __init__(
        self, estimator="gaussian", kernel_width=None, prop_k=None, dyn_corr_excl=None
    ):
        if estimator not in _ESTIMATORS:
            raise ValueError(
                f"Unknown estimator {estimator!r}; expected one of "
                f"{sorted(_ESTIMATORS)}."
            )

        supplied = {
            "kernel_width": kernel_width,
            "prop_k": prop_k,
            "dyn_corr_excl": dyn_corr_excl,
        }
        for name, value in supplied.items():
            owners = self._PARAM_OWNER[name]
            if value is not None and estimator not in owners:
                raise ValueError(
                    f"{name}={value!r} is not used by estimator={estimator!r} "
                    f"(it applies to {'/'.join(owners)}). Remove it, or select "
                    f"the estimator it belongs to."
                )

        self._estimator = estimator
        # Defaults applied after validation so "not supplied" stays
        # distinguishable from "supplied with the default value".
        self._kernel_width = 0.5 if kernel_width is None else kernel_width
        self._prop_k = 4 if prop_k is None else prop_k
        self._dyn_corr_excl = dyn_corr_excl
        self._entropy_calc = self._getcalc("entropy")

        self.identifier = self.identifier + "_" + estimator
        if estimator == "kraskov":
            # Only the measures with a genuine KSG implementation may accept it.
            # The composed measures (joint/conditional/crossmap/causal entropy,
            # directed info, stochastic interaction) are built from marginal
            # entropies, and _getcalc hands them GaussianEntropyCalculator for
            # "kraskov" -- so they returned exactly the Gaussian result while
            # advertising kraskov_NN-<k> in the identifier. Reporting a k-NN
            # estimate that was never computed is worse than refusing.
            if not isinstance(self, (MutualInfo, TimeLaggedMutualInfo, TransferEntropy)):
                raise NotImplementedError(
                    f"The kraskov estimator is not implemented for "
                    f"{type(self).__name__}: it is composed from marginal "
                    f"entropies, and no KSG estimator exists for that "
                    f"composition. Use estimator='kozachenko' for a "
                    f"k-nearest-neighbour entropy, or 'gaussian'."
                )
            self.identifier = self.identifier + "_NN-{}".format(self._prop_k)
            self.labels = self.labels + ["nonlinear"]
        elif estimator == "kernel":
            self.identifier = self.identifier + "_W-{}".format(self._kernel_width)
            self.labels = self.labels + ["nonlinear"]
        elif estimator == "symbolic":
            if not isinstance(self, TransferEntropy):
                raise NotImplementedError(
                    "Symbolic estimator is only available for transfer entropy."
                )
            self.labels = self.labels + ["symbolic"]
            self._dyn_corr_excl = None
            return
        elif estimator == "kozachenko":
            # Kozachenko-Leonenko estimates *entropy* from k-NN distances. The
            # measures below are computed directly rather than as a sum of
            # marginal entropies, and there is no KL path for them: composing
            # them from separate KL entropies is biased, since the per-space
            # biases do not cancel (avoiding exactly that is why KSG couples
            # its radii across spaces -- use estimator="kraskov" instead).
            # Fail here rather than returning NaN at compute time.
            if isinstance(self, (MutualInfo, TimeLaggedMutualInfo, TransferEntropy)):
                raise NotImplementedError(
                    f"The kozachenko estimator is not available for "
                    f"{type(self).__name__}; use estimator='kraskov' for a "
                    f"k-nearest-neighbour estimate of this measure."
                )
            # k-NN based, so nonlinear -- not "linear" as gaussian is.
            self.labels = self.labels + ["nonlinear"]
            self._dyn_corr_excl = None
        else:
            self.labels = self.labels + ["linear"]
            self._dyn_corr_excl = None

        if self._dyn_corr_excl:
            self.identifier = self.identifier + "_DCE"

    def __getstate__(self):
        state = dict(self.__dict__)
        unserializable_objects = ["_entropy_calc", "_calc"]
        for k in unserializable_objects:
            if k in state.keys():
                del state[k]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._entropy_calc = self._getcalc("entropy")

    def __deepcopy__(self, memo):
        newone = type(self)()
        newone.__dict__.update(self.__dict__)
        for attr in newone.__dict__:
            setattr(newone, attr, copy.deepcopy(getattr(self, attr), memo))
        return newone

    def _getkey(self):
        if self._estimator == "kernel":
            return (self._estimator, self._kernel_width)
        elif self._estimator == "kraskov":
            return (self._estimator, self._prop_k)
        else:
            return (self._estimator,)

    def _getcalc(self, measure):
        est = self._estimator

        # --- Pure-numpy calculators (no JIDT/JVM needed) ---

        if measure == "entropy":
            if est == 'kozachenko':
                return KLEntropyCalculator()
            if est in ('gaussian', 'kraskov'):
                return GaussianEntropyCalculator()
            if est == 'kernel':
                calc = KernelEntropyCalculator()
                calc.setProperty("KERNEL_WIDTH", str(self._kernel_width))
                return calc
            if est == 'symbolic':
                return None  # symbolic TE doesn't use entropy calculator

        if measure == "MutualInfo":
            if est in ('gaussian', 'kraskov', 'kozachenko'):
                return GaussianEntropyCalculator()  # dummy; multivariate bypasses
            if est == 'kernel':
                calc = KernelMICalculator()
                calc.setProperty("KERNEL_WIDTH", str(self._kernel_width))
                return calc

        if measure == "TransferEntropy":
            if est in ('gaussian', 'kraskov', 'kozachenko'):
                return _DummyTECalculator()
            if est == 'kernel':
                calc = KernelTECalculator()
                calc.setProperty("KERNEL_WIDTH", str(self._kernel_width))
                return calc
            if est == 'symbolic':
                return SymbolicTECalculator()

        raise TypeError(f"Unknown measure/estimator: {measure}/{est}")

    @parse_univariate
    def _compute_entropy(self, data, i=None):
        if not hasattr(data, "entropy"):
            data.entropy = {}

        key = self._getkey()
        if key not in data.entropy:
            data.entropy[key] = np.full((data.n_processes,), -np.inf)

        if data.entropy[key][i] == -np.inf:
            x = np.squeeze(data.to_numpy()[i])
            est = self._estimator

            if est in ('gaussian', 'kraskov'):
                data.entropy[key][i] = _gaussian_entropy_from_data(x.reshape(-1, 1))
            else:
                # kozachenko, kernel — all have numpy calculators
                self._entropy_calc.initialise(1)
                self._entropy_calc.setObservations(x)
                data.entropy[key][i] = self._entropy_calc.computeAverageLocalOfObservations()

        return data.entropy[key][i]

    @parse_bivariate
    def _compute_joint_entropy(self, data, i, j):
        if not hasattr(data, "joint_entropy"):
            data.joint_entropy = {}

        key = self._getkey()
        if key not in data.joint_entropy:
            data.joint_entropy[key] = np.full((data.n_processes, data.n_processes), -np.inf)

        if data.joint_entropy[key][i, j] == -np.inf:
            x, y = data.to_numpy()[[i, j]]
            joint = np.concatenate([x, y], axis=1)
            est = self._estimator

            if est in ('gaussian', 'kraskov'):
                val = _gaussian_entropy_from_data(joint)
            else:
                # kozachenko, kernel — all have numpy calculators
                self._entropy_calc.initialise(2)
                self._entropy_calc.setObservations(joint)
                val = self._entropy_calc.computeAverageLocalOfObservations()

            data.joint_entropy[key][i, j] = val
            data.joint_entropy[key][j, i] = val

        return data.joint_entropy[key][i, j]

    def _compute_conditional_entropy(self, X, Y):
        est = self._estimator
        if est in ('gaussian', 'kraskov'):
            XY = np.concatenate([X, Y], axis=1)
            return _gaussian_entropy_from_data(XY) - _gaussian_entropy_from_data(Y)
        else:
            # kozachenko, kernel — all have numpy calculators
            XY = np.concatenate([X, Y], axis=1)
            self._entropy_calc.initialise(XY.shape[1])
            self._entropy_calc.setObservations(XY)
            H_XY = self._entropy_calc.computeAverageLocalOfObservations()
            self._entropy_calc.initialise(Y.shape[1])
            self._entropy_calc.setObservations(Y)
            H_Y = self._entropy_calc.computeAverageLocalOfObservations()
            return H_XY - H_Y

    def _resolve_theiler(self, data, i, j):
        """Theiler/dynamic-correlation-exclusion window for pair (i, j).

        None -> 0 (no window); an integer -> that window; "AUTO" -> the
        autocorrelation time 2*<acf(x_i), acf(x_j)>, cached per dataset.
        Shared by MI/TLMI/TE; only the kNN (kraskov) paths consume it.
        """
        raw_w = getattr(self, '_dyn_corr_excl', None)
        if raw_w is None:
            return 0
        if raw_w == "AUTO":
            if not hasattr(data, 'theiler'):
                z = data.to_numpy()
                M = data.n_processes
                theiler = -np.ones((M, M))
                for _i in range(M):
                    for _j in range(_i + 1, M):
                        theiler[_i, _j] = 2 * np.dot(
                            utils.acf(z[_i]), utils.acf(z[_j])
                        )
                        theiler[_j, _i] = theiler[_i, _j]
                data.theiler = theiler
            return int(data.theiler[i, j])
        return int(raw_w)


class JointEntropy(InfoTheoryBase, Undirected):

    name = "Joint entropy"
    identifier = "je"
    labels = ["unsigned", "infotheory", "unordered", "undirected"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        return self._compute_joint_entropy(data, i=i, j=j)

    @parse_multivariate
    def multivariate(self, data):
        if self._estimator == 'gaussian':
            Z = data.to_numpy(squeeze=True)
            M = Z.shape[0]
            R = np.corrcoef(Z)
            variances = np.var(Z, axis=1, ddof=1)
            log_var = np.log(np.maximum(variances, 1e-300))
            r2 = np.clip(R ** 2, 0, 1 - 1e-15)
            JE = (np.log(2 * np.pi * np.e)
                  + 0.5 * log_var[:, None]
                  + 0.5 * log_var[None, :]
                  + 0.5 * np.log(1 - r2))
            np.fill_diagonal(JE, np.nan)
            return JE
        return super().multivariate(data)


class ConditionalEntropy(InfoTheoryBase, Directed):

    name = "Conditional entropy"
    identifier = "ce"
    labels = ["unsigned", "infotheory", "unordered", "directed"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        return self._compute_joint_entropy(data, i=i, j=j) - self._compute_entropy(
            data, i=i
        )

    @parse_multivariate
    def multivariate(self, data):
        if self._estimator == 'gaussian':
            Z = data.to_numpy(squeeze=True)
            M = Z.shape[0]
            R = np.corrcoef(Z)
            variances = np.var(Z, axis=1, ddof=1)
            log_var = np.log(np.maximum(variances, 1e-300))
            r2 = np.clip(R ** 2, 0, 1 - 1e-15)
            H_marginal = 0.5 * np.log(2 * np.pi * np.e * np.maximum(variances, 1e-300))
            JE = (np.log(2 * np.pi * np.e)
                  + 0.5 * log_var[:, None]
                  + 0.5 * log_var[None, :]
                  + 0.5 * np.log(1 - r2))
            CE = JE - H_marginal[:, None]
            np.fill_diagonal(CE, np.nan)
            return CE
        return super().multivariate(data)


class MutualInfo(InfoTheoryBase, Undirected):
    name = "Mutual information"
    identifier = "mi"
    labels = ["unsigned", "infotheory", "unordered", "undirected"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._calc = self._getcalc("MutualInfo")

    def __setstate__(self, state):
        super().__setstate__(state)
        self._calc = self._getcalc("MutualInfo")

    @parse_bivariate
    def bivariate(self, data, i=None, j=None, verbose=False):
        """Compute mutual information between Y and X"""
        if self._estimator in ('gaussian', 'kraskov'):
            # Handled by multivariate; bivariate fallback
            z = data.to_numpy(squeeze=True)
            if self._estimator == 'gaussian':
                r = np.corrcoef(z[i], z[j])[0, 1]
                r2 = np.clip(r ** 2, 0, 1 - 1e-15)
                return -0.5 * np.log(1 - r2)
            else:
                # kraskov bivariate
                tree_x = cKDTree(z[i].reshape(-1, 1))
                tree_y = cKDTree(z[j].reshape(-1, 1))
                k = int(self._prop_k)
                w = self._resolve_theiler(data, i, j)
                return _ksg_mi_pair(z[i], z[j], k, w, tree_x, tree_y)

        # kernel estimator: use numpy KernelMICalculator
        if self._estimator == 'kernel':
            src, targ = data.to_numpy(squeeze=True)[[i, j]]
            self._calc.initialise(1, 1)
            self._calc.setObservations(src, targ)
            return self._calc.computeAverageLocalOfObservations()

        # Fallback should not be reached (all estimators handled above)
        logging.warning(f"MI bivariate: unhandled estimator '{self._estimator}'")
        return np.nan

    @parse_multivariate
    def multivariate(self, data):
        if self._estimator == 'gaussian':
            Z = data.to_numpy(squeeze=True)
            R = np.corrcoef(Z)
            r2 = np.clip(R ** 2, 0, 1 - 1e-15)
            MI = -0.5 * np.log(1 - r2)
            np.fill_diagonal(MI, np.nan)
            return MI
        elif self._estimator == 'kraskov':
            Z = data.to_numpy(squeeze=True)
            M, N = Z.shape
            k = int(self._prop_k)
            marginal_trees = [cKDTree(Z[i].reshape(-1, 1)) for i in range(M)]
            result = np.full((M, M), np.nan)
            for i in range(M):
                for j in range(i + 1, M):
                    w = self._resolve_theiler(data, i, j)
                    mi = _ksg_mi_pair(Z[i], Z[j], k, w,
                                      marginal_trees[i], marginal_trees[j])
                    result[i, j] = result[j, i] = mi
            return result
        return super().multivariate(data)


class TimeLaggedMutualInfo(InfoTheoryBase, Directed):
    name = "Time-lagged mutual information"
    identifier = "tlmi"
    labels = ["unsigned", "infotheory", "temporal", "directed"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._calc = self._getcalc("MutualInfo")

    def __setstate__(self, state):
        super().__setstate__(state)
        self._calc = self._getcalc("MutualInfo")

    @parse_bivariate
    def bivariate(self, data, i=None, j=None, verbose=False):
        if self._estimator in ('gaussian', 'kraskov'):
            z = data.to_numpy(squeeze=True)
            src = z[i][:-1]
            tgt = z[j][1:]
            if self._estimator == 'gaussian':
                r = np.corrcoef(src, tgt)[0, 1]
                r2 = np.clip(r ** 2, 0, 1 - 1e-15)
                return -0.5 * np.log(1 - r2)
            else:
                tree_x = cKDTree(src.reshape(-1, 1))
                tree_y = cKDTree(tgt.reshape(-1, 1))
                k = int(self._prop_k)
                w = self._resolve_theiler(data, i, j)
                return _ksg_mi_pair(src, tgt, k, w, tree_x, tree_y)

        # kernel estimator: use numpy KernelMICalculator
        if self._estimator == 'kernel':
            src, targ = data.to_numpy(squeeze=True)[[i, j]]
            src = src[:-1]
            targ = targ[1:]
            self._calc.initialise(1, 1)
            self._calc.setObservations(src, targ)
            return self._calc.computeAverageLocalOfObservations()

        logging.warning(f"TLMI bivariate: unhandled estimator '{self._estimator}'")
        return np.nan

    @parse_multivariate
    def multivariate(self, data):
        if self._estimator == 'gaussian':
            Z = data.to_numpy(squeeze=True)
            M, T = Z.shape
            Z_src = Z[:, :-1]
            Z_tgt = Z[:, 1:]
            stacked = np.vstack([Z_src, Z_tgt])
            R = np.corrcoef(stacked)
            r_cross = R[:M, M:]
            r2 = np.clip(r_cross ** 2, 0, 1 - 1e-15)
            TLMI = -0.5 * np.log(1 - r2)
            np.fill_diagonal(TLMI, np.nan)
            return TLMI
        elif self._estimator == 'kraskov':
            Z = data.to_numpy(squeeze=True)
            M, T = Z.shape
            k = int(self._prop_k)
            Z_src = Z[:, :-1]
            Z_tgt = Z[:, 1:]
            src_trees = [cKDTree(Z_src[i].reshape(-1, 1)) for i in range(M)]
            tgt_trees = [cKDTree(Z_tgt[j].reshape(-1, 1)) for j in range(M)]
            result = np.full((M, M), np.nan)
            for i in range(M):
                for j in range(M):
                    if i == j:
                        continue
                    w = self._resolve_theiler(data, i, j)
                    mi = _ksg_mi_pair(Z_src[i], Z_tgt[j], k, w,
                                      src_trees[i], tgt_trees[j])
                    result[i, j] = mi
            return result
        return super().multivariate(data)


class TransferEntropy(InfoTheoryBase, Directed):

    name = "Transfer entropy"
    identifier = "te"
    labels = ["unsigned", "embedding", "infotheory", "temporal", "directed"]

    def __init__(
        self,
        auto_embed_method=None,
        k_search_max=None,
        tau_search_max=None,
        k_history=1,
        k_tau=1,
        l_history=1,
        l_tau=1,
        **kwargs,
    ):
        if "estimator" not in kwargs.keys() or kwargs["estimator"] == "gaussian":
            self.identifier = "gc"
        super().__init__(**kwargs)

        if auto_embed_method is not None and auto_embed_method not in _AUTO_EMBED_METHODS:
            # The value was previously never inspected: any non-None string
            # took the auto-embed branch, so a typo silently ran MAX_CORR_AIS.
            raise ValueError(
                f"Unknown auto_embed_method {auto_embed_method!r}; implemented: "
                f"{sorted(_AUTO_EMBED_METHODS)}. (Ragwitz-style local-prediction "
                f"selection is not implemented; the search here maximises AIS.)"
            )

        if self._estimator == "symbolic" and int(k_history) < 2:
            # An ordinal pattern of length 1 has exactly one possible symbol, so
            # every entropy term is zero and TE is identically zero. It is not a
            # degenerate edge case, it is a guaranteed-null statistic.
            raise ValueError(
                "estimator='symbolic' requires k_history >= 2: a length-1 "
                "ordinal pattern has a single symbol, so the transfer entropy "
                "is identically zero."
            )

        self._calc = self._getcalc("TransferEntropy")

        # Store embedding params for numpy path
        self._auto_embed_method = auto_embed_method
        self._k_search_max = k_search_max if k_search_max is not None else 10
        self._tau_search_max = tau_search_max if tau_search_max is not None else 4
        self._k_history = k_history
        self._k_tau = k_tau
        self._l_history = l_history
        self._l_tau = l_tau

        # Auto-embedding
        if auto_embed_method is not None:
            self._calc.setProperty(self._AUTO_EMBED_METHOD_PROP_NAME, auto_embed_method)
            self._calc.setProperty(self._K_SEARCH_MAX_PROP_NAME, str(k_search_max))
            if self._estimator != "kernel":
                self.identifier = self.identifier + "_k-max-{}_tau-max-{}".format(
                    k_search_max, tau_search_max
                )
                self._calc.setProperty(
                    self._TAU_SEARCH_MAX_PROP_NAME, str(tau_search_max)
                )
            else:
                self.identifier = self.identifier + "_k-max-{}".format(k_search_max)
        else:
            self._calc.setProperty(self._K_HISTORY_PROP_NAME, str(k_history))
            if self._estimator != "kernel":
                self._calc.setProperty(self._K_TAU_PROP_NAME, str(k_tau))
                self._calc.setProperty(self._L_HISTORY_PROP_NAME, str(l_history))
                self._calc.setProperty(self._L_TAU_PROP_NAME, str(l_tau))
                self.identifier = self.identifier + "_k-{}_kt-{}_l-{}_lt-{}".format(
                    k_history, k_tau, l_history, l_tau
                )
            else:
                self.identifier = self.identifier + "_k-{}".format(k_history)

    def __setstate__(self, state):
        super().__setstate__(state)
        self._calc = self._getcalc("TransferEntropy")

    @parse_bivariate
    def bivariate(self, data, i=None, j=None, verbose=False):
        est = self._estimator
        auto = self._auto_embed_method

        src, targ = data.to_numpy(squeeze=True)[[i, j]]

        # Pure-numpy path for gaussian/kraskov with fixed embedding
        if est in ('gaussian', 'kraskov') and auto is None:
            if est == 'gaussian':
                return _gaussian_te_bivariate(
                    src, targ, self._k_history, self._k_tau,
                    self._l_history, self._l_tau
                )
            else:
                k_nn = int(self._prop_k)
                w = self._resolve_theiler(data, i, j)
                return _kraskov_te_bivariate(
                    src, targ, self._k_history, self._k_tau,
                    self._l_history, self._l_tau, k_nn, w
                )

        # Auto-embedding via AIS criterion (Ragwitz-style)
        if est in ('gaussian', 'kraskov') and auto is not None:
            k_max = self._k_search_max
            tau_max = self._tau_search_max
            if est == 'gaussian':
                return _auto_embed_gaussian_te(src, targ, k_max, tau_max)
            else:
                # Estimator-consistent: select the embedding by maximising the
                # KSG AIS (same estimator as the final TE), then run kraskov TE.
                # Using Gaussian AIS here would pick the embedding by *linear*
                # predictability for a nonlinear estimator (Wibral et al. 2014;
                # JIDT MAX_CORR_AIS uses the destination's own estimator).
                k_nn = int(self._prop_k)
                w = self._resolve_theiler(data, i, j)
                best_k, best_tau, best_ais = 1, 1, -np.inf
                for k in range(1, k_max + 1):
                    for tau in range(1, tau_max + 1):
                        # Skip candidates the estimator cannot support, instead
                        # of ranking them and rejecting the winner afterwards.
                        # On kuramoto M7/T100 an invalid (k=10, tau=4, w=33)
                        # candidate won and the whole SPI then failed, even
                        # though valid smaller embeddings existed.
                        n_eff = targ.size - (k - 1) * tau
                        try:
                            _validate_ksg_sample(n_eff, k_nn, w)
                        except ValueError:
                            continue
                        ais = _ksg_ais(targ, k, tau, k_nn, w)
                        if ais > best_ais:
                            best_ais, best_k, best_tau = ais, k, tau
                return _kraskov_te_bivariate(src, targ, best_k, best_tau, 1, 1, k_nn, w)

        # kernel/symbolic path: numpy calculators
        if est in ('kernel', 'symbolic'):
            self._calc.initialise()
            self._calc.setObservations(src, targ)
            return self._calc.computeAverageLocalOfObservations()

        logging.warning(f"TE bivariate: unhandled estimator '{est}'")
        return np.nan


class CrossmapEntropy(InfoTheoryBase, Directed):

    name = "Cross-map entropy"
    identifier = "xme"
    labels = ["unsigned", "infotheory", "temporal", "directed"]

    def __init__(self, history_length=10, **kwargs):
        super().__init__(**kwargs)
        self.identifier += f"_k{history_length}"
        self._history_length = history_length

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        src, targ = data.to_numpy(squeeze=True)[[i, j]]
        k = self._history_length
        targ_future = targ[k:]
        src_past = np.expand_dims(src[k - 1 : -1], axis=1)
        for idx in range(2, k):
            src_past = np.append(
                src_past, np.expand_dims(src[k - idx : -idx], axis=1), axis=1
            )

        joint = np.concatenate([src_past, np.expand_dims(targ_future, axis=1)], axis=1)

        # All estimators now have numpy entropy calculators
        self._entropy_calc.initialise(joint.shape[1])
        self._entropy_calc.setObservations(joint)
        H_xy = self._entropy_calc.computeAverageLocalOfObservations()

        self._entropy_calc.initialise(src_past.shape[1])
        self._entropy_calc.setObservations(src_past)
        H_y = self._entropy_calc.computeAverageLocalOfObservations()

        return H_xy - H_y


class CausalEntropy(InfoTheoryBase, Directed):

    name = "Causally conditioned entropy"
    identifier = "cce"
    labels = ["unsigned", "infotheory", "temporal", "directed"]

    def __init__(self, n=5, **kwargs):
        super().__init__(**kwargs)
        if int(n) < 1:
            raise ValueError(f"Horizon n must be >= 1, got {n}.")
        self._n = int(n)
        # n changes the measure, so it must reach the identifier.
        self.identifier += f"_n-{self._n}"

    def _compute_causal_entropy(self, src, targ):
        src = np.squeeze(src)
        targ = np.squeeze(targ)

        causal_entropy = 0
        for i in range(1, self._n + 1):
            Yp = _numpy_delay_embedding(targ, i - 1)[:-1]
            Xp = _numpy_delay_embedding(src, i)
            XYp = np.concatenate([Yp, Xp], axis=1)
            Yf = np.expand_dims(targ[i - 1:], 1)
            causal_entropy += self._compute_conditional_entropy(Yf, XYp)
        return causal_entropy

    def _getkey(self):
        return super(CausalEntropy, self)._getkey() + (self._n,)

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        if not hasattr(data, "causal_entropy"):
            data.causal_entropy = {}

        key = self._getkey()
        if key not in data.causal_entropy:
            data.causal_entropy[key] = np.full(
                (data.n_processes, data.n_processes), -np.inf
            )

        if data.causal_entropy[key][i, j] == -np.inf:
            z = data.to_numpy(squeeze=True)
            data.causal_entropy[key][i, j] = self._compute_causal_entropy(z[i], z[j])

        return data.causal_entropy[key][i, j]


class DirectedInfo(CausalEntropy, Directed):

    name = "Directed information"
    identifier = "di"
    labels = ["unsigned", "infotheory", "temporal", "directed"]

    def __init__(self, n=5, **kwargs):
        # n is handled by CausalEntropy; re-appending here doubled the suffix.
        super().__init__(n=n, **kwargs)

    def _entropy_of(self, M):
        """Joint entropy of the columns of ``M``; 0 for a zero-column matrix."""
        if M.shape[1] == 0:
            return 0.0
        if self._estimator == "gaussian":
            return _gaussian_entropy_from_data(M)
        self._entropy_calc.initialise(M.shape[1])
        self._entropy_calc.setObservations(M)
        return self._entropy_calc.computeAverageLocalOfObservations()

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        r"""Directed information from process ``i`` to process ``j``.

        Massey's finite-horizon definition:

        .. math::
            I(X^n \to Y^n) = \sum_{i=1}^{n} I(X^i; Y_i \mid Y^{i-1})
                           = \sum_{i=1}^{n} \left[ H(Y_i \mid Y^{i-1})
                             - H(Y_i \mid Y^{i-1}, X^i) \right]

        The previous implementation summed :math:`H(Y^i)/i` and subtracted the
        causal entropy. That is not the above and is not a dependence measure:
        with a source statistically independent of the target it returned 0.007
        at target autocorrelation 0, rising to 1.53 at 0.95 -- it grew with how
        predictable the *target* was from its own past, with no source coupling
        present at all.

        Each conditional entropy is expanded as a difference of joint entropies
        over one common row window, so the terms telescope correctly and every
        entropy is estimated on identically aligned samples.
        """
        z = data.to_numpy(squeeze=True)
        src, targ = np.asarray(z[i], float), np.asarray(z[j], float)
        n = self._n
        T = targ.size
        if T <= n + 1:
            return np.nan

        # One aligned window for every term: rows are t = n .. T-1.
        y_now = targ[n:].reshape(-1, 1)
        y_lags = [targ[n - k: T - k].reshape(-1, 1) for k in range(1, n + 1)]
        # X^i includes the current source sample (same time index as y_i).
        x_lags = [src[n - k: T - k].reshape(-1, 1) for k in range(0, n)]

        total = 0.0
        for order in range(1, n + 1):
            Ypast = np.hstack(y_lags[: order - 1]) if order > 1 else y_now[:, :0]
            Xpast = np.hstack(x_lags[:order])

            # H(Y_i | Y^{i-1}) - H(Y_i | Y^{i-1}, X^i)
            h_y_given_ypast = (
                self._entropy_of(np.hstack([y_now, Ypast])) - self._entropy_of(Ypast)
            )
            h_y_given_ypast_x = (
                self._entropy_of(np.hstack([y_now, Ypast, Xpast]))
                - self._entropy_of(np.hstack([Ypast, Xpast]))
            )
            total += h_y_given_ypast - h_y_given_ypast_x

        return total


class StochasticInteraction(InfoTheoryBase, Undirected):

    name = "Stochastic interaction"
    identifier = "si"
    labels = ["unsigned", "infotheory", "temporal", "undirected"]

    def __init__(self, delay=1, **kwargs):
        super().__init__(**kwargs)
        self._delay = delay
        self.identifier += f"_k-{delay}"

    @parse_bivariate
    def bivariate(self, data, i=None, j=None, verbose=False):
        x, y = data.to_numpy()[[i, j]]
        xy = np.concatenate([x, y], axis=1)
        tau = self._delay

        H_joint = self._compute_conditional_entropy(xy[tau:], xy[:-tau])
        H_src = self._compute_conditional_entropy(x[tau:], x[:-tau])
        H_targ = self._compute_conditional_entropy(y[tau:], y[:-tau])

        return H_src + H_targ - H_joint


class IntegratedInformation(Undirected, Unsigned):

    name = "Integrated information"
    identifier = "phi"
    labels = ["linear", "unsigned", "infotheory", "temporal", "undirected"]

    def __init__(self, phitype="star", delay=1, normalization=0):
        self._phitype = phitype
        self._delay = delay
        self._normalization = normalization
        self.identifier += f"_{phitype}_t-{delay}_norm-{normalization}"

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        try:
            from pyspi.lib.phi_native import phi_comp

            P = np.array([1, 2])
            X = data.to_numpy(squeeze=True)[[i, j]]

            params = {"tau": self._delay}
            options = {
                "type_of_phi": self._phitype,
                "type_of_dist": "Gauss",
                "normalization": self._normalization
            }

            return phi_comp(X, P, params, options)

        except Exception as e:
            logging.error(f"Integrated information computation failed: {e}")
            return np.nan
