import math
import numpy as np


def log_comb(n, k):
    """Return log(binomial(n, k))."""
    if k < 0 or k > n:
        return -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def log_elementary_symmetric_dp(w, k, rescale_threshold=1e100):
    """Return log e_k(w) using exact DP with rescaling."""
    T = len(w)
    if k < 0 or k > T:
        return -math.inf
    if k == 0:
        return 0.0

    w = np.asarray(w, dtype=np.float64)
    E = np.zeros(k + 1, dtype=np.float64)
    E[0] = 1.0

    log_scale = 0.0

    for i in range(T):
        upper = min(i + 1, k)

        # Reverse update:
        # E[r] <- E[r] + w[i] * E[r-1]
        prev = E[:upper].copy()
        E[1:upper + 1] += w[i] * prev

        # Rescale only when numbers get too large.
        m = E[1:upper + 1].max()
        if m >= rescale_threshold:
            E[:upper + 1] /= m
            log_scale += math.log(m)

    return math.log(E[k]) + log_scale


def _log_elementary_symmetric_dp_from_log_weights(log_w, k):
    """Return ``log(e_k(exp(log_w)))`` using a log-space DP.

    The elementary-symmetric recurrence is

    ``E_r <- E_r + exp(log_w[i]) * E_{r-1}``.

    Representing every ``E_r`` by its logarithm turns that update into a
    ``logaddexp``.  In particular, no individual weight is exponentiated, so
    a finite log-weight cannot disappear merely because it is more than about
    745 below the largest one.
    """
    T = len(log_w)
    if k < 0 or k > T:
        return -math.inf
    if k == 0:
        return 0.0

    log_E = np.full(k + 1, -math.inf, dtype=np.float64)
    log_E[0] = 0.0

    for i, log_weight in enumerate(log_w):
        upper = min(i + 1, k)
        log_E[1:upper + 1] = np.logaddexp(
            log_E[1:upper + 1],
            log_weight + log_E[:upper],
        )

    return float(log_E[k])


def compute_bis_weights(y, sigma):
    """Compute the weights w used in the BIS log-likelihood ratio."""
    return np.exp(compute_bis_log_weights(y, sigma))


def compute_bis_log_weights(y, sigma):
    """Compute ``log(w_i)`` for the BIS likelihood ratio.

    Working in log space avoids overflow in the inexpensive screening step.
    The exact elementary-symmetric-polynomial calculation also consumes these
    values directly, without exponentiating individual weights; see
    :func:`bis_log_likelihood_ratio_from_log_weights`.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    sigma2 = sigma * sigma
    y_arr = np.asarray(y, dtype=np.float64)
    return y_arr / sigma2 - 0.5 / sigma2


def bis_log_likelihood_ratio(w, sigma, k):
    """Return log(P(y)/Q(y)) for BIS."""
    T = len(w)
    return log_elementary_symmetric_dp(w, k) - log_comb(T, k)


def bis_log_likelihood_ratio_from_log_weights(log_w, k):
    """Return ``log(P(y) / Q(y))`` from BIS log-weights.

    If ``m = max_i log(w_i)``, homogeneity gives
    ``log(e_k(w)) = k m + log(e_k(exp(log(w) - m)))``.  The dynamic
    program evaluates the second term directly from the shifted log-weights,
    avoiding both overflow and the loss of small but required weights to
    underflow.
    """
    log_w = np.asarray(log_w, dtype=np.float64)
    if log_w.ndim != 1 or log_w.size == 0:
        raise ValueError("log_w must be a non-empty 1D array.")
    T = log_w.size
    if k < 0 or k > T:
        raise ValueError("Need 0 <= k <= len(log_w).")
    if k == 0:
        return 0.0
    if k == T and np.all(np.isfinite(log_w)):
        # There is exactly one degree-T monomial.  Summing its logarithms
        # directly avoids needless shift/subtract cancellation at this
        # endpoint; fsum also gives a more accurate binary64 sum.
        return math.fsum(float(value) for value in log_w)
    shift = float(np.max(log_w))
    return (
        k * shift
        + _log_elementary_symmetric_dp_from_log_weights(log_w - shift, k)
        - log_comb(T, k)
    )


def bis_log_likelihood_bounds(log_w, k):
    """Return certified lower and upper bounds on the BIS log likelihood.

    For positive weights, Maclaurin's inequalities imply

    ``k * mean(log(w)) <= log(e_k(w) / C(T,k))``
    ``                         <= k * log(mean(w))``.

    The upper bound screens forward samples; the lower bound screens reverse
    samples.  Both bounds cost ``O(T)``, whereas the exact dynamic program
    costs ``O(T k)``.
    """
    log_w = np.asarray(log_w, dtype=np.float64)
    if log_w.ndim != 1 or log_w.size == 0:
        raise ValueError("log_w must be a non-empty 1D array.")
    T = log_w.size
    if k < 0 or k > T:
        raise ValueError("Need 0 <= k <= len(log_w).")
    if k == 0:
        return 0.0, 0.0
    shift = float(np.max(log_w))
    log_mean_w = shift + math.log(float(np.mean(np.exp(log_w - shift))))
    lower = k * float(np.mean(log_w))
    upper = k * log_mean_w
    return lower, upper


def generate_y(T, k, sigma):
    """Generate one BIS sample y of length T."""
    y = np.random.normal(loc=0.0, scale=sigma, size=T)
    y[:k] += 1.0
    return y


def generate_llr_sample(T, k, sigma, target_eps=None):
    """Generate one sample and its likelihood ratio; exact whenever it is above target_eps"""
    log_w = compute_bis_log_weights(generate_y(T, k, sigma), sigma)
    _, ub = bis_log_likelihood_bounds(log_w, k)
    if target_eps is not None and ub <= target_eps:
        return ub
    else:
        return bis_log_likelihood_ratio_from_log_weights(log_w, k)


# if __name__ == '__main__':
#     from tqdm import tqdm
#
#     T, k, sigma = 2000, 655, 10
#     num_samples = 100000000
#     target_delta = 1e-5
#     log_likelihood_ratios = []
#     for _ in tqdm(range(num_samples)):
#         log_likelihood_ratios.append(generate_llr_sample(T, k, sigma, 6))
#
#     print(delta_from_epsilon_and_samples(6, log_likelihood_ratios))
