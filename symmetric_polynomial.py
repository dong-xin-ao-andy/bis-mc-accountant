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


def compute_bis_weights(y, sigma):
    """Compute the weights w used in the BIS log-likelihood ratio."""
    sigma2 = sigma * sigma
    y_arr = np.asarray(y, dtype=np.float64)
    return np.exp(y_arr / sigma2 - 0.5 / sigma2)


def bis_log_likelihood_ratio(w, sigma, k):
    """Return log(P(y)/Q(y)) for BIS."""
    T = len(w)
    return log_elementary_symmetric_dp(w, k) - log_comb(T, k)


def generate_y(T, k, sigma):
    """Generate one BIS sample y of length T."""
    y = np.random.normal(loc=0.0, scale=sigma, size=T)
    y[:k] += 1.0
    return y


def generate_llr_sample(T, k, sigma, target_eps=None):
    """Generate one sample and its likelihood ratio; exact whenever it is above target_eps"""
    w = compute_bis_weights(generate_y(T, k, sigma), sigma)
    ub = k * np.log(np.sum(w) / T)
    if target_eps is not None and ub <= target_eps:
        return ub
    else:
        return bis_log_likelihood_ratio(w, sigma, k)