"""Wall-clock benchmark for the BIS forward Maclaurin screen.

The benchmark isolates likelihood evaluation: it pre-generates BIS log-weights,
then compares exact dynamic-program evaluation on every draw with the production
screen-then-exact path.  Random-number generation is deliberately excluded from
both timings.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import numpy as np

from symmetric_polynomial import bis_log_likelihood_ratio_from_log_weights


def _contribution(epsilon: float, privacy_loss: float) -> float:
    return float(-np.expm1(min(epsilon - privacy_loss, 0.0)))


def _exact_all(log_weights: np.ndarray, k: int, epsilon: float) -> float:
    total = 0.0
    for row in log_weights:
        loss = bis_log_likelihood_ratio_from_log_weights(row, k)
        total += _contribution(epsilon, loss)
    return total


def _screen_then_exact(
    log_weights: np.ndarray, k: int, epsilon: float
) -> tuple[float, int]:
    shifts = np.max(log_weights, axis=1)
    log_mean_weights = shifts + np.log(
        np.mean(np.exp(log_weights - shifts[:, None]), axis=1)
    )
    active = k * log_mean_weights > epsilon

    total = 0.0
    for row in log_weights[active]:
        loss = bis_log_likelihood_ratio_from_log_weights(row, k)
        total += _contribution(epsilon, loss)
    return total, int(np.count_nonzero(active))


def _audit_forward_screen(
    *,
    T: int,
    k: int,
    sigma: float,
    epsilon: float,
    samples: int,
    batch_size: int,
    seed: int,
) -> int:
    """Count exact-DP triggers without evaluating the likelihoods.

    This is kept separate from the timing benchmark so a large, statistically
    useful trigger audit does not force an exact likelihood evaluation for
    every draw.
    """
    rng = np.random.default_rng(seed)
    sigma2 = sigma * sigma
    active_count = 0
    for start in range(0, samples, batch_size):
        current = min(batch_size, samples - start)
        y = rng.normal(0.0, sigma, size=(current, T))
        y[:, :k] += 1.0
        log_weights = y / sigma2 - 0.5 / sigma2
        shifts = np.max(log_weights, axis=1)
        log_mean_weights = shifts + np.log(
            np.mean(np.exp(log_weights - shifts[:, None]), axis=1)
        )
        active_count += int(
            np.count_nonzero(k * log_mean_weights > epsilon)
        )
    return active_count


def _timed(callable_):
    start = time.perf_counter()
    value = callable_()
    return time.perf_counter() - start, value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=2000)
    parser.add_argument("--k", type=int, default=655)
    parser.add_argument("--sigma", type=float, default=20.5)
    parser.add_argument("--epsilon", type=float, default=3.0)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument(
        "--audit-samples",
        type=int,
        default=0,
        help=(
            "If positive, independently audit the forward screen on this "
            "many additional draws without running the exact DP."
        ),
    )
    parser.add_argument("--audit-batch-size", type=int, default=1000)
    args = parser.parse_args()

    if not 0 <= args.k <= args.T:
        raise ValueError("Need 0 <= k <= T.")
    if args.sigma <= 0 or args.samples <= 0 or args.repeats <= 0:
        raise ValueError("sigma, samples, and repeats must be positive.")
    if args.audit_samples < 0 or args.audit_batch_size <= 0:
        raise ValueError("audit-samples must be nonnegative and batch size positive.")

    rng = np.random.default_rng(args.seed)
    y = rng.normal(0.0, args.sigma, size=(args.samples, args.T))
    y[:, : args.k] += 1.0
    sigma2 = args.sigma * args.sigma
    log_weights = y / sigma2 - 0.5 / sigma2

    # Warm up NumPy dispatch and the dynamic-program function.
    bis_log_likelihood_ratio_from_log_weights(log_weights[0], args.k)
    _screen_then_exact(log_weights[:1], args.k, args.epsilon)

    exact_seconds = []
    screened_seconds = []
    exact_value = None
    screened_value = None
    active_count = None

    # Alternate order to reduce systematic effects from cache or CPU state.
    for repeat in range(args.repeats):
        if repeat % 2 == 0:
            elapsed, exact_value = _timed(
                lambda: _exact_all(log_weights, args.k, args.epsilon)
            )
            exact_seconds.append(elapsed)
            elapsed, screened = _timed(
                lambda: _screen_then_exact(
                    log_weights, args.k, args.epsilon
                )
            )
            screened_seconds.append(elapsed)
        else:
            elapsed, screened = _timed(
                lambda: _screen_then_exact(
                    log_weights, args.k, args.epsilon
                )
            )
            screened_seconds.append(elapsed)
            elapsed, exact_value = _timed(
                lambda: _exact_all(log_weights, args.k, args.epsilon)
            )
            exact_seconds.append(elapsed)
        screened_value, active_count = screened
        if not np.isclose(exact_value, screened_value, rtol=1e-12, atol=1e-14):
            raise AssertionError(
                "Screened and exact hockey-stick sums do not agree: "
                f"{screened_value} versus {exact_value}."
            )

    exact_median = statistics.median(exact_seconds)
    screened_median = statistics.median(screened_seconds)
    result = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "T": args.T,
        "k": args.k,
        "sigma": args.sigma,
        "epsilon": args.epsilon,
        "samples": args.samples,
        "repeats": args.repeats,
        "active_count": active_count,
        "active_fraction": active_count / args.samples,
        "exact_seconds_median": exact_median,
        "screened_seconds_median": screened_median,
        "likelihood_stage_speedup": exact_median / screened_median,
        "hockey_stick_sum": exact_value,
    }
    if args.audit_samples:
        audit_active = _audit_forward_screen(
            T=args.T,
            k=args.k,
            sigma=args.sigma,
            epsilon=args.epsilon,
            samples=args.audit_samples,
            batch_size=args.audit_batch_size,
            seed=args.seed + 1,
        )
        audit_fraction = audit_active / args.audit_samples
        result.update(
            {
                "audit_samples": args.audit_samples,
                "audit_active_count": audit_active,
                "audit_active_fraction": audit_fraction,
                "audit_operation_model_speedup": (
                    args.k / (1.0 + audit_fraction * args.k)
                ),
            }
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
