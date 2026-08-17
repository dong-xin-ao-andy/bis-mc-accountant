#!/usr/bin/env python3

import argparse
import math
import os
from dataclasses import dataclass
from multiprocessing import get_context
from numbers import Integral
from typing import Optional, Tuple, Union

import numpy as np

from delta_calculation import get_base_delta, delta_from_epsilon_and_samples
from symmetric_polynomial import (
    bis_log_likelihood_bounds,
    bis_log_likelihood_ratio_from_log_weights,
    compute_bis_log_weights,
)


def _require_finite(name: str, value: float) -> float:
    """Convert ``value`` to float and reject NaN or either infinity."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("{} must be finite.".format(name))
    return value


def _require_integer(name: str, value: int, *, minimum: int) -> int:
    """Reject booleans, floats, and integers below ``minimum``."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("{} must be an integer.".format(name))
    value = int(value)
    if value < minimum:
        raise ValueError("{} must be at least {}.".format(name, minimum))
    return value


def _validate_weighted_samples(samples, counts, label):
    """Validate a compressed empirical distribution and return its exact size.

    Counts arrive as float arrays for compatibility with NumPy's weighted
    average, but they represent multiplicities. Summing converted Python
    integers avoids rounding a floating-point total into a different draw
    count.
    """
    samples = np.asarray(samples, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    if samples.ndim != 1 or counts.ndim != 1:
        raise ValueError("{} samples and counts must be 1D.".format(label))
    if samples.size != counts.size:
        raise ValueError(
            "{} samples and counts must have the same length.".format(label)
        )
    if not np.all(np.isfinite(samples)):
        raise ValueError("{} samples must all be finite.".format(label))
    if not np.all(np.isfinite(counts)):
        raise ValueError("{} counts must all be finite.".format(label))
    if np.any(counts < 0):
        raise ValueError("{} counts must be non-negative.".format(label))
    if not np.all(counts == np.floor(counts)):
        raise ValueError("{} counts must be integer-valued.".format(label))
    total = sum(int(count) for count in counts)
    if total <= 0:
        raise ValueError("{} total count must be positive.".format(label))
    return samples, counts, total


def _delta_estimate_or_infinity(epsilon, samples, counts) -> float:
    """Return a valid empirical divergence, failing closed on bad numerics."""
    estimate = float(
        delta_from_epsilon_and_samples(epsilon, samples, counts)
    )
    if not math.isfinite(estimate) or estimate < 0.0 or estimate > 1.0:
        return math.inf
    return estimate


def _validate_sampling_parameters(T, k, sigma, eps=None):
    T = _require_integer("T", T, minimum=1)
    k = _require_integer("k", k, minimum=0)
    if k > T:
        raise ValueError("Need 0 <= k <= T.")
    sigma = _require_finite("sigma", sigma)
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    if eps is not None:
        eps = _require_finite("epsilon", eps)
        if eps < 0:
            raise ValueError("epsilon must be non-negative.")
    return T, k, sigma, eps


# ----------------------------
# Sequential two-sided calibrator
# ----------------------------

@dataclass
class SequentialCalibrationResult:
    passed: bool
    candidate_index: int
    candidate_delta: float
    forward_delta: float
    reverse_delta: float
    base_delta: float
    done: bool
    best_index_so_far: Optional[int]


class SequentialCalibration:
    """
    Streaming version of ``perform_calibration_from_samples``.

    Forward samples contain ``log(P/Q)`` for draws from ``P``.  Optional reverse
    samples contain ``log(Q/P)`` for draws from ``Q``.  A candidate passes only
    when every requested direction is at most the same certified ``base_delta``,
    matching DeepMind's Monte Carlo calibration convention.

    This class requires the same number of draws for every requested direction
    and candidate.  Candidates must be ordered from most private to least
    private:
      - compute base_delta once from the sample count,
      - stop at the first failure,
      - return False, base_delta if the first candidate fails,
      - otherwise return True, previous_index.
    """

    def __init__(
        self,
        epsilon: float,
        target_delta: float,
    ):
        epsilon = _require_finite("epsilon", epsilon)
        target_delta = _require_finite("target_delta", target_delta)
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative.")
        if target_delta <= 0 or target_delta > 1:
            raise ValueError("target_delta must be in (0, 1].")

        self.epsilon = float(epsilon)
        self.target_delta = float(target_delta)
        self.base_delta = None  # type: Optional[float]
        self.reference_sample_count = None  # type: Optional[int]

        self.num_candidates_seen = 0
        self.best_index_so_far = None  # type: Optional[int]
        self.done = False

    def add_candidate(
        self,
        forward_samples: np.ndarray,
        forward_counts: np.ndarray,
        reverse_samples: Optional[np.ndarray] = None,
        reverse_counts: Optional[np.ndarray] = None,
    ) -> SequentialCalibrationResult:
        if self.done:
            raise RuntimeError("Calibration is already finished.")

        forward_samples, forward_counts, forward_total = (
            _validate_weighted_samples(
                forward_samples, forward_counts, "forward"
            )
        )
        if (reverse_samples is None) != (reverse_counts is None):
            raise ValueError(
                "reverse_samples and reverse_counts must either both be set or both be None."
            )
        if reverse_samples is None:
            reverse_total = forward_total
        else:
            reverse_samples, reverse_counts, reverse_total = (
                _validate_weighted_samples(
                    reverse_samples, reverse_counts, "reverse"
                )
            )
            if reverse_total != forward_total:
                raise ValueError(
                    "Forward and reverse sample counts must match; got {} and {}.".format(
                        forward_total, reverse_total
                    )
                )

        total_count = forward_total

        if self.base_delta is None:
            candidate_base_delta = float(
                get_base_delta(total_count, self.target_delta)
            )
            if (
                not math.isfinite(candidate_base_delta)
                or candidate_base_delta <= 0
                or candidate_base_delta > self.target_delta
            ):
                raise ArithmeticError(
                    "get_base_delta returned an invalid certification threshold."
                )
            self.reference_sample_count = total_count
            self.base_delta = candidate_base_delta
        elif total_count != self.reference_sample_count:
            raise ValueError(
                "Expected total sample count {}, got {}.".format(
                    self.reference_sample_count, total_count
                )
            )

        forward_delta = _delta_estimate_or_infinity(
            self.epsilon,
            forward_samples,
            forward_counts,
        )
        if reverse_samples is None:
            reverse_delta = 0.0
        else:
            reverse_delta = _delta_estimate_or_infinity(
                self.epsilon,
                reverse_samples,
                reverse_counts,
            )
        candidate_delta = max(forward_delta, reverse_delta)

        idx = self.num_candidates_seen
        self.num_candidates_seen += 1

        if candidate_delta > self.base_delta:
            self.done = True
            return SequentialCalibrationResult(
                passed=False,
                candidate_index=idx,
                candidate_delta=float(candidate_delta),
                forward_delta=float(forward_delta),
                reverse_delta=float(reverse_delta),
                base_delta=float(self.base_delta),
                done=True,
                best_index_so_far=self.best_index_so_far,
            )

        self.best_index_so_far = idx
        return SequentialCalibrationResult(
            passed=True,
            candidate_index=idx,
            candidate_delta=float(candidate_delta),
            forward_delta=float(forward_delta),
            reverse_delta=float(reverse_delta),
            base_delta=float(self.base_delta),
            done=False,
            best_index_so_far=self.best_index_so_far,
        )

    def final_outcome(self) -> Tuple[bool, Union[int, float]]:
        if self.base_delta is None:
            raise ValueError("No candidates processed.")
        if self.best_index_so_far is None:
            return False, float(self.base_delta)
        return True, int(self.best_index_so_far)


# Backward-compatible name for callers that used the former one-sided class.
SequentialPositiveCalibration = SequentialCalibration


# ----------------------------
# Compressed privacy-loss sampler
# ----------------------------

_VALID_SAMPLE_DIRECTIONS = ("forward", "reverse")


def _validate_sample_direction(direction: str) -> None:
    if direction not in _VALID_SAMPLE_DIRECTIONS:
        raise ValueError(
            "direction must be one of {}; got {!r}.".format(
                _VALID_SAMPLE_DIRECTIONS, direction
            )
        )


def generate_y_with_rng(
    T: int,
    k: int,
    sigma: float,
    rng: np.random.Generator,
    direction: str = "forward",
) -> np.ndarray:
    """Draw from the numerator distribution for one privacy direction.

    ``forward`` draws from the BIS mixture ``P``.  By permutation symmetry we
    may put the ``k`` ones in the first coordinates.  ``reverse`` draws from
    the zero-out distribution ``Q``.  The privacy-loss samples returned below
    are respectively ``log(P/Q)`` and ``log(Q/P)``.
    """
    _validate_sample_direction(direction)
    T, k, sigma, _ = _validate_sampling_parameters(T, k, sigma)
    y = rng.normal(loc=0.0, scale=sigma, size=T)
    if direction == "forward":
        y[:k] += 1.0
    return y


def generate_privacy_loss_sample_compressed(
    T: int,
    k: int,
    sigma: float,
    eps: float,
    rng: np.random.Generator,
    direction: str = "forward",
) -> float:
    """Return one compressed privacy-loss sample for a requested direction.

    A sample whose directional privacy loss is at most ``eps`` contributes
    exactly zero to the hockey-stick estimator, so it is represented by 0.0.
    Otherwise the exact directional privacy loss is returned.  Maclaurin's
    upper bound screens the forward tail and its geometric-mean lower bound
    screens the reverse tail before the ``O(T k)`` dynamic program is invoked.
    """
    _validate_sample_direction(direction)
    T, k, sigma, eps = _validate_sampling_parameters(T, k, sigma, eps)
    y = generate_y_with_rng(T, k, sigma, rng, direction=direction)
    log_w = compute_bis_log_weights(y, sigma)
    lower, upper = bis_log_likelihood_bounds(log_w, k)

    if direction == "forward" and upper <= eps:
        return 0.0
    if direction == "reverse" and lower >= -eps:
        return 0.0

    llr = bis_log_likelihood_ratio_from_log_weights(log_w, k)
    privacy_loss = llr if direction == "forward" else -llr
    if privacy_loss <= eps:
        return 0.0

    return float(privacy_loss)


def generate_llr_sample_compressed(
    T: int,
    k: int,
    sigma: float,
    eps: float,
    rng: np.random.Generator,
) -> float:
    """Backward-compatible alias for a compressed forward sample."""
    return generate_privacy_loss_sample_compressed(
        T, k, sigma, eps, rng, direction="forward"
    )


def _worker_generate_chunk(args):
    T, k, sigma, eps, num_samples, seed, direction = args
    _validate_sample_direction(direction)
    T, k, sigma, eps = _validate_sampling_parameters(T, k, sigma, eps)
    num_samples = _require_integer("num_samples", num_samples, minimum=1)
    seed = _require_integer("seed", seed, minimum=0)
    rng = np.random.default_rng(seed)

    sigma2 = sigma * sigma
    zero_count = 0
    exact_evaluation_count = 0
    positive_privacy_losses = []

    batch_size = 5000
    num_done = 0

    while num_done < num_samples:
        cur_batch = min(batch_size, num_samples - num_done)

        y_batch = rng.normal(loc=0.0, scale=sigma, size=(cur_batch, T))
        if direction == "forward":
            y_batch[:, :k] += 1.0

        log_w_batch = y_batch / sigma2 - 0.5 / sigma2
        if k == 0:
            active_mask = np.zeros(cur_batch, dtype=bool)
        elif direction == "forward":
            shifts = np.max(log_w_batch, axis=1)
            log_mean_w = shifts + np.log(
                np.mean(np.exp(log_w_batch - shifts[:, None]), axis=1)
            )
            upper_batch = k * log_mean_w
            active_mask = upper_batch > eps
        else:
            lower_batch = k * np.mean(log_w_batch, axis=1)
            active_mask = lower_batch < -eps

        zero_count += int(cur_batch - np.count_nonzero(active_mask))
        exact_evaluation_count += int(np.count_nonzero(active_mask))

        if np.any(active_mask):
            active_log_w = log_w_batch[active_mask]
            for log_w in active_log_w:
                llr = bis_log_likelihood_ratio_from_log_weights(log_w, k)
                privacy_loss = llr if direction == "forward" else -llr
                if privacy_loss <= eps:
                    zero_count += 1
                else:
                    positive_privacy_losses.append(privacy_loss)

        num_done += cur_batch

    if positive_privacy_losses:
        positive_privacy_losses = np.asarray(
            positive_privacy_losses, dtype=np.float64
        )
    else:
        positive_privacy_losses = np.empty(0, dtype=np.float64)

    return zero_count, positive_privacy_losses, exact_evaluation_count


def _split_work(total: int, num_workers: int):
    base = total // num_workers
    rem = total % num_workers
    return [base + (1 if i < rem else 0) for i in range(num_workers)]


def generate_compressed_samples_parallel(
    T: int,
    k: int,
    sigma: float,
    eps: float,
    total_samples: int,
    num_workers: int,
    base_seed: int,
    direction: str = "forward",
    return_stats: bool = False,
):
    """
    Generate compressed Monte Carlo samples using multiple processes.

    Output format is compatible with delta_from_epsilon_and_samples:
      - samples[0] = 0.0 with count = zero_count, if any
      - remaining entries are directional privacy losses > eps, each with count 1
    """
    _validate_sample_direction(direction)
    T, k, sigma, eps = _validate_sampling_parameters(T, k, sigma, eps)
    total_samples = _require_integer("total_samples", total_samples, minimum=1)
    num_workers = _require_integer("num_workers", num_workers, minimum=1)
    base_seed = _require_integer("base_seed", base_seed, minimum=0)

    worker_counts = _split_work(total_samples, num_workers)
    seeds = [base_seed + 1000003 * i for i in range(num_workers)]

    tasks = [
        (T, k, sigma, eps, worker_counts[i], seeds[i], direction)
        for i in range(num_workers)
        if worker_counts[i] > 0
    ]

    # "spawn" is safer than fork for numpy RNG independence and BLAS weirdness.
    ctx = get_context("spawn")
    with ctx.Pool(processes=len(tasks)) as pool:
        results = pool.map(_worker_generate_chunk, tasks)

    zero_count = 0
    exact_evaluation_count = 0
    positive_parts = []

    for zc, pos, exact_count in results:
        zero_count += int(zc)
        exact_evaluation_count += int(exact_count)
        if pos.size > 0:
            positive_parts.append(pos)

    if positive_parts:
        positive_privacy_losses = np.concatenate(positive_parts)
    else:
        positive_privacy_losses = np.empty(0, dtype=np.float64)

    if zero_count > 0:
        samples = np.concatenate(([0.0], positive_privacy_losses))
        counts = np.concatenate((
            np.array([float(zero_count)], dtype=np.float64),
            np.ones(positive_privacy_losses.shape[0], dtype=np.float64),
        ))
    else:
        samples = positive_privacy_losses
        counts = np.ones(positive_privacy_losses.shape[0], dtype=np.float64)

    # In the extremely unlikely event that every sample is positive and there are none,
    # fall back to one zero sample with count total_samples. This shouldn't happen here,
    # but it keeps the interface robust.
    if samples.size == 0:
        samples = np.array([0.0], dtype=np.float64)
        counts = np.array([float(total_samples)], dtype=np.float64)

    if return_stats:
        return samples, counts, {
            "draw_count": int(total_samples),
            "exact_evaluation_count": int(exact_evaluation_count),
            "exact_evaluation_fraction": (
                float(exact_evaluation_count) / float(total_samples)
            ),
        }
    return samples, counts


def _num_uncompressed_positive(samples: np.ndarray, counts: np.ndarray) -> int:
    """Count draws with nonzero hockey-stick contribution after compression."""
    if samples.size > 0 and samples[0] == 0.0:
        return sum(int(count) for count in counts[1:])
    return sum(int(count) for count in counts)


# ----------------------------
# Main calibration loop
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate BIS with certified Monte Carlo accounting. By default "
            "both H(P||Q) and H(Q||P) are verified."
        )
    )
    parser.add_argument("--eps", type=float, required=True)
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--initial-sigma", type=float, required=True)
    parser.add_argument("--samples-per-sigma", type=int, required=True)
    parser.add_argument("--sigma-step", type=float, default=0.1)
    parser.add_argument("--min-sigma", type=float, default=0.1)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional deterministic seed for reproducible research. If "
            "omitted, fresh OS entropy is used, as required for an "
            "operational randomized EVR calibration."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--directions",
        choices=("both", "forward"),
        default="both",
        help=(
            "Privacy directions to verify. 'both' (default) certifies the "
            "two-sided zero-out DP guarantee; 'forward' is a one-sided "
            "diagnostic/reproduction mode."
        ),
    )
    args = parser.parse_args()

    args.eps = _require_finite("eps", args.eps)
    args.delta = _require_finite("delta", args.delta)
    args.initial_sigma = _require_finite("initial_sigma", args.initial_sigma)
    args.sigma_step = _require_finite("sigma_step", args.sigma_step)
    args.min_sigma = _require_finite("min_sigma", args.min_sigma)

    if args.seed is None:
        calibration_seed = int.from_bytes(os.urandom(16), "big")
        seed_source = "fresh OS entropy"
        displayed_seed = "<fresh verifier seed not displayed>"
    else:
        calibration_seed = int(args.seed)
        if calibration_seed < 0:
            raise ValueError("seed must be non-negative.")
        seed_source = "user-supplied reproducibility seed"
        displayed_seed = str(calibration_seed)

    if args.k < 0 or args.k > args.T:
        raise ValueError("Need 0 <= k <= T.")
    if args.initial_sigma <= 0:
        raise ValueError("initial_sigma must be positive.")
    if args.sigma_step <= 0:
        raise ValueError("sigma_step must be positive.")
    if args.min_sigma <= 0:
        raise ValueError("min_sigma must be positive.")
    if args.samples_per_sigma <= 0:
        raise ValueError("samples_per_sigma must be positive.")
    if args.eps < 0:
        raise ValueError("eps must be non-negative.")
    if args.delta <= 0 or args.delta > 1:
        raise ValueError("delta must be in (0, 1].")
    if args.T <= 0:
        raise ValueError("T must be positive.")

    if args.num_workers is None:
        num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    else:
        num_workers = args.num_workers

    if num_workers <= 0:
        raise ValueError("num_workers must be positive.")

    print("=== Job configuration ===")
    print("eps                = {}".format(args.eps))
    print("target_delta       = {}".format(args.delta))
    print("T                  = {}".format(args.T))
    print("k                  = {}".format(args.k))
    print("initial_sigma      = {}".format(args.initial_sigma))
    print("samples_per_direction = {}".format(args.samples_per_sigma))
    print("directions         = {}".format(args.directions))
    print("sigma_step         = {}".format(args.sigma_step))
    print("min_sigma          = {}".format(args.min_sigma))
    print("num_workers        = {}".format(num_workers))
    print("seed               = {}".format(displayed_seed))
    print("seed_source        = {}".format(seed_source))
    print()

    base_delta = float(get_base_delta(args.samples_per_sigma, args.delta))
    if (
        not math.isfinite(base_delta)
        or base_delta <= 0
        or base_delta > args.delta
    ):
        raise ArithmeticError(
            "get_base_delta returned an invalid certification threshold."
        )
    print("certified base_delta = {:.8e}".format(base_delta))
    print()

    sigma = args.initial_sigma
    candidate_index = 0

    best_certified_sigma = None
    best_optimistic_sigma = None

    certified_failed = False

    while sigma >= args.min_sigma - 1e-12:
        sigma = round(sigma, 10)

        print(
            "--- Candidate {}: sigma={:.3f} ---".format(candidate_index, sigma),
            flush=True,
        )

        print(
            "Generating {} forward samples with {} workers...".format(
                args.samples_per_sigma, num_workers
            ),
            flush=True,
        )

        forward_samples, forward_counts, forward_stats = (
            generate_compressed_samples_parallel(
            T=args.T,
            k=args.k,
            sigma=sigma,
            eps=args.eps,
            total_samples=args.samples_per_sigma,
            num_workers=num_workers,
            base_seed=calibration_seed + 10000019 * candidate_index,
            direction="forward",
            return_stats=True,
        ))
        forward_positive = _num_uncompressed_positive(
            forward_samples, forward_counts
        )
        forward_delta = _delta_estimate_or_infinity(
            args.eps,
            forward_samples,
            forward_counts,
        )

        if args.directions == "both":
            print(
                "Generating {} reverse samples with {} workers...".format(
                    args.samples_per_sigma, num_workers
                ),
                flush=True,
            )
            reverse_samples, reverse_counts, reverse_stats = (
                generate_compressed_samples_parallel(
                T=args.T,
                k=args.k,
                sigma=sigma,
                eps=args.eps,
                total_samples=args.samples_per_sigma,
                num_workers=num_workers,
                base_seed=(
                    calibration_seed
                    + 10000019 * candidate_index
                    + 500000003
                ),
                direction="reverse",
                return_stats=True,
            ))
            reverse_positive = _num_uncompressed_positive(
                reverse_samples, reverse_counts
            )
            reverse_delta = _delta_estimate_or_infinity(
                args.eps,
                reverse_samples,
                reverse_counts,
            )
        else:
            reverse_positive = None
            reverse_delta = None
            reverse_stats = None

        directional_deltas = [forward_delta]
        if reverse_delta is not None:
            directional_deltas.append(reverse_delta)
        candidate_delta = max(directional_deltas)

        certified_pass = all(d <= base_delta for d in directional_deltas)
        optimistic_pass = all(d <= args.delta for d in directional_deltas)
        if reverse_delta is None:
            dominating_direction = "forward (reverse not evaluated)"
        elif forward_delta >= reverse_delta:
            dominating_direction = "forward"
        else:
            dominating_direction = "reverse"

        print("forward_positive_count  = {}".format(forward_positive))
        print(
            "forward_exact_count     = {} ({:.8e})".format(
                forward_stats["exact_evaluation_count"],
                forward_stats["exact_evaluation_fraction"],
            )
        )
        print("forward_delta_hat       = {:.8e}".format(forward_delta))
        if reverse_delta is not None:
            print("reverse_positive_count  = {}".format(reverse_positive))
            print(
                "reverse_exact_count     = {} ({:.8e})".format(
                    reverse_stats["exact_evaluation_count"],
                    reverse_stats["exact_evaluation_fraction"],
                )
            )
            print("reverse_delta_hat       = {:.8e}".format(reverse_delta))
        print("max_delta_hat           = {:.8e}".format(candidate_delta))
        print("dominating_direction    = {}".format(dominating_direction))
        print("base_delta              = {:.8e}".format(base_delta))
        print("target_delta            = {:.8e}".format(args.delta))
        print("certified_pass          = {}".format(certified_pass))
        print("optimistic_pass         = {}".format(optimistic_pass))
        print(flush=True)

        # Algorithm 5's ordered calibration freezes at the first certified
        # failure.  Later Monte Carlo fluctuations must never reopen the sweep.
        if not certified_failed:
            if certified_pass:
                best_certified_sigma = sigma
            else:
                certified_failed = True

        if optimistic_pass:
            best_optimistic_sigma = sigma
        else:
            # Since we sweep from more private to less private, once optimistic fails
            # there is no point continuing.
            break

        sigma -= args.sigma_step
        candidate_index += 1

    print("========== Final outcome ==========")
    print("eps   = {}".format(args.eps))
    print("delta = {}".format(args.delta))
    print("T     = {}".format(args.T))
    print("k     = {}".format(args.k))
    print("directions = {}".format(args.directions))

    if best_certified_sigma is None:
        print("Best certified sigma  = None (>{:.3f})".format(args.initial_sigma))
        print(
            "Certified fallback     = data-independent output / no training "
            "(0,0)-DP"
        )
        print(
            "Do not deploy an MC-tested candidate: the most-private candidate "
            "did not pass verification."
        )
    else:
        print("Best certified sigma  = {:.3f}".format(best_certified_sigma))

    if best_optimistic_sigma is None:
        print("Best optimistic sigma = None (>{:.3f})".format(args.initial_sigma))
    else:
        print("Best optimistic sigma = {:.3f}".format(best_optimistic_sigma))


if __name__ == "__main__":
    main()
