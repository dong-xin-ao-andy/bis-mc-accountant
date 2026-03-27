#!/usr/bin/env python3

import argparse
import os
from dataclasses import dataclass
from multiprocessing import get_context
from typing import Optional, Tuple, Union

import numpy as np

from delta_calculation import get_base_delta, delta_from_epsilon_and_samples
from symmetric_polynomial import (
    compute_bis_weights,
    bis_log_likelihood_ratio,
)


# ----------------------------
# New compressed LLR sampler
# ----------------------------

def generate_y_with_rng(
        T: int, k: int, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    y = rng.normal(loc=0.0, scale=sigma, size=T)
    y[:k] += 1.0
    return y


def generate_llr_sample_compressed(
        T: int,
        k: int,
        sigma: float,
        eps: float,
        rng: np.random.Generator,
) -> float:
    y = generate_y_with_rng(T, k, sigma, rng)
    w = compute_bis_weights(y, sigma)

    ub = k * np.log(np.sum(w) / T)
    if ub <= eps:
        return 0.0

    exact_llr = bis_log_likelihood_ratio(w, sigma, k)
    if exact_llr <= eps:
        return 0.0

    return float(exact_llr)


def _worker_generate_chunk(args):
    T, k, sigma, eps, num_samples, seed = args
    rng = np.random.default_rng(seed)

    sigma2 = sigma * sigma
    zero_count = 0
    positive_llrs = []

    batch_size = 5000
    num_done = 0

    while num_done < num_samples:
        cur_batch = min(batch_size, num_samples - num_done)

        y_batch = rng.normal(loc=0.0, scale=sigma, size=(cur_batch, T))
        y_batch[:, :k] += 1.0

        w_batch = np.exp(y_batch / sigma2 - 0.5 / sigma2)
        ub_batch = k * np.log(np.sum(w_batch, axis=1) / T)

        active_mask = ub_batch > eps
        zero_count += int(cur_batch - np.count_nonzero(active_mask))

        if np.any(active_mask):
            active_w = w_batch[active_mask]
            for w in active_w:
                exact_llr = bis_log_likelihood_ratio(w, sigma, k)
                if exact_llr <= eps:
                    zero_count += 1
                else:
                    positive_llrs.append(exact_llr)

        num_done += cur_batch

    if positive_llrs:
        positive_llrs = np.asarray(positive_llrs, dtype=np.float64)
    else:
        positive_llrs = np.empty(0, dtype=np.float64)

    return zero_count, positive_llrs


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
):
    if total_samples <= 0:
        raise ValueError("total_samples must be positive.")
    if num_workers <= 0:
        raise ValueError("num_workers must be positive.")

    worker_counts = _split_work(total_samples, num_workers)
    seeds = [base_seed + 1000003 * i for i in range(num_workers)]

    tasks = [
        (T, k, sigma, eps, worker_counts[i], seeds[i])
        for i in range(num_workers)
        if worker_counts[i] > 0
    ]

    ctx = get_context("spawn")
    with ctx.Pool(processes=len(tasks)) as pool:
        results = pool.map(_worker_generate_chunk, tasks)

    zero_count = 0
    positive_parts = []

    for zc, pos in results:
        zero_count += int(zc)
        if pos.size > 0:
            positive_parts.append(pos)

    if positive_parts:
        positive_llrs = np.concatenate(positive_parts)
    else:
        positive_llrs = np.empty(0, dtype=np.float64)

    if zero_count > 0:
        samples = np.concatenate(([0.0], positive_llrs))
        counts = np.concatenate((
            np.array([float(zero_count)], dtype=np.float64),
            np.ones(positive_llrs.shape[0], dtype=np.float64),
        ))
    else:
        samples = positive_llrs
        counts = np.ones(positive_llrs.shape[0], dtype=np.float64)

    if samples.size == 0:
        samples = np.array([0.0], dtype=np.float64)
        counts = np.array([float(total_samples)], dtype=np.float64)

    return samples, counts


# ----------------------------
# Main calibration loop
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps", type=float, required=True)
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--initial-sigma", type=float, required=True)
    parser.add_argument("--samples-per-sigma", type=int, required=True)
    parser.add_argument("--sigma-step", type=float, default=0.1)
    parser.add_argument("--min-sigma", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--run-optimistic", action="store_true",
                        help="Simultaneously compute and sweep based on optimistic Monte Carlo bounds.")
    args = parser.parse_args()

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
    if args.T <= 0:
        raise ValueError("T must be positive.")

    if args.num_workers is None:
        num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    else:
        num_workers = args.num_workers

    if num_workers <= 0:
        num_workers = 1

    print("=== Job configuration ===")
    print(f"eps                = {args.eps}")
    print(f"target_delta       = {args.delta}")
    print(f"T                  = {args.T}")
    print(f"k                  = {args.k}")
    print(f"initial_sigma      = {args.initial_sigma}")
    print(f"samples_per_sigma  = {args.samples_per_sigma}")
    print(f"sigma_step         = {args.sigma_step}")
    print(f"min_sigma          = {args.min_sigma}")
    print(f"num_workers        = {num_workers}")
    print(f"seed               = {args.seed}")
    print(f"run_optimistic     = {args.run_optimistic}\n")

    base_delta = get_base_delta(args.samples_per_sigma, args.delta)
    print(f"certified base_delta = {base_delta:.8e}\n")

    sigma = args.initial_sigma
    candidate_index = 0

    best_certified_sigma = None
    best_optimistic_sigma = None

    while sigma >= args.min_sigma - 1e-12:
        sigma = round(sigma, 10)

        print(f"--- Candidate {candidate_index}: sigma={sigma:.3f} ---", flush=True)
        print(f"Generating {args.samples_per_sigma} samples with {num_workers} workers...", flush=True)

        samples, counts = generate_compressed_samples_parallel(
            T=args.T,
            k=args.k,
            sigma=sigma,
            eps=args.eps,
            total_samples=args.samples_per_sigma,
            num_workers=num_workers,
            base_seed=args.seed + 10000019 * candidate_index,
        )

        zero_count = int(round(counts[0])) if samples.size > 0 and samples[0] == 0.0 else 0
        num_positive = samples.size - (1 if zero_count > 0 else 0)

        candidate_delta = delta_from_epsilon_and_samples(args.eps, samples, counts)

        certified_pass = candidate_delta <= base_delta

        print(f"positive_llr_count      = {num_positive}")
        print(f"delta_hat               = {candidate_delta:.8e}")
        print(f"base_delta              = {base_delta:.8e}")
        print(f"target_delta            = {args.delta:.8e}")
        print(f"certified_pass          = {certified_pass}")

        if args.run_optimistic:
            optimistic_pass = candidate_delta <= args.delta
            print(f"optimistic_pass         = {optimistic_pass}")

            if optimistic_pass:
                best_optimistic_sigma = sigma
            else:
                print(flush=True)
                break

            if certified_pass:
                best_certified_sigma = sigma
        else:
            if certified_pass:
                best_certified_sigma = sigma
            else:
                print(flush=True)
                break

        print(flush=True)
        sigma -= args.sigma_step
        candidate_index += 1

    print("========== Final outcome ==========")
    print(f"eps   = {args.eps}")
    print(f"delta = {args.delta}")
    print(f"T     = {args.T}")
    print(f"k     = {args.k}")

    if best_certified_sigma is None:
        print(f"Best certified sigma  = None (>{args.initial_sigma:.3f})")
    else:
        print(f"Best certified sigma  = {best_certified_sigma:.3f}")

    if args.run_optimistic:
        if best_optimistic_sigma is None:
            print(f"Best optimistic sigma = None (>{args.initial_sigma:.3f})")
        else:
            print(f"Best optimistic sigma = {best_optimistic_sigma:.3f}")


if __name__ == "__main__":
    main()