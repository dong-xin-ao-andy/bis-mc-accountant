import itertools
import math
import unittest
from unittest import mock

import numpy as np

from bis_calibration import (
    SequentialCalibration,
    _worker_generate_chunk,
    generate_privacy_loss_sample_compressed,
    generate_y_with_rng,
)
from symmetric_polynomial import (
    bis_log_likelihood_bounds,
    bis_log_likelihood_ratio,
    bis_log_likelihood_ratio_from_log_weights,
)


def _brute_force_log_likelihood(log_w, k):
    terms = np.asarray(
        [sum(log_w[i] for i in subset) for subset in itertools.combinations(range(len(log_w)), k)],
        dtype=np.float64,
    )
    if k == 0:
        return 0.0
    shift = float(np.max(terms))
    return (
        shift
        + math.log(float(np.sum(np.exp(terms - shift))))
        - math.log(math.comb(len(log_w), k))
    )


class SymmetricPolynomialTest(unittest.TestCase):
    def test_log_space_exact_value_matches_brute_force(self):
        rng = np.random.default_rng(20260817)
        for T in range(2, 9):
            for k in range(T + 1):
                log_w = rng.normal(loc=0.0, scale=2.0, size=T)
                actual = bis_log_likelihood_ratio_from_log_weights(log_w, k)
                expected = _brute_force_log_likelihood(log_w, k)
                self.assertAlmostEqual(actual, expected, places=11)

    def test_maclaurin_bounds_enclose_exact_value(self):
        rng = np.random.default_rng(20260818)
        for T in range(2, 12):
            for k in range(T + 1):
                for _ in range(10):
                    log_w = rng.normal(loc=0.0, scale=5.0, size=T)
                    lower, upper = bis_log_likelihood_bounds(log_w, k)
                    exact = bis_log_likelihood_ratio_from_log_weights(log_w, k)
                    self.assertLessEqual(lower, exact + 1e-11)
                    self.assertLessEqual(exact, upper + 1e-11)

    def test_log_space_and_weight_space_agree(self):
        log_w = np.asarray([-3.0, -0.5, 0.2, 1.7, 4.0])
        for k in range(6):
            stable = bis_log_likelihood_ratio_from_log_weights(log_w, k)
            legacy = bis_log_likelihood_ratio(np.exp(log_w), 1.0, k)
            self.assertAlmostEqual(stable, legacy, places=11)

    def test_required_tiny_weight_does_not_underflow(self):
        log_w = np.asarray([0.0, -1000.0])
        actual = bis_log_likelihood_ratio_from_log_weights(log_w, k=2)
        self.assertEqual(actual, -1000.0)

        # Exercise the general DP as well as the k=T endpoint: every
        # degree-2 monomial still needs at least one formerly underflowing
        # weight.
        log_w = np.asarray([0.0, -1000.0, -1001.0])
        actual = bis_log_likelihood_ratio_from_log_weights(log_w, k=2)
        expected = _brute_force_log_likelihood(log_w, k=2)
        self.assertAlmostEqual(actual, expected, places=11)

    def test_log_space_endpoints_with_extreme_log_range(self):
        log_w = np.asarray([-10_000.0, -1000.0, 0.0, 1000.0, 10_000.0])
        self.assertEqual(
            bis_log_likelihood_ratio_from_log_weights(log_w, k=0),
            0.0,
        )
        self.assertAlmostEqual(
            bis_log_likelihood_ratio_from_log_weights(log_w, k=len(log_w)),
            math.fsum(log_w),
            places=11,
        )

    def test_extreme_log_range_matches_brute_force_for_all_degrees(self):
        log_w = np.asarray([-1200.0, -800.0, 0.0, 700.0, 1100.0])
        for k in range(len(log_w) + 1):
            with self.subTest(k=k):
                actual = bis_log_likelihood_ratio_from_log_weights(log_w, k)
                expected = _brute_force_log_likelihood(log_w, k)
                self.assertAlmostEqual(actual, expected, places=11)


class DirectionalSamplingTest(unittest.TestCase):
    def _check_compressed_sample(self, direction):
        T, k, sigma, eps, seed = 8, 3, 1.1, 0.2, 913
        got = generate_privacy_loss_sample_compressed(
            T,
            k,
            sigma,
            eps,
            np.random.default_rng(seed),
            direction=direction,
        )
        y = generate_y_with_rng(
            T,
            k,
            sigma,
            np.random.default_rng(seed),
            direction=direction,
        )
        log_w = y / (sigma * sigma) - 0.5 / (sigma * sigma)
        llr = bis_log_likelihood_ratio_from_log_weights(log_w, k)
        privacy_loss = llr if direction == "forward" else -llr
        expected = privacy_loss if privacy_loss > eps else 0.0
        self.assertAlmostEqual(got, expected, places=12)

    def test_forward_sample_is_log_p_over_q_under_p(self):
        self._check_compressed_sample("forward")

    def test_reverse_sample_is_log_q_over_p_under_q(self):
        self._check_compressed_sample("reverse")

    def test_worker_compression_preserves_draw_count_and_threshold(self):
        for direction in ("forward", "reverse"):
            zero_count, positive, exact_count = _worker_generate_chunk(
                (7, 3, 1.2, 0.1, 500, 1729, direction)
            )
            self.assertEqual(zero_count + positive.size, 500)
            self.assertGreaterEqual(exact_count, positive.size)
            self.assertLessEqual(exact_count, 500)
            self.assertTrue(np.all(positive > 0.1))

    def test_zero_participations_have_zero_privacy_loss(self):
        for direction in ("forward", "reverse"):
            sample = generate_privacy_loss_sample_compressed(
                5,
                0,
                1.0,
                0.0,
                np.random.default_rng(1),
                direction=direction,
            )
            self.assertEqual(sample, 0.0)

    def test_nonfinite_sampling_parameters_are_rejected(self):
        for bad_sigma in (math.nan, math.inf, -math.inf):
            with self.subTest(sigma=bad_sigma), self.assertRaises(ValueError):
                generate_privacy_loss_sample_compressed(
                    5,
                    2,
                    bad_sigma,
                    1.0,
                    np.random.default_rng(1),
                )
        for bad_epsilon in (math.nan, math.inf, -math.inf):
            with self.subTest(epsilon=bad_epsilon), self.assertRaises(ValueError):
                generate_privacy_loss_sample_compressed(
                    5,
                    2,
                    1.0,
                    bad_epsilon,
                    np.random.default_rng(1),
                )


class SequentialCalibrationTest(unittest.TestCase):
    def test_nonfinite_privacy_parameters_are_rejected(self):
        for bad_epsilon in (math.nan, math.inf, -math.inf):
            with self.subTest(epsilon=bad_epsilon), self.assertRaises(ValueError):
                SequentialCalibration(epsilon=bad_epsilon, target_delta=0.1)
        for bad_delta in (math.nan, math.inf, -math.inf):
            with self.subTest(delta=bad_delta), self.assertRaises(ValueError):
                SequentialCalibration(epsilon=1.0, target_delta=bad_delta)

    def test_invalid_samples_and_counts_are_rejected(self):
        invalid = (
            (np.asarray([math.nan]), np.asarray([100_000.0])),
            (np.asarray([math.inf]), np.asarray([100_000.0])),
            (np.asarray([0.0]), np.asarray([math.nan])),
            (np.asarray([0.0]), np.asarray([math.inf])),
            (np.asarray([0.0]), np.asarray([-1.0])),
            (np.asarray([0.0]), np.asarray([1.5])),
            (np.asarray([0.0]), np.asarray([0.0])),
        )
        for samples, counts in invalid:
            with self.subTest(samples=samples, counts=counts):
                calibration = SequentialCalibration(
                    epsilon=1.0, target_delta=0.1
                )
                with self.assertRaises(ValueError):
                    calibration.add_candidate(samples, counts)

    def test_reverse_failure_stops_two_sided_calibration(self):
        calibration = SequentialCalibration(epsilon=1.0, target_delta=0.1)
        zeros = np.asarray([0.0])
        bad = np.asarray([10.0])
        counts = np.asarray([100_000.0])

        first = calibration.add_candidate(zeros, counts, zeros, counts)
        self.assertTrue(first.passed)
        second = calibration.add_candidate(zeros, counts, bad, counts)
        self.assertFalse(second.passed)
        self.assertGreater(second.reverse_delta, second.forward_delta)
        self.assertEqual(calibration.final_outcome(), (True, 0))
        with self.assertRaises(RuntimeError):
            calibration.add_candidate(zeros, counts, zeros, counts)

    def test_forward_only_mode_remains_available(self):
        calibration = SequentialCalibration(epsilon=1.0, target_delta=0.1)
        result = calibration.add_candidate(
            np.asarray([0.0]), np.asarray([100_000.0])
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.reverse_delta, 0.0)

    def test_first_failure_returns_safe_fallback_and_freezes(self):
        calibration = SequentialCalibration(epsilon=1.0, target_delta=0.1)
        bad = np.asarray([10.0])
        counts = np.asarray([100_000.0])

        result = calibration.add_candidate(bad, counts)
        self.assertFalse(result.passed)
        self.assertTrue(result.done)
        passed, fallback_delta = calibration.final_outcome()
        self.assertFalse(passed)
        self.assertEqual(fallback_delta, result.base_delta)
        with self.assertRaises(RuntimeError):
            calibration.add_candidate(np.asarray([0.0]), counts)

    def test_nonfinite_estimator_fails_closed(self):
        calibration = SequentialCalibration(epsilon=1.0, target_delta=0.1)
        zeros = np.asarray([0.0])
        counts = np.asarray([100_000.0])

        with mock.patch(
            "bis_calibration.delta_from_epsilon_and_samples",
            return_value=math.nan,
        ):
            result = calibration.add_candidate(zeros, counts)

        self.assertFalse(result.passed)
        self.assertTrue(result.done)
        self.assertTrue(math.isinf(result.forward_delta))
        self.assertFalse(calibration.final_outcome()[0])


if __name__ == "__main__":
    unittest.main()
