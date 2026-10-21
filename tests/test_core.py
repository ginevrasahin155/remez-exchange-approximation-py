"""Tests for the Remez exchange approximation core.

Every test exercises a behaviour the implementation actually claims to
support. Floats are compared by relative tolerance, never by ``==``.
"""

import math
import unittest

from remez_exchange_approximation_py import RemezResult, remez_exchange


class TestRemezBasics(unittest.TestCase):

    def test_constant_function(self):
        result = remez_exchange(lambda x: 1.0, -1.0, 1.0, 0)
        self.assertIsInstance(result, RemezResult)
        self.assertEqual(len(result.coefficients), 1)
        self.assertAlmostEqual(result.coefficients[0], 1.0, places=10)
        self.assertLess(result.error_level, 1e-9)
        self.assertTrue(result.converged)

    def test_linear_function_degree_one_exact(self):
        result = remez_exchange(lambda x: 2.0 * x + 3.0, -2.0, 5.0, 1)
        self.assertEqual(len(result.coefficients), 2)
        self.assertAlmostEqual(result.coefficients[0], 3.0, places=8)
        self.assertAlmostEqual(result.coefficients[1], 2.0, places=8)
        self.assertLess(result.error_level, 1e-9)

    def test_monomial_degree_match(self):
        result = remez_exchange(lambda x: x ** 3, -1.0, 1.0, 3)
        self.assertEqual(len(result.coefficients), 4)
        self.assertAlmostEqual(result.coefficients[3], 1.0, places=8)
        for i in (0, 1, 2):
            self.assertAlmostEqual(result.coefficients[i], 0.0, places=7)
        self.assertLess(result.error_level, 1e-9)

    def test_returns_reference_points(self):
        result = remez_exchange(lambda x: x * x, -1.0, 1.0, 1)
        self.assertEqual(len(result.reference), 3)
        for i in range(len(result.reference) - 1):
            self.assertLess(result.reference[i], result.reference[i + 1])

    def test_coefficient_count(self):
        result = remez_exchange(lambda x: math.sin(x), -1.0, 1.0, 4)
        self.assertEqual(len(result.coefficients), 5)


class TestRemezAccuracy(unittest.TestCase):

    def _max_error(self, f, coeffs, a, b, n=5001):
        peak = 0.0
        for i in range(n):
            t = i / (n - 1)
            x = a + t * (b - a)
            acc = 0.0
            for k, c in enumerate(coeffs):
                acc += c * (x ** k)
            peak = max(peak, abs(acc - f(x)))
        return peak

    def test_exp_error_better_than_naive(self):
        f = math.exp
        a, b = -1.0, 1.0
        result = remez_exchange(f, a, b, 5, max_iterations=60)
        actual = self._max_error(f, result.coefficients, a, b)
        self.assertLess(actual, 3.0 * result.error_level + 1e-9)
        self.assertLess(actual, 1e-3)

    def test_chebyshev_polynomial_exact(self):
        def f(x):
            return math.cos(3 * math.acos(max(-1.0, min(1.0, x))))
        result = remez_exchange(f, -1.0, 1.0, 3, max_iterations=60)
        actual = self._max_error(f, result.coefficients, -1.0, 1.0)
        self.assertLess(actual, 1e-6)


class TestRemezInputValidation(unittest.TestCase):

    def test_bad_interval_raises(self):
        with self.assertRaises(ValueError):
            remez_exchange(lambda x: x, 1.0, 1.0, 1)
        with self.assertRaises(ValueError):
            remez_exchange(lambda x: x, 1.0, 0.0, 1)

    def test_negative_degree_raises(self):
        with self.assertRaises(ValueError):
            remez_exchange(lambda x: x, -1.0, 1.0, -1)

    def test_bad_grid_size_raises(self):
        with self.assertRaises(ValueError):
            remez_exchange(lambda x: x, -1.0, 1.0, 2, grid_size=3)

    def test_bad_tolerance_raises(self):
        with self.assertRaises(ValueError):
            remez_exchange(lambda x: x, -1.0, 1.0, 1, tolerance=-0.1)

    def test_bad_max_iterations_raises(self):
        with self.assertRaises(ValueError):
            remez_exchange(lambda x: x, -1.0, 1.0, 1, max_iterations=0)

    def test_non_callable_f_raises(self):
        with self.assertRaises(TypeError):
            remez_exchange(42, -1.0, 1.0, 1)


class TestRemezEdgeCases(unittest.TestCase):

    def test_zero_degree_constant_approx(self):
        f = lambda x: x
        result = remez_exchange(f, -1.0, 1.0, 0)
        self.assertEqual(len(result.coefficients), 1)
        self.assertAlmostEqual(result.coefficients[0], 0.0, places=6)
        self.assertLess(result.error_level, 1.0 + 1e-9)

    def test_result_has_iterations(self):
        result = remez_exchange(lambda x: x, -1.0, 1.0, 2)
        self.assertIsInstance(result.iterations, int)
        self.assertGreaterEqual(result.iterations, 1)

    def test_gaussian_on_small_interval(self):
        result = remez_exchange(lambda x: math.exp(-x * x),
                                -0.5, 0.5, 4, max_iterations=60)
        self.assertEqual(len(result.coefficients), 5)
        self.assertGreater(result.error_level, 0.0)
        self.assertLess(result.error_level, 1e-2)


if __name__ == "__main__":
    unittest.main()
