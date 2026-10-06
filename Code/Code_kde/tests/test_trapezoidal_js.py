"""Accuracy and integration checks for spatial Jensen-Shannon quadrature."""

from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import numpy as np

from kde_tools.density import gaussian_kde_values
from kde_tools.experiments import run_one_consistency_experiment
from kde_tools.quadrature import (
    compute_quadrature_similarity_matrices,
    quadrature_js_divergence_matrix,
    quadrature_similarity_matrices,
    select_quadrature_similarity_bandwidths,
)


class TrapezoidalJSTests(unittest.TestCase):
    def test_identical_densities_and_matrix_invariants(self) -> None:
        grid = np.linspace(-10.0, 10.0, 4001)
        densities = gaussian_kde_values(
            np.array([-0.8, 0.1, 0.9]), np.array([0.2, 0.6, 0.2]), grid
        )
        result = quadrature_js_divergence_matrix(densities, grid)
        np.testing.assert_array_equal(result, result.T)
        np.testing.assert_array_equal(np.diag(result), 0.0)
        self.assertLess(abs(result[0, 2]), 2e-15)
        self.assertTrue(np.all((result >= 0.0) & (result <= np.log(2.0))))

    def test_disjoint_support_has_known_value_and_handles_zeros(self) -> None:
        grid = np.array([0.0, 0.2, 0.5, 1.0])
        # Trapezoidal masses are exactly one, with disjoint node supports.
        densities = np.array([[10.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 4.0]])
        result = quadrature_js_divergence_matrix(densities, grid)
        self.assertAlmostEqual(result[0, 1], np.log(2.0), places=14)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_partial_masses_are_not_renormalized_or_capped(self) -> None:
        grid = np.array([0.0, 1.0])
        unit_mass = np.array([[2.0, 0.0], [0.0, 2.0]])
        for factor in (0.2, 2.0):
            result = quadrature_js_divergence_matrix(factor * unit_mass, grid)
            self.assertAlmostEqual(result[0, 1], factor * np.log(2.0), places=14)

    def test_gaussian_pairs_match_independent_adaptive_integration(self) -> None:
        try:
            from scipy.integrate import quad
        except ImportError:
            self.skipTest("Independent adaptive reference requires SciPy.")
        bandwidths = np.array([0.2, 0.7, 1.1])
        grid = np.linspace(-12.0, 12.0, 12001)
        densities = gaussian_kde_values(np.array([0.0]), bandwidths, grid)
        result = quadrature_js_divergence_matrix(densities, grid)
        for i in range(len(bandwidths)):
            for j in range(i + 1, len(bandwidths)):
                def integrand(x):
                    p = math.exp(-0.5 * (x / bandwidths[i]) ** 2) / (
                        bandwidths[i] * math.sqrt(2.0 * math.pi)
                    )
                    q = math.exp(-0.5 * (x / bandwidths[j]) ** 2) / (
                        bandwidths[j] * math.sqrt(2.0 * math.pi)
                    )
                    mixture = 0.5 * p + 0.5 * q
                    return 0.5 * (
                        (p * math.log(p / mixture) if p else 0.0)
                        + (q * math.log(q / mixture) if q else 0.0)
                    )
                reference, _ = quad(integrand, -12.0, 12.0, epsabs=1e-12)
                self.assertAlmostEqual(result[i, j], reference, places=11)

    def test_similarity_uses_raw_divergence_with_original_scaling(self) -> None:
        grid = np.linspace(-8.0, 8.0, 4001)
        densities = gaussian_kde_values(
            np.array([-1.0, 0.0, 1.0]), np.array([0.2, 0.7]), grid
        )
        raw = quadrature_js_divergence_matrix(densities, grid)
        similarity = quadrature_similarity_matrices(densities, grid, "js_similarity")
        np.testing.assert_allclose(similarity, 1.0 - raw / np.log(2.0), atol=1e-15)

    def test_sample_wrapper_honors_supplied_spatial_grid(self) -> None:
        samples = np.array([-0.5, 0.0, 0.8])
        bandwidths = np.array([0.2, 0.7])
        grid = np.linspace(-3.0, 3.0, 301)
        result = compute_quadrature_similarity_matrices(
            samples, bandwidths, "js_similarity", x_grid=grid
        )
        np.testing.assert_array_equal(result["x_grid"], grid)
        expected = quadrature_similarity_matrices(
            gaussian_kde_values(samples, bandwidths, grid), grid, "js_similarity"
        )
        np.testing.assert_array_equal(result["js_similarity"], expected)

    def test_selection_wrapper_uses_requested_node_count(self) -> None:
        result = select_quadrature_similarity_bandwidths(
            np.array([-0.5, 0.0, 0.8]), np.array([0.2, 0.7]),
            "js_similarity", m_x=257,
        )
        self.assertEqual(len(result["x_grid"]), 257)
        self.assertIn(result["choice"]["selected_h"], (0.2, 0.7))

    def test_consistency_experiment_evaluates_densities_once(self) -> None:
        with patch("kde_tools.experiments.gaussian_kde_values", wraps=gaussian_kde_values) as evaluator:
            rows = run_one_consistency_experiment(
                seed=23, n=20, sigma2=1.0,
                h_grid=np.array([0.2, 0.5, 0.9]), m_x=501,
            )
        self.assertEqual(evaluator.call_count, 1)
        self.assertEqual({row["method"] for row in rows},
                         {"Oracle ISE", "Hellinger affinity", "JS similarity"})

    def test_invalid_density_values_are_rejected(self) -> None:
        grid = np.array([0.0, 1.0])
        for values in (np.ones(2), np.ones((1, 3)), np.empty((0, 2)),
                       np.array([[-1.0, 1.0]]), np.array([[np.nan, 1.0]]),
                       np.array([[np.inf, 1.0]])):
            with self.subTest(values=values), self.assertRaises(ValueError):
                quadrature_js_divergence_matrix(values, grid)

    def test_invalid_grids_and_log_floors_are_rejected(self) -> None:
        for grid in (np.array([0.0]), np.array([1.0, 0.0]),
                     np.array([0.0, 0.0]), np.array([0.0, np.nan]),
                     np.array([0.0, np.inf]), np.array([[0.0, 1.0]])):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                quadrature_js_divergence_matrix(np.ones((2, 2)), grid)
        for eps in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(eps=eps), self.assertRaises(ValueError):
                quadrature_js_divergence_matrix(np.ones((2, 2)), np.array([0.0, 1.0]), eps)


if __name__ == "__main__":
    unittest.main()
