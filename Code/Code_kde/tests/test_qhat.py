"""Independent checks of continuous sample Q_hat minimization."""

import unittest
from unittest.mock import patch

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

from kde_tools.qhat import (
    GaussianQhatObjective,
    bounded_qhat_minimum,
    grid_brent_qhat_minimum,
    continuous_qhat_minimum,
    gaussian_qhat,
)


class ContinuousQhatTests(unittest.TestCase):
    def test_grid_brent_finds_multiple_valleys_without_derivatives(self):
        base = np.array([-3., -1., 0., 1., 3.])
        sample = (base[:, None] + .01*base[None, :]).ravel()
        with patch("kde_tools.qhat._energy_at_h", side_effect=AssertionError("derivative evaluation")):
            result = grid_brent_qhat_minimum(sample, .005, 100., r=13/14)
        self.assertEqual(result["valleys_detected"], 2)
        np.testing.assert_allclose([p["h"] for p in result["valley_candidates"]],
                                   [.0159405444385612, 1.5938089305185468], rtol=1e-6)
        self.assertEqual(result["h"], 100.)
        self.assertEqual(result["boundary"], 1)
        self.assertEqual(result["grid_function_evaluations"], 257)

    def test_grid_brent_handles_no_valleys(self):
        for spacing in ("geometric", "linear"):
            result = grid_brent_qhat_minimum([0.], grid_spacing=spacing)
            self.assertEqual(result["valleys_detected"], 0)
            self.assertEqual(result["minimizer_function_evaluations"], 0)
            self.assertEqual(result["h"], 2.)
        with self.assertRaises(ValueError):
            grid_brent_qhat_minimum([0.], grid_spacing="unknown")

    def test_grid_brent_both_spacings_find_known_minimum(self):
        sample = np.random.default_rng(74).normal(size=100)
        results = [grid_brent_qhat_minimum(sample, grid_spacing=spacing)
                   for spacing in ("geometric", "linear")]
        self.assertAlmostEqual(results[0]["h"], results[1]["h"], delta=1e-7)

    def test_direct_bandwidth_objective_and_derivatives(self):
        sample = np.random.default_rng(14).normal(size=40)
        objective = GaussianQhatObjective(sample)
        for r in (0.0, 0.7, 1.0):
            for h in (0.15, 0.5, 1.2):
                value, first, second = objective.derivatives(h, r)
                step = 1e-4*h
                low, high = (objective.energy(h+s*step, r) for s in (-1, 1))
                self.assertAlmostEqual(value, objective.energy(h, r), delta=1e-12)
                self.assertAlmostEqual(first, (high-low)/(2*step), delta=1e-6)
                self.assertAlmostEqual(second, (high+low-2*value)/step**2, delta=1e-4)

    def test_direct_bounded_minimum_checks_endpoints_and_r(self):
        result = bounded_qhat_minimum([0.0])
        self.assertEqual(result["h"], 2.0)
        self.assertEqual(result["boundary"], 1)
        sample = np.random.default_rng(8).normal(size=70)
        for r in (0.0, 0.7):
            result = bounded_qhat_minimum(sample, r=r)
            objective = GaussianQhatObjective(sample)
            self.assertGreaterEqual(result["h"], 0.01)
            self.assertLessEqual(result["h"], 2.0)
            self.assertAlmostEqual(result["energy"], objective.energy(result["h"], r))
            self.assertLessEqual(result["energy"], min(p["energy"] for p in result["endpoints"]))

    def test_gaussian_pair_formula_matches_direct_derivative_integration(self):
        samples = np.array([-0.9, -0.2, 0.15, 1.2])
        for h in (0.08, 0.3, 0.9):
            def integrand(x):
                z = (x - samples) / h
                log_derivative = np.mean((z*z - 1) * np.exp(-0.5*z*z))
                log_derivative /= np.sqrt(2*np.pi) * h
                return 0.5 * log_derivative**2

            reference = quad(integrand, -10, 10, epsabs=1e-12, points=samples)[0]
            self.assertAlmostEqual(float(gaussian_qhat(samples, h)), reference, places=11)

    def test_continuous_search_matches_independent_dense_exact_search(self):
        for seed in (12, 74, 20260823):
            samples = np.random.default_rng(seed).normal(size=24)
            result = continuous_qhat_minimum(samples)
            h_grid = np.geomspace(0.01, 2, 1025)
            values = gaussian_qhat(samples, h_grid)
            candidates = [values[0], values[-1]]
            for index in np.flatnonzero((values[1:-1] < values[:-2])
                                        & (values[1:-1] < values[2:])) + 1:
                optimum = minimize_scalar(
                    lambda h: float(gaussian_qhat(samples, h)),
                    bounds=tuple(h_grid[[index-1, index+1]]), method="bounded",
                    options={"xatol": 1e-12},
                )
                candidates.append(optimum.fun)
            self.assertAlmostEqual(result["qhat"], min(candidates), delta=1e-11)
            refined = continuous_qhat_minimum(samples, scan_points=513, spatial_points=65536)
            self.assertAlmostEqual(result["h"], refined["h"], delta=1e-8)

    def test_boundary_is_compared_and_minimum_is_not_restricted_to_plot_grid(self):
        result = continuous_qhat_minimum(np.array([0.0]))
        self.assertEqual(result["h"], 2.0)
        self.assertTrue(result["boundary"])
        result = continuous_qhat_minimum(np.random.default_rng(74).normal(size=100))
        self.assertGreater(np.min(np.abs(np.geomspace(0.01, 2, 201) - result["h"])), 1e-7)
        self.assertLess(abs(result["log_derivative"]), 1e-8 * result["qhat"])
        self.assertGreater(result["log_curvature"], 0)


if __name__ == "__main__":
    unittest.main()
