"""Independent checks of continuous sample Q_hat minimization."""

# Permit running individual tests and unittest discovery from the project root.
import sys as _sys
from pathlib import Path as _Path
_code_root = str(_Path(__file__).resolve().parents[1])
if _code_root not in _sys.path:
    _sys.path.insert(0, _code_root)

import unittest

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

from kde_tools.qhat import (
    GaussianQhatObjective,
    continuous_qhat_minimum,
    gaussian_qhat,
)


class ContinuousQhatTests(unittest.TestCase):
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

    def test_analytic_log_derivatives_match_finite_differences(self):
        samples = np.random.default_rng(74).normal(size=20)
        objective = GaussianQhatObjective(samples)
        z, step = np.log(0.23), 1e-4
        h = np.exp(z)
        value, first_h, second_h = objective.derivatives(h)
        gradient, curvature = h * first_h, h * first_h + h*h * second_h
        lower, upper = (objective.energy(np.exp(z + s*step)) for s in (-1, 1))
        self.assertAlmostEqual(gradient, (upper-lower)/(2*step), delta=1e-8)
        self.assertAlmostEqual(curvature, (upper+lower-2*value)/step**2, delta=1e-7)

    def test_continuous_search_matches_independent_dense_exact_search(self):
        for seed in (12, 74, 20260823):
            samples = np.random.default_rng(seed).normal(size=24)
            result = continuous_qhat_minimum(samples)
            log_h = np.linspace(np.log(0.01), np.log(2), 1025)
            values = gaussian_qhat(samples, np.exp(log_h))
            candidates = [values[0], values[-1]]
            for index in np.flatnonzero((values[1:-1] < values[:-2])
                                        & (values[1:-1] < values[2:])) + 1:
                optimum = minimize_scalar(
                    lambda z: float(gaussian_qhat(samples, np.exp(z))),
                    bounds=tuple(log_h[[index-1, index+1]]), method="bounded",
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
