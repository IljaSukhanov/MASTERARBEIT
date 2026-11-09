"""Independent objective and continuous-search checks for the four selectors."""

import unittest

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import pdist

from kde_tools.continuous_minimizers import (
    METHODS, R_VALUES, continuous_ise_qr_minima, ise_log_energy,
)
from kde_tools.density import exact_oracle_ise_values
from kde_tools.qhat import continuous_qhat_minimum, GaussianQhatObjective


class ContinuousComparisonTests(unittest.TestCase):
    def setUp(self):
        self.samples = np.random.default_rng(20260830).normal(size=30)
        self.d2 = pdist(self.samples[:, None], metric="sqeuclidean")

    def test_ise_matches_existing_exact_gaussian_formula(self):
        grid = np.array([0.015, 0.1, 0.35, 0.9, 2.0])
        for variance in (0.6, 1.0, 1.7):
            expected = exact_oracle_ise_values(self.samples, grid, variance)
            actual = [ise_log_energy(self.d2, self.samples, np.log(h), variance)[0] for h in grid]
            np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=2e-13)

    def test_log_derivatives_match_finite_differences(self):
        functions = [lambda z: ise_log_energy(self.d2, self.samples, z)]
        z, step = np.log(0.31), 1e-4
        for function in functions:
            value, gradient, curvature = function(z)
            low, high = function(z-step)[0], function(z+step)[0]
            self.assertAlmostEqual(gradient, (high-low)/(2*step), delta=1e-7)
            self.assertAlmostEqual(curvature, (high+low-2*value)/step**2, delta=1e-6)

    def test_all_selectors_match_dense_exact_search_and_refined_scan(self):
        result = continuous_ise_qr_minima(self.samples)
        refined = continuous_ise_qr_minima(self.samples, scan_points=513, spatial_points=65536)
        h_grid = np.geomspace(.01, 2, 1025)
        qhat = GaussianQhatObjective(self.samples)
        for method in METHODS:
            function = ((lambda h: ise_log_energy(self.d2, self.samples, np.log(h))) if method == "ISE"
                        else (lambda h: qhat.derivatives(h, R_VALUES[method])))
            values = np.array([function(h)[0] for h in h_grid])
            candidates = [values[0], values[-1]]
            for index in np.flatnonzero((values[1:-1] < values[:-2])
                                        & (values[1:-1] < values[2:]))+1:
                fit = minimize_scalar(lambda h: function(h)[0], bounds=h_grid[[index-1, index+1]],
                                      method="bounded", options={"xatol": 1e-12})
                candidates.append(fit.fun)
            self.assertAlmostEqual(result[method]["objective"], min(candidates), delta=1e-11)
            self.assertAlmostEqual(result[method]["h"], refined[method]["h"], delta=1e-8)

    def test_q0_agrees_with_previous_continuous_selector(self):
        expected = continuous_qhat_minimum(self.samples, half_width=16)
        actual = continuous_ise_qr_minima(self.samples)["Q0"]
        self.assertAlmostEqual(actual["h"], expected["h"], delta=1e-8)
        self.assertAlmostEqual(actual["objective"], expected["qhat"], delta=1e-12)

    def test_endpoints_are_not_replaced_by_interior_local_minima(self):
        samples = np.random.default_rng(20260830).normal(size=100)
        result = continuous_ise_qr_minima(samples)
        for method in ("Q1", "Q13_14"):
            self.assertEqual(result[method]["h"], 2.0)
            self.assertEqual(result[method]["boundary"], 1)
        self.assertEqual(result["ISE"]["boundary"], 0)


if __name__ == "__main__":
    unittest.main()
