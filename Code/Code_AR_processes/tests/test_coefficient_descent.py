import unittest

import numpy as np
from scipy.optimize._numdiff import approx_derivative

from Code.Code_AR_processes.ar_tools.coefficient_descent import (
    coefficient_width_descent,
    direction_from_coefficients,
    normalized_width_value_gradient,
)
from Code.Code_AR_processes.ar_tools.descent import projected_subgradient_empirical_width


class CoefficientDescentTests(unittest.TestCase):
    def setUp(self):
        self.points = np.random.default_rng(340).normal(size=(31, 4))
        self.phi = np.array([0.3, -0.1, 0.2])

    def test_original_objective_is_preserved_including_denominator(self):
        value, _ = normalized_width_value_gradient(self.points, self.phi)
        residuals = self.points[:, 0] - self.points[:, 1:] @ self.phi
        expected = np.ptp(residuals) / np.sqrt(1 + self.phi @ self.phi)
        self.assertAlmostEqual(value, expected, places=14)
        direction = direction_from_coefficients(self.phi)
        np.testing.assert_allclose(-direction[1:] / direction[0], self.phi)
        self.assertAlmostEqual(np.linalg.norm(direction), 1.0, places=14)

    def test_gradient_matches_independent_finite_differences(self):
        _, gradient = normalized_width_value_gradient(self.points, self.phi)

        def objective(phi):
            residuals = self.points[:, 0] - self.points[:, 1:] @ phi
            return np.ptp(residuals) / np.sqrt(1 + phi @ phi)

        numerical = approx_derivative(objective, self.phi, method="3-point").ravel()
        np.testing.assert_allclose(gradient, numerical, atol=1e-8, rtol=1e-8)

    def test_tied_extrema_select_an_active_branch(self):
        points = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 0.0]])
        value, gradient = normalized_width_value_gradient(points, [0.0])
        self.assertEqual(value, 2.0)
        np.testing.assert_array_equal(gradient, [-1.0])
        # Each active branch derivative belongs to the limiting max model.
        for displacement in [-1e-6, 1e-6]:
            actual, _ = normalized_width_value_gradient(points, [displacement])
            branch = (2.0 - displacement) / np.sqrt(1 + displacement**2)
            self.assertGreaterEqual(actual + 1e-14, branch)

    def test_sphere_step_inverted_each_time_matches_original_steps(self):
        settings = dict(iterations=15, initial_step=0.001, geometric_decay=0.97)
        sphere = projected_subgradient_empirical_width(
            self.points, initial_direction=direction_from_coefficients(self.phi),
            patience=16, improvement_tolerance=0.0, step_schedule="geometric",
            record_directions=True, **settings,
        )
        mapped = coefficient_width_descent(
            self.points, self.phi, geometry="sphere", **settings,
        )
        np.testing.assert_allclose(mapped["width_history"], sphere["width_history"], atol=1e-13)
        expected_phi = -sphere["direction_history"][:, 1:] / sphere["direction_history"][:, :1]
        np.testing.assert_allclose(mapped["coefficient_history"], expected_phi, atol=1e-13)

    def test_direction_recording_does_not_change_original_search(self):
        settings = dict(iterations=23, initial_direction=direction_from_coefficients(self.phi))
        plain = projected_subgradient_empirical_width(self.points, **settings)
        traced = projected_subgradient_empirical_width(self.points, record_directions=True, **settings)
        self.assertNotIn("direction_history", plain)
        np.testing.assert_array_equal(plain["direction"], traced["direction"])
        np.testing.assert_array_equal(plain["width_history"], traced["width_history"])
        widths = np.ptp(self.points @ traced["direction_history"].T, axis=0)
        np.testing.assert_allclose(widths, traced["width_history"], atol=1e-14)

    def test_history_best_state_and_patience_remain_aligned(self):
        result = coefficient_width_descent(
            self.points, self.phi, iterations=30, patience=2,
            improvement_tolerance=100, initial_step=0.001,
        )
        self.assertEqual(result["termination_reason"], "patience")
        self.assertEqual(result["evaluations"], 3)
        self.assertEqual(len(result["coefficient_history"]), 3)
        np.testing.assert_array_equal(result["coefficients"], self.phi)
        value, _ = normalized_width_value_gradient(self.points, result["coefficients"])
        self.assertAlmostEqual(value, result["minimum_width"], places=14)

    def test_euclidean_step_uses_gradient_of_normalized_width(self):
        _, gradient = normalized_width_value_gradient(self.points, self.phi)
        result = coefficient_width_descent(self.points, self.phi, iterations=2, initial_step=0.03)
        np.testing.assert_allclose(result["coefficient_history"][1], self.phi - 0.03 * gradient)

    def test_exact_chart_pole_stops_without_unrecorded_state(self):
        # u=(1,1)/sqrt(2), g=(2,1), tangent gradient=(1/2,-1/2).
        points = np.array([[2.0, 1.0], [0.0, 0.0]])
        result = coefficient_width_descent(
            points, [-1.0], initial_step=np.sqrt(2), iterations=5, geometry="sphere",
        )
        self.assertEqual(result["termination_reason"], "chart_pole")
        self.assertEqual(result["evaluations"], 1)
        np.testing.assert_array_equal(result["coefficients"], [-1.0])

    def test_invalid_inputs_are_rejected(self):
        for kwargs in [dict(iterations=0), dict(initial_step=np.inf), dict(geometric_decay=1),
                       dict(geometry="residual_span"), dict(patience=False)]:
            with self.assertRaises(ValueError):
                coefficient_width_descent(self.points, self.phi, **kwargs)
        with self.assertRaises(ValueError):
            normalized_width_value_gradient(self.points, [1, 2])


if __name__ == "__main__":
    unittest.main()
