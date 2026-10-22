import unittest

import numpy as np

from Code.Code_AR_processes.ar_tools.descent import projected_subgradient_empirical_width


class DescentDiagnosticsTests(unittest.TestCase):

    def test_constant_direction_keeps_full_history(self):
        result = projected_subgradient_empirical_width(
            np.array([[-2.0, 0.0], [1.0, 0.0]]),
            iterations=10,
            initial_direction=np.array([1.0, 0.0]),
        )
        self.assertEqual(result["termination_reason"], "iteration_limit")
        self.assertEqual(len(result["width_history"]), 10)
        np.testing.assert_allclose(result["width_history"], np.full(10, 3.0))
        np.testing.assert_allclose(result["best_width_history"], np.full(10, 3.0))

    def test_constant_width_uses_patience_counter(self):
        result = projected_subgradient_empirical_width(
            np.array([[-2.0, 0.0], [1.0, 0.0]]),
            iterations=10,
            patience=3,
            initial_direction=np.array([1.0, 0.0]),
        )
        self.assertEqual(result["termination_reason"], "patience")
        self.assertEqual(len(result["width_history"]), 4)
        np.testing.assert_allclose(result["width_history"], np.full(4, 3.0))

    def test_patience_records_aligned_width_and_step_histories(self):
        result = projected_subgradient_empirical_width(
            np.array([[0.0, 0.0], [3.0, 4.0]]),
            iterations=10,
            patience=1,
            improvement_tolerance=100.0,
            initial_direction=np.array([1.0, 0.0]),
        )
        self.assertEqual(result["termination_reason"], "patience")
        self.assertEqual(len(result["width_history"]), 2)
        for key in ("best_width_history", "step_history"):
            self.assertEqual(len(result[key]), len(result["width_history"]))
            self.assertTrue(np.isfinite(result[key]).all())

    def test_delta_stops_without_discarding_smaller_best_width_updates(self):
        result = projected_subgradient_empirical_width(
            np.array([[0., 0.], [3., 4.]]), iterations=20, patience=2,
            initial_direction=[1., 0.], initial_step=.001,
            step_schedule="geometric", geometric_decay=.99,
            improvement_tolerance=0., delta=1.,
            record_directions=True, record_best_coefficients=True,
        )
        self.assertEqual(result["termination_reason"], "patience")
        self.assertEqual(len(result["width_history"]), 3)
        self.assertTrue(np.all(np.diff(result["width_history"]) < 0))
        self.assertLess(result["width_history"][0] - result["minimum_width"], 1.)
        self.assertEqual(result["minimum_width"], min(result["width_history"]))
        np.testing.assert_array_equal(result["direction"], result["direction_history"][-1])
        np.testing.assert_array_equal(result["best_update_history"], [True, True, True])
        np.testing.assert_array_equal(result["coefficients"],
                                      -result["direction"][1:] / result["direction"][0])

    def test_delta_accumulates_smaller_gains_before_resetting_patience(self):
        result = projected_subgradient_empirical_width(
            np.array([[0., 0.], [3., 4.]]), iterations=8, patience=2,
            initial_direction=[1., 0.], initial_step=.001,
            step_schedule="geometric", geometric_decay=.99,
            improvement_tolerance=0., delta=.025,
        )
        gains = -np.diff(result["best_width_history"])
        self.assertEqual(result["termination_reason"], "iteration_limit")
        self.assertEqual(len(result["width_history"]), 8)
        self.assertTrue(np.all((gains > 0) & (gains < .025)))
        self.assertGreater(sum(gains[:2]), .025)

    def test_delta_zero_preserves_strict_progress_and_exact_threshold_qualifies(self):
        arguments = dict(
            points=np.array([[0., 0.], [3., 4.]]), iterations=8, patience=1,
            initial_direction=[1., 0.], initial_step=.001,
            step_schedule="geometric", geometric_decay=.99,
            improvement_tolerance=0.,
        )
        strict = projected_subgradient_empirical_width(**arguments, delta=0.)
        self.assertEqual(len(strict["width_history"]), 8)
        self.assertEqual(strict["termination_reason"], "iteration_limit")
        first_gain = strict["width_history"][0] - strict["width_history"][1]
        threshold = projected_subgradient_empirical_width(**arguments, delta=first_gain)
        self.assertGreaterEqual(len(threshold["width_history"]), 3)
        np.testing.assert_array_equal(threshold["width_history"],
                                      strict["width_history"][:len(threshold["width_history"])])

    def test_delta_default_and_validation(self):
        cloud = np.array([[-2., 0.], [1., 0.]])
        result = projected_subgradient_empirical_width(cloud, iterations=2)
        self.assertEqual(result["delta"], 1e-8)
        for invalid in (-1., float("inf"), float("nan"), True, "1e-8", None):
            with self.subTest(delta=invalid), self.assertRaisesRegex(ValueError, "delta"):
                projected_subgradient_empirical_width(cloud, delta=invalid)


if __name__ == "__main__":
    unittest.main()
