"""Checkpoint decisions, fresh budgets, RNG replay and discarded incumbents."""

import unittest
from unittest.mock import patch

import numpy as np

from Code.Code_AR_processes.ar_tools.descent import projected_subgradient_empirical_width as fit


class StationarityRestartTests(unittest.TestCase):
    def test_disabled_option_keeps_the_original_search(self):
        points = np.random.default_rng(4).normal(size=(20, 4))
        settings = dict(iterations=24, patience=6, rng=7, record_directions=True,
                        record_best_coefficients=True, step_schedule="geometric")
        original = fit(points, **settings)
        disabled = fit(points, **settings, stationarity_check=False)
        self.assertEqual(original.keys(), disabled.keys())
        for key, value in original.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, disabled[key])
            else:
                self.assertEqual(value, disabled[key])

    def test_restarts_discard_failed_best_and_reset_budget_and_schedule(self):
        points = np.array([[-1., 0.], [1., 0.]])
        settings = dict(iterations=7, patience=2, initial_step=.01, geometric_decay=.9,
                        step_schedule="geometric", record_directions=True, record_best_coefficients=True)
        run = fit(points, initial_direction=[0., 1.], rng=3,
                  stationarity_check=True, max_restarts=20, **settings)
        self.assertTrue(run["stationarity_check_passed"])
        self.assertFalse(run["restart_limit_reached"])
        self.assertGreater(run["restart_count"], 0)
        self.assertEqual(run["attempt_count"], run["restart_count"] + 1)
        self.assertEqual(run["attempts"][0]["minimum_width"], 0.)
        self.assertGreater(run["minimum_width"], 0.)  # Discarded optimum had undefined coefficients.
        self.assertEqual(len(run["width_history"]), 7)  # Fresh full budget after aborted prefixes.
        self.assertEqual(run["total_evaluations"], 2 * run["restart_count"] + 7)
        draws = np.random.default_rng(3)
        for index, attempt in enumerate(run["attempts"]):
            if index:
                expected = draws.normal(size=2)
                expected /= np.linalg.norm(expected)
                np.testing.assert_array_equal(attempt["initial_direction"], expected)
            baseline = fit(points, initial_direction=attempt["initial_direction"], **settings)
            length = len(attempt["width_history"])
            np.testing.assert_allclose(attempt["direction_history"], baseline["direction_history"][:length], atol=1e-14)
            np.testing.assert_allclose(attempt["width_history"], baseline["width_history"][:length], atol=1e-14)
            np.testing.assert_allclose(attempt["step_history"], .01 * .9 ** np.arange(length))
            self.assertEqual(attempt["checkpoint"]["evaluated_states"], 2)
        again = fit(points, initial_direction=[0., 1.], rng=3,
                    stationarity_check=True, max_restarts=20, **settings)
        np.testing.assert_array_equal(run["direction_history"], again["direction_history"])
        self.assertEqual(run["restart_count"], again["restart_count"])

    def test_checkpoint_uses_best_direction_and_only_checks_once_after_pass(self):
        points = np.array([[0., 0.], [3., 4.]])
        with patch("Code.Code_AR_processes.ar_tools.descent.is_stationary",
                   return_value=(True, np.array([.5]))) as check:
            run = fit(points, iterations=8, patience=2, initial_direction=[1., .2],
                      improvement_tolerance=100., stationarity_check=True,
                      record_directions=True, record_best_coefficients=True)
        check.assert_called_once()
        np.testing.assert_allclose(check.call_args.args[0], [-.2])
        self.assertFalse(np.allclose(run["direction_history"][0], run["direction_history"][1]))
        self.assertEqual(run["checkpoint"]["minimum_width"], run["width_history"][0])
        self.assertEqual(run["restart_count"], 0)
        self.assertEqual(run["termination_reason"], "patience")

    def test_spectral_radius_handles_negative_complex_and_boundary_eigenvalues(self):
        for phi, stationary in [([1.], False), ([-1.], False),
                                ([0., -1.21], False), ([0., -.81], True)]:
            with self.subTest(phi=phi):
                direction = np.r_[1., -np.asarray(phi)]
                points = np.vstack([np.zeros(len(direction)), np.ones(len(direction))])
                run = fit(points, iterations=1, patience=1, initial_direction=direction,
                          stationarity_check=True, max_restarts=0)
                self.assertEqual(run["stationarity_check_passed"], stationary)
                self.assertEqual(run["restart_limit_reached"], not stationary)
                self.assertAlmostEqual(run["checkpoint"]["spectral_radius"],
                                       max(abs(np.roots(np.r_[1., -np.asarray(phi)]))))

    def test_restart_limit_is_explicit_and_never_reports_a_success(self):
        with patch("Code.Code_AR_processes.ar_tools.descent.is_stationary",
                   return_value=(False, np.array([2.]))):
            run = fit(np.array([[0., -1.], [0., 1.]]), iterations=5, patience=2,
                      initial_direction=[1., 2.], rng=11, stationarity_check=True, max_restarts=2)
        self.assertTrue(run["restart_limit_reached"])
        self.assertFalse(run["stationarity_check_passed"])
        self.assertEqual(run["termination_reason"], "stationarity_abort")
        self.assertEqual(run["attempt_count"], 3)
        self.assertEqual(run["total_evaluations"], 6)

    def test_a_passed_checkpoint_does_not_add_a_final_stationarity_filter(self):
        run = fit(np.array([[-1., 0.], [1., 0.]]), iterations=2, patience=1,
                  initial_direction=[1., -.9], initial_step=.5,
                  stationarity_check=True, record_best_coefficients=True)
        self.assertTrue(run["checkpoint"]["stationary"])
        self.assertAlmostEqual(run["checkpoint"]["spectral_radius"], .9)
        self.assertTrue(run["stationarity_check_passed"])
        self.assertEqual(run["restart_count"], 0)
        self.assertGreater(abs(run["coefficients"][0]), 1.)

    def test_invalid_checkpoint_options_are_rejected(self):
        points = np.array([[0., 0.], [1., 1.]])
        for kwargs in [dict(stationarity_check=1), dict(max_restarts=True),
                       dict(max_restarts=-1), dict(max_restarts=1.5),
                       dict(stationarity_check=True, iterations=1, patience=2)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                fit(points, **kwargs)
        with self.assertRaises(ValueError):
            fit(np.array([[0.], [1.]]), stationarity_check=True)


if __name__ == "__main__":
    unittest.main()
