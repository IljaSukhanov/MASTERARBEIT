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


if __name__ == "__main__":
    unittest.main()
