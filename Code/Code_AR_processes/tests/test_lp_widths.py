import unittest

import numpy as np

from Code.Code_AR_processes.ar_tools.lp_widths import (
    lp_width_descent, lp_width_curve, exact_empirical_width_low_dimension,
)
from Code.Code_AR_processes.ar3_descent_chain_lp import earlier_bend_certificate


class LPWidthTests(unittest.TestCase):
    def test_lp_steps_retain_feasible_nonincreasing_widths(self):
        points = np.random.default_rng(0).normal(size=(50, 3))
        fit = lp_width_descent(points, [1, 0, 0], max_iterations=20)
        self.assertTrue(np.all(np.diff(fit["history"]) <= 0))
        self.assertAlmostEqual(np.linalg.norm(fit["direction"]), 1.0)
        self.assertAlmostEqual(fit["width"], float(np.ptp(points @ fit["direction"])))

    def test_curve_is_monotone_on_common_times(self):
        series = np.random.default_rng(1).uniform(-1, 1, 70)
        widths, _, _ = lp_width_curve(series, max_k=5)
        self.assertTrue(np.all(np.diff(widths) <= 1e-12))

    def test_global_hull_audit_matches_a_rectangle(self):
        points = np.array([[-2, -1], [-2, 1], [2, -1], [2, 1]], dtype=float)
        width, _, _ = exact_empirical_width_low_dimension(points)
        self.assertAlmostEqual(width, 2.0)

    def test_earlier_bend_certificate_cannot_be_repaired_by_lower_tail(self):
        prefix = np.array([10., 4., 2., 1.9])
        result = earlier_bend_certificate(prefix, 1.8)
        self.assertTrue(result["earlier_bend_wins_for_all_smaller_terminal_widths"])
        j = result["earlier_order"]
        for terminal in [1.8, 1.0, 0.0]:
            earlier = (prefix[j - 1] - prefix[j]) / (prefix[j] - terminal)
            true = (prefix[-2] - prefix[-1]) / (prefix[-1] - terminal)
            self.assertGreater(earlier, true)

    def test_certificate_rejects_a_larger_true_drop(self):
        result = earlier_bend_certificate([10., 9., 8., 2.], 1.9)
        self.assertFalse(result["earlier_bend_wins_for_all_smaller_terminal_widths"])


if __name__ == "__main__":
    unittest.main()
