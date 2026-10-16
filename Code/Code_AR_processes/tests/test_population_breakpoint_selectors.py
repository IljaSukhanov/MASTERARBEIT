"""Population rules should recognize a final drop with an exact flat tail."""

import unittest

import numpy as np

from Code.Code_AR_processes.ar_tools.breakpoint_selectors import (
    _sse, all_population_selectors, width_selectors,
)


class PopulationBreakpointTests(unittest.TestCase):
    def test_difference_selector_methods(self):
        self.assertEqual(width_selectors([7, 5, 3, 1, 1, 1, 1, 1]), {
            "First difference": 1,
            "Second difference": 3,
        })

    def test_population_selector_methods(self):
        self.assertEqual(set(all_population_selectors([7, 5, 3, 1, 1, 1, 1, 1])), {
            "First difference", "Second difference", "Descent chain", "Local drift",
            "Drop-to-tail energy", "Tail SSE ratio", "Tail variance ratio",
            "Minimum width — first tied minimum",
        })

    def test_cumulative_rules_ignore_a_flat_prefix(self):
        results = all_population_selectors([5, 5, 5, 1, 1, 1, 1, 1])
        for name in ["Descent chain", "Local drift", "Drop-to-tail energy",
                     "Tail SSE ratio", "Tail variance ratio", "Minimum width — first tied minimum"]:
            with self.subTest(method=name):
                self.assertEqual(results[name], 3)

    def test_sse_of_an_analytical_plateau_is_exactly_zero(self):
        for length in range(2, 11):
            self.assertEqual(_sse(np.repeat(np.pi, length)), 0.0)

    def test_first_difference_preserves_earliest_tie_rule(self):
        results = all_population_selectors([7, 5, 3, 1, 1, 1, 1, 1])
        self.assertEqual(results["First difference"], 1)
        self.assertEqual(results["Second difference"], 3)


if __name__ == "__main__":
    unittest.main()
