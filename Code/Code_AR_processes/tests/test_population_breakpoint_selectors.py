"""Population rules should recognize a final drop with an exact flat tail."""

import unittest

from Code.Code_AR_processes.ar_tools.breakpoint_selectors import (
    all_population_selectors, width_selectors,
)


class PopulationBreakpointTests(unittest.TestCase):
    def test_difference_selector_methods(self):
        self.assertEqual(width_selectors([7, 5, 3, 1, 1, 1, 1, 1]), {
            "First difference": 1,
            "Second difference": 3,
        })

    def test_population_selector_methods(self):
        self.assertEqual(set(all_population_selectors([7, 5, 3, 1, 1, 1, 1, 1])), {
            "First difference", "Second difference", "Descent chain",
        })

    def test_descent_chain_ignores_a_flat_prefix(self):
        results = all_population_selectors([5, 5, 5, 1, 1, 1, 1, 1])
        self.assertEqual(results["Descent chain"], 3)

    def test_first_difference_preserves_earliest_tie_rule(self):
        results = all_population_selectors([7, 5, 3, 1, 1, 1, 1, 1])
        self.assertEqual(results["First difference"], 1)
        self.assertEqual(results["Second difference"], 3)


if __name__ == "__main__":
    unittest.main()
