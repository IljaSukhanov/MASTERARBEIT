"""Checks for the descent-chain comparison on the warm-start rerun."""

import unittest

import numpy as np

from Code.Code_AR_processes.ar_tools.breakpoint_selectors import descent_chain


class DescentChainTests(unittest.TestCase):
    def test_flat_prefix_does_not_hide_later_genuine_drop(self):
        result = descent_chain([5, 5, 5, 1, 1, 1, 1, 1])
        self.assertEqual(result[0], 3)
        self.assertEqual(int(np.argmax(result[4][1:-1]) + 1), 3)
        self.assertAlmostEqual(float(result[4].sum()), 1.0)

    def test_one_ulp_tail_difference_does_not_create_a_false_breakpoint(self):
        m = [5, 5, 5, np.nextafter(1.0, 2.0), 1, 1, 1, 1]
        self.assertEqual(descent_chain(m)[0], 3)

    def test_substantive_upward_step_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonincreasing"):
            descent_chain([5, 4, 4.01, 2, 1, 1])


if __name__ == "__main__":
    unittest.main()
