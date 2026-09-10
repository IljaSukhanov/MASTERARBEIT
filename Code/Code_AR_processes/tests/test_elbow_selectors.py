"""Tests for the two unpenalized minimum-width elbow selectors."""

import unittest

import numpy as np

from Code.Code_AR_processes.functions import (
    affine_constant_l_method_selector,
    two_affine_l_method_selector,
)


class ElbowSelectorTests(unittest.TestCase):
    def test_both_methods_find_an_exact_order_three_elbow(self) -> None:
        k_values = np.arange(1, 9)
        widths = np.array([7.0, 5.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        classical = two_affine_l_method_selector(k_values, widths)
        modified = affine_constant_l_method_selector(k_values, widths)

        self.assertEqual(classical["selected_order"], 3)
        self.assertEqual(classical["selected_q"], 4)
        self.assertAlmostEqual(classical["score"], 0.0)
        self.assertEqual(modified["selected_order"], 3)
        self.assertEqual(modified["selected_q"], 4)
        self.assertAlmostEqual(modified["score"], 0.0)

    def test_modified_method_can_select_order_one(self) -> None:
        k_values = np.arange(1, 7)
        widths = np.array([4.0, 2.0, 2.0, 2.0, 2.0, 2.0])

        classical = two_affine_l_method_selector(k_values, widths)
        modified = affine_constant_l_method_selector(k_values, widths)

        self.assertGreaterEqual(classical["selected_order"], 2)
        self.assertEqual(modified["selected_order"], 1)
        self.assertEqual(modified["selected_q"], 2)

    def test_rejects_nonconsecutive_embedding_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "consecutive"):
            affine_constant_l_method_selector(
                np.array([1, 2, 4, 5]),
                np.array([4.0, 2.0, 2.0, 2.0]),
            )


if __name__ == "__main__":
    unittest.main()
