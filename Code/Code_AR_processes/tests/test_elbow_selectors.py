"""Tests for the two unpenalized minimum-width elbow selectors."""

import unittest

import numpy as np

from Code.Code_AR_processes.ar_tools.L_method import (
    affine_constant_l_method_selector,
    shared_breakpoint_l_method_selector,
    two_affine_l_method_selector,
)


class ElbowSelectorTests(unittest.TestCase):
    def test_shared_breakpoint_methods_admit_order_one(self) -> None:
        k = np.arange(1, 7)
        widths = np.array([4., 2., 2., 2., 2., 2.])
        for tail_model in ("affine", "constant"):
            fit = shared_breakpoint_l_method_selector(k, widths, tail_model=tail_model)
            self.assertEqual(fit["selected_order"], 1)
            self.assertEqual(fit["head_k"][-1], fit["tail_k"][0])

    def test_shared_constant_matches_existing_modified_method(self) -> None:
        k = np.arange(1, 9)
        widths = np.array([8., 4.5, 3., 1.4, 1.6, 1.3, 1.5, 1.2])
        existing = affine_constant_l_method_selector(k, widths)
        shared = shared_breakpoint_l_method_selector(k, widths, tail_model="constant")
        self.assertEqual(shared["selected_order"], existing["selected_order"])
        np.testing.assert_allclose(shared["candidate_scores"]["score"], existing["candidate_scores"]["score"])

    def test_affine_tail_recovers_a_nonhorizontal_two_line_curve(self) -> None:
        k = np.arange(1, 9)
        widths = np.array([10., 7., 4., 3.8, 3.6, 3.4, 3.2, 3.])
        for criterion in ("weighted_rmse", "squared_error"):
            fit = shared_breakpoint_l_method_selector(k, widths, score_criterion=criterion)
            self.assertEqual(fit["selected_order"], 2)
            self.assertAlmostEqual(fit["tail_slope"], -0.2)
            self.assertAlmostEqual(fit["score"], 0.)

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
