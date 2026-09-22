"""Tests for the configurable minimum-width L-method selector."""

import unittest

import numpy as np

from Code.Code_AR_processes.ar_tools.L_method import l_method_selector


class LMethodSelectorTests(unittest.TestCase):
    def test_all_sixteen_configurations_find_exact_elbow(self) -> None:
        widths = np.array([7.0, 5.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        for weighted in (False, True):
            for common_breakpoint in (False, True):
                for affine_affine in (False, True):
                    for squared_loss in (False, True):
                        with self.subTest(
                            weighted=weighted,
                            common_breakpoint=common_breakpoint,
                            affine_affine=affine_affine,
                            squared_loss=squared_loss,
                        ):
                            fit = l_method_selector(
                                widths,
                                weighted=weighted,
                                common_breakpoint=common_breakpoint,
                                affine_affine=affine_affine,
                                squared_loss=squared_loss,
                            )
                            self.assertEqual(fit["selected_order"], 3)
                            self.assertEqual(fit["plateau_onset"], 4)
                            self.assertAlmostEqual(fit["score"], 0.0)

    def test_common_breakpoint_uses_shared_observation(self) -> None:
        widths = np.array([8.0, 5.0, 3.0, 1.5, 1.4, 1.3])
        fit = l_method_selector(
            widths,
            weighted=True,
            common_breakpoint=True,
            affine_affine=True,
            squared_loss=True,
        )
        self.assertEqual(fit["left_k"][-1], fit["right_k"][0])
        self.assertEqual(fit["left_count"] + fit["right_count"], len(widths) + 1)
        self.assertEqual(len(fit["fitted_values"]), len(widths))

    def test_disjoint_segments_do_not_share_observation(self) -> None:
        widths = np.array([8.0, 5.0, 3.0, 1.5, 1.4, 1.3])
        fit = l_method_selector(
            widths,
            weighted=True,
            common_breakpoint=False,
            affine_affine=True,
            squared_loss=True,
        )
        self.assertEqual(fit["left_k"][-1] + 1, fit["right_k"][0])
        self.assertEqual(fit["left_count"] + fit["right_count"], len(widths))
        self.assertEqual(len(fit["fitted_values"]), len(widths))

    def test_weighted_squared_score_is_size_weighted_rmse(self) -> None:
        widths = np.array([8.0, 5.1, 3.2, 1.5, 1.4, 1.2])
        fit = l_method_selector(
            widths,
            weighted=True,
            common_breakpoint=True,
            affine_affine=False,
            squared_loss=True,
        )
        row = fit["candidate_scores"].iloc[0]
        expected = (
            row.left_count * row.left_average_error
            + row.right_count * row.right_average_error
        ) / (row.left_count + row.right_count)
        self.assertAlmostEqual(row.score, expected)

    def test_unweighted_squared_score_is_total_squared_error(self) -> None:
        widths = np.array([8.0, 5.1, 3.2, 1.5, 1.4, 1.2])
        fit = l_method_selector(
            widths,
            weighted=False,
            common_breakpoint=False,
            affine_affine=False,
            squared_loss=True,
        )
        row = fit["candidate_scores"].iloc[0]
        self.assertAlmostEqual(
            row.score,
            row.left_loss_sum + row.right_loss_sum,
        )

    def test_absolute_loss_uses_lad_and_mae_scores(self) -> None:
        widths = np.array([8.0, 5.0, 3.0, 1.0, 1.0, 1.0, 20.0])
        weighted = l_method_selector(
            widths,
            weighted=True,
            common_breakpoint=True,
            affine_affine=False,
            squared_loss=False,
        )
        unweighted = l_method_selector(
            widths,
            weighted=False,
            common_breakpoint=True,
            affine_affine=False,
            squared_loss=False,
        )
        self.assertEqual(weighted["right_intercept"], 1.0)
        self.assertEqual(weighted["selected_order"], unweighted["selected_order"])

    def test_rejects_non_boolean_configuration(self) -> None:
        with self.assertRaisesRegex(TypeError, "weighted must be a boolean"):
            l_method_selector(
                np.array([4.0, 3.0, 2.0, 2.0]),
                weighted=1,
                common_breakpoint=True,
                affine_affine=False,
                squared_loss=True,
            )


if __name__ == "__main__":
    unittest.main()
