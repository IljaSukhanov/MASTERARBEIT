import unittest                                                                                 ### CHANGED

import numpy as np                                                                              ### CHANGED

from Code.Code_AR_processes.ar_tools.descent import batched_geometric_width_curves              ### CHANGED
from Code.Code_AR_processes.ar_tools.functions import sliding_window_point_cloud                ### CHANGED


class DescentWarmStartTests(unittest.TestCase):                                                 ### CHANGED
    def setUp(self):                                                                            ### CHANGED
        self.series = np.array([0.0, 2.0, -1.0, 3.0, 1.0, -2.0, 4.0])                           ### CHANGED
        self.arguments = dict(                                                                  ### CHANGED
            series=self.series, max_k=4, seed=0, replication=0,                                 ### CHANGED
            starts=1, iterations=1, return_directions=True,                                     ### CHANGED
        )                                                                                       ### CHANGED

    def test_previous_direction_is_padded_from_dimension_two(self):
        baseline = batched_geometric_width_curves(**self.arguments)                             ### CHANGED
        result = batched_geometric_width_curves(                                                ### CHANGED
            **self.arguments, warm_start_previous_direction=True,                               ### CHANGED
        )                                                                                       ### CHANGED
        np.testing.assert_allclose(result[0][:1], baseline[0][:1])
        for k in (2, 3, 4):
            expected = np.r_[result[3][k - 2], 0.0]                                             ### CHANGED
            np.testing.assert_allclose(result[3][k - 1], expected)                              ### CHANGED
            np.testing.assert_allclose(                                                         ### CHANGED
                result[0][k - 1],                                                               ### CHANGED
                np.ptp(sliding_window_point_cloud(self.series, k) @ expected),                  ### CHANGED
            )                                                                                   ### CHANGED

    def test_padded_candidates_keep_computed_widths_nonincreasing(self):
        result = batched_geometric_width_curves(
            self.series, max_k=6, seed=2, replication=0,
            starts=3, iterations=30, return_directions=True,
            warm_start_previous_direction=True,
        )
        self.assertTrue(np.all(np.diff(result[1]) <= 1e-12))
        for k, (width, direction) in enumerate(zip(result[1], result[3]), start=1):
            cloud = sliding_window_point_cloud(self.series, k)
            self.assertAlmostEqual(width, float(np.ptp(cloud @ direction)))
            self.assertAlmostEqual(float(np.linalg.norm(direction)), 1.0)

    def test_previous_coefficients_define_a_unit_start(self):                                   ### CHANGED
        result = batched_geometric_width_curves(                                                ### CHANGED
            **self.arguments, warm_start_previous_coefficients=True,                            ### CHANGED
        )                                                                                       ### CHANGED
        for k in (3, 4):                                                                        ### CHANGED
            previous_direction = result[3][k - 2]                                               ### CHANGED
            phi = -previous_direction[1:] / previous_direction[0]                               ### CHANGED
            expected = np.r_[1.0, -phi, 0.0]                                                    ### CHANGED
            expected /= np.linalg.norm(expected)                                                ### CHANGED
            np.testing.assert_allclose(result[3][k - 1], expected)                              ### CHANGED
            self.assertAlmostEqual(np.linalg.norm(result[3][k - 1]), 1.0)                       ### CHANGED

    def test_warm_start_options_are_boolean_and_mutually_exclusive(self):                       ### CHANGED
        with self.assertRaisesRegex(ValueError, "Only one warm-start"):                         ### CHANGED
            batched_geometric_width_curves(                                                     ### CHANGED
                **self.arguments, warm_start_previous_direction=True,                           ### CHANGED
                warm_start_previous_coefficients=True,                                          ### CHANGED
            )                                                                                   ### CHANGED
        for option in (                                                                         ### CHANGED
            "warm_start_previous_direction",                                                    ### CHANGED
            "warm_start_previous_coefficients",                                                 ### CHANGED
        ):                                                                                      ### CHANGED
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, option):       ### CHANGED
                batched_geometric_width_curves(**self.arguments, **{option: 1})                 ### CHANGED


if __name__ == "__main__":                                                                      ### CHANGED
    unittest.main()                                                                             ### CHANGED
