"""Regression checks for exact convex-hull width acceleration."""

import unittest

import numpy as np

from Code.Code_AR_processes.ar_tools.descent import (
    batched_geometric_width_curves,
)
from Code.Code_AR_processes.ar_tools.functions import (
    simulate_uniform_ar,
    sliding_window_point_cloud,
)


class GeometricWidthHullTests(unittest.TestCase):
    def compare_full_and_hull(self, series, max_k, iterations=80):
        arguments = dict(
            series=series, max_k=max_k, seed=71, replication=3,
            starts=5, iterations=iterations, initial_step=0.04,
            geometric_decay=0.95, return_directions=True,
        )
        full = batched_geometric_width_curves(**arguments)
        hull = batched_geometric_width_curves(**arguments, use_convex_hull=True)
        np.testing.assert_allclose(full[0], hull[0], atol=1e-11, rtol=1e-11)
        np.testing.assert_allclose(full[1], hull[1], atol=1e-11, rtol=1e-11)
        for k, full_projections, hull_projections, full_u, hull_u in zip(
            range(1, max_k + 1), full[2], hull[2], full[3], hull[3]
        ):
            self.assertEqual(len(hull_projections), len(series) - k + 1)
            np.testing.assert_allclose(full_projections, hull_projections, atol=1e-10)
            np.testing.assert_allclose(full_u, hull_u, atol=1e-10)
            cloud = sliding_window_point_cloud(series, k)
            np.testing.assert_allclose(hull_projections, cloud @ hull_u, atol=1e-12)

    def test_preserves_descent_results_and_full_projections(self):
        series = simulate_uniform_ar([0.6, -0.3, 0.2], 400, rng=23)
        self.compare_full_and_hull(series, 4)

    def test_falls_back_for_a_lower_dimensional_cloud(self):
        self.compare_full_and_hull(np.linspace(-1.0, 1.0, 50), 3, iterations=20)

    def test_rejects_a_non_boolean_option(self):
        with self.assertRaisesRegex(ValueError, "use_convex_hull"):
            batched_geometric_width_curves(
                np.arange(10.0), 2, seed=1, replication=1,
                use_convex_hull="yes",
            )

    def test_returns_every_dimension_through_max_k(self):
        series = np.arange(10.0)
        widths, best_widths, projections, directions = batched_geometric_width_curves(
            series, 4, seed=1, replication=1, starts=2, iterations=5,
            return_directions=True,
        )
        self.assertEqual(len(widths), 4)
        self.assertEqual(len(best_widths), 4)
        for k, projection, direction in zip(range(1, 5), projections, directions):
            self.assertEqual(direction.shape, (k,))
            np.testing.assert_allclose(
                projection, sliding_window_point_cloud(series, k) @ direction,
            )

    def test_rejects_invalid_max_k(self):
        for max_k in (0, -1, 2.5, True, [2]):
            with self.subTest(max_k=max_k), self.assertRaisesRegex(ValueError, "max_k"):
                batched_geometric_width_curves(
                    np.arange(10.0), max_k, seed=1, replication=1,
                )


if __name__ == "__main__":
    unittest.main()
