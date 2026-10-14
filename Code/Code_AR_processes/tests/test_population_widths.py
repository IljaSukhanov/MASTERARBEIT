"""Compare angular sweeps with independent exhaustive facet evaluation."""

import unittest

import numpy as np

from Code.Code_AR_processes.ar_tools.population_widths import (
    minimum_zonotope_width_2d,
    minimum_zonotope_width_3d,
    population_ar3_widths,
)


class PopulationWidthTests(unittest.TestCase):
    def test_two_dimensional_sweep_matches_every_facet(self):
        for seed in range(5):
            g = np.random.default_rng(seed).normal(size=(17, 2))
            normals = np.column_stack((-g[:, 1], g[:, 0]))
            normals /= np.linalg.norm(normals, axis=1, keepdims=True)
            expected = float(np.min(2 * np.abs(g @ normals.T).sum(axis=0)))
            actual, _, residual = minimum_zonotope_width_2d(g)
            self.assertAlmostEqual(actual, expected, places=11)
            self.assertLess(residual, 1e-10)

    def test_three_dimensional_sweep_matches_every_generator_pair(self):
        for seed in range(5):
            g = np.random.default_rng(seed).normal(size=(19, 3))
            a, b = np.triu_indices(len(g), 1)
            normals = np.cross(g[a], g[b])
            normals /= np.linalg.norm(normals, axis=1, keepdims=True)
            expected = float(np.min(2 * np.abs(g @ normals.T).sum(axis=0)))
            actual, _, residual = minimum_zonotope_width_3d(g)
            self.assertAlmostEqual(actual, expected, places=10)
            self.assertLess(residual, 1e-10)

    def test_seasonal_ar3_has_flat_prefix_and_exact_plateau(self):
        result = population_ar3_widths([0, 0, 0.6])
        np.testing.assert_allclose(result["widths"][:3], 5, rtol=0, atol=1e-8)
        np.testing.assert_allclose(result["widths"][3:], 2 / np.sqrt(1.36), rtol=0, atol=1e-14)
        self.assertLessEqual(float(result["tail_bounds"].max()), 1e-8)


if __name__ == "__main__":
    unittest.main()
