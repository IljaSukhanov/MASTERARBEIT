"""Regression checks for the shared AR support geometry module."""

import unittest

import numpy as np

from Code.Code_AR_processes.ar_tools import functions, support_geometry


class SupportGeometryTests(unittest.TestCase):
    def test_existing_imports_reexport_the_moved_functions(self):
        for name in support_geometry.__all__:
            with self.subTest(name=name):
                self.assertIs(getattr(functions, name), getattr(support_geometry, name))

    def test_negative_ar1_has_exact_translated_parallelogram(self):
        boundary = support_geometry.theoretical_support_zonoid_2d(
            [-0.5], (-1.0, 3.0)
        )
        centre = np.full(2, 2.0 / 3.0)
        past = np.array([4.0, -2.0])
        new = np.array([0.0, 2.0])
        expected = np.array([
            centre + past + new,
            centre + past - new,
            centre - past + new,
            centre - past - new,
        ])

        self.assertEqual(boundary.shape, (5, 2))
        np.testing.assert_array_equal(boundary[0], boundary[-1])
        np.testing.assert_allclose(
            boundary[:-1][np.lexsort(boundary[:-1].T[::-1])],
            expected[np.lexsort(expected.T[::-1])],
            atol=1e-14,
        )

    def test_ar2_history_sets_the_finite_support_extrema(self):
        boundary = support_geometry.theoretical_support_zonoid_2d(
            [0.4, 0.2], (-1.0, 3.0), history=3
        )
        centre = 2.5
        half_width = 2.0
        expected_radii = half_width * np.array([
            1.0 + 0.4 + 0.36,
            1.0 + 0.4 + 0.36 + 0.224,
        ])

        np.testing.assert_allclose(
            np.min(boundary, axis=0), centre - expected_radii
        )
        np.testing.assert_allclose(
            np.max(boundary, axis=0), centre + expected_radii
        )
        np.testing.assert_array_equal(boundary[0], boundary[-1])

    def test_rejects_nonstationary_coefficients_and_reversed_bounds(self):
        with self.assertRaisesRegex(ValueError, "stationary"):
            support_geometry.theoretical_support_zonoid_2d([1.0], (-1.0, 1.0))
        with self.assertRaisesRegex(ValueError, "ordered endpoints"):
            support_geometry.theoretical_support_zonoid_2d([0.5], (1.0, -1.0))


if __name__ == "__main__":
    unittest.main()
