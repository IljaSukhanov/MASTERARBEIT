"""Regression tests for the explicitly optional Fourier subpackage."""

from __future__ import annotations

# Permit running individual tests and unittest discovery from the project root.
import sys as _sys
from pathlib import Path as _Path
_code_root = str(_Path(__file__).resolve().parents[1])
if _code_root not in _sys.path:
    _sys.path.insert(0, _code_root)

import unittest

import numpy as np

from kde_tools.bandwidths import bandwidth_grid_linear_in_psi_beta
from kde_tools.fourier import (
    gaussian_kde_l2_gram_fourier,
    beta2_divergence_matrix_fourier,
    beta2_similarity_matrix_fourier,
    beta2_divergence_matrix_fft,
    beta2_similarity_matrix_fft,
    kde_family_fft,
)
from kde_tools.similarities import (
    beta2_divergence_matrix,
    spectral_choice_from_similarity,
)


class OptionalFourierTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(20260817)
        self.samples = rng.normal(size=35)
        self.bandwidths = bandwidth_grid_linear_in_psi_beta(
            h_min=0.15,
            h_max=0.9,
            beta=2.0,
            grid_size=6,
        )

    def test_reconstructed_family_is_normalized(self) -> None:
        densities, dx = kde_family_fft(
            self.samples,
            self.bandwidths,
            half_width=12.0,
            spatial_points=4096,
        )
        np.testing.assert_allclose(densities.sum(axis=1) * dx, 1.0, atol=1e-12)

    def test_fft_beta2_matches_exact_gaussian_formula(self) -> None:
        exact = beta2_divergence_matrix(self.samples, self.bandwidths)
        approximate = beta2_divergence_matrix_fft(
            self.samples,
            self.bandwidths,
            half_width=12.0,
            spatial_points=4096,
        )
        np.testing.assert_allclose(approximate, exact, rtol=1e-10, atol=1e-12)

        similarity = beta2_similarity_matrix_fft(
            self.samples,
            self.bandwidths,
            lambda_value=1.7,
            half_width=12.0,
            spatial_points=4096,
        )
        np.testing.assert_allclose(similarity, np.exp(-1.7 * exact), atol=1e-12)


    def test_frequency_gram_matches_literal_trapezoidal_integrals(self) -> None:
        # A short interval deliberately leaves nonzero integrand values at
        # both endpoints, so an incorrect Nyquist or endpoint weight is visible.
        sample = np.array([-0.7, 0.2, 1.1])
        bandwidths = np.array([0.12, 0.4, 1.3])
        omega = np.linspace(0.0, 3.0, 18)
        empirical_cf = np.mean(np.exp(-1j * np.outer(omega, sample)), axis=1)
        trapezoid = getattr(np, "trapezoid", None)
        if trapezoid is None:
            trapezoid = np.trapz
        expected = np.empty((3, 3))
        for i, h in enumerate(bandwidths):
            for j, s in enumerate(bandwidths):
                integrand = np.abs(empirical_cf)**2 * np.exp(
                    -0.5 * (h*h+s*s) * omega**2
                )
                expected[i, j] = trapezoid(integrand, omega) / np.pi
        actual = gaussian_kde_l2_gram_fourier(
            sample, bandwidths, frequency_cutoff=3.0, frequency_points=18
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-16)

    def test_frequency_cutoff_converges_for_narrow_bandwidths(self) -> None:
        bandwidths = np.geomspace(0.01, 2.0, 12)
        exact = beta2_divergence_matrix(self.samples, bandwidths)
        coarse = beta2_divergence_matrix_fourier(
            self.samples, bandwidths, half_width=12.0, spatial_points=512
        )
        fine = beta2_divergence_matrix_fourier(
            self.samples, bandwidths, half_width=12.0, spatial_points=8192
        )
        self.assertGreater(np.max(np.abs(coarse-exact)), 1e-3)
        np.testing.assert_allclose(fine, exact, rtol=1e-10, atol=1e-12)

    def test_frequency_distances_preserve_symmetry_zero_and_translation(self) -> None:
        bandwidths = np.array([0.03, 0.1, 0.1, 0.6])
        actual = beta2_divergence_matrix_fourier(self.samples, bandwidths)
        shifted = beta2_divergence_matrix_fourier(self.samples+7.25, bandwidths)
        np.testing.assert_array_equal(actual, actual.T)
        np.testing.assert_array_equal(np.diag(actual), np.zeros(len(bandwidths)))
        self.assertEqual(actual[1, 2], 0.0)
        self.assertTrue(np.all(actual >= 0.0))
        np.testing.assert_allclose(shifted, actual, rtol=1e-11, atol=1e-12)

    def test_legacy_names_use_the_direct_frequency_route(self) -> None:
        expected = beta2_divergence_matrix_fourier(self.samples, self.bandwidths)
        np.testing.assert_array_equal(
            beta2_divergence_matrix_fft(self.samples, self.bandwidths), expected
        )
        np.testing.assert_array_equal(
            beta2_similarity_matrix_fft(self.samples, self.bandwidths),
            beta2_similarity_matrix_fourier(self.samples, self.bandwidths),
        )

    def test_frequency_grid_rejects_invalid_inputs(self) -> None:
        for sample, bandwidths, cutoff, points in (
            ([], [0.1], 4.0, 20),
            ([np.nan], [0.1], 4.0, 20),
            ([0.0], [np.inf], 4.0, 20),
            ([0.0], [0.0], 4.0, 20),
            ([0.0], [0.1], np.inf, 20),
            ([0.0], [0.1], -1.0, 20),
            ([0.0], [0.1], 4.0, 1),
            ([0.0], [0.1], 4.0, 20.5),
        ):
            with self.subTest(sample=sample, bandwidths=bandwidths,
                              cutoff=cutoff, points=points):
                with self.assertRaises(ValueError):
                    gaussian_kde_l2_gram_fourier(
                        sample, bandwidths,
                        frequency_cutoff=cutoff, frequency_points=points,
                    )


if __name__ == "__main__":
    unittest.main()
