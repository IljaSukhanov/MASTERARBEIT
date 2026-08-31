"""Regression tests for the explicitly optional Fourier subpackage."""

from __future__ import annotations

import unittest

import numpy as np

from kde_tools.bandwidths import bandwidth_grid_linear_in_psi_beta
from kde_tools.experiments import fourier_beta2_bandwidth_selection_counts
from kde_tools.fourier import (
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
            tau=1.7,
            half_width=12.0,
            spatial_points=4096,
        )
        np.testing.assert_allclose(similarity, np.exp(-1.7 * exact), atol=1e-12)

    def test_fourier_bandwidth_selection_counts(self) -> None:
        n = 20
        repetitions = 4
        tau = 1.7
        seed = 1234
        spatial_points = 2_048
        counts = fourier_beta2_bandwidth_selection_counts(
            n=n,
            h_grid=self.bandwidths,
            M=repetitions,
            sigma2=1.0,
            tau=tau,
            half_width=12.0,
            spatial_points=spatial_points,
            seed=seed,
        )

        rng = np.random.default_rng(seed)
        expected = np.zeros(len(self.bandwidths), dtype=int)
        for _ in range(repetitions):
            samples = rng.normal(0.0, 1.0, size=n)
            similarity = beta2_similarity_matrix_fft(
                samples=samples,
                h_grid=self.bandwidths,
                tau=tau,
                half_width=12.0,
                spatial_points=spatial_points,
            )
            choice = spectral_choice_from_similarity(similarity)
            expected[int(choice["selected_index"])] += 1

        np.testing.assert_array_equal(counts, expected)
        self.assertEqual(counts.shape, self.bandwidths.shape)
        self.assertEqual(int(counts.sum()), repetitions)


if __name__ == "__main__":
    unittest.main()
