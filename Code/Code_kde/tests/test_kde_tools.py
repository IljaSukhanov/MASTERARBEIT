"""Numerical regression tests for the shared KDE helpers."""

from __future__ import annotations

import unittest

import numpy as np

from kde_tools.bandwidths import (
    gaussian_mise_oracle_bandwidth,
)
from kde_tools.density import (
    exact_gaussian_kde_l2_gram,
    exact_mixture_ise_values,
    gaussian_kde_values,
    mixture_density,
    trapezoidal_integral,
)
from kde_tools.similarities import (
    beta2_divergence_matrix,
    beta2_similarity_matrix,
    centered_beta2_gram,
    centered_scale_transition_descriptors,
    directional_kl_divergence_matrix,
    first_order_perron_vector,
    leading_centered_beta2_coordinate,
    leading_principal_determinants,
    principal_logdet_increments,
    perron_vector,
    population_gaussian_beta2_divergence_matrix,
    quadrature_similarity_matrices,
    renyi_divergence_matrix,
    select_spectral_bandwidth,
    squared_l2_distance_from_gram,
)


class DensityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(20260817)
        self.samples = self.rng.normal(size=35)
        self.bandwidths = np.geomspace(0.15, 0.9, 6)
        self.x_grid = np.linspace(-9.0, 9.0, 12_001)

    def test_density_rows_have_unit_mass(self) -> None:
        densities = gaussian_kde_values(self.samples, self.bandwidths, self.x_grid)
        masses = trapezoidal_integral(densities, self.x_grid, axis=1)
        np.testing.assert_allclose(masses, 1.0, atol=1e-12)

    def test_exact_gram_matches_fine_quadrature(self) -> None:
        densities = gaussian_kde_values(self.samples, self.bandwidths, self.x_grid)
        numerical = trapezoidal_integral(
            densities[:, None, :] * densities[None, :, :],
            self.x_grid,
            axis=2,
        )
        exact = exact_gaussian_kde_l2_gram(self.samples, self.bandwidths)
        np.testing.assert_allclose(numerical, exact, rtol=2e-10, atol=2e-12)

    def test_exact_mixture_ise_matches_quadrature(self) -> None:
        weights = np.array([0.6, 0.4])
        means = np.array([-1.5, 2.0])
        sigmas = np.array([0.5, 1.1])
        h_grid = np.array([0.2, 0.5, 0.9])
        target = mixture_density(self.x_grid, weights, means, sigmas)
        densities = gaussian_kde_values(self.samples, h_grid, self.x_grid)
        numerical = trapezoidal_integral(
            (densities - target[None, :]) ** 2, self.x_grid, axis=1
        )
        exact = exact_mixture_ise_values(
            self.samples, h_grid, weights, means, sigmas
        )
        np.testing.assert_allclose(numerical, exact, rtol=2e-9, atol=2e-11)

class SimilarityTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(17)
        self.samples = rng.normal(size=30)
        self.bandwidths = np.geomspace(0.15, 0.9, 7)
        self.x_grid = np.linspace(-8.0, 8.0, 4001)
        self.densities = gaussian_kde_values(
            self.samples, self.bandwidths, self.x_grid
        )

    def test_similarity_invariants(self) -> None:
        for name in (
            "tv_overlap",
            "hellinger_affinity",
            "exponential_l2_similarity",
            "js_similarity",
        ):
            matrix = quadrature_similarity_matrices(
                self.densities,
                self.x_grid,
                similarity_name=name,
            )
            np.testing.assert_allclose(matrix, matrix.T, atol=1e-13)
            np.testing.assert_allclose(np.diag(matrix), 1.0, atol=1e-13)
            self.assertTrue(np.all((matrix >= 0.0) & (matrix <= 1.0)))

    def test_exact_selector_uses_exact_l2_distances(self) -> None:
        result = select_spectral_bandwidth(self.samples, self.bandwidths)
        gram = exact_gaussian_kde_l2_gram(self.samples, self.bandwidths)
        expected = squared_l2_distance_from_gram(gram)
        np.testing.assert_allclose(result["l2_squared"], expected, atol=1e-14)
        self.assertIn(result["selected_index"], range(len(self.bandwidths)))

    def test_exact_beta2_divergence_matches_fine_quadrature(self) -> None:
        numerical = 0.5 * trapezoidal_integral(
            (
                self.densities[:, None, :]
                - self.densities[None, :, :]
            ) ** 2,
            self.x_grid,
            axis=2,
        )
        exact = beta2_divergence_matrix(self.samples, self.bandwidths)
        np.testing.assert_allclose(exact, numerical, rtol=2e-9, atol=2e-11)
        np.testing.assert_allclose(exact, exact.T, atol=1e-14)
        np.testing.assert_allclose(np.diag(exact), 0.0, atol=1e-14)

    def test_beta2_similarity_is_exponential_of_divergence(self) -> None:
        divergence = beta2_divergence_matrix(self.samples, self.bandwidths)
        similarity = beta2_similarity_matrix(
            self.samples,
            self.bandwidths,
            tau=1.7,
        )
        np.testing.assert_allclose(similarity, np.exp(-1.7 * divergence))
        np.testing.assert_allclose(np.diag(similarity), 1.0)

    def test_directional_kl_quadrature_invariants(self) -> None:
        divergence = directional_kl_divergence_matrix(
            self.densities,
            self.x_grid,
        )
        self.assertEqual(divergence.shape, (len(self.bandwidths),) * 2)
        np.testing.assert_allclose(np.diag(divergence), 0.0)
        self.assertTrue(np.all(divergence >= 0.0))

    def test_renyi_alpha_one_matches_kl_quadrature(self) -> None:
        kl = directional_kl_divergence_matrix(self.densities, self.x_grid)
        renyi = renyi_divergence_matrix(
            self.densities,
            self.x_grid,
            alpha=1.0,
            bandwidths=self.bandwidths,
        )
        np.testing.assert_allclose(renyi, kl, atol=1e-14)

    def test_population_gaussian_beta2_matrix_matches_quadrature(self) -> None:
        population = population_gaussian_beta2_divergence_matrix(
            self.bandwidths,
            target_sigma=1.0,
        )
        variances = 1.0 + self.bandwidths**2
        population_densities = np.stack(
            [
                np.exp(-self.x_grid**2 / (2.0 * variance))
                / np.sqrt(2.0 * np.pi * variance)
                for variance in variances
            ]
        )
        numerical = 0.5 * trapezoidal_integral(
            (
                population_densities[:, None, :]
                - population_densities[None, :, :]
            ) ** 2,
            self.x_grid,
            axis=2,
        )
        np.testing.assert_allclose(population, numerical, atol=1e-13)

    def test_first_order_perron_vector_has_second_order_error(self) -> None:
        divergence = population_gaussian_beta2_divergence_matrix(
            self.bandwidths,
            target_sigma=1.0,
        )
        tau = 1e-3
        exact = perron_vector(np.exp(-tau * divergence))
        approximation = first_order_perron_vector(divergence, tau=tau)
        np.testing.assert_allclose(approximation.sum(), 1.0, atol=1e-15)
        self.assertLess(np.max(np.abs(exact - approximation)), 1e-9)

    def test_centered_beta2_gram_recovers_centered_l2_gram(self) -> None:
        gram = exact_gaussian_kde_l2_gram(self.samples, self.bandwidths)
        divergence = 0.5 * squared_l2_distance_from_gram(gram)
        number_of_scales = len(self.bandwidths)
        centering = (
            np.eye(number_of_scales)
            - np.ones((number_of_scales, number_of_scales))
            / number_of_scales
        )
        expected = centering @ gram @ centering
        recovered = centered_beta2_gram(divergence)
        np.testing.assert_allclose(recovered, expected, atol=1e-14)

        result = leading_centered_beta2_coordinate(
            divergence,
            self.bandwidths,
        )
        np.testing.assert_allclose(result["coordinate"].sum(), 0.0, atol=1e-14)
        np.testing.assert_allclose(
            np.dot(result["coordinate"], result["coordinate"]),
            result["eigenvalue"],
            atol=1e-14,
        )
        self.assertGreaterEqual(
            np.dot(
                result["eigenvector"],
                np.log(self.bandwidths) - np.log(self.bandwidths).mean(),
            ),
            0.0,
        )
        descriptors = centered_scale_transition_descriptors(
            result["coordinate"],
            self.bandwidths,
            edge_exclusion=1,
        )
        self.assertIn(
            descriptors["zero_crossing_index"],
            range(1, len(self.bandwidths) - 2),
        )
        self.assertGreater(descriptors["zero_crossing_h"], 0.0)

    def test_principal_logdet_increments_match_nested_determinants(self) -> None:
        similarity = np.array(
            [
                [1.0, 0.4, 0.2],
                [0.4, 1.0, 0.3],
                [0.2, 0.3, 1.0],
            ]
        )
        regularization = 1e-10
        regularized = (
            similarity + regularization * np.eye(len(similarity))
        ) / (1.0 + regularization)

        forward = principal_logdet_increments(
            similarity,
            regularization=regularization,
        )
        expected_forward_logdets = np.array([
            np.linalg.slogdet(regularized[:size, :size])[1]
            for size in range(1, len(similarity) + 1)
        ])
        np.testing.assert_allclose(
            np.cumsum(forward),
            expected_forward_logdets,
            atol=1e-14,
        )

        backward = principal_logdet_increments(
            similarity,
            regularization=regularization,
            reverse=True,
        )
        expected_backward_logdets = np.array([
            np.linalg.slogdet(regularized[index:, index:])[1]
            for index in range(len(similarity))
        ])
        np.testing.assert_allclose(
            np.cumsum(backward[::-1])[::-1],
            expected_backward_logdets,
            atol=1e-14,
        )

        raw_determinants = leading_principal_determinants(similarity)
        expected_raw_determinants = np.array([
            np.linalg.det(similarity[:size, :size])
            for size in range(1, len(similarity) + 1)
        ])
        np.testing.assert_allclose(
            raw_determinants,
            expected_raw_determinants,
            atol=1e-15,
        )

    def test_standard_normal_mise_oracle_regression_value(self) -> None:
        self.assertAlmostEqual(gaussian_mise_oracle_bandwidth(200), 0.383039449, places=8)


if __name__ == "__main__":
    unittest.main()
