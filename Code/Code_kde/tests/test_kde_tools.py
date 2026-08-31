"""Numerical regression tests for the shared KDE helpers."""

from __future__ import annotations

import unittest

import numpy as np

from kde_tools.bandwidths import (
    bandwidth_grid_linear_in_psi_beta,
    estimate_sample_scale,
    gaussian_mise_oracle_bandwidth,
    sample_scaled_bandwidth_grid,
)
from kde_tools.density import (
    exact_gaussian_kde_l2_gram,
    gaussian_kde_values,
)
from kde_tools.gaussian_mixtures import (
    exact_mixture_ise_values,
    exact_mixture_mise_values,
    mixture_density,
)
from kde_tools.kullback_leibler import directional_kl_divergence_matrix
from kde_tools.quadrature import (
    quadrature_beta2_divergence_matrix,
    quadrature_similarity_matrices,
    trapezoidal_integral,
)
from kde_tools.similarities import (
    beta2_divergence_matrix,
    beta2_similarity_matrix,
    first_order_perron_vector,
    perron_vector,
    population_gaussian_beta2_divergence_matrix,
    select_spectral_bandwidth,
    squared_l2_distance_from_gram,
)
from kde_tools.experiments import (
    exact_beta2_bandwidth_selection_counts,
    quadrature_beta2_bandwidth_selection_counts,
    run_one_spectral_vs_oracle_experiment,
)


class DensityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(20260817)
        self.samples = self.rng.normal(size=35)
        self.bandwidths = bandwidth_grid_linear_in_psi_beta(
            h_min=0.15,
            h_max=0.9,
            beta=2.0,
            grid_size=6,
        )
        self.x_grid = np.linspace(-9.0, 9.0, 12_001)

    def test_density_rows_have_unit_mass(self) -> None:
        densities = gaussian_kde_values(self.samples, self.bandwidths, self.x_grid)
        masses = trapezoidal_integral(densities, self.x_grid, axis=1)
        np.testing.assert_allclose(masses, 1.0, atol=1e-12)

    def test_psi_beta_bandwidth_grid_special_cases(self) -> None:
        geometric = bandwidth_grid_linear_in_psi_beta(grid_size=7)
        np.testing.assert_allclose(
            geometric,
            np.geomspace(0.01, 2.0, 7),
        )

        linear = bandwidth_grid_linear_in_psi_beta(
            h_min=0.1,
            h_max=1.6,
            beta=0.0,
            grid_size=7,
        )
        np.testing.assert_allclose(linear, np.linspace(0.1, 1.6, 7))

    def test_psi_beta_bandwidth_grid_is_linear_in_transformed_scale(self) -> None:
        beta = 1.0
        grid = bandwidth_grid_linear_in_psi_beta(
            h_min=0.1,
            h_max=2.0,
            beta=beta,
            grid_size=9,
        )
        exponent = 1.0 - beta / 2.0
        transformed = (grid**exponent - 1.0) / exponent

        self.assertAlmostEqual(grid[0], 0.1)
        self.assertAlmostEqual(grid[-1], 2.0)
        self.assertTrue(np.all(np.diff(grid) > 0.0))
        np.testing.assert_allclose(
            np.diff(transformed),
            np.diff(transformed)[0],
        )

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
        h_grid = bandwidth_grid_linear_in_psi_beta(
            h_min=0.2,
            h_max=0.9,
            beta=2.0,
            grid_size=3,
        )
        target = mixture_density(self.x_grid, weights, means, sigmas)
        densities = gaussian_kde_values(self.samples, h_grid, self.x_grid)
        numerical = trapezoidal_integral(
            (densities - target[None, :]) ** 2, self.x_grid, axis=1
        )
        exact = exact_mixture_ise_values(
            self.samples, h_grid, weights, means, sigmas
        )
        np.testing.assert_allclose(numerical, exact, rtol=2e-9, atol=2e-11)

    def test_exact_mixture_mise_matches_quadrature(self) -> None:
        weights = np.array([0.6, 0.4])
        means = np.array([-1.5, 2.0])
        sigmas = np.array([0.5, 1.1])
        h_grid = bandwidth_grid_linear_in_psi_beta(
            h_min=0.2,
            h_max=0.9,
            beta=2.0,
            grid_size=3,
        )
        n = len(self.samples)
        target = mixture_density(self.x_grid, weights, means, sigmas)
        smoothed_targets = np.asarray(
            [
                mixture_density(
                    self.x_grid,
                    weights,
                    means,
                    np.sqrt(sigmas**2 + h**2),
                )
                for h in h_grid
            ]
        )
        integrated_bias_squared = trapezoidal_integral(
            (smoothed_targets - target[None, :]) ** 2,
            self.x_grid,
            axis=1,
        )
        integrated_variance = (
            1.0 / (2.0 * np.sqrt(np.pi) * n * h_grid)
            - trapezoidal_integral(
                smoothed_targets**2,
                self.x_grid,
                axis=1,
            )
            / n
        )
        numerical = integrated_variance + integrated_bias_squared
        exact = exact_mixture_mise_values(
            n,
            h_grid,
            weights,
            means,
            sigmas,
        )
        np.testing.assert_allclose(numerical, exact, rtol=2e-9, atol=2e-11)

    def test_exact_mixture_mise_contains_standard_normal_case(self) -> None:
        h_grid = bandwidth_grid_linear_in_psi_beta(
            h_min=0.2,
            h_max=0.9,
            beta=2.0,
            grid_size=3,
        )
        n = 35
        exact = exact_mixture_mise_values(
            n=n,
            h_grid=h_grid,
            weights=np.array([1.0]),
            means=np.array([0.0]),
            sigmas=np.array([1.0]),
        )
        expected = (
            1.0 / (2.0 * np.sqrt(np.pi) * n * h_grid)
            + (1.0 - 1.0 / n)
            / (2.0 * np.sqrt(np.pi) * np.sqrt(1.0 + h_grid**2))
            - 2.0 / np.sqrt(2.0 * np.pi * (2.0 + h_grid**2))
            + 1.0 / (2.0 * np.sqrt(np.pi))
        )
        np.testing.assert_allclose(exact, expected, atol=1e-15)

    def test_sample_scaled_bandwidth_grid_retains_normalized_coordinates(self) -> None:
        samples = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        bandwidths, dimensionless, scale = sample_scaled_bandwidth_grid(
            samples,
            grid_size=5,
            multiplier_min=0.1,
            multiplier_max=1.6,
            rate_exponent=0.2,
            scale_method="iqr",
        )
        expected_scale = estimate_sample_scale(samples, method="iqr")
        expected_dimensionless = (
            np.geomspace(0.1, 1.6, 5) * len(samples) ** (-0.2)
        )
        self.assertAlmostEqual(scale, expected_scale)
        np.testing.assert_allclose(dimensionless, expected_dimensionless)
        np.testing.assert_allclose(bandwidths, scale * dimensionless)

    def test_sample_scale_methods_use_gaussian_normalization(self) -> None:
        samples = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        normal_quartile = 0.6744897501960817
        self.assertAlmostEqual(
            estimate_sample_scale(samples, method="iqr"),
            1.0 / normal_quartile,
        )
        self.assertAlmostEqual(
            estimate_sample_scale(samples, method="mad"),
            1.0 / normal_quartile,
        )
        self.assertAlmostEqual(
            estimate_sample_scale(samples, method="std"),
            np.std(samples, ddof=1),
        )

class SimilarityTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(17)
        self.samples = rng.normal(size=30)
        self.bandwidths = bandwidth_grid_linear_in_psi_beta(
            h_min=0.15,
            h_max=0.9,
            beta=2.0,
            grid_size=7,
        )
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
        expected = 0.5 * squared_l2_distance_from_gram(gram)
        np.testing.assert_allclose(
            result["beta2_divergence"],
            expected,
            atol=1e-14,
        )
        self.assertIn(result["selected_index"], range(len(self.bandwidths)))

    def test_exact_selector_uses_supplied_gamma(self) -> None:
        divergence = beta2_divergence_matrix(self.samples, self.bandwidths)
        gamma = 2.3

        result = select_spectral_bandwidth(
            self.samples,
            self.bandwidths,
            gamma=gamma,
        )

        np.testing.assert_allclose(
            result["similarity_matrix"],
            np.exp(-gamma * divergence),
        )

    def test_exact_beta2_bandwidth_selection_counts(self) -> None:
        n = 20
        repetitions = 6
        tau = 2.3
        seed = 1234
        counts = exact_beta2_bandwidth_selection_counts(
            n=n,
            h_grid=self.bandwidths,
            M=repetitions,
            sigma2=1.0,
            tau=tau,
            seed=seed,
        )

        rng = np.random.default_rng(seed)
        expected = np.zeros(len(self.bandwidths), dtype=int)
        for _ in range(repetitions):
            samples = rng.normal(0.0, 1.0, size=n)
            selection = select_spectral_bandwidth(
                samples=samples,
                h_grid=self.bandwidths,
                gamma=tau,
            )
            expected[int(selection["selected_index"])] += 1

        np.testing.assert_array_equal(counts, expected)
        self.assertEqual(counts.shape, self.bandwidths.shape)
        self.assertEqual(int(counts.sum()), repetitions)

    def test_quadrature_beta2_bandwidth_selection_counts(self) -> None:
        counts = quadrature_beta2_bandwidth_selection_counts(
            n=20,
            h_grid=self.bandwidths,
            M=4,
            m_x=501,
            sigma2=1.0,
            tau=2.3,
            seed=1234,
        )
        repeated = quadrature_beta2_bandwidth_selection_counts(
            n=20,
            h_grid=self.bandwidths,
            M=4,
            m_x=501,
            sigma2=1.0,
            tau=2.3,
            seed=1234,
        )

        np.testing.assert_array_equal(counts, repeated)
        self.assertEqual(counts.shape, self.bandwidths.shape)
        self.assertEqual(int(counts.sum()), 4)

    def test_quadrature_similarity_uses_supplied_gamma(self) -> None:
        divergence = quadrature_beta2_divergence_matrix(
            self.densities,
            self.x_grid,
        )
        gamma = 1.7
        similarity = quadrature_similarity_matrices(
            self.densities,
            self.x_grid,
            similarity_name="exponential_l2_similarity",
            gamma=gamma,
        )
        np.testing.assert_allclose(similarity, np.exp(-gamma * divergence))

    def test_spectral_vs_oracle_runner_reports_supplied_gamma(self) -> None:
        gamma = 1.4
        result = run_one_spectral_vs_oracle_experiment(
            seed=123,
            n=20,
            sigma2=1.0,
            h_grid=bandwidth_grid_linear_in_psi_beta(
                h_min=0.1,
                h_max=1.0,
                beta=2.0,
                grid_size=8,
            ),
            gamma=gamma,
        )
        self.assertAlmostEqual(result["gamma"], gamma)

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

    def test_standard_normal_mise_oracle_regression_value(self) -> None:
        self.assertAlmostEqual(gaussian_mise_oracle_bandwidth(200), 0.383039449, places=8)


if __name__ == "__main__":
    unittest.main()
