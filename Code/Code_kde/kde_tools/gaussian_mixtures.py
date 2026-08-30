"""Utilities for one-dimensional Gaussian mixture models."""

from __future__ import annotations

import numpy as np

from .bandwidths import (
    direct_plugin_bandwidth,
    least_squares_cv_bandwidth_index,
    nearest_bandwidth_grid_index,
    silverman_rule_of_thumb_bandwidth,
)
from .density import gaussian_density_with_variance, gaussian_kde_values
from .quadrature import (
    quadrature_grid_for_kdes,
    quadrature_similarity_matrices,
)
from .similarities import spectral_choice_from_similarity


# sample n points from a Gaussian mixture with given weights, means, and sigmas
def sample_gaussian_mixture(
        rng: np.random.Generator,
        n: int,
        weights: np.ndarray,
        means: np.ndarray,
        sigmas: np.ndarray,
) -> np.ndarray:
    component_indices = rng.choice(len(weights), size=n, p=weights / np.sum(weights))
    return rng.normal(
        loc=np.asarray(means)[component_indices],
        scale=np.asarray(sigmas)[component_indices],
    )


# compute the mean and variance of a Gaussian mixture with given weights, means, and sigmas
def gaussian_mixture_mean_and_variance(
        weights: np.ndarray,
        means: np.ndarray,
        sigmas: np.ndarray,
) -> tuple[float, float]:
    normalized_weights = np.asarray(weights, dtype=float) / np.sum(weights)
    means = np.asarray(means, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    mean = float(np.sum(normalized_weights * means))
    second_moment = float(np.sum(normalized_weights * (sigmas**2 + means**2)))
    return mean, second_moment - mean**2


# input: x = (x_1, ..., x_n), weights, means, sigmas; output: f(x) = sum_i weights[i] * phi_{means[i],sigmas[i]^2}(x)
def mixture_density(
        x: np.ndarray,
        weights: np.ndarray,
        means: np.ndarray,
        sigmas: np.ndarray,
) -> np.ndarray:
    density = np.zeros_like(np.asarray(x, dtype=float), dtype=float)
    for weight, mean, sigma in zip(weights, means, sigmas):
        density += weight * gaussian_density_with_variance(
            x=x,
            mean=mean,
            variance=sigma**2,
        )
    return density


def exact_mixture_ise_values(
        samples: np.ndarray,
        h_grid: np.ndarray,
        weights: np.ndarray,
        means: np.ndarray,
        sigmas: np.ndarray,
) -> np.ndarray:
    samples = np.asarray(samples, dtype=float)
    h_grid = np.asarray(h_grid, dtype=float)
    weights = np.asarray(weights, dtype=float) / np.sum(weights)
    means = np.asarray(means, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    n = len(samples)
    squared_differences = (samples[:, None] - samples[None, :]) ** 2

    kde_squared_integrals = []
    cross_integrals = []
    for h in h_grid:
        variance = 2.0 * h**2
        kde_squared = np.exp(-squared_differences / (2.0 * variance)).sum()
        kde_squared /= n**2 * np.sqrt(2.0 * np.pi * variance)
        kde_squared_integrals.append(kde_squared)

        cross = 0.0
        for weight, mean, sigma in zip(weights, means, sigmas):
            cross += weight * gaussian_density_with_variance(
                x=samples,
                mean=mean,
                variance=h**2 + sigma**2,
            ).mean()
        cross_integrals.append(cross)

    target_squared_integral = 0.0
    for weight_a, mean_a, sigma_a in zip(weights, means, sigmas):
        for weight_b, mean_b, sigma_b in zip(weights, means, sigmas):
            target_squared_integral += (
                weight_a
                * weight_b
                * gaussian_density_with_variance(
                    x=mean_a,
                    mean=mean_b,
                    variance=sigma_a**2 + sigma_b**2,
                )
            )

    return (
        np.asarray(kde_squared_integrals)
        - 2.0 * np.asarray(cross_integrals)
        + target_squared_integral
    )


def exact_mixture_mise_values(
        n: int,
        h_grid: np.ndarray,
        weights: np.ndarray,
        means: np.ndarray,
        sigmas: np.ndarray,
) -> np.ndarray:
    """Return the exact Gaussian-kernel MISE for a Gaussian-mixture target."""
    h_grid = np.asarray(h_grid, dtype=float)
    weights = np.asarray(weights, dtype=float)
    means = np.asarray(means, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)

    if int(n) != n or n < 1:
        raise ValueError("n must be a positive integer.")
    if h_grid.ndim != 1 or len(h_grid) == 0:
        raise ValueError("h_grid must be a nonempty one-dimensional array.")
    if not np.all(np.isfinite(h_grid)) or np.any(h_grid <= 0.0):
        raise ValueError("bandwidths must be finite and strictly positive.")
    if weights.ndim != 1 or len(weights) == 0:
        raise ValueError("weights must be a nonempty one-dimensional array.")
    if means.shape != weights.shape or sigmas.shape != weights.shape:
        raise ValueError("weights, means, and sigmas must have the same shape.")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("weights must be finite and nonnegative.")
    if np.sum(weights) <= 0.0:
        raise ValueError("at least one mixture weight must be positive.")
    if not np.all(np.isfinite(means)):
        raise ValueError("means must be finite.")
    if not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0.0):
        raise ValueError("sigmas must be finite and strictly positive.")

    n = int(n)
    weights = weights / np.sum(weights)
    pair_weights = weights[:, None] * weights[None, :]
    squared_mean_differences = (means[:, None] - means[None, :]) ** 2
    component_variances = sigmas[:, None] ** 2 + sigmas[None, :] ** 2

    h_squared = h_grid[:, None, None] ** 2

    smoothed_square_variances = component_variances[None, :, :] + 2.0 * h_squared
    integrated_smoothed_square = np.sum(
        pair_weights[None, :, :]
        * np.exp(
            -squared_mean_differences[None, :, :]
            / (2.0 * smoothed_square_variances)
        )
        / np.sqrt(2.0 * np.pi * smoothed_square_variances),
        axis=(1, 2),
    )

    overlap_variances = component_variances[None, :, :] + h_squared
    smoothed_truth_overlap = np.sum(
        pair_weights[None, :, :]
        * np.exp(
            -squared_mean_differences[None, :, :]
            / (2.0 * overlap_variances)
        )
        / np.sqrt(2.0 * np.pi * overlap_variances),
        axis=(1, 2),
    )

    integrated_truth_square = float(
        np.sum(
            pair_weights
            * np.exp(-squared_mean_differences / (2.0 * component_variances))
            / np.sqrt(2.0 * np.pi * component_variances)
        )
    )

    integrated_kernel_square = 1.0 / (2.0 * np.sqrt(np.pi))
    return (
        integrated_kernel_square / (n * h_grid)
        + (1.0 - 1.0 / n) * integrated_smoothed_square
        - 2.0 * smoothed_truth_overlap
        + integrated_truth_square
    )


def run_one_mixture_bandwidth_experiment(
    seed: int,
    n: int,
    h_grid: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    sigmas: np.ndarray,
    m_x: int,
    gamma: float = 1.0,
) -> dict[str, float | int]:
    """Run oracle, spectral, plug-in, LSCV, and Silverman selectors once."""
    rng = np.random.default_rng(seed)
    samples = sample_gaussian_mixture(rng, n, weights, means, sigmas)
    ise = exact_mixture_ise_values(samples, h_grid, weights, means, sigmas)
    oracle_index = int(np.argmin(ise))
    x_grid = quadrature_grid_for_kdes(samples, h_grid, m_x=m_x)
    densities = gaussian_kde_values(samples, h_grid, x_grid)
    similarities = {
        name: quadrature_similarity_matrices(
            densities,
            x_grid,
            similarity_name=name,
            gamma=gamma,
        )
        for name in (
            "tv_overlap",
            "hellinger_affinity",
            "exponential_l2_similarity",
            "js_similarity",
        )
    }
    indices = {
        name: int(
            spectral_choice_from_similarity(matrix)["selected_index"]
        )
        for name, matrix in similarities.items()
    }

    plugin_raw_h = direct_plugin_bandwidth(samples, x_grid)
    silverman_raw_h = silverman_rule_of_thumb_bandwidth(samples)
    plugin_index = nearest_bandwidth_grid_index(plugin_raw_h, h_grid)
    silverman_index = nearest_bandwidth_grid_index(silverman_raw_h, h_grid)
    lscv_index, lscv_scores = least_squares_cv_bandwidth_index(samples, h_grid)

    result: dict[str, float | int] = {
        "seed": seed,
        "oracle_index": oracle_index,
        "oracle_h": float(h_grid[oracle_index]),
        "oracle_l2_error": float(np.sqrt(max(ise[oracle_index], 0.0))),
        "plugin_index": plugin_index,
        "plugin_h": float(h_grid[plugin_index]),
        "plugin_raw_h": float(plugin_raw_h),
        "plugin_l2_error": float(np.sqrt(max(ise[plugin_index], 0.0))),
        "lscv_index": lscv_index,
        "lscv_h": float(h_grid[lscv_index]),
        "lscv_score": float(lscv_scores[lscv_index]),
        "lscv_l2_error": float(np.sqrt(max(ise[lscv_index], 0.0))),
        "silverman_index": silverman_index,
        "silverman_h": float(h_grid[silverman_index]),
        "silverman_raw_h": float(silverman_raw_h),
        "silverman_l2_error": float(np.sqrt(max(ise[silverman_index], 0.0))),
        "gamma": float(gamma),
    }
    key_prefix = {
        "tv_overlap": "tv",
        "hellinger_affinity": "hellinger",
        "exponential_l2_similarity": "exponential_l2_similarity",
        "js_similarity": "js",
    }
    for name, index in indices.items():
        prefix = key_prefix[name]
        result[f"{prefix}_index"] = index
        result[f"{prefix}_h"] = float(h_grid[index])
        result[f"{prefix}_l2_error"] = float(np.sqrt(max(ise[index], 0.0)))
    return result
