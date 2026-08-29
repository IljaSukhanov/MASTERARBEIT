"""Utilities for one-dimensional Gaussian mixture models."""

from __future__ import annotations

import numpy as np

from .density import gaussian_density_with_variance


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
