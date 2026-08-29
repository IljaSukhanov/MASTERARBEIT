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
