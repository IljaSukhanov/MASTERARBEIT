"""Density evaluation and exact Gaussian integral formulas."""

from __future__ import annotations

import numpy as np


def trapezoidal_integral(
    values: np.ndarray,
    x_grid: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    """Integrate values along an explicitly supplied one-dimensional grid."""
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return trapezoid(values, x=x_grid, axis=axis)


def gaussian_density_with_variance(
    x: np.ndarray | float,
    mean: float,
    variance: float,
) -> np.ndarray:
    """Evaluate a univariate normal density parameterized by its variance."""
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * (x - mean) ** 2 / variance) / np.sqrt(
        2.0 * np.pi * variance
    )


def gaussian_kde_values(
    samples: np.ndarray,
    h_grid: np.ndarray,
    x_grid: np.ndarray,
) -> np.ndarray:
    """Evaluate a family of Gaussian KDEs.

    Returns an array with shape ``(number of bandwidths, number of x values)``.
    """
    samples = np.asarray(samples, dtype=float)
    h_grid = np.asarray(h_grid, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    z = (x_grid[None, None, :] - samples[None, :, None]) / h_grid[:, None, None]
    kernels = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    return kernels.mean(axis=1) / h_grid[:, None]


def kde_family_fft(
    samples: np.ndarray,
    bandwidths: np.ndarray,
    half_width: float,
    spatial_points: int,
) -> tuple[np.ndarray, float]:
    """Evaluate Gaussian KDEs on an equispaced periodic grid by Fourier inversion."""
    samples = np.asarray(samples, dtype=float)
    bandwidths = np.asarray(bandwidths, dtype=float)
    dx = 2.0 * half_width / spatial_points
    frequencies = 2.0 * np.pi * np.fft.rfftfreq(spatial_points, d=dx)
    shifted_samples = samples + half_width
    empirical_cf = np.exp(
        -1j * frequencies[:, None] * shifted_samples[None, :]
    ).mean(axis=1)
    smoothers = np.exp(
        -0.5 * bandwidths[:, None] ** 2 * frequencies[None, :] ** 2
    )
    coefficients = smoothers * empirical_cf[None, :] / dx
    densities = np.fft.irfft(coefficients, n=spatial_points, axis=1)
    densities = np.maximum(densities, np.finfo(float).tiny)
    densities /= densities.sum(axis=1, keepdims=True) * dx
    return densities, dx


def quadrature_grid_for_kdes(
    samples: np.ndarray,
    h_grid: np.ndarray,
    sigma2: float | None = None,
    m_x: int = 2_001,
    padding: float = 6.0,
) -> np.ndarray:
    """Build a finite grid covering the data and Gaussian KDE tails."""
    samples = np.asarray(samples, dtype=float)
    h_grid = np.asarray(h_grid, dtype=float)
    empirical_scale = np.std(samples, ddof=1) if len(samples) > 1 else 1.0
    reference_scale = (
        empirical_scale
        if sigma2 is None
        else max(empirical_scale, float(np.sqrt(sigma2)))
    )
    radius = padding * (reference_scale + float(np.max(h_grid)))
    left = min(float(np.min(samples)) - radius, -radius)
    right = max(float(np.max(samples)) + radius, radius)
    return np.linspace(left, right, int(m_x))


def normalize_density_rows(values: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """Normalize every density row to unit trapezoidal mass."""
    values = np.asarray(values, dtype=float)
    masses = trapezoidal_integral(values, x_grid, axis=1)
    if np.any(masses <= 0):
        raise ValueError("Every density row must have positive quadrature mass.")
    return values / masses[:, None]


def exact_gaussian_kde_l2_gram(
    samples: np.ndarray,
    h_grid: np.ndarray,
) -> np.ndarray:
    """Return ``<f_hat_h_i, f_hat_h_j>`` exactly for univariate Gaussian KDEs."""
    samples = np.asarray(samples, dtype=float)
    h_grid = np.asarray(h_grid, dtype=float)
    n = len(samples)
    squared_differences = (samples[:, None] - samples[None, :]) ** 2
    gram = np.empty((len(h_grid), len(h_grid)), dtype=float)

    for i, h_i in enumerate(h_grid):
        for j, h_j in enumerate(h_grid[i:], start=i):
            variance = h_i**2 + h_j**2
            value = np.exp(-squared_differences / (2.0 * variance)).sum()
            value /= n**2 * np.sqrt(2.0 * np.pi * variance)
            gram[i, j] = value
            gram[j, i] = value
    return gram


def exact_oracle_ise_values(
    samples: np.ndarray,
    h_grid: np.ndarray,
    sigma2: float,
) -> np.ndarray:
    """Exact ISE of Gaussian KDEs against the target ``N(0, sigma2)``."""
    samples = np.asarray(samples, dtype=float)
    h_grid = np.asarray(h_grid, dtype=float)
    sigma = float(np.sqrt(sigma2))
    n = len(samples)
    squared_differences = (samples[:, None] - samples[None, :]) ** 2

    kde_squared_integrals = []
    cross_integrals = []
    for h in h_grid:
        variance = 2.0 * h**2
        kde_squared = np.exp(-squared_differences / (2.0 * variance)).sum()
        kde_squared /= n**2 * np.sqrt(2.0 * np.pi * variance)
        kde_squared_integrals.append(kde_squared)

        variance = sigma2 + h**2
        cross = np.exp(-samples**2 / (2.0 * variance)).mean()
        cross /= np.sqrt(2.0 * np.pi * variance)
        cross_integrals.append(cross)

    true_squared_integral = 1.0 / (2.0 * np.sqrt(np.pi) * sigma)
    return (
        np.asarray(kde_squared_integrals)
        - 2.0 * np.asarray(cross_integrals)
        + true_squared_integral
    )


def mixture_density(
    x: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    sigmas: np.ndarray,
) -> np.ndarray:
    """Evaluate a univariate Gaussian-mixture density."""
    density = np.zeros_like(np.asarray(x, dtype=float), dtype=float)
    for weight, mean, sigma in zip(weights, means, sigmas):
        density += weight * gaussian_density_with_variance(x, mean, sigma**2)
    return density


def sample_gaussian_mixture(
    rng: np.random.Generator,
    n: int,
    weights: np.ndarray,
    means: np.ndarray,
    sigmas: np.ndarray,
) -> np.ndarray:
    """Draw ``n`` independent values from a Gaussian mixture."""
    normalized_weights = np.asarray(weights, dtype=float) / np.sum(weights)
    component_indices = rng.choice(len(weights), size=n, p=normalized_weights)
    return rng.normal(
        loc=np.asarray(means)[component_indices],
        scale=np.asarray(sigmas)[component_indices],
    )


def gaussian_mixture_mean_and_variance(
    weights: np.ndarray,
    means: np.ndarray,
    sigmas: np.ndarray,
) -> tuple[float, float]:
    """Return the mean and variance of a univariate Gaussian mixture."""
    normalized_weights = np.asarray(weights, dtype=float) / np.sum(weights)
    means = np.asarray(means, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    mean = float(np.sum(normalized_weights * means))
    second_moment = float(np.sum(normalized_weights * (sigmas**2 + means**2)))
    return mean, second_moment - mean**2


def exact_mixture_ise_values(
    samples: np.ndarray,
    h_grid: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    sigmas: np.ndarray,
) -> np.ndarray:
    """Exact ISE of Gaussian KDEs against a Gaussian-mixture target."""
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
                samples, mean, h**2 + sigma**2
            ).mean()
        cross_integrals.append(cross)

    target_squared_integral = 0.0
    for weight_a, mean_a, sigma_a in zip(weights, means, sigmas):
        for weight_b, mean_b, sigma_b in zip(weights, means, sigmas):
            target_squared_integral += (
                weight_a
                * weight_b
                * gaussian_density_with_variance(
                    mean_a,
                    mean_b,
                    sigma_a**2 + sigma_b**2,
                )
            )

    return (
        np.asarray(kde_squared_integrals)
        - 2.0 * np.asarray(cross_integrals)
        + target_squared_integral
    )


def quadrature_grid_for_gaussian_mixture_kdes(
    samples: np.ndarray,
    h_grid: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    sigmas: np.ndarray,
    m_x: int = 2_001,
    padding: float = 6.0,
) -> np.ndarray:
    """Build a grid covering a Gaussian mixture, the sample, and KDE tails."""
    del weights  # Kept in the signature because callers naturally pass the full mixture.
    samples = np.asarray(samples, dtype=float)
    h_grid = np.asarray(h_grid, dtype=float)
    means = np.asarray(means, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    left = min(float(np.min(samples)), float(np.min(means - padding * sigmas)))
    right = max(float(np.max(samples)), float(np.max(means + padding * sigmas)))
    tail_padding = padding * float(np.max(h_grid))
    return np.linspace(left - tail_padding, right + tail_padding, int(m_x))
