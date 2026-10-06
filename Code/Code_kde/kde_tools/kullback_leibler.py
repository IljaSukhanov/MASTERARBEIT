from __future__ import annotations

import numpy as np

from .quadrature import trapezoidal_integral




def directional_kl_divergence_matrix(
        density_values: np.ndarray,
        x_grid: np.ndarray,
        eps: float = 1e-300,
) -> np.ndarray:
    """Compute ``KL(p_i || p_j)`` by ordinary spatial quadrature."""
    values = np.asarray(density_values, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    if values.ndim != 2:
        raise ValueError("density_values must be a two-dimensional array.")
    if values.shape[1] != len(x_grid):
        raise ValueError("x_grid must match the density grid axis.")
    if np.any(values < 0.0):
        raise ValueError("Density values must be nonnegative.")
    if eps <= 0.0:
        raise ValueError("eps must be strictly positive.")

    positive_values = np.maximum(values, eps)
    log_values = np.log(positive_values)
    own_log_density = trapezoidal_integral(
        values=values * log_values,
        x_grid=x_grid,
        axis=1,
    )
    cross_log_density = np.empty((len(values), len(values)), dtype=float)
    for j, log_density in enumerate(log_values):
        cross_log_density[:, j] = trapezoidal_integral(
            values=values * log_density[None, :],
            x_grid=x_grid,
            axis=1,
        )
    divergence = np.maximum(own_log_density[:, None] - cross_log_density, 0.0)
    np.fill_diagonal(divergence, 0.0)
    return divergence


def directional_kl_similarity(
        density_values: np.ndarray,
        x_grid: np.ndarray,
        lambda_value: float = 1.0,
        eps: float = 1e-300,
) -> np.ndarray:
    """Return ``exp(-lambda * KL(p_i || p_j))`` using spatial quadrature."""
    lambda_value = float(lambda_value)
    if not np.isfinite(lambda_value) or lambda_value <= 0.0:
        raise ValueError("lambda_value must be finite and strictly positive.")
    divergence = directional_kl_divergence_matrix(
        density_values=density_values,
        x_grid=x_grid,
        eps=eps,
    )
    similarity = np.exp(-lambda_value * divergence)
    np.fill_diagonal(similarity, 1.0)
    return similarity


def _closed_periodic_grid(
        densities: np.ndarray,
        dx: float,
) -> tuple[np.ndarray, np.ndarray]:
    densities = np.asarray(densities, dtype=float)
    if densities.ndim != 2:
        raise ValueError("densities must be a two-dimensional array.")
    if dx <= 0.0:
        raise ValueError("dx must be strictly positive.")
    x_grid = np.arange(densities.shape[1] + 1, dtype=float) * float(dx)
    periodic_densities = np.concatenate((densities, densities[:, :1]), axis=1)
    return periodic_densities, x_grid


def directional_kl_similarity_fft(
        densities: np.ndarray,
        dx: float,
        lambda_value: float = 1.0,
        eps: float = 1e-300,
) -> np.ndarray:
    """Compute directional KL similarities on a periodic FFT grid."""
    lambda_value = float(lambda_value)
    if not np.isfinite(lambda_value) or lambda_value <= 0.0:
        raise ValueError("lambda_value must be finite and strictly positive.")
    values, x_grid = _closed_periodic_grid(densities, dx)
    values = np.maximum(values, eps)
    log_values = np.log(values)
    own_log_density = trapezoidal_integral(
        values=values * log_values,
        x_grid=x_grid,
        axis=1,
    )
    cross_log_density = np.empty((len(values), len(values)), dtype=float)
    for j, log_density in enumerate(log_values):
        cross_log_density[:, j] = trapezoidal_integral(
            values=values * log_density[None, :],
            x_grid=x_grid,
            axis=1,
        )
    divergence = np.maximum(own_log_density[:, None] - cross_log_density, 0.0)
    similarity = np.exp(-lambda_value * divergence)
    np.fill_diagonal(similarity, 1.0)
    return similarity
