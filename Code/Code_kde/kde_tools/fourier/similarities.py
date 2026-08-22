"""Similarity helpers tied specifically to periodic Fourier grids."""

from __future__ import annotations

import numpy as np

from ..density import trapezoidal_integral
from ..similarities import (
    renyi_similarity_matrix,
    squared_l2_distance_from_gram,
)
from .reconstruction import kde_family_fft


def beta2_divergence_matrix_fft(
    samples: np.ndarray,
    h_grid: np.ndarray,
    half_width: float = 12.0,
    spatial_points: int = 8192,
) -> np.ndarray:
    """Approximate the beta-2 divergence matrix on a periodic FFT grid."""
    densities, dx = kde_family_fft(
        samples=samples,
        bandwidths=h_grid,
        half_width=half_width,
        spatial_points=spatial_points,
    )
    gram = float(dx) * (densities @ densities.T)
    return 0.5 * squared_l2_distance_from_gram(gram)


def beta2_similarity_matrix_fft(
    samples: np.ndarray,
    h_grid: np.ndarray,
    tau: float = 1.0,
    half_width: float = 12.0,
    spatial_points: int = 8192,
) -> np.ndarray:
    """Return ``exp(-tau * D_2)`` using periodic FFT reconstruction."""
    if tau <= 0.0:
        raise ValueError("tau must be positive.")
    divergence = beta2_divergence_matrix_fft(
        samples=samples,
        h_grid=h_grid,
        half_width=half_width,
        spatial_points=spatial_points,
    )
    similarity = np.exp(-float(tau) * divergence)
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
    gamma: float = 1.0,
    eps: float = 1e-300,
) -> np.ndarray:
    """Compute directional KL similarities on a periodic FFT grid."""
    if gamma <= 0.0:
        raise ValueError("gamma must be strictly positive.")
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
    similarity = np.exp(-float(gamma) * divergence)
    np.fill_diagonal(similarity, 1.0)
    return similarity


def symmetrized_renyi_similarity_fft(
    densities: np.ndarray,
    dx: float,
    alpha: float,
    gamma: float = 1.0,
    bandwidths: np.ndarray | None = None,
    eps: float = 1e-300,
) -> np.ndarray:
    """Compute a symmetrized Rényi similarity on a periodic FFT grid."""
    values, x_grid = _closed_periodic_grid(densities, dx)
    return renyi_similarity_matrix(
        density_values=values,
        x_grid=x_grid,
        alpha=alpha,
        gamma=gamma,
        symmetrized=True,
        bandwidths=bandwidths,
        eps=eps,
    )
