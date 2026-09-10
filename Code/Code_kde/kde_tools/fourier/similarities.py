"""Similarity helpers tied specifically to periodic Fourier grids."""

from __future__ import annotations

import numpy as np

from ..similarities import squared_l2_distance_from_gram
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
    lambda_value: float = 1.0,
    half_width: float = 12.0,
    spatial_points: int = 8192,
) -> np.ndarray:
    """Return ``exp(-lambda * D_2)`` using periodic FFT reconstruction."""
    if lambda_value <= 0.0:
        raise ValueError("lambda_value must be positive.")
    divergence = beta2_divergence_matrix_fft(
        samples=samples,
        h_grid=h_grid,
        half_width=half_width,
        spatial_points=spatial_points,
    )
    similarity = np.exp(-float(lambda_value) * divergence)
    np.fill_diagonal(similarity, 1.0)
    return similarity
