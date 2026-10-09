from __future__ import annotations

import numpy as np

from .density import exact_gaussian_kde_l2_gram
from .perron import (
    doob_mass_from_perron,
    perron_eigenpair,
    perron_eigenpair_from_divergence,
    perron_vector,
    perron_vector_from_divergence,
)

# 0 ... worked through
# 1 ... inspect later


# 0
def squared_l2_distance_from_gram(gram: np.ndarray) -> np.ndarray:
    diagonal = np.diag(gram)
    return np.maximum(diagonal[:, None] + diagonal[None, :] - 2.0 * gram, 0.0)


# 0
def beta2_divergence_matrix(
        samples: np.ndarray,
        h_grid: np.ndarray,
) -> np.ndarray:
    """Return the exact beta-divergence matrix at beta=2 for Gaussian KDEs.

    We use the density-power-divergence convention

        D_2(p, q) = (1 / 2) * integral (p(x) - q(x))**2 dx.

    All inner products between the Gaussian KDEs are evaluated analytically;
    no spatial grid or numerical quadrature is involved.
    """
    gram = exact_gaussian_kde_l2_gram(samples=samples, h_grid=h_grid)
    return 0.5 * squared_l2_distance_from_gram(gram)


# 0
def beta2_similarity_matrix(
        samples: np.ndarray,
        h_grid: np.ndarray,
        lambda_value: float = 1.0,
) -> np.ndarray:
    """Return ``exp(-lambda * D_2)`` using the exact beta=2 divergence."""
    if lambda_value <= 0.0:
        raise ValueError("lambda_value must be positive.")
    divergence = beta2_divergence_matrix(samples=samples, h_grid=h_grid)
    similarity = np.exp(-float(lambda_value) * divergence)
    np.fill_diagonal(similarity, 1.0)
    return similarity


# 1
def population_gaussian_beta2_divergence_matrix(
        h_grid: np.ndarray,
        target_sigma: float = 1.0,
) -> np.ndarray:
    """Return the exact population beta=2 matrix for Gaussian data and kernels.

    For data distributed as ``N(0, target_sigma**2)``, the population-smoothed
    density at bandwidth ``h`` is ``N(0, target_sigma**2 + h**2)``.
    """
    h_grid = np.asarray(h_grid, dtype=float)
    if h_grid.ndim != 1 or len(h_grid) < 1:
        raise ValueError("h_grid must be a nonempty one-dimensional array.")
    if np.any(h_grid < 0.0):
        raise ValueError("bandwidths must be nonnegative.")
    if target_sigma <= 0.0:
        raise ValueError("target_sigma must be positive.")

    variances = float(target_sigma) ** 2 + h_grid**2
    gram = 1.0 / np.sqrt(
        2.0 * np.pi * (variances[:, None] + variances[None, :])
    )
    return 0.5 * squared_l2_distance_from_gram(gram)


# 1
def first_order_perron_vector(
        divergence_matrix: np.ndarray,
        lambda_value: float = 1.0,
) -> np.ndarray:
    """Return the sum-normalized first-order Perron approximation.

    For ``S_ij = exp(-lambda * D_ij)``, the expansion around the uniform Perron
    vector of the all-ones matrix is

        v_i = 1/N - lambda * (r_i - mean(r)) / N**2 + O(lambda**2),

    where ``r_i = sum_j D_ij``.
    """
    divergence = np.asarray(divergence_matrix, dtype=float)
    if divergence.ndim != 2 or divergence.shape[0] != divergence.shape[1]:
        raise ValueError("divergence_matrix must be square.")
    if not np.allclose(divergence, divergence.T, atol=1e-12):
        raise ValueError("divergence_matrix must be symmetric.")
    if lambda_value < 0.0:
        raise ValueError("lambda_value must be nonnegative.")

    number_of_scales = divergence.shape[0]
    row_sums = divergence.sum(axis=1)
    approximation = (
        np.full(number_of_scales, 1.0 / number_of_scales)
        - float(lambda_value)
        * (row_sums - row_sums.mean())
        / number_of_scales**2
    )
    return approximation / approximation.sum()


def doob_stationary_mass(matrix: np.ndarray) -> np.ndarray:
    """Compute symmetric Doob mass using the centralized Perron solver once."""
    return doob_mass_from_perron(perron_vector(matrix))


# 0
def spectral_choice_from_similarity(
        similarity_matrix: np.ndarray,
) -> dict[str, np.ndarray | int | float]:
    result = perron_eigenpair(similarity_matrix)
    vector = result.vector
    selected_index = result.selected_index
    return {
        "selected_index": selected_index,
        "perron_vector": vector,
        "selected_weight": float(vector[selected_index]),
        "relative_residual": result.relative_residual,
        "relative_gap": result.relative_gap,
    }


def spectral_choice_from_divergence(
    divergence_matrix: np.ndarray,
    lambda_value: float = 1.0,
) -> dict[str, np.ndarray | int | float]:
    """Select exp(-lambda*D)'s Perron peak before exponent underflow occurs."""
    result = perron_eigenpair_from_divergence(divergence_matrix, lambda_value)
    selected_index = result.selected_index
    return {
        "selected_index": selected_index,
        "perron_vector": result.vector,
        "selected_weight": float(result.vector[selected_index]),
        "relative_residual": result.relative_residual,
        "relative_gap": result.relative_gap,
    }


# 0
# Compute the beta-2 divergence matrix, its similarity transformation,
# the Perron vector, and the selected bandwidth index.
def select_spectral_bandwidth(
        samples: np.ndarray,
        h_grid: np.ndarray,
        lambda_value: float = 1.0,
) -> dict[str, np.ndarray | int]:
    lambda_value = float(lambda_value)
    if not np.isfinite(lambda_value) or lambda_value <= 0.0:
        raise ValueError("lambda_value must be finite and strictly positive.")
    d2 = beta2_divergence_matrix(samples=samples, h_grid=h_grid)
    similarity = np.exp(-lambda_value * d2)
    np.fill_diagonal(similarity, 1.0)
    result = perron_eigenpair_from_divergence(d2, lambda_value)
    return {
        "selected_index": result.selected_index,
        "perron_vector": result.vector,
        "similarity_matrix": similarity,
        "beta2_divergence": d2,
    }
