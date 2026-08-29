from __future__ import annotations

import numpy as np

from .density import (
    exact_gaussian_kde_l2_gram,
    gaussian_kde_values,
    quadrature_grid_for_kdes,
)


SIMILARITY_NAMES = (
    "tv_overlap",
    "hellinger_affinity",
    "exponential_l2_similarity",
    "js_similarity"
)

# 0 ... worked through
# 1 ... inspect later
# 2 ... quadrature related (put in new .py called quadrature?)


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
        tau: float = 1.0,
) -> np.ndarray:
    """Return ``exp(-tau * D_2)`` using the exact beta=2 divergence."""
    if tau <= 0.0:
        raise ValueError("tau must be positive.")
    divergence = beta2_divergence_matrix(samples=samples, h_grid=h_grid)
    similarity = np.exp(-float(tau) * divergence)
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
        tau: float = 1.0,
) -> np.ndarray:
    """Return the sum-normalized first-order Perron approximation.

    For ``S_ij = exp(-tau * D_ij)``, the expansion around the uniform Perron
    vector of the all-ones matrix is

        v_i = 1/N - tau * (r_i - mean(r)) / N**2 + O(tau**2),

    where ``r_i = sum_j D_ij``.
    """
    divergence = np.asarray(divergence_matrix, dtype=float)
    if divergence.ndim != 2 or divergence.shape[0] != divergence.shape[1]:
        raise ValueError("divergence_matrix must be square.")
    if not np.allclose(divergence, divergence.T, atol=1e-12):
        raise ValueError("divergence_matrix must be symmetric.")
    if tau < 0.0:
        raise ValueError("tau must be nonnegative.")

    number_of_scales = divergence.shape[0]
    row_sums = divergence.sum(axis=1)
    approximation = (
        np.full(number_of_scales, 1.0 / number_of_scales)
        - float(tau)
        * (row_sums - row_sums.mean())
        / number_of_scales**2
    )
    return approximation / approximation.sum()


# 0
# compute the normalized Perron vector of a symmetric nonnegative matrix
def perron_vector(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("The matrix must be square.")

    if not np.allclose(matrix, matrix.T, atol=1e-12):
        raise ValueError("The matrix must be symmetric.")

    if np.any(matrix < 0):
        raise ValueError("The matrix must be nonnegative.")

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    vector = eigenvectors[:, int(np.argmax(eigenvalues))]
    if np.sum(vector) < 0:
        vector = -vector
    return vector / np.sum(vector)

# 1
# computes the normalized Perron vector of a not necessarily symmetric nonnegative matrix using the power method
def power_perron_vector(
    matrix: np.ndarray,
    tolerance: float = 1e-13,
    max_iterations: int = 10_000,
) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    vector = np.full(matrix.shape[0], 1.0 / matrix.shape[0])
    for _ in range(max_iterations):
        updated = matrix @ vector
        updated /= updated.sum()
        if np.max(np.abs(updated - vector)) < tolerance:
            return updated
        vector = updated
    return vector


# 1
def doob_stationary_mass(matrix: np.ndarray) -> np.ndarray:
    right = power_perron_vector(matrix=matrix)
    left = power_perron_vector(matrix=np.asarray(matrix).T)
    mass = left * right
    return mass / mass.sum()


# 0
def spectral_choice_from_similarity(
        similarity_matrix: np.ndarray,
) -> dict[str, np.ndarray | int | float]:
    vector = perron_vector(matrix=similarity_matrix)
    return {
        "selected_index": int(np.argmax(vector)),
        "perron_vector": vector,
        "selected_weight": float(np.max(vector)),
    }


# continue here

# Compute the beta-2 divergence matrix, its similarity transformation,
# the Perron vector, and the selected bandwidth index.
def select_spectral_bandwidth(
        samples: np.ndarray,
        h_grid: np.ndarray,
        gamma: float = 1.0,
) -> dict[str, np.ndarray | int]:
    gamma = float(gamma)
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be finite and strictly positive.")
    d2 = beta2_divergence_matrix(samples=samples, h_grid=h_grid)
    similarity = np.exp(-gamma * d2)
    np.fill_diagonal(similarity, 1.0)
    vector = perron_vector(matrix=similarity)
    return {
        "selected_index": int(np.argmax(vector)),
        "perron_vector": vector,
        "similarity_matrix": similarity,
        "beta2_divergence": d2,
    }


# 2
def quadrature_beta2_divergence_matrix(
        density_values: np.ndarray,
        x_grid: np.ndarray,
) -> np.ndarray:
    """Return ``D_2 = 0.5 * integral (p_i - p_j)^2`` by quadrature."""
    values = np.asarray(density_values, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    if values.ndim != 2:
        raise ValueError("density_values must be a two-dimensional array.")
    if values.shape[1] != len(x_grid):
        raise ValueError("x_grid must match the density grid axis.")
    if np.any(values < 0.0):
        raise ValueError("Density values must be nonnegative.")

    weights = _trapezoidal_weights(x_grid)
    gram = (values * weights[None, :]) @ values.T
    divergence = 0.5 * squared_l2_distance_from_gram(gram=gram)
    np.fill_diagonal(divergence, 0.0)
    return divergence


# 2
# input: grid a=x_1<...<x_m=b, density_values: y_{jk}=\hat{f}_{h_j}(x_k)
# computes quadrature integrals of the chosen symmetric divergences
def quadrature_similarity_matrices(
        density_values: np.ndarray,
        x_grid: np.ndarray,
        similarity_name: str,
        gamma: float = 1.0,
        eps: float = 1e-300,
) -> np.ndarray:
    if similarity_name not in SIMILARITY_NAMES:
        raise ValueError(
            f"Unknown similarity {similarity_name!r}. Choose from {SIMILARITY_NAMES}."
        )
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
    gamma = float(gamma)
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be finite and strictly positive.")

    number_of_densities = len(values)
    weights = _trapezoidal_weights(x_grid)

    if similarity_name == "tv_overlap":
        matrix = np.empty((number_of_densities, number_of_densities), dtype=float)
        for i in range(number_of_densities):
            for j in range(i, number_of_densities):
                value = float(np.dot(weights, np.minimum(values[i], values[j])))
                matrix[i, j] = value
                matrix[j, i] = value
    elif similarity_name == "hellinger_affinity":
        square_roots = np.sqrt(values)
        matrix = (square_roots * weights[None, :]) @ square_roots.T
    elif similarity_name == "exponential_l2_similarity":
        d2 = quadrature_beta2_divergence_matrix(values, x_grid)
        matrix = np.exp(-gamma * d2)
    else:
        entropy = -np.sum(
            weights[None, :] * values * np.log(np.maximum(values, eps)),
            axis=1,
        )
        matrix = np.empty((number_of_densities, number_of_densities), dtype=float)
        for i in range(number_of_densities):
            for j in range(i, number_of_densities):
                mixture = 0.5 * (values[i] + values[j])
                mixture_entropy = -float(
                    np.dot(weights, mixture * np.log(np.maximum(mixture, eps)))
                )
                divergence = max(
                    mixture_entropy - 0.5 * entropy[i] - 0.5 * entropy[j],
                    0.0,
                )
                value = 1.0 - divergence / np.log(2.0)
                matrix[i, j] = value
                matrix[j, i] = value

    return np.clip(matrix, 0.0, 1.0)

# 2
def compute_quadrature_similarity_matrices(
        samples: np.ndarray,
        h_grid: np.ndarray,
        similarity_name: str,
        x_grid: np.ndarray | None = None,
        m_x: int = 2_001,
        gamma: float = 1.0,
) -> dict[str, np.ndarray | float]:
    if x_grid is None:
        x_grid = quadrature_grid_for_kdes(samples=samples, h_grid=h_grid, m_x=m_x)
    density_values = gaussian_kde_values(samples=samples, h_grid=h_grid, x_grid=x_grid)
    matrix = quadrature_similarity_matrices(density_values=density_values, x_grid=x_grid, similarity_name=similarity_name, gamma=gamma,)
    return {
        similarity_name: matrix,
        "x_grid": x_grid,
    }

#2
def select_quadrature_similarity_bandwidths(
        samples: np.ndarray,
        h_grid: np.ndarray,
        similarity_name: str,
        x_grid: np.ndarray | None = None,
        m_x: int = 2_001,
        gamma: float = 1.0,
) -> dict[str, object]:
    result = compute_quadrature_similarity_matrices(
        samples=samples,
        h_grid=h_grid,
        similarity_name=similarity_name,
        x_grid=x_grid,
        m_x=m_x,
        gamma=gamma,
    )
    choice = spectral_choice_from_similarity(similarity_matrix=result[similarity_name])
    result["choice"] = {
        **choice,
        "selected_h": float(h_grid[int(choice["selected_index"])])
    }
    return result

# 2
def _trapezoidal_weights(x_grid: np.ndarray) -> np.ndarray:
    """Return composite-trapezoidal weights for a one-dimensional grid."""
    x_grid = np.asarray(x_grid, dtype=float)
    if x_grid.ndim != 1 or len(x_grid) < 2:
        raise ValueError("x_grid must be one-dimensional with at least two nodes.")

    steps = np.diff(x_grid)
    if np.any(steps <= 0.0):
        raise ValueError("x_grid must be strictly increasing.")

    weights = np.empty_like(x_grid)
    weights[0] = steps[0] / 2.0
    weights[-1] = steps[-1] / 2.0
    weights[1:-1] = (steps[:-1] + steps[1:]) / 2.0
    return weights
