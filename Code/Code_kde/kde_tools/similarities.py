from __future__ import annotations

import numpy as np

from .density import (
    exact_gaussian_kde_l2_gram,
    gaussian_kde_values,
    quadrature_grid_for_kdes,
    trapezoidal_integral,
)


SIMILARITY_NAMES = (
    "tv_overlap",
    "hellinger_affinity",
    "exponential_l2_similarity",
    "js_similarity"
)



def squared_l2_distance_from_gram(gram: np.ndarray) -> np.ndarray:
    diagonal = np.diag(gram)
    return np.maximum(diagonal[:, None] + diagonal[None, :] - 2.0 * gram, 0.0)



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


def doob_stationary_mass(matrix: np.ndarray) -> np.ndarray:
    right = power_perron_vector(matrix=matrix)
    left = power_perron_vector(matrix=np.asarray(matrix).T)
    mass = left * right
    return mass / mass.sum()


def spectral_choice_from_similarity(
        similarity_matrix: np.ndarray,
) -> dict[str, np.ndarray | int | float]:
    vector = perron_vector(matrix=similarity_matrix)
    return {
        "selected_index": int(np.argmax(vector)),
        "perron_vector": vector,
        "selected_weight": float(np.max(vector)),
    }

# computes the l2 distance matrix D, similarity transformation S, Perron vector v of S, arg max v_i and max v_i
def select_spectral_bandwidth(
        samples: np.ndarray,
        h_grid: np.ndarray,
        gamma: float | str = 1.0,
) -> dict[str, np.ndarray | int | float]:
    gram = exact_gaussian_kde_l2_gram(samples=samples, h_grid=h_grid)
    l2_squared = squared_l2_distance_from_gram(gram)
    similarity = np.exp(-float(gamma) * l2_squared)
    return {
        **spectral_choice_from_similarity(similarity),
        "similarity_matrix": similarity,
        "l2_squared": l2_squared,
    }


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
    values = np.maximum(np.asarray(density_values, dtype=float), 0.0)
    x_grid = np.asarray(x_grid, dtype=float)

    if similarity_name == "tv_overlap":
        matrix = trapezoidal_integral(values=np.minimum(values[:, None, :], values[None, :, :]), x_grid=x_grid, axis=2)
    elif similarity_name == "hellinger_affinity":
        matrix = trapezoidal_integral(values=np.sqrt(values[:, None, :] * values[None, :, :]), x_grid=x_grid, axis=2)
    elif similarity_name == "exponential_l2_similarity":
        gram = trapezoidal_integral(values=values[:, None, :] * values[None, :, :], x_grid=x_grid, axis=2)
        l2_squared = squared_l2_distance_from_gram(gram=gram)
        d2 = 0.5 * l2_squared
        matrix = np.exp(-gamma * d2)
    else:
        entropy = - trapezoidal_integral(values=values*np.log(np.maximum(values, eps)), x_grid=x_grid, axis=1)
        mixtures = 0.5 * (values[:, None, :] + values[None, :, :])
        mixture_entropy = -trapezoidal_integral(values=mixtures * np.log(np.maximum(mixtures, eps)), x_grid=x_grid, axis=2)
        js_divergence = np.maximum(mixture_entropy - 0.5 * entropy[:, None] - 0.5 * entropy[None, :], 0.0)
        matrix = 1.0 - js_divergence / np.log(2.0)

    return matrix


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
    result = quadrature_similarity_matrices(density_values=density_values, x_grid=x_grid, similarity_name=similarity_name, gamma=gamma,)
    result["x_grid"] = x_grid
    return result


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

# continue here

def symmetrized_renyi_similarity(
    densities: np.ndarray,
    dx: float,
    alpha: float,
) -> np.ndarray:
    """Return ``exp(-D)`` for the symmetrized Renyi divergence of gridded KDEs."""
    if alpha <= 0 or np.isclose(alpha, 1.0):
        raise ValueError("alpha must be positive and different from one.")
    left = np.power(densities, alpha)
    right = np.power(densities, 1.0 - alpha)
    overlap = dx * (left @ right.T)
    overlap = np.clip(overlap, np.finfo(float).tiny, 1.0)
    divergence = (np.log(overlap) + np.log(overlap.T)) / (2.0 * (alpha - 1.0))
    similarity = np.exp(-divergence)
    np.fill_diagonal(similarity, 1.0)
    return similarity


def directional_kl_similarity(densities: np.ndarray, dx: float) -> np.ndarray:
    """Return the directed matrix ``exp(-KL(p_i || p_j))``."""
    densities = np.maximum(np.asarray(densities, dtype=float), np.finfo(float).tiny)
    log_densities = np.log(densities)
    own_log_density = dx * np.sum(densities * log_densities, axis=1)
    cross_log_density = dx * (densities @ log_densities.T)
    divergence = np.maximum(own_log_density[:, None] - cross_log_density, 0.0)
    similarity = np.exp(-divergence)
    np.fill_diagonal(similarity, 1.0)
    return similarity

def toeplitz_projection(matrix: np.ndarray, symmetric: bool = True) -> np.ndarray:
    """Project a square matrix onto Toeplitz matrices by diagonal averaging."""
    matrix = np.asarray(matrix, dtype=float)
    n = matrix.shape[0]
    indices = np.arange(n)
    if symmetric:
        lag_means = np.array(
            [np.diagonal((matrix + matrix.T) / 2.0, offset=d).mean() for d in range(n)]
        )
        return lag_means[np.abs(indices[:, None] - indices[None, :])]

    lag_means = {
        offset: float(np.diagonal(matrix, offset=offset).mean())
        for offset in range(-(n - 1), n)
    }
    result = np.empty_like(matrix)
    rows = indices[:, None]
    columns = indices[None, :]
    for offset, value in lag_means.items():
        result[columns - rows == offset] = value
    return result


def toeplitz_corrected_choice(
    similarity: np.ndarray,
    bandwidths: np.ndarray,
    directed: bool = False,
) -> tuple[float, int, np.ndarray]:
    """Select the maximum Perron score relative to a Toeplitz null matrix."""
    null = toeplitz_projection(similarity, symmetric=not directed)
    if directed:
        score = doob_stationary_mass(similarity) / doob_stationary_mass(null)
    else:
        score = power_perron_vector(similarity) / power_perron_vector(null)
    index = int(np.argmax(score))
    return float(bandwidths[index]), index, score
