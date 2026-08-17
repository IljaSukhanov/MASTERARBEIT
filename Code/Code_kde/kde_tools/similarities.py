"""Divergence matrices, similarity transforms, and spectral selection."""

from __future__ import annotations

import numpy as np

from .density import (
    exact_gaussian_kde_l2_gram,
    gaussian_kde_values,
    normalize_density_rows,
    quadrature_grid_for_kdes,
    trapezoidal_integral,
)

SIMILARITY_NAMES = ("tv_overlap", "hellinger_affinity", "rbf_l2", "js_similarity")


def perron_vector(matrix: np.ndarray) -> np.ndarray:
    """Return the normalized principal vector of a symmetric nonnegative matrix."""
    matrix = np.asarray(matrix, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    vector = np.real(eigenvectors[:, int(np.argmax(eigenvalues))])
    if np.sum(vector) < 0:
        vector = -vector
    vector = np.maximum(vector, 0.0)
    if np.sum(vector) == 0:
        vector = np.abs(np.real(eigenvectors[:, int(np.argmax(eigenvalues))]))
    return vector / np.sum(vector)


def power_perron_vector(
    matrix: np.ndarray,
    tolerance: float = 1e-13,
    max_iterations: int = 10_000,
) -> np.ndarray:
    """Return a right Perron vector by power iteration, including nonsymmetric cases."""
    matrix = np.asarray(matrix, dtype=float)
    vector = np.full(matrix.shape[0], 1.0 / matrix.shape[0])
    for _ in range(max_iterations):
        updated = matrix @ vector
        updated /= updated.sum()
        if np.max(np.abs(updated - vector)) < tolerance:
            return updated
        vector = updated
    return vector


def spectral_choice_from_similarity(
    similarity_matrix: np.ndarray,
) -> dict[str, np.ndarray | int | float]:
    """Choose the index at which the normalized Perron vector is maximal."""
    vector = perron_vector(similarity_matrix)
    return {
        "selected_index": int(np.argmax(vector)),
        "perron_vector": vector,
        "selected_weight": float(np.max(vector)),
    }


def _auto_gamma(distance_matrix: np.ndarray) -> float:
    positive = distance_matrix[distance_matrix > 0]
    return 1.0 if len(positive) == 0 else float(1.0 / np.median(positive))


def squared_l2_distance_from_gram(gram: np.ndarray) -> np.ndarray:
    """Convert a Gram matrix to a matrix of squared Hilbert-space distances."""
    diagonal = np.diag(gram)
    return np.maximum(diagonal[:, None] + diagonal[None, :] - 2.0 * gram, 0.0)


def select_spectral_bandwidth(
    samples: np.ndarray,
    h_grid: np.ndarray,
    gamma: float | str = "auto",
) -> dict[str, np.ndarray | int | float]:
    """Select a Gaussian bandwidth from exact squared-L2 RBF similarities."""
    gram = exact_gaussian_kde_l2_gram(samples, h_grid)
    l2_squared = squared_l2_distance_from_gram(gram)
    gamma_value = _auto_gamma(l2_squared) if gamma == "auto" else float(gamma)
    similarity = np.exp(-gamma_value * l2_squared)
    np.fill_diagonal(similarity, 1.0)
    return {
        **spectral_choice_from_similarity(similarity),
        "gamma": gamma_value,
        "similarity_matrix": similarity,
        "l2_squared": l2_squared,
    }


def quadrature_similarity_matrices(
    density_values: np.ndarray,
    x_grid: np.ndarray,
    gamma: float | str = "auto",
    renormalize: bool = True,
    eps: float = 1e-300,
) -> dict[str, np.ndarray | float]:
    """Compute TV, Hellinger, squared-L2 RBF, and JS KDE similarities."""
    values = np.maximum(np.asarray(density_values, dtype=float), 0.0)
    x_grid = np.asarray(x_grid, dtype=float)
    if renormalize:
        values = normalize_density_rows(values, x_grid)

    tv_overlap = trapezoidal_integral(
        np.minimum(values[:, None, :], values[None, :, :]), x_grid, axis=2
    )
    hellinger_affinity = trapezoidal_integral(
        np.sqrt(values[:, None, :] * values[None, :, :]), x_grid, axis=2
    )

    gram = trapezoidal_integral(
        values[:, None, :] * values[None, :, :], x_grid, axis=2
    )
    l2_squared = squared_l2_distance_from_gram(gram)
    gamma_value = _auto_gamma(l2_squared) if gamma == "auto" else float(gamma)
    rbf_l2 = np.exp(-gamma_value * l2_squared)

    entropy = -trapezoidal_integral(
        values * np.log(np.maximum(values, eps)), x_grid, axis=1
    )
    mixtures = 0.5 * (values[:, None, :] + values[None, :, :])
    mixture_entropy = -trapezoidal_integral(
        mixtures * np.log(np.maximum(mixtures, eps)), x_grid, axis=2
    )
    js_divergence = np.maximum(
        mixture_entropy - 0.5 * entropy[:, None] - 0.5 * entropy[None, :], 0.0
    )
    js_similarity = 1.0 - js_divergence / np.log(2.0)

    matrices = {
        "tv_overlap": tv_overlap,
        "hellinger_affinity": hellinger_affinity,
        "rbf_l2": rbf_l2,
        "js_similarity": js_similarity,
    }
    for name, matrix in matrices.items():
        matrix = np.clip((matrix + matrix.T) / 2.0, 0.0, 1.0)
        np.fill_diagonal(matrix, 1.0)
        matrices[name] = matrix

    return {
        **matrices,
        "l2_squared": l2_squared,
        "js_divergence": js_divergence,
        "gamma": gamma_value,
        "density_values": values,
    }


def compute_quadrature_similarity_matrices(
    samples: np.ndarray,
    h_grid: np.ndarray,
    x_grid: np.ndarray | None = None,
    sigma2: float | None = None,
    m_x: int = 2_001,
    gamma: float | str = "auto",
    renormalize: bool = True,
) -> dict[str, np.ndarray | float]:
    """Evaluate Gaussian KDEs and their four standard similarity matrices."""
    if x_grid is None:
        x_grid = quadrature_grid_for_kdes(samples, h_grid, sigma2=sigma2, m_x=m_x)
    density_values = gaussian_kde_values(samples, h_grid, x_grid)
    result = quadrature_similarity_matrices(
        density_values, x_grid, gamma=gamma, renormalize=renormalize
    )
    result["x_grid"] = x_grid
    return result


def select_quadrature_similarity_bandwidths(
    samples: np.ndarray,
    h_grid: np.ndarray,
    x_grid: np.ndarray | None = None,
    sigma2: float | None = None,
    m_x: int = 2_001,
    gamma: float | str = "auto",
    renormalize: bool = True,
) -> dict[str, object]:
    """Compute four similarities and select one bandwidth from every Perron vector."""
    result = compute_quadrature_similarity_matrices(
        samples,
        h_grid,
        x_grid=x_grid,
        sigma2=sigma2,
        m_x=m_x,
        gamma=gamma,
        renormalize=renormalize,
    )
    choices = {}
    for name in SIMILARITY_NAMES:
        choice = spectral_choice_from_similarity(result[name])
        choices[name] = {
            **choice,
            "selected_h": float(h_grid[int(choice["selected_index"])]),
        }
    result["choices"] = choices
    return result


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


def doob_stationary_mass(matrix: np.ndarray) -> np.ndarray:
    """Return stationary mass proportional to left-times-right Perron vectors."""
    right = power_perron_vector(matrix)
    left = power_perron_vector(np.asarray(matrix).T)
    mass = left * right
    return mass / mass.sum()


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
