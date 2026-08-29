"""Numerical-quadrature helpers for KDE divergences and similarities."""

from __future__ import annotations

import numpy as np

from .density import gaussian_kde_values
from .similarities import (
    spectral_choice_from_similarity,
    squared_l2_distance_from_gram,
)


SIMILARITY_NAMES = (
    "tv_overlap",
    "hellinger_affinity",
    "exponential_l2_similarity",
    "js_similarity",
)


def trapezoidal_integral(
    values: np.ndarray,
    x_grid: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return trapezoid(y=values, x=x_grid, axis=axis)


def quadrature_grid_for_kdes(
    samples: np.ndarray,
    h_grid: np.ndarray,
    m_x: int = 2_001,
    padding: float = 6.0,
) -> np.ndarray:
    """Construct an integration grid covering the Gaussian KDE tails."""
    samples = np.asarray(samples, dtype=float)
    h_grid = np.asarray(h_grid, dtype=float)
    if samples.ndim != 1 or len(samples) == 0:
        raise ValueError("samples must be a nonempty one-dimensional array.")
    if h_grid.ndim != 1 or len(h_grid) == 0 or np.any(h_grid <= 0.0):
        raise ValueError("h_grid must contain strictly positive bandwidths.")
    if int(m_x) < 2:
        raise ValueError("m_x must be at least two.")
    if padding <= 0.0:
        raise ValueError("padding must be strictly positive.")
    left = float(np.min(samples)) - padding * float(np.max(h_grid))
    right = float(np.max(samples)) + padding * float(np.max(h_grid))
    return np.linspace(left, right, int(m_x))


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


def quadrature_similarity_matrices(
    density_values: np.ndarray,
    x_grid: np.ndarray,
    similarity_name: str,
    gamma: float = 1.0,
    eps: float = 1e-300,
) -> np.ndarray:
    """Compute a similarity matrix from densities evaluated on a grid."""
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


def compute_quadrature_similarity_matrices(
    samples: np.ndarray,
    h_grid: np.ndarray,
    similarity_name: str,
    x_grid: np.ndarray | None = None,
    m_x: int = 2_001,
    gamma: float = 1.0,
) -> dict[str, np.ndarray | float]:
    if x_grid is None:
        x_grid = quadrature_grid_for_kdes(
            samples=samples,
            h_grid=h_grid,
            m_x=m_x,
        )
    density_values = gaussian_kde_values(
        samples=samples,
        h_grid=h_grid,
        x_grid=x_grid,
    )
    matrix = quadrature_similarity_matrices(
        density_values=density_values,
        x_grid=x_grid,
        similarity_name=similarity_name,
        gamma=gamma,
    )
    return {
        similarity_name: matrix,
        "x_grid": x_grid,
    }


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
    choice = spectral_choice_from_similarity(
        similarity_matrix=result[similarity_name]
    )
    result["choice"] = {
        **choice,
        "selected_h": float(h_grid[int(choice["selected_index"])]),
    }
    return result


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
