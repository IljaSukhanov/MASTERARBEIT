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


# is this function needed?
def trapezoidal_integral(
    values: np.ndarray,
    x_grid: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = np.trapz
    return trapezoid(y=values, x=x_grid, axis=axis)

### construct the quadrature nodes to approximate int \hat{f}_h dx for a given grid of bandwidths h_grid and samples X_i ###
def quadrature_grid_for_kdes(
    samples: np.ndarray,
    h_grid: np.ndarray,
    m_x: int = 2_001,
    padding: float = 6.0,
) -> np.ndarray:
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


def quadrature_js_divergence_matrix(
    density_values: np.ndarray,
    x_grid: np.ndarray,
    eps: float = 1e-300,
) -> np.ndarray:
    r"""Approximate pairwise JS divergences by composite trapezoidal integration.

    For p, q and m = (p + q)/2, integrate
    (p log(p/m) + q log(q/m))/2 on the supplied spatial grid. Natural
    logarithms are used. The implementation expands this as
    (p log(p) + q log(q))/2 - m log(m), caching the individual integrals.
    Zero-density terms contribute zero; eps protects logarithms in the tails.

    See eq:JS_KL_formula and the composite trapezoidal rule in
    Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex, and
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.trapezoid.html

    Densities are not renormalized on the finite interval. The returned matrix
    is symmetric with zero diagonal. Roundoff below zero is clipped; values
    are not capped at log(2), so an underresolved grid remains diagnosable.
    Tail truncation and spatial discretization errors are not certified here;
    compare wider intervals and finer grids to assess convergence.
    """
    values = np.asarray(density_values, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    weights = _trapezoidal_weights(x_grid)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("density_values must be a nonempty two-dimensional array.")
    if values.shape[1] != len(x_grid):
        raise ValueError("x_grid must match the density grid axis.")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Density values must be finite and nonnegative.")
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and strictly positive.")

    integrated_xlogx = (values * np.log(np.maximum(values, eps))) @ weights
    divergence = np.zeros((len(values), len(values)), dtype=float)
    for i in range(len(values) - 1):
        mixtures = 0.5 * values[i] + 0.5 * values[i + 1:]
        mixture_xlogx = (mixtures * np.log(np.maximum(mixtures, eps))) @ weights
        pair_values = np.maximum(
            0.5 * integrated_xlogx[i] + 0.5 * integrated_xlogx[i + 1:]
            - mixture_xlogx,
            0.0,
        )
        divergence[i, i + 1:] = pair_values
        divergence[i + 1:, i] = pair_values
    return divergence


### is this function still needed when we have a faster Fourier-based method? ###
def quadrature_beta2_divergence_matrix(
    density_values: np.ndarray, # matrix F of shape (len(h_grid), len(x_grid)) with entries F[j, k] = \hat{f}_{h_j}(x_k)
    x_grid: np.ndarray,         # x_grid [x_0, x_1, ..., x_{m-1}] constructed by quadrature_grid_for_kdes
) -> np.ndarray:
    values = np.asarray(density_values, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    if values.ndim != 2:
        raise ValueError("density_values must be a two-dimensional array.")
    if values.shape[1] != len(x_grid):
        raise ValueError("x_grid must match the density grid axis.")
    if np.any(values < 0.0):
        raise ValueError("Density values must be nonnegative.")

    weights = _trapezoidal_weights(x_grid)          # trapezoidal weights [w_0, w_1, ..., w_{m-1}]
    gram = (values * weights[None, :]) @ values.T   # gram[j, k] = \sum_{l=0}^{m-1}w_l \hat{f}_{h_j}(x_l) \hat{f}_{h_k}(x_l) \approx \int \hat{f}_{h_j}(x) \hat{f}_{h_k}(x) dx
    divergence = 0.5 * squared_l2_distance_from_gram(gram=gram)
    np.fill_diagonal(divergence, 0.0)
    return divergence


### inspect later ###
def quadrature_similarity_matrices(
    density_values: np.ndarray,
    x_grid: np.ndarray,
    similarity_name: str,
    lambda_value: float = 1.0,
    eps: float = 1e-300,
) -> np.ndarray:
    """Compute similarities by composite trapezoidal integration on x_grid."""
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
    lambda_value = float(lambda_value)
    if not np.isfinite(lambda_value) or lambda_value <= 0.0:
        raise ValueError("lambda_value must be finite and strictly positive.")

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
        matrix = np.exp(-lambda_value * d2)
    else:
        divergence = quadrature_js_divergence_matrix(values, x_grid, eps=eps)
        matrix = 1.0 - divergence / np.log(2.0)

    return np.clip(matrix, 0.0, 1.0)


### inspect later ###
def compute_quadrature_similarity_matrices(
    samples: np.ndarray,
    h_grid: np.ndarray,
    similarity_name: str,
    x_grid: np.ndarray | None = None,
    m_x: int = 2_001,
    lambda_value: float = 1.0,
) -> dict[str, np.ndarray]:
    """Construct KDE similarities on a shared composite trapezoidal grid."""
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
        lambda_value=lambda_value,
    )
    return {
        similarity_name: matrix,
        "x_grid": x_grid,
    }


### inspect later ###
def select_quadrature_similarity_bandwidths(
    samples: np.ndarray,
    h_grid: np.ndarray,
    similarity_name: str,
    x_grid: np.ndarray | None = None,
    m_x: int = 2_001,
    lambda_value: float = 1.0,
) -> dict[str, object]:
    result = compute_quadrature_similarity_matrices(
        samples=samples,
        h_grid=h_grid,
        similarity_name=similarity_name,
        x_grid=x_grid,
        m_x=m_x,
        lambda_value=lambda_value,
    )
    choice = spectral_choice_from_similarity(
        similarity_matrix=result[similarity_name]
    )
    result["choice"] = {
        **choice,
        "selected_h": float(h_grid[int(choice["selected_index"])]),
    }
    return result


# can we make it a regular function instead of a private one?
### gives the trapezoidal weights w_k = (x_{k+1} - x_{k-1}) / 2 for k = 1, ..., m-2 and w_0 = (x_1 - x_0) / 2, w_{m-1} = (x_{m-1} - x_{m-2}) / 2 ###
def _trapezoidal_weights(x_grid: np.ndarray) -> np.ndarray:
    x_grid = np.asarray(x_grid, dtype=float)
    if x_grid.ndim != 1 or len(x_grid) < 2 or not np.all(np.isfinite(x_grid)):
        raise ValueError("x_grid must be finite and one-dimensional with at least two nodes.")

    steps = np.diff(x_grid)
    if np.any(steps <= 0.0):
        raise ValueError("x_grid must be strictly increasing.")

    weights = np.empty_like(x_grid)
    weights[0] = steps[0] / 2.0
    weights[-1] = steps[-1] / 2.0
    weights[1:-1] = (steps[:-1] + steps[1:]) / 2.0
    return weights
