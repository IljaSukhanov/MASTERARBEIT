from __future__ import annotations

import numpy as np

from .density import (
    exact_gaussian_kde_l2_gram,
    gaussian_kde_values,
    quadrature_grid_for_kdes,
    trapezoidal_integral,
)
from .fft import kde_family_fft


SIMILARITY_NAMES = (
    "tv_overlap",
    "hellinger_affinity",
    "exponential_l2_similarity",
    "js_similarity"
)



def squared_l2_distance_from_gram(gram: np.ndarray) -> np.ndarray:
    diagonal = np.diag(gram)
    return np.maximum(diagonal[:, None] + diagonal[None, :] - 2.0 * gram, 0.0)


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


def beta2_divergence_matrix_fft(
        samples: np.ndarray,
        h_grid: np.ndarray,
        half_width: float = 12.0,
        spatial_points: int = 8192,
) -> np.ndarray:
    """Approximate the beta=2 divergence matrix by periodic FFT quadrature.

    The KDE family is evaluated once on a common spatial grid. Its L2 Gram
    matrix is then obtained by the matrix product ``dx * densities @
    densities.T``. This avoids the sample-pair sum in the analytic formula and
    is substantially faster for large samples or bandwidth grids.
    """
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
    """Return ``exp(-tau * D_2)`` using FFT-based beta=2 quadrature."""
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
        gamma: float = 1.0,
) -> dict[str, np.ndarray | int | float]:
    d2 = beta2_divergence_matrix(samples=samples, h_grid=h_grid)
    l2_squared = 2.0 * d2
    similarity = np.exp(-float(gamma) * d2)
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
    matrix = quadrature_similarity_matrices(density_values=density_values, x_grid=x_grid, similarity_name=similarity_name, gamma=gamma,)
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
    choice = spectral_choice_from_similarity(similarity_matrix=result[similarity_name])
    result["choice"] = {
        **choice,
        "selected_h": float(h_grid[int(choice["selected_index"])])
    }
    return result


# computes S=(S_{ij}), S_{ij}=exp{-D_{ij}}, D_{ij}=D_{KL}(\hat{f}_i,\hat{f}_j)
def directional_kl_similarity(densities: np.ndarray, dx: float) -> np.ndarray:
    densities = np.maximum(np.asarray(densities, dtype=float), 1e-300)
    log_densities = np.log(densities)
    x_grid = np.arange(densities.shape[1] + 1, dtype=float) * float(dx)
    periodic_densities = np.concatenate((densities, densities[:, :1]), axis=1)
    periodic_log_densities = np.concatenate((log_densities, log_densities[:, :1]), axis=1)
    own_log_density = trapezoidal_integral(values=periodic_densities * periodic_log_densities, x_grid=x_grid, axis=1)
    cross_log_density = np.empty((len(densities), len(densities)), dtype=float)
    for j, log_density in enumerate(periodic_log_densities):
        cross_log_density[:, j] = trapezoidal_integral(values=periodic_densities * log_density[None, :], x_grid=x_grid, axis=1)
    divergence = np.maximum(own_log_density[:, None] - cross_log_density, 0.0)
    similarity = np.exp(-divergence)
    return similarity


# computes Toeplitz projection S^{(0)} for any matrix S
def toeplitz_projection(matrix: np.ndarray, symmetric: bool=True) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    n = matrix.shape[0]
    indices = np.arange(n)
    if symmetric:
        lag_means = np.array([np.diagonal(matrix / 2.0, offset=d).mean() for d in range(n)])
        return lag_means[np.abs(indices[:, None] - indices[None, :])]
    lag_means = {offset: float(np.diagonal(matrix, offset=offset).mean()) for offset in range(-(n - 1), n)}
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
    null = toeplitz_projection(matrix=similarity, symmetric=not directed)
    if directed:
        score = doob_stationary_mass(similarity) / doob_stationary_mass(null)
    else:
        score = power_perron_vector(similarity) / power_perron_vector(null)
    index = int(np.argmax(score))
    return float(bandwidths[index]), index, score


################################################################################
# Rényi divergence and similarity functions
################################################################################


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


def _log_trapezoidal_integral(
        log_values: np.ndarray,
        x_grid: np.ndarray,
        axis: int = -1,
) -> np.ndarray:
    """Evaluate a trapezoidal integral stably from logarithmic values."""
    log_values = np.asarray(log_values, dtype=float)
    axis = np.core.numeric.normalize_axis_index(axis, log_values.ndim)
    weights = _trapezoidal_weights(x_grid)
    if log_values.shape[axis] != len(weights):
        raise ValueError("x_grid must match the selected integration axis.")

    weight_shape = [1] * log_values.ndim
    weight_shape[axis] = len(weights)
    weighted_log_values = log_values + np.log(weights).reshape(weight_shape)
    maximum = np.max(weighted_log_values, axis=axis, keepdims=True)
    return np.squeeze(
        maximum
        + np.log(
            np.sum(
                np.exp(weighted_log_values - maximum),
                axis=axis,
                keepdims=True,
            )
        ),
        axis=axis,
    )


def renyi_divergence_matrix(
        density_values: np.ndarray,
        x_grid: np.ndarray,
        alpha: float,
        bandwidths: np.ndarray | None = None,
        eps: float = 1e-300,
) -> np.ndarray:
    """Compute all directed Rényi divergences by stable quadrature.

    Rows and columns correspond to densities, so entry ``(i, j)`` is
    ``R_alpha(p_i || p_j)``. At ``alpha = 1`` the KL limit is used.
    Supplying Gaussian-KDE bandwidths also detects divergent full-line pairs
    when ``alpha > 1``.
    """
    alpha = float(alpha)
    if alpha <= 0.0:
        raise ValueError("alpha must be strictly positive.")
    if eps <= 0.0:
        raise ValueError("eps must be strictly positive.")

    values = np.asarray(density_values, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    if values.ndim != 2:
        raise ValueError("density_values must be a two-dimensional array.")
    if values.shape[1] != len(x_grid):
        raise ValueError("x_grid must match the density grid axis.")
    if np.any(values < 0.0):
        raise ValueError("Density values must be nonnegative.")

    values = np.maximum(values, eps)
    log_values = np.log(values)
    number_of_densities = len(values)
    divergence = np.empty((number_of_densities, number_of_densities), dtype=float)

    if np.isclose(alpha, 1.0, rtol=0.0, atol=1e-12):
        for i in range(number_of_densities):
            divergence[i] = trapezoidal_integral(
                values=values[i][None, :]
                * (log_values[i][None, :] - log_values),
                x_grid=x_grid,
                axis=1,
            )
    else:
        for i in range(number_of_densities):
            log_integrands = (
                alpha * log_values[i][None, :]
                + (1.0 - alpha) * log_values
            )
            log_integrals = _log_trapezoidal_integral(
                log_values=log_integrands,
                x_grid=x_grid,
                axis=1,
            )
            divergence[i] = log_integrals / (alpha - 1.0)

    finite = np.isfinite(divergence)
    divergence[finite] = np.maximum(divergence[finite], 0.0)
    np.fill_diagonal(divergence, 0.0)

    if alpha > 1.0 and bandwidths is not None:
        bandwidths = np.asarray(bandwidths, dtype=float)
        if bandwidths.ndim != 1 or len(bandwidths) != number_of_densities:
            raise ValueError("bandwidths must contain one value per density.")
        if np.any(bandwidths <= 0.0):
            raise ValueError("Bandwidths must be strictly positive.")

        h_i = bandwidths[:, None]
        h_j = bandwidths[None, :]
        divergent_tails = alpha / h_i**2 + (1.0 - alpha) / h_j**2 <= 0.0
        divergence[divergent_tails] = np.inf

    return divergence


def symmetrized_renyi_divergence_matrix(
        density_values: np.ndarray,
        x_grid: np.ndarray,
        alpha: float,
        bandwidths: np.ndarray | None = None,
        eps: float = 1e-300,
) -> np.ndarray:
    """Return the arithmetic symmetrization of the directed Rényi matrix."""
    divergence = renyi_divergence_matrix(
        density_values=density_values,
        x_grid=x_grid,
        alpha=alpha,
        bandwidths=bandwidths,
        eps=eps,
    )
    return 0.5 * (divergence + divergence.T)


def renyi_similarity_matrix(
        density_values: np.ndarray,
        x_grid: np.ndarray,
        alpha: float,
        gamma: float = 1.0,
        symmetrized: bool = False,
        bandwidths: np.ndarray | None = None,
        eps: float = 1e-300,
) -> np.ndarray:
    """Transform a directed or symmetrized Rényi matrix using ``exp(-gamma D)``."""
    gamma = float(gamma)
    if gamma <= 0.0:
        raise ValueError("gamma must be strictly positive.")

    divergence = renyi_divergence_matrix(
        density_values=density_values,
        x_grid=x_grid,
        alpha=alpha,
        bandwidths=bandwidths,
        eps=eps,
    )
    if symmetrized:
        divergence = 0.5 * (divergence + divergence.T)

    similarity = np.exp(-gamma * divergence)
    np.fill_diagonal(similarity, 1.0)
    return similarity


def symmetrized_renyi_similarity(
        densities: np.ndarray,
        dx: float,
        alpha: float,
        gamma: float = 1.0,
        bandwidths: np.ndarray | None = None,
        eps: float = 1e-300,
) -> np.ndarray:
    """Compute a symmetrized Rényi similarity on a periodic FFT grid."""
    densities = np.asarray(densities, dtype=float)
    if densities.ndim != 2:
        raise ValueError("densities must be a two-dimensional array.")
    if dx <= 0.0:
        raise ValueError("dx must be strictly positive.")

    x_grid = np.arange(densities.shape[1] + 1, dtype=float) * float(dx)
    periodic_densities = np.concatenate((densities, densities[:, :1]), axis=1)
    return renyi_similarity_matrix(
        density_values=periodic_densities,
        x_grid=x_grid,
        alpha=alpha,
        gamma=gamma,
        symmetrized=True,
        bandwidths=bandwidths,
        eps=eps,
    )
