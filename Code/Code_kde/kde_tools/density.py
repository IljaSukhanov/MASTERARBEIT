from __future__ import annotations

import numpy as np

# 0 ... worked through
# 1 ... inspect later


# 1
# computes \hat{f}_{h_j}(x_k) for all j and k
def gaussian_kde_values(
        samples: np.ndarray,
        h_grid: np.ndarray,
        x_grid: np.ndarray,
) -> np.ndarray:
    samples = np.asarray(samples, dtype=float)
    h_grid = np.asarray(h_grid, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    if samples.ndim != 1 or len(samples) == 0:
        raise ValueError("samples must be a nonempty one-dimensional array.")
    if h_grid.ndim != 1 or len(h_grid) == 0:
        raise ValueError("h_grid must be a nonempty one-dimensional array.")
    if np.any(h_grid <= 0.0):
        raise ValueError("bandwidths must be strictly positive.")
    if x_grid.ndim != 1 or len(x_grid) < 2:
        raise ValueError("x_grid must contain at least two points.")

    # Work one bandwidth at a time.  The previous fully broadcasted form used
    # an N-by-n-by-m temporary array, which made ordinary quadrature needlessly
    # memory intensive for the main consistency experiments.
    densities = np.empty((len(h_grid), len(x_grid)), dtype=float)
    normalization = np.sqrt(2.0 * np.pi)
    differences = x_grid[None, :] - samples[:, None]
    for index, bandwidth in enumerate(h_grid):
        z = differences / bandwidth
        densities[index] = np.exp(-0.5 * z**2).mean(axis=0)
        densities[index] /= normalization * bandwidth
    return densities


# 0
def exact_gaussian_kde_l2_gram(
        samples: np.ndarray,
        h_grid: np.ndarray,
) -> np.ndarray:
    samples = np.asarray(samples, dtype=float)
    h_grid = np.asarray(h_grid, dtype=float)
    n = len(samples)
    squared_differences = (samples[:, None] - samples[None, :]) ** 2 # (X_i - X_j)^2
    gram = np.empty((len(h_grid), len(h_grid)), dtype=float)

    for i, h_i in enumerate(h_grid):
        for j, h_j in enumerate(h_grid[i:], start=i):
            variance = h_i**2 + h_j**2
            value = np.exp(-squared_differences / (2.0 * variance)).sum()
            value /= n**2 * np.sqrt(2.0 * np.pi * variance)
            gram[i, j] = value
            gram[j, i] = value
    return gram

# 0
# this function computes the true integrated squared error between \hat{f}_h and f, where f is the density of N(0,sigma2) for a given grid of bandwidths h_grid.
def exact_oracle_ise_values(
    samples: np.ndarray,
    h_grid: np.ndarray,
    sigma2: float,
) -> np.ndarray:
    """Exact ISE of Gaussian KDEs against the target ``N(0, sigma2)``."""
    samples = np.asarray(samples, dtype=float)
    h_grid = np.asarray(h_grid, dtype=float)
    sigma = float(np.sqrt(sigma2))
    n = len(samples)
    squared_differences = (samples[:, None] - samples[None, :]) ** 2

    kde_squared_integrals = []
    cross_integrals = []
    for h in h_grid:
        variance = 2.0 * h**2
        kde_squared = np.exp(-squared_differences / (2.0 * variance)).sum()
        kde_squared /= n**2 * np.sqrt(2.0 * np.pi * variance)
        kde_squared_integrals.append(kde_squared)

        variance = sigma2 + h**2
        cross = np.exp(-samples**2 / (2.0 * variance)).mean()
        cross /= np.sqrt(2.0 * np.pi * variance)
        cross_integrals.append(cross)

    true_squared_integral = 1.0 / (2.0 * np.sqrt(np.pi) * sigma)
    return (
        np.asarray(kde_squared_integrals)
        - 2.0 * np.asarray(cross_integrals)
        + true_squared_integral
    )
