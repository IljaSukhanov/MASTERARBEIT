from __future__ import annotations

import numpy as np

# 0 ... worked through
# 1 ... inspect later


# 1
def trapezoidal_integral(
    values: np.ndarray,
    x_grid: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return trapezoid(y=values, x=x_grid, axis=axis)


# 1
# input: x = (x_1, ..., x_n), mean, variance; output: phi_{mean,variance}(x)
def gaussian_density_with_variance(
        x: np.ndarray | float,
        mean: float,
        variance: float,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * (x - mean) ** 2 / variance) / np.sqrt(2.0 * np.pi * variance)


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


# 1
# computes grid on which to evaluate the KDEs: a=x_1<...<x_m=b, where a = min(samples) - padding * max(h_grid), b = max(samples) + padding * max(h_grid)
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

# 3
def exact_mixture_ise_values(
        samples: np.ndarray,
        h_grid: np.ndarray,
        weights: np.ndarray,
        means: np.ndarray,
        sigmas: np.ndarray,
) -> np.ndarray:
    samples = np.asarray(samples, dtype=float)
    h_grid = np.asarray(h_grid, dtype=float)
    weights = np.asarray(weights, dtype=float) / np.sum(weights)
    means = np.asarray(means, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    n = len(samples)
    squared_differences = (samples[:, None] - samples[None, :]) ** 2 # (X_i - X_j)^2

    kde_squared_integrals = []
    cross_integrals = []
    for h in h_grid:
        variance = 2.0 * h**2
        kde_squared = np.exp(-squared_differences / (2.0 * variance)).sum()
        kde_squared /= n**2 * np.sqrt(2.0 * np.pi * variance)
        kde_squared_integrals.append(kde_squared)

        cross = 0.0
        for weight, mean, sigma in zip(weights, means, sigmas):
            cross += weight * gaussian_density_with_variance(x=samples, mean=mean, variance=h**2 + sigma**2).mean()
        cross_integrals.append(cross)

    target_squared_integral = 0.0
    for weight_a, mean_a, sigma_a in zip(weights, means, sigmas):
        for weight_b, mean_b, sigma_b in zip(weights, means, sigmas):
            target_squared_integral += (weight_a * weight_b * gaussian_density_with_variance(x=mean_a, mean=mean_b, variance=sigma_a**2 + sigma_b**2))

    return np.asarray(kde_squared_integrals) - 2.0 * np.asarray(cross_integrals) + target_squared_integral



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
