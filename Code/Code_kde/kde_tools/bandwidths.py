from __future__ import annotations

import numpy as np

from .density import gaussian_mixture_mean_and_variance, trapezoidal_integral

# classical selectors

def silverman_rule_of_thumb_bandwidth(samples: np.ndarray) -> float:
    """Silverman's robust rule of thumb, ``0.9 A n^(-1/5)``."""
    samples = np.asarray(samples, dtype=float)
    return 0.9 * robust_scale_for_bandwidth(samples) * len(samples) ** (-1.0 / 5.0)

def least_squares_cv_bandwidth_index(
    samples: np.ndarray,
    h_grid: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Select a Gaussian-KDE bandwidth by exact least-squares cross-validation."""
    samples = np.asarray(samples, dtype=float)
    h_grid = np.asarray(h_grid, dtype=float)
    n = len(samples)
    squared_differences = (samples[:, None] - samples[None, :]) ** 2
    off_diagonal = ~np.eye(n, dtype=bool)
    scores = []

    for h in h_grid:
        variance = 2.0 * h**2
        integral_squared = np.exp(-squared_differences / (2.0 * variance)).sum()
        integral_squared /= n**2 * np.sqrt(2.0 * np.pi * variance)

        off_diagonal_sum = np.exp(
            -squared_differences[off_diagonal] / (2.0 * h**2)
        ).sum()
        leave_one_out = 2.0 * off_diagonal_sum
        leave_one_out /= n * (n - 1) * h * np.sqrt(2.0 * np.pi)
        scores.append(integral_squared - leave_one_out)

    scores = np.asarray(scores)
    return int(np.argmin(scores)), scores

def direct_plugin_bandwidth(
    samples: np.ndarray,
    x_grid: np.ndarray,
    pilot_h: float | None = None,
) -> float:
    """Estimate an AMISE plug-in bandwidth using a pilot estimate of ``R(f'')``."""
    samples = np.asarray(samples, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    n = len(samples)
    if pilot_h is None:
        pilot_h = 1.06 * robust_scale_for_bandwidth(samples) * n ** (-1.0 / 5.0)
    if not np.isfinite(pilot_h) or pilot_h <= 0:
        return silverman_rule_of_thumb_bandwidth(samples)

    z = (x_grid[None, :] - samples[:, None]) / pilot_h
    kernel = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    second_derivative = ((z**2 - 1.0) * kernel).mean(axis=0) / pilot_h**3
    roughness = float(trapezoidal_integral(second_derivative**2, x_grid, axis=0))
    if not np.isfinite(roughness) or roughness <= 0:
        return silverman_rule_of_thumb_bandwidth(samples)

    kernel_roughness = 1.0 / (2.0 * np.sqrt(np.pi))
    return float((kernel_roughness / (n * roughness)) ** (1.0 / 5.0))


# computes h_oracle given that f ~ N(0,1) using nested intervals
def gaussian_mise_oracle_bandwidth(sample_size: int) -> float:
    root_pi = np.sqrt(np.pi)

    def mise(bandwidth: float) -> float:
        integrated_kernel_square = 1.0 / (2.0 * root_pi)
        integrated_smoothed_square = 1.0 / (2.0 * root_pi * np.sqrt(1.0 + bandwidth**2))
        smoothed_truth_overlap = 1.0 / np.sqrt(2.0 * np.pi * (2.0 + bandwidth**2))
        integrated_truth_square = 1.0 / (2.0 * root_pi)
        return (
            integrated_kernel_square / (sample_size * bandwidth)
            + (1.0 - 1.0 / sample_size) * integrated_smoothed_square
            - 2.0 * smoothed_truth_overlap
            + integrated_truth_square
        )

    left, right = 1e-8, 5.0
    golden_ratio = (np.sqrt(5.0) - 1.0) / 2.0
    inner_right = left + golden_ratio * (right - left)
    inner_left = right - golden_ratio * (right - left)
    value_left = mise(inner_left)
    value_right = mise(inner_right)
    for _ in range(120):
        if value_left < value_right:
            right, inner_right, value_right = inner_right, inner_left, value_left
            inner_left = right - golden_ratio * (right - left)
            value_left = mise(inner_left)
        else:
            left, inner_left, value_left = inner_left, inner_right, value_right
            inner_right = left + golden_ratio * (right - left)
            value_right = mise(inner_right)
    return float((left + right) / 2.0)


# continue here


def bandwidth_grid_for_gaussian_experiment(
    n: int,
    sigma2: float,
    n_h: int = 80,
    h_min_factor: float = 0.25,
    h_max_factor: float = 2.50,
) -> np.ndarray:
    """Return a logarithmic grid centered on the one-dimensional KDE rate."""
    center = float(np.sqrt(sigma2)) * n ** (-1.0 / 5.0)
    return np.geomspace(h_min_factor * center, h_max_factor * center, int(n_h))


def bandwidth_grid_for_gaussian_mixture_experiment(
    n: int,
    weights: np.ndarray,
    means: np.ndarray,
    sigmas: np.ndarray,
    n_h: int = 80,
    h_min_factor: float = 0.08,
    h_max_factor: float = 2.50,
) -> np.ndarray:
    """Return a logarithmic grid scaled by the target mixture variance."""
    _, variance = gaussian_mixture_mean_and_variance(weights, means, sigmas)
    center = np.sqrt(variance) * n ** (-1.0 / 5.0)
    return np.geomspace(h_min_factor * center, h_max_factor * center, int(n_h))


def nearest_bandwidth_grid_index(h_value: float, h_grid: np.ndarray) -> int:
    """Return the index nearest to a positive bandwidth on a logarithmic scale."""
    h_value = float(h_value)
    h_grid = np.asarray(h_grid, dtype=float)
    if not np.isfinite(h_value) or h_value <= 0:
        return int(len(h_grid) // 2)
    return int(np.argmin(np.abs(np.log(h_grid) - np.log(h_value))))


def robust_scale_for_bandwidth(samples: np.ndarray) -> float:
    """Return ``min(sample standard deviation, IQR / 1.349)``."""
    samples = np.asarray(samples, dtype=float)
    sample_std = float(np.std(samples, ddof=1)) if len(samples) > 1 else 1.0
    q25, q75 = np.quantile(samples, [0.25, 0.75])
    iqr_scale = float((q75 - q25) / 1.349) if q75 > q25 else sample_std
    scale = min(sample_std, iqr_scale)
    if not np.isfinite(scale) or scale <= 0:
        scale = sample_std if sample_std > 0 else 1.0
    return float(scale)
