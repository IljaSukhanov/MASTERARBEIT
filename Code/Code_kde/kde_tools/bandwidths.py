from __future__ import annotations

import numpy as np

from .density import trapezoidal_integral

# classical selectors

_STANDARD_NORMAL_UPPER_QUARTILE = 0.6744897501960817


def estimate_sample_scale(
    samples: np.ndarray,
    method: str = "iqr",
) -> float:
    """Estimate a sample's physical scale without selecting a KDE bandwidth.

    ``method`` may be ``"iqr"``, ``"mad"``, or ``"std"``.  The IQR and MAD
    estimates are normalized to equal the standard deviation for Gaussian
    data.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 1 or len(samples) < 2:
        raise ValueError(
            "samples must be a one-dimensional array with at least two entries."
        )
    if not np.all(np.isfinite(samples)):
        raise ValueError("samples must contain only finite values.")

    if method == "iqr":
        q25, q75 = np.quantile(samples, [0.25, 0.75])
        scale = (q75 - q25) / (2.0 * _STANDARD_NORMAL_UPPER_QUARTILE)
    elif method == "mad":
        median = np.median(samples)
        scale = (
            np.median(np.abs(samples - median))
            / _STANDARD_NORMAL_UPPER_QUARTILE
        )
    elif method == "std":
        scale = np.std(samples, ddof=1)
    else:
        raise ValueError('method must be "iqr", "mad", or "std".')

    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("The estimated sample scale must be finite and positive.")
    return float(scale)


def sample_scaled_bandwidth_grid(
    samples: np.ndarray,
    grid_size: int,
    multiplier_min: float = 0.01,
    multiplier_max: float = 2.0,
    rate_exponent: float = 1.0 / 5.0,
    scale_method: str = "iqr",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Construct ``H_n = scale_hat * n**(-rate_exponent) * C``.

    ``C`` is a geometric grid from ``multiplier_min`` to ``multiplier_max``.
    The return value is ``(bandwidths, dimensionless_grid, scale_hat)`` so an
    experiment can retain both the data-unit bandwidths and their normalized
    coordinates.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 1 or len(samples) < 2:
        raise ValueError(
            "samples must be a one-dimensional array with at least two entries."
        )
    if isinstance(grid_size, (bool, np.bool_)) or int(grid_size) != grid_size:
        raise ValueError("grid_size must be an integer.")
    grid_size = int(grid_size)
    if grid_size < 2:
        raise ValueError("grid_size must be at least two.")
    if not 0.0 < multiplier_min < multiplier_max:
        raise ValueError("Require 0 < multiplier_min < multiplier_max.")
    if not np.isfinite(rate_exponent) or rate_exponent <= 0.0:
        raise ValueError("rate_exponent must be finite and positive.")

    scale = estimate_sample_scale(samples, method=scale_method)
    dimensionless_grid = np.geomspace(
        multiplier_min,
        multiplier_max,
        grid_size,
    ) * len(samples) ** (-float(rate_exponent))
    return scale * dimensionless_grid, dimensionless_grid, scale

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


# for given h and grid h_1<...<h_N it finds i^* = argmin{|log h - log h_i|}
def nearest_bandwidth_grid_index(h_value: float, h_grid: np.ndarray) -> int:
    h_value = float(h_value)
    h_grid = np.asarray(h_grid, dtype=float)
    if not np.isfinite(h_value) or h_value <= 0:
        return int(len(h_grid) // 2)
    return int(np.argmin(np.abs(np.log(h_grid) - np.log(h_value))))
