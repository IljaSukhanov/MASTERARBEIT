from __future__ import annotations

import numpy as np


def bandwidth_grid_linear_in_psi_beta(
    h_min: float = 0.01,
    h_max: float = 2.0,
    beta: float = 2.0,
    grid_size: int = 201,
) -> np.ndarray:
    """Return a bandwidth grid that is linear in ``psi_beta``.

    Here ``psi_beta(h) = (h**(1 - beta / 2) - 1) / (1 - beta / 2)``.
    Its continuous limit at ``beta=2`` is ``log(h)``, so the default is a
    geometric grid.  The case ``beta=0`` gives an ordinary linear grid.
    """
    h_min = float(h_min)
    h_max = float(h_max)
    beta = float(beta)
    grid_size = int(grid_size)

    if not 0.0 < h_min < h_max:
        raise ValueError("Require 0 < h_min < h_max.")
    if not np.isfinite(beta):
        raise ValueError("beta must be finite.")
    if grid_size < 2:
        raise ValueError("grid_size must be at least two.")

    exponent = 1.0 - beta / 2.0
    if np.isclose(exponent, 0.0, rtol=0.0, atol=1e-12):
        return np.geomspace(h_min, h_max, grid_size)

    transformed_grid = np.linspace(
        h_min**exponent,
        h_max**exponent,
        grid_size,
    )
    return transformed_grid ** (1.0 / exponent)


#############################################
### oracle bandwidth (ultimate benchmark) ###
#############################################

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
