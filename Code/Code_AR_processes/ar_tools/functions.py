from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .support_geometry import (
    ar_impulse_response,
    ar_window_zonotope_generators,
    fibonacci_sphere_directions,
    theoretical_ar_width_in_direction,
    theoretical_support_zonoid_2d,
    theoretical_width_2d_from_angle,
    zonotope_polygon_2d,
    zonotope_support_points,
)

__all__ = [
    "jones_ar_coefficients",
    "centered_beta_support",
    "simulate_beta_ar",
    "simulate_uniform_ar",
    "sliding_window_point_cloud",
    "ar_impulse_response",
    "theoretical_ar_width_in_direction",
    "companion_matrix",
    "is_stationary",
    "theoretical_width_2d_from_angle",
    "ar_window_zonotope_generators",
    "zonotope_polygon_2d",
    "fibonacci_sphere_directions",
    "zonotope_support_points",
    "theoretical_support_zonoid_2d",
    "stationarity_root_margin",
    "coefficients_from_minimizing_direction",
]


### work through later:
def jones_ar_coefficients(
    p: int,
    rng: np.random.Generator | int | None = None,
    *,
    return_pacf: bool = False,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
    if not isinstance(p, (int, np.integer)) or p < 1:
        raise ValueError(f"p must be a positive integer, got {p}")

    generator = np.random.default_rng(rng)
    lag_indices = np.arange(1, p + 1)
    beta_shape_a = (lag_indices + 1) // 2
    beta_shape_b = lag_indices // 2 + 1
    beta_draws = generator.beta(beta_shape_a, beta_shape_b)
    pacf = 2.0 * beta_draws - 1.0

    phi = np.array([pacf[0]], dtype=float)
    for order in range(2, p + 1):
        reflection_coefficient = pacf[order - 1]
        previous_phi = phi
        phi = np.empty(order, dtype=float)
        phi[:-1] = (
            previous_phi - reflection_coefficient * previous_phi[::-1]
        )
        phi[-1] = reflection_coefficient

    if return_pacf:
        return phi, pacf
    return phi


### work through later:
def centered_beta_support(
    alpha: float = 1.0,
    beta: float = 1.0,
    support_width: float = 2.0,
) -> tuple[float, float]:
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be a positive finite number")
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be a positive finite number")
    if not np.isfinite(support_width) or support_width <= 0:
        raise ValueError("support_width must be a positive finite number")

    beta_mean = alpha / (alpha + beta)
    return (
        float(-support_width * beta_mean),
        float(support_width * (1.0 - beta_mean)),
    )


### work through later:
def simulate_beta_ar(
    phi: ArrayLike,
    sample_size: int,
    burnin: int = 500,
    support_width: float = 2.0,
    alpha: float = 1.0,
    beta: float = 1.0,
    rng: np.random.Generator | int | None = None,
    *,
    intercept: float = 0.0,
) -> NDArray[np.float64]:
    phi = np.asarray(phi, dtype=float)
    if phi.ndim != 1 or len(phi) == 0 or not np.all(np.isfinite(phi)):
        raise ValueError("phi must be a nonempty finite one-dimensional vector")
    if not isinstance(sample_size, (int, np.integer)) or sample_size < 1:
        raise ValueError("sample_size must be a positive integer")
    if not isinstance(burnin, (int, np.integer)) or burnin < 0:
        raise ValueError("burnin must be a non-negative integer")
    if not np.isfinite(intercept):
        raise ValueError("intercept must be finite")

    innovation_lower, innovation_upper = centered_beta_support(
        alpha=alpha,
        beta=beta,
        support_width=support_width,
    )

    generator = np.random.default_rng(rng)
    order = len(phi)
    total_length = burnin + order + sample_size
    if alpha == 1.0 and beta == 1.0:
        innovations = generator.uniform(
            innovation_lower,
            innovation_upper,
            size=total_length,
        )
    else:
        beta_draws = generator.beta(alpha, beta, size=total_length)
        innovations = support_width * (
            beta_draws - alpha / (alpha + beta)
        )
    series = np.empty(total_length, dtype=float)
    series[:order] = intercept / (1.0 - np.sum(phi))

    for time_index in range(order, total_length):
        recent_values = series[time_index - np.arange(1, order + 1)]
        series[time_index] = intercept + phi @ recent_values + innovations[time_index]

    return series[burnin + order:]


def simulate_uniform_ar(
    phi: ArrayLike,
    sample_size: int,
    burnin: int = 500,
    half_width: float = 1.0,
    rng: np.random.Generator | int | None = None,
    *,
    intercept: float = 0.0,
) -> NDArray[np.float64]:
    if not np.isfinite(half_width) or half_width <= 0:
        raise ValueError("half_width must be a positive finite number")
    return simulate_beta_ar(
        phi=phi,
        sample_size=sample_size,
        burnin=burnin,
        support_width=2.0 * half_width,
        alpha=1.0,
        beta=1.0,
        rng=rng,
        intercept=intercept,
    )


# explain output mathematically
def sliding_window_point_cloud(
        series: ArrayLike, # why not np.ndarray?
        embedding_dimension: int,
        *, # what is the asterisk for?
        newest_first: bool = True,
) -> NDArray[np.float64]: # why not np.ndarray?
    series = np.asarray(series, dtype=float)

    ### input checks ###
    if series.ndim != 1:
        raise ValueError("series must be a one-dimensional array")
    if (
        not isinstance(embedding_dimension, (int, np.integer))
        or not 1 <= embedding_dimension <= len(series)
    ):
        raise ValueError(
            "embedding_dimension must be a positive integer less than or "
            "equal to the length of the series"
        )
    ### end input checks ###

    dimension = int(embedding_dimension)
    window_count = len(series) - dimension + 1
    offsets = (
        range(dimension -1, -1, -1)
        if newest_first
        else range(dimension)
    )
    return np.column_stack(
        [series[offset:window_count + offset] for offset in offsets]
    )


### companion matrix and stationarity check
def companion_matrix(phi):
    phi = np.asarray(phi, dtype=float)
    order = len(phi)
    matrix = np.zeros((order, order))
    matrix[0, :] = phi
    if order > 1:
        matrix[1:, :-1] = np.eye(order - 1)
    return matrix

def is_stationary(phi):
    eigenvalues = np.linalg.eigvals(companion_matrix(phi))
    return bool(np.all(np.abs(eigenvalues) < 1)), eigenvalues


def stationarity_root_margin(row, *, order):
    coefficients = np.array([
        row[f"true_phi_{index}"] for index in range(1, order + 1)
    ])
    roots = np.roots(np.concatenate((-coefficients[::-1], [1.0])))
    return float(np.min(abs(roots)) - 1.0)


def coefficients_from_minimizing_direction(direction):
    """Recover phi_j = -u_j / u_0 as in Remark 3."""
    direction = np.asarray(direction, dtype=float)
    if direction.ndim != 1 or len(direction) < 2:
        raise ValueError("direction must contain u_0 and at least one lag coefficient")
    if np.isclose(direction[0], 0.0):
        raise ValueError("cannot recover coefficients when u_0 is zero")
    return -direction[1:] / direction[0]
