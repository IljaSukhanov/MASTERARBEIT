from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

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
]


### work through later:
def jones_ar_coefficients(
    p: int,
    rng: np.random.Generator | int | None = None,
    *,
    return_pacf: bool = False,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Sample a stationary AR(p) coefficient vector using Jones' method."""
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
    """Return the support bounds of a mean-zero scaled Beta distribution."""
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
    """Simulate an AR process with centered, scaled Beta innovations.

    If ``B ~ Beta(alpha, beta)``, the innovations are
    ``support_width * (B - E[B])``. Their mean is zero and their support has
    total width ``support_width``. The defaults use the original
    ``Uniform[-1, 1]`` generator path, preserving seeded uniform experiments.
    """
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
    """Simulate an AR process with Uniform[-half_width, half_width] noise."""
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


### compute the coefficients of the MA(infinity) representation of an AR(p) process
def ar_impulse_response(phi, maximum_lag):
    phi = np.asarray(phi, dtype=float)
    order = len(phi) # p
    psi = np.zeros(maximum_lag + 1) # psi_0, psi_1, ..., psi_maximum_lag
    psi[0] = 1.0 # psi_0 = 1
    for lag in range(1, maximum_lag + 1):
        psi[lag] = sum(
            phi[j] * psi[lag - j - 1] for j in range(min(lag, order))
        )
    return psi


def theoretical_ar_width_in_direction(
        phi,
        direction,
        half_width=1.0,
        history=10_000,
):
    direction = np.asarray(direction, dtype=float) # u
    direction_norm = np.linalg.norm(direction) # ||u||

    ### input checks ###
    if direction.ndim != 1 or len(direction) == 0:
        raise ValueError("direction must be a nonempty one-dimensional array.")
    if direction_norm == 0:
        raise ValueError("direction must be nonzero.")
    if half_width <= 0 or history < 0:
        raise ValueError("Require half_width > 0 and history >= 0.")
    ### end input checks ###

    unit_direction = direction / direction_norm # u / ||u||
    psi = ar_impulse_response(phi=phi, maximum_lag=history) # psi_0, psi_1, ..., psi_history
    generator_projections = np.convolve(a=psi, v=unit_direction) # c_j = sum_{r=0}^{k-1} u_r psi_{j-r} for j = 0, 1, ..., history + k - 1
    return float(2.0 * half_width * np.sum(np.abs(generator_projections[: history + 1]))) # 2 * half_width * sum_{j=0}^{history} |c_j|

    


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


# continue here

def theoretical_width_2d_from_angle(
    angle,
    impulse_response,
    lagged_impulse_response,
    half_width=1.0,
):
    """Evaluate a two-dimensional AR support width at a given angle."""
    direction = np.array([np.cos(angle), np.sin(angle)])
    generator_projections = (
        direction[0] * impulse_response
        + direction[1] * lagged_impulse_response
    )
    return float(2.0 * half_width * np.sum(np.abs(generator_projections)))


def ar_window_zonotope_generators(
    phi,
    embedding_dimension,
    history,
    half_width=1.0,
):
    """Construct finite-history zonotope generators for an AR embedding."""
    psi = ar_impulse_response(phi, history)
    generators = np.zeros((embedding_dimension, history + 1))
    for innovation_lag in range(history + 1):
        for coordinate in range(embedding_dimension):
            response_lag = innovation_lag - coordinate
            if response_lag >= 0:
                generators[coordinate, innovation_lag] = (
                    half_width * psi[response_lag]
                )
    return generators[:, np.linalg.norm(generators, axis=0) > 1e-13]


def zonotope_polygon_2d(generators):
    """Return the ordered boundary vertices of a planar zonotope."""
    lengths = np.linalg.norm(generators, axis=0)
    angles = np.mod(np.arctan2(generators[1], generators[0]), np.pi)
    oriented = lengths[:, None] * np.column_stack((
        np.cos(angles),
        np.sin(angles),
    ))
    oriented = oriented[np.argsort(angles)]

    current = -np.sum(oriented, axis=0)
    vertices = [current.copy()]
    for generator in oriented:
        current = current + 2.0 * generator
        vertices.append(current.copy())
    for generator in oriented:
        current = current - 2.0 * generator
        vertices.append(current.copy())
    return np.asarray(vertices)


def fibonacci_sphere_directions(number_of_directions):
    """Construct approximately uniform directions on the unit sphere."""
    index = np.arange(number_of_directions)
    z = 1.0 - 2.0 * (index + 0.5) / number_of_directions
    radius = np.sqrt(1.0 - z**2)
    angle = np.pi * (3.0 - np.sqrt(5.0)) * index
    return np.column_stack((
        radius * np.cos(angle),
        radius * np.sin(angle),
        z,
    ))


def zonotope_support_points(generators, directions):
    """Return zonotope support points for the supplied directions."""
    signs = np.sign(generators.T @ directions.T)
    signs[signs == 0.0] = 1.0
    return (generators @ signs).T
