from __future__ import annotations

import numpy as np

__all__ = [
    "ar_impulse_response",
    "theoretical_ar_width_in_direction",
    "theoretical_width_2d_from_angle",
    "ar_window_zonotope_generators",
    "zonotope_polygon_2d",
    "fibonacci_sphere_directions",
    "zonotope_support_points",
    "theoretical_support_zonoid_2d",
]


def ar_impulse_response(phi, maximum_lag):
    phi = np.asarray(phi, dtype=float)
    order = len(phi)
    psi = np.zeros(maximum_lag + 1)
    psi[0] = 1.0
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
    direction = np.asarray(direction, dtype=float)
    direction_norm = np.linalg.norm(direction)

    if direction.ndim != 1 or len(direction) == 0:
        raise ValueError("direction must be a nonempty one-dimensional array.")
    if direction_norm == 0:
        raise ValueError("direction must be nonzero.")
    if half_width <= 0 or history < 0:
        raise ValueError("Require half_width > 0 and history >= 0.")

    unit_direction = direction / direction_norm
    psi = ar_impulse_response(phi=phi, maximum_lag=history)
    generator_projections = np.convolve(a=psi, v=unit_direction)
    return float(
        2.0 * half_width * np.sum(np.abs(generator_projections[: history + 1]))
    )


def theoretical_width_2d_from_angle(
    angle,
    impulse_response,
    lagged_impulse_response,
    half_width=1.0,
):
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
    signs = np.sign(generators.T @ directions.T)
    signs[signs == 0.0] = 1.0
    return (generators @ signs).T


def theoretical_support_zonoid_2d(phi, innovation_bounds, history=2_000):
    phi = np.asarray(phi, dtype=float)
    if phi.ndim != 1 or len(phi) == 0 or not np.all(np.isfinite(phi)):
        raise ValueError("phi must be a nonempty finite one-dimensional vector.")
    if np.any(np.abs(np.roots(np.r_[1.0, -phi])) >= 1.0):
        raise ValueError("phi must define a stationary AR process.")

    bounds = np.asarray(innovation_bounds, dtype=float)
    if (
        bounds.shape != (2,)
        or not np.all(np.isfinite(bounds))
        or bounds[0] > bounds[1]
    ):
        raise ValueError("innovation_bounds must be finite ordered endpoints.")
    lower, upper = bounds
    half_width = 0.5 * (upper - lower)
    midpoint = 0.5 * (lower + upper)
    centre = np.full(2, midpoint / (1.0 - np.sum(phi)))

    if len(phi) == 1:
        # Past-innovation generators are collinear even for negative phi.
        coefficient = float(phi[0])
        generators = half_width * np.column_stack((
            np.array([1.0, coefficient]) / (1.0 - abs(coefficient)),
            np.array([0.0, 1.0]),
        ))
    else:
        if not isinstance(history, (int, np.integer)) or history < 1:
            raise ValueError("history must be a positive integer.")
        # The generator routine orders coordinates newest first.
        generators = ar_window_zonotope_generators(
            phi, embedding_dimension=2, history=history,
            half_width=half_width,
        )[::-1]

    boundary = zonotope_polygon_2d(generators) + centre
    boundary[-1] = boundary[0]
    return boundary
