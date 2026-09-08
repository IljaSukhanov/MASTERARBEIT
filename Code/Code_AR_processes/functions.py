"""Shared simulation utilities for the autoregressive-process experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linprog, minimize_scalar

__all__ = [
    "jones_ar_coefficients",
    "simulate_uniform_ar",
    "sliding_window_point_cloud",
    "ar_impulse_response",
    "projected_subgradient_empirical_width",
    "batched_geometric_width_curves",
    "theoretical_ar_width_in_direction",
    "population_minimizing_ar2_direction",
    "horizontal_tail_lad_selector",
    "companion_matrix",
    "is_stationary",
    "theoretical_width_2d_from_angle",
    "ar_window_zonotope_generators",
    "zonotope_polygon_2d",
    "fibonacci_sphere_directions",
    "zonotope_support_points",
]


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


def simulate_uniform_ar(
    phi: ArrayLike,
    sample_size: int,
    burnin: int = 500,
    half_width: float = 1.0,
    rng: np.random.Generator | int | None = None,
    *,
    intercept: float = 0.0,
) -> NDArray[np.float64]:
    """Simulate a stationary AR process with Uniform[-a, a] innovations."""
    phi = np.asarray(phi, dtype=float)
    if phi.ndim != 1 or len(phi) == 0 or not np.all(np.isfinite(phi)):
        raise ValueError("phi must be a nonempty finite one-dimensional vector")
    if not isinstance(sample_size, (int, np.integer)) or sample_size < 1:
        raise ValueError("sample_size must be a positive integer")
    if not isinstance(burnin, (int, np.integer)) or burnin < 0:
        raise ValueError("burnin must be a non-negative integer")
    if not np.isfinite(half_width) or half_width <= 0:
        raise ValueError("half_width must be a positive finite number")
    if not np.isfinite(intercept):
        raise ValueError("intercept must be finite")

    generator = np.random.default_rng(rng)
    order = len(phi)
    total_length = burnin + order + sample_size
    innovations = generator.uniform(-half_width, half_width, size=total_length)
    series = np.empty(total_length, dtype=float)
    series[:order] = intercept / (1.0 - np.sum(phi))

    for time_index in range(order, total_length):
        recent_values = series[time_index - np.arange(1, order + 1)]
        series[time_index] = intercept + phi @ recent_values + innovations[time_index]

    return series[burnin + order:]


def sliding_window_point_cloud(
    series: ArrayLike,
    embedding_dimension: int,
    *,
    newest_first: bool = True,
) -> NDArray[np.float64]:
    """Construct consecutive sliding-window vectors from a time series.

    By default, each row is ordered from newest to oldest, matching the
    minimum-width calculations. Set ``newest_first=False`` for the forward
    ordering ``(x_t, ..., x_{t+k-1})`` used by the interactive embeddings.
    """
    series = np.asarray(series, dtype=float)
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

    dimension = int(embedding_dimension)
    offsets = (
        range(dimension - 1, -1, -1)
        if newest_first
        else range(dimension)
    )
    window_count = len(series) - dimension + 1
    return np.column_stack(
        [series[offset:window_count + offset] for offset in offsets]
    )


def ar_impulse_response(phi, maximum_lag):
    phi = np.asarray(phi, dtype=float)
    order = len(phi)
    psi = np.zeros(maximum_lag + 1)
    psi[0] = 1.0
    for lag in range(1, maximum_lag + 1):
        psi[lag] = sum(
            phi[j - 1] * psi[lag - j]
            for j in range(1, min(order, lag) + 1)
        )
    return psi


def projected_subgradient_empirical_width(
    points,
    iterations=2_000,
    initial_step=0.2,
    patience=200,
    improvement_tolerance=1e-10,
    rng=None,
    initial_direction=None,
    step_schedule="inverse_sqrt",
    geometric_decay=0.99,
):
    """Minimize empirical directional width over the unit sphere.

    step_schedule="inverse_sqrt" uses eta_t = eta_0 / sqrt(t + 1).
    step_schedule="geometric" uses eta_t = eta_0 * beta**t, as in
    the local convergence result of Li et al. (Theorem 4).
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or len(points) < 2:
        raise ValueError("points must have shape (n, k) with n >= 2.")
    if iterations < 1:
        raise ValueError("iterations must be positive.")
    if initial_step <= 0:
        raise ValueError("initial_step must be positive.")
    valid_step_schedules = {"inverse_sqrt", "geometric"}
    if step_schedule not in valid_step_schedules:
        raise ValueError(
            f"step_schedule must be one of {sorted(valid_step_schedules)}."
        )
    if not 0.0 < geometric_decay < 1.0:
        raise ValueError("geometric_decay must lie strictly between zero and one.")
    if not isinstance(patience, (int, np.integer)) or patience < 1:
        raise ValueError("patience must be a positive integer.")
    if improvement_tolerance < 0:
        raise ValueError("improvement_tolerance must be nonnegative.")

    generator = np.random.default_rng(rng)
    dimension = points.shape[1]

    if initial_direction is None:
        direction = generator.normal(size=dimension)
    else:
        direction = np.asarray(initial_direction, dtype=float)
        if direction.shape != (dimension,):
            raise ValueError(f"initial_direction must have shape ({dimension},).")

    direction_norm = np.linalg.norm(direction)
    if direction_norm <= np.finfo(float).eps:
        raise ValueError("initial_direction must be nonzero.")
    direction = direction / direction_norm

    width_history = np.empty(iterations)
    best_width_history = np.empty(iterations)
    step_history = np.empty(iterations)
    best_width = np.inf
    best_direction = direction.copy()
    iterations_without_improvement = 0

    for iteration in range(iterations):
        projections = points @ direction
        maximum_index = int(np.argmax(projections))
        minimum_index = int(np.argmin(projections))
        width = projections[maximum_index] - projections[minimum_index]

        if width < best_width - improvement_tolerance:
            best_width = float(width)
            best_direction = direction.copy()
            iterations_without_improvement = 0
        else:
            iterations_without_improvement += 1

        width_history[iteration] = width
        best_width_history[iteration] = best_width
        if step_schedule == "inverse_sqrt":
            step = initial_step / np.sqrt(iteration + 1.0)
        else:
            step = initial_step * geometric_decay**iteration
        step_history[iteration] = step

        if iterations_without_improvement >= patience:
            width_history = width_history[:iteration + 1]
            best_width_history = best_width_history[:iteration + 1]
            step_history = step_history[:iteration + 1]
            break

        subgradient = points[maximum_index] - points[minimum_index]
        riemannian_subgradient = (
            subgradient - (direction @ subgradient) * direction
        )
        if np.linalg.norm(riemannian_subgradient) <= np.finfo(float).eps:
            width_history = width_history[:iteration + 1]
            best_width_history = best_width_history[:iteration + 1]
            step_history = step_history[:iteration + 1]
            break

        candidate = direction - step * riemannian_subgradient
        candidate_norm = np.linalg.norm(candidate)
        if candidate_norm <= np.finfo(float).eps:
            candidate = generator.normal(size=dimension)
            candidate_norm = np.linalg.norm(candidate)
        direction = candidate / candidate_norm

    return {
        "minimum_width": best_width,
        "direction": best_direction,
        "width_history": width_history,
        "best_width_history": best_width_history,
        "step_history": step_history,
        "step_schedule": step_schedule,
        "geometric_decay": geometric_decay,
    }


def batched_geometric_width_curves(
    series,
    k_values,
    *,
    seed,
    replication,
    starts=3,
    iterations=150,
    initial_step=0.2,
    geometric_decay=0.97,
    return_directions=False,
):
    """Update several Riemannian subgradient starts together."""
    single_widths, best_widths, best_projections = [], [], []
    best_directions = []
    for k in k_values:
        cloud_k = sliding_window_point_cloud(series, int(k))
        if k == 1:
            initial_directions = [np.ones(1)]
        else:
            covariance_k = np.cov(cloud_k, rowvar=False)
            _, eigenvectors_k = np.linalg.eigh(covariance_k)
            initial_directions = [eigenvectors_k[:, 0]]
        initial_generator = np.random.default_rng(
            np.random.SeedSequence([seed, replication, int(k), 20])
        )
        initial_directions.extend(
            initial_generator.normal(size=int(k))
            for _ in range(starts - 1)
        )
        directions = np.asarray(initial_directions, dtype=float)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        best_start_widths = np.full(starts, np.inf)
        best_start_directions = directions.copy()

        for iteration in range(iterations):
            projections = cloud_k @ directions.T
            maximum_indices = np.argmax(projections, axis=0)
            minimum_indices = np.argmin(projections, axis=0)
            current_widths = (
                projections[maximum_indices, np.arange(starts)]
                - projections[minimum_indices, np.arange(starts)]
            )
            improved = current_widths < best_start_widths
            best_start_widths[improved] = current_widths[improved]
            best_start_directions[improved] = directions[improved]

            subgradients = cloud_k[maximum_indices] - cloud_k[minimum_indices]
            tangent_subgradients = subgradients - (
                np.sum(directions * subgradients, axis=1, keepdims=True)
                * directions
            )
            step = initial_step * geometric_decay**iteration
            candidates = directions - step * tangent_subgradients
            candidate_norms = np.linalg.norm(
                candidates,
                axis=1,
                keepdims=True,
            )
            valid = candidate_norms[:, 0] > np.finfo(float).eps
            directions[valid] = candidates[valid] / candidate_norms[valid]

        single_widths.append(best_start_widths[0])
        best_start = int(np.argmin(best_start_widths))
        best_widths.append(best_start_widths[best_start])
        best_projections.append(cloud_k @ best_start_directions[best_start])
        best_directions.append(best_start_directions[best_start].copy())

    standard_result = (
        np.asarray(single_widths),
        np.asarray(best_widths),
        best_projections,
    )
    if return_directions:
        return (*standard_result, best_directions)
    return standard_result


def theoretical_ar_width_in_direction(
    phi,
    direction,
    half_width=1.0,
    history=10_000,
):
    """Evaluate the theoretical AR support width in a supplied direction."""
    direction = np.asarray(direction, dtype=float)
    direction_norm = np.linalg.norm(direction)

    if direction.ndim != 1 or len(direction) == 0:
        raise ValueError("direction must be a nonempty one-dimensional array.")
    if direction_norm == 0:
        raise ValueError("direction must be nonzero.")
    if half_width <= 0 or history < 0:
        raise ValueError("Require half_width > 0 and history >= 0.")

    unit_direction = direction / direction_norm
    psi = ar_impulse_response(phi, maximum_lag=history)

    generator_projections = np.convolve(psi, unit_direction)[: history + 1]
    return float(2.0 * half_width * np.sum(np.abs(generator_projections)))


def population_minimizing_ar2_direction(phi, k, history=10_000):
    """Return a population-minimizing direction for an AR(2) process."""
    phi = np.asarray(phi, dtype=float)
    if phi.shape != (2,):
        raise ValueError("phi must contain exactly two AR coefficients.")
    if k < 1:
        raise ValueError("k must be positive.")
    if k == 1:
        return np.ones(1)
    if k >= 3:
        direction = np.r_[1.0, -phi, np.zeros(k - 3)]
        return direction / np.linalg.norm(direction)

    psi = ar_impulse_response(phi, maximum_lag=history)
    lagged_psi = np.r_[0.0, psi[:-1]]

    def width_from_angle(angle):
        generator_projections = (
            np.cos(angle) * psi + np.sin(angle) * lagged_psi
        )
        return float(2.0 * np.sum(np.abs(generator_projections)))

    angle_grid = np.linspace(0.0, np.pi, 4_097)
    grid_widths = np.asarray([
        width_from_angle(angle) for angle in angle_grid
    ])
    best_index = int(np.argmin(grid_widths))
    lower_index = max(0, best_index - 1)
    upper_index = min(len(angle_grid) - 1, best_index + 1)
    local_fit = minimize_scalar(
        width_from_angle,
        bounds=(angle_grid[lower_index], angle_grid[upper_index]),
        method="bounded",
        options={"xatol": 1e-12},
    )
    best_angle = (
        float(local_fit.x)
        if local_fit.success
        else float(angle_grid[best_index])
    )
    return np.array([np.cos(best_angle), np.sin(best_angle)])


def horizontal_tail_lad_selector(k_values, widths):
    """Fit c + s(q-k)_+ by LAD for each q, subject to s >= 0."""
    k_values = np.asarray(k_values, dtype=float)
    widths = np.asarray(widths, dtype=float)
    candidate_rows = []

    for selected_q in k_values[1:-1].astype(int):
        hinge = np.maximum(selected_q - k_values, 0.0)
        sample_count = len(widths)
        identity = np.eye(sample_count)
        objective = np.concatenate(([0.0, 0.0], np.ones(sample_count)))
        constraints = np.vstack((
            np.column_stack((-np.ones(sample_count), -hinge, -identity)),
            np.column_stack((np.ones(sample_count), hinge, -identity)),
        ))
        bounds = [(None, None), (0.0, None)] + [(0.0, None)] * sample_count
        fit = linprog(
            objective,
            A_ub=constraints,
            b_ub=np.concatenate((-widths, widths)),
            bounds=bounds,
            method="highs",
        )
        if not fit.success:
            raise RuntimeError(
                f"LAD fit failed for q={selected_q}: {fit.message}"
            )
        level, slope = fit.x[:2]
        fitted_values = level + slope * hinge
        candidate_rows.append({
            "selected_q": selected_q,
            "score": float(np.sum(np.abs(widths - fitted_values))),
            "level": float(level),
            "slope": float(slope),
            "fitted_values": fitted_values,
        })

    best = min(
        candidate_rows,
        key=lambda row: (row["score"], row["selected_q"]),
    )
    return {
        **best,
        "selected_order": best["selected_q"] - 1,
        "candidate_scores": pd.DataFrame(candidate_rows).drop(
            columns="fitted_values"
        ),
    }


def companion_matrix(phi):
    """Return the companion matrix for an AR(p) coefficient vector."""
    phi = np.asarray(phi, dtype=float)
    order = len(phi)
    matrix = np.zeros((order, order))
    matrix[0, :] = phi
    if order > 1:
        matrix[1:, :-1] = np.eye(order - 1)
    return matrix


def is_stationary(phi):
    """Return a stationarity flag and the companion-matrix eigenvalues."""
    eigenvalues = np.linalg.eigvals(companion_matrix(phi))
    return bool(np.all(np.abs(eigenvalues) < 1)), eigenvalues


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
