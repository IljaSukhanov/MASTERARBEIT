"""Shared simulation utilities for the autoregressive-process experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize_scalar

__all__ = [
    "jones_ar_coefficients",
    "centered_beta_support",
    "simulate_beta_ar",
    "simulate_uniform_ar",
    "sliding_window_point_cloud",
    "ar_impulse_response",
    "projected_subgradient_empirical_width",
    "batched_geometric_width_curves",
    "batched_riemannian_adam_width_curves",
    "theoretical_ar_width_in_direction",
    "population_minimizing_ar2_direction",
    "two_affine_l_method_selector",
    "affine_constant_l_method_selector",
    "aic_bic_ar_order",
    "yule_walker_pacf",
    "pacf_cutoff_order",
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


def batched_riemannian_adam_width_curves(
    series,
    k_values,
    *,
    seed,
    replication,
    starts=3,
    iterations=150,
    initial_step=0.2,
    geometric_decay=0.97,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8,
    return_directions=False,
):
    """Minimize width using sphere-aware Adam with a scalar second moment.

    The active width subgradient and first moment are projected onto the
    tangent space. A scalar second moment keeps the update invariant under
    rotations of the point cloud. The update is retracted to the sphere by
    normalization and uses ``initial_step * geometric_decay**iteration``.
    """
    if not isinstance(starts, (int, np.integer)) or starts < 1:
        raise ValueError("starts must be a positive integer.")
    if not isinstance(iterations, (int, np.integer)) or iterations < 1:
        raise ValueError("iterations must be a positive integer.")
    if not np.isfinite(initial_step) or initial_step <= 0.0:
        raise ValueError("initial_step must be positive and finite.")
    if not 0.0 < geometric_decay < 1.0:
        raise ValueError("geometric_decay must lie strictly between zero and one.")
    if not 0.0 <= beta_1 < 1.0:
        raise ValueError("beta_1 must lie in [0, 1).")
    if not 0.0 <= beta_2 < 1.0:
        raise ValueError("beta_2 must lie in [0, 1).")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be positive and finite.")

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

        first_moments = np.zeros_like(directions)
        second_moments = np.zeros(starts)
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

            # Projecting the previous moment is a vector transport into the
            # current tangent space of the sphere.
            first_moments -= (
                np.sum(directions * first_moments, axis=1, keepdims=True)
                * directions
            )
            first_moments = (
                beta_1 * first_moments
                + (1.0 - beta_1) * tangent_subgradients
            )
            second_moments = (
                beta_2 * second_moments
                + (1.0 - beta_2)
                * np.sum(tangent_subgradients**2, axis=1)
            )

            bias_corrected_first = first_moments / (
                1.0 - beta_1 ** (iteration + 1)
            )
            bias_corrected_second = second_moments / (
                1.0 - beta_2 ** (iteration + 1)
            )
            step = initial_step * geometric_decay**iteration
            adaptive_direction = bias_corrected_first / (
                np.sqrt(bias_corrected_second)[:, None] + epsilon
            )
            candidates = directions - step * adaptive_direction
            directions = candidates / np.linalg.norm(
                candidates,
                axis=1,
                keepdims=True,
            )

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


def _validated_elbow_data(k_values, widths, minimum_length=4):
    """Validate an embedding-dimension width curve for elbow fitting."""
    k_values = np.asarray(k_values, dtype=float)
    widths = np.asarray(widths, dtype=float)
    if k_values.ndim != 1 or widths.ndim != 1:
        raise ValueError("k_values and widths must be one-dimensional.")
    if len(k_values) != len(widths) or len(k_values) < minimum_length:
        raise ValueError(
            "k_values and widths must have the same length of at least "
            f"{minimum_length}."
        )
    if not np.all(np.isfinite(k_values)) or not np.all(np.isfinite(widths)):
        raise ValueError("k_values and widths must contain only finite values.")
    if not np.all(np.diff(k_values) > 0):
        raise ValueError("k_values must be strictly increasing.")
    if k_values[0] != 1 or not np.all(np.diff(k_values) == 1):
        raise ValueError(
            "k_values must be consecutive embedding dimensions starting at one."
        )
    return k_values, widths


def _affine_fit(x_values, y_values):
    """Return intercept, slope, predictions, and RMSE for an affine fit."""
    design = np.column_stack((np.ones(len(x_values)), x_values))
    coefficients, *_ = np.linalg.lstsq(design, y_values, rcond=None)
    fitted_values = design @ coefficients
    rmse = float(np.sqrt(np.mean((y_values - fitted_values) ** 2)))
    return (
        float(coefficients[0]),
        float(coefficients[1]),
        fitted_values,
        rmse,
    )


def two_affine_l_method_selector(k_values, widths):
    """Classical L-method based on two disjoint affine least-squares fits.

    Each candidate puts at least two observations in both segments. The
    standard segment-size-weighted RMSE chooses the boundary, and the selected
    AR order is the last embedding dimension in the left segment. Consequently
    this classical construction cannot select AR order one.
    """
    k_values, widths = _validated_elbow_data(k_values, widths)
    candidate_rows = []
    candidate_fits = []
    point_count = len(k_values)

    for split_index in range(1, point_count - 2):
        left_k = k_values[: split_index + 1]
        left_widths = widths[: split_index + 1]
        right_k = k_values[split_index + 1 :]
        right_widths = widths[split_index + 1 :]
        left_intercept, left_slope, left_fit, left_rmse = _affine_fit(
            left_k, left_widths
        )
        right_intercept, right_slope, right_fit, right_rmse = _affine_fit(
            right_k, right_widths
        )
        score = (
            len(left_k) * left_rmse + len(right_k) * right_rmse
        ) / point_count
        selected_order = int(k_values[split_index])
        selected_q = selected_order + 1
        candidate_rows.append({
            "selected_order": selected_order,
            "selected_q": selected_q,
            "score": float(score),
            "left_intercept": left_intercept,
            "left_slope": left_slope,
            "right_intercept": right_intercept,
            "right_slope": right_slope,
        })
        candidate_fits.append({
            "left_k": left_k.copy(),
            "left_fitted_values": left_fit,
            "right_k": right_k.copy(),
            "right_fitted_values": right_fit,
        })

    best_index = min(
        range(len(candidate_rows)),
        key=lambda index: (
            candidate_rows[index]["score"],
            candidate_rows[index]["selected_order"],
        ),
    )
    return {
        **candidate_rows[best_index],
        **candidate_fits[best_index],
        "candidate_scores": pd.DataFrame(candidate_rows),
    }


def affine_constant_l_method_selector(k_values, widths):
    """Modified L-method with an affine head and a constant tail.

    The candidate plateau onset q is included in both fits. This overlap gives
    the affine head two observations when q=2, so AR order one remains
    admissible. The objective is the segment-size-weighted RMSE and contains no
    order or dimension penalty.
    """
    k_values, widths = _validated_elbow_data(k_values, widths)
    candidate_rows = []
    candidate_fits = []

    for breakpoint_index in range(1, len(k_values) - 1):
        head_k = k_values[: breakpoint_index + 1]
        head_widths = widths[: breakpoint_index + 1]
        tail_k = k_values[breakpoint_index:]
        tail_widths = widths[breakpoint_index:]

        head_intercept, head_slope, head_fit, head_rmse = _affine_fit(
            head_k, head_widths
        )
        tail_level = float(np.mean(tail_widths))
        tail_fit = np.full(len(tail_k), tail_level)
        tail_rmse = float(np.sqrt(np.mean((tail_widths - tail_level) ** 2)))
        score = (
            len(head_k) * head_rmse + len(tail_k) * tail_rmse
        ) / (len(head_k) + len(tail_k))
        selected_q = int(k_values[breakpoint_index])
        candidate_rows.append({
            "selected_order": selected_q - 1,
            "selected_q": selected_q,
            "score": float(score),
            "head_intercept": head_intercept,
            "head_slope": head_slope,
            "tail_level": tail_level,
        })
        candidate_fits.append({
            "head_k": head_k.copy(),
            "head_fitted_values": head_fit,
            "tail_k": tail_k.copy(),
            "tail_fitted_values": tail_fit,
        })

    best_index = min(
        range(len(candidate_rows)),
        key=lambda index: (
            candidate_rows[index]["score"],
            candidate_rows[index]["selected_q"],
        ),
    )
    return {
        **candidate_rows[best_index],
        **candidate_fits[best_index],
        "candidate_scores": pd.DataFrame(candidate_rows),
    }


def aic_bic_ar_order(series, maximum_order):
    """Select conditional least-squares AR orders by Gaussian AIC and BIC."""
    series = np.asarray(series, dtype=float)
    maximum_order = int(maximum_order)
    if series.ndim != 1 or len(series) <= maximum_order + 1:
        raise ValueError("series is too short for the requested maximum_order.")

    # Use the same response times for every candidate so the criteria are comparable.
    response = series[maximum_order:]
    lag_matrix = np.column_stack([
        series[maximum_order - lag:len(series) - lag]
        for lag in range(1, maximum_order + 1)
    ])
    observation_count = len(response)
    criterion_rows = []

    for order in range(maximum_order + 1):
        if order == 0:
            design = np.ones((observation_count, 1))
        else:
            design = np.column_stack([
                np.ones(observation_count),
                lag_matrix[:, :order],
            ])

        coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
        residuals = response - design @ coefficients
        residual_sum_of_squares = max(
            float(residuals @ residuals),
            np.finfo(float).tiny,
        )
        parameter_count = order + 2
        gaussian_deviance = observation_count * np.log(
            residual_sum_of_squares / observation_count
        )
        criterion_rows.append({
            "order": order,
            "AIC": gaussian_deviance + 2.0 * parameter_count,
            "BIC": gaussian_deviance + np.log(observation_count) * parameter_count,
        })

    criterion_table = pd.DataFrame(criterion_rows)
    aic_order = int(criterion_table.loc[criterion_table["AIC"].idxmin(), "order"])
    bic_order = int(criterion_table.loc[criterion_table["BIC"].idxmin(), "order"])
    return aic_order, bic_order, criterion_table


def yule_walker_pacf(series, maximum_lag):
    """Estimate PACF values with the Durbin-Levinson/Yule-Walker recursion."""
    series = np.asarray(series, dtype=float)
    centered = series - np.mean(series)
    sample_size = len(centered)
    if maximum_lag < 1 or maximum_lag >= sample_size:
        raise ValueError("Require 1 <= maximum_lag < len(series).")

    autocovariance = np.array([
        centered[lag:] @ centered[:sample_size - lag] / sample_size
        for lag in range(maximum_lag + 1)
    ])
    if autocovariance[0] <= np.finfo(float).eps:
        raise ValueError("Cannot estimate PACF from a constant series.")

    pacf_values = np.empty(maximum_lag + 1)
    pacf_values[0] = 1.0
    previous_phi = np.empty(0)

    for order in range(1, maximum_lag + 1):
        if order == 1:
            reflection = autocovariance[1] / autocovariance[0]
        else:
            numerator = autocovariance[order] - (
                previous_phi @ autocovariance[order - 1:0:-1]
            )
            denominator = autocovariance[0] - (
                previous_phi @ autocovariance[1:order]
            )
            reflection = numerator / max(denominator, np.finfo(float).eps)

        pacf_values[order] = reflection
        updated_phi = np.empty(order)
        if order > 1:
            updated_phi[:-1] = previous_phi - reflection * previous_phi[::-1]
        updated_phi[-1] = reflection
        previous_phi = updated_phi

    return pacf_values


def pacf_cutoff_order(series, maximum_order, confidence_multiplier=1.96):
    """Choose the largest lag whose PACF exceeds the approximate confidence band."""
    pacf_values = yule_walker_pacf(series, maximum_order)
    cutoff = confidence_multiplier / np.sqrt(len(series))
    significant_lags = np.flatnonzero(np.abs(pacf_values[1:]) > cutoff) + 1
    selected_order = int(significant_lags[-1]) if len(significant_lags) else 0
    return selected_order, pacf_values, cutoff


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
