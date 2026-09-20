from __future__ import annotations

import numpy as np

from .functions import sliding_window_point_cloud

__all__ = [
    "projected_subgradient_empirical_width",
    "batched_geometric_width_curves",
    "batched_riemannian_adam_width_curves",
]


def projected_subgradient_empirical_width(
        points,
        iterations=2_000,
        initial_step=0.2, # gamma_0
        patience=200,
        improvement_tolerance=1e-10,
        rng=None,
        initial_direction=None,
        step_schedule="inverse_sqrt",
        geometric_decay=0.99, # beta
):
    points = np.asarray(points, dtype=float)

    ### input checks ###
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
    ### end input checks ###

    generator = np.random.default_rng(rng)
    dimension = points.shape[1] # k

    if initial_direction is None:
        direction = generator.normal(size=dimension) # u_0 ~ N(0, I_k)
    else:
        direction = np.asarray(initial_direction, dtype=float)
        if direction.shape != (dimension,):
            raise ValueError(f"initial_direction must have shape ({dimension},).")

    direction_norm = np.linalg.norm(direction) # ||u_0||
    if direction_norm <= np.finfo(float).eps:
        raise ValueError("initial_direction must be nonzero.")
    direction = direction / direction_norm # u_0 <- u_0 / ||u_0|| (u_0 ~ U(S^{k-1}))

    width_history = np.empty(iterations)
    best_width_history = np.empty(iterations)
    step_history = np.empty(iterations)
    best_width = np.inf
    best_direction = direction.copy()
    iterations_without_improvement = 0

    # main loop
    for iteration in range(iterations):
        projections = points @ direction # p_i = x_i^T u_t
        maximum_index = int(np.argmax(projections)) # i_max = argmax_i p_i
        minimum_index = int(np.argmin(projections)) # i_min = argmin_i p_i
        width = projections[maximum_index] - projections[minimum_index] # w(u_t) = p_{i_max} - p_{i_min}

        if width < best_width - improvement_tolerance:
            best_width = float(width)
            best_direction = direction.copy()
            iterations_without_improvement = 0
        else:
            iterations_without_improvement += 1

        width_history[iteration] = width
        best_width_history[iteration] = best_width

        if step_schedule == "inverse_sqrt":
            step = initial_step / np.sqrt(iteration + 1.0) # gamma_t = gamma_0 / sqrt(t+1)
        else:
            step = initial_step * geometric_decay**iteration # gamma_t = gamma_0 * beta^t

        step_history[iteration] = step

        if iterations_without_improvement >= patience:
            width_history = width_history[:iteration + 1]
            best_width_history = best_width_history[:iteration + 1]
            step_history = step_history[:iteration + 1]
            break

        subgradient = points[maximum_index] - points[minimum_index] # g_t = x_{i_max} - x_{i_min}
        riemannian_subgradient = (
            subgradient - (direction @ subgradient) * direction
        ) # xi_t = g_t - (u_t^T g_t) u_t
        if np.linalg.norm(riemannian_subgradient) <= np.finfo(float).eps:
            width_history = width_history[:iteration + 1]
            best_width_history = best_width_history[:iteration + 1]
            step_history = step_history[:iteration + 1]
            break

        candidate = direction - step * riemannian_subgradient # u_{t+1} = u_t - gamma_t xi_t
        candidate_norm = np.linalg.norm(candidate)
        # is the fallback mathematically necessary?
        if candidate_norm <= np.finfo(float).eps:
            candidate = generator.normal(size=dimension) # u_{t+1} ~ N(0, I_k)
            candidate_norm = np.linalg.norm(candidate) # ||u_{t+1}||
        direction = candidate / candidate_norm # u_{t+1} <- u_{t+1} / ||u_{t+1}|| (u_{t+1} ~ U(S^{k-1}))

    return {
        "minimum_width": best_width,
        "direction": best_direction,
        "width_history": width_history,
        "best_width_history": best_width_history,
        "step_history": step_history,
        "step_schedule": step_schedule,
        "geometric_decay": geometric_decay,
    }



# continue here


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
    use_convex_hull=False,
):
    """Update several Riemannian subgradient starts together.

    ``use_convex_hull=True`` evaluates widths and subgradients using only
    hull vertices. Covariance initialization and returned projections still
    use the full point cloud. This preserves the directional-width objective.
    Degenerate clouds fall back to the full cloud.
    """
    if not isinstance(use_convex_hull, (bool, np.bool_)):
        raise ValueError("use_convex_hull must be a boolean.")
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
        optimization_cloud = cloud_k
        if use_convex_hull and int(k) >= 2 and len(cloud_k) > int(k):
            from scipy.spatial import ConvexHull, QhullError

            try:
                hull = ConvexHull(cloud_k)
                # Sorted original indices preserve the argmax/argmin tie order.
                optimization_cloud = cloud_k[np.sort(hull.vertices)]
            except QhullError:
                pass
        best_start_widths = np.full(starts, np.inf)
        best_start_directions = directions.copy()

        for iteration in range(iterations):
            projections = optimization_cloud @ directions.T
            maximum_indices = np.argmax(projections, axis=0)
            minimum_indices = np.argmin(projections, axis=0)
            current_widths = (
                projections[maximum_indices, np.arange(starts)]
                - projections[minimum_indices, np.arange(starts)]
            )
            improved = current_widths < best_start_widths
            best_start_widths[improved] = current_widths[improved]
            best_start_directions[improved] = directions[improved]

            subgradients = (
                optimization_cloud[maximum_indices]
                - optimization_cloud[minimum_indices]
            )
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
