from __future__ import annotations

import json
from time import perf_counter

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import pdist

from .experiment_settings import WARM_START_VARIANTS
from .functions import sliding_window_point_cloud

__all__ = [
    "projected_subgradient_empirical_width",
    "batched_geometric_width_curves",
    "fit_minimum_width_known_order",
    "fit_ar3_mse_prefix",
    "fit_ar3_warm_start_rows",
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
    termination_reason = "iteration_limit"

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
            termination_reason = "patience"
            width_history = width_history[:iteration + 1]
            best_width_history = best_width_history[:iteration + 1]
            step_history = step_history[:iteration + 1]
            break

        subgradient = points[maximum_index] - points[minimum_index] # g_t = x_{i_max} - x_{i_min}
        riemannian_subgradient = (
            subgradient - (direction @ subgradient) * direction
        ) # xi_t = g_t - (u_t^T g_t) u_t
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
        "termination_reason": termination_reason,
        "step_schedule": step_schedule,
        "geometric_decay": geometric_decay,
    }



def batched_geometric_width_curves(
        series,
        max_k,
        *,
        seed,
        replication,
        starts=3, # total number of initial directions for each k                               ### CHANGED
        iterations=150,
        initial_step=0.2,
        geometric_decay=0.97,
        return_directions=False,
        use_convex_hull=False,
        warm_start_previous_direction=False,                                                    ### CHANGED
        warm_start_previous_coefficients=False,                                                 ### CHANGED
):
    ### input checks ###
    if (
        isinstance(max_k, (bool, np.bool_))
        or not isinstance(max_k, (int, np.integer))
        or max_k < 1
    ):
        raise ValueError("max_k must be a positive integer.")
    if not isinstance(use_convex_hull, (bool, np.bool_)):
        raise ValueError("use_convex_hull must be a boolean.")
    if not isinstance(starts, (int, np.integer)) or starts < 1:
        raise ValueError("starts must be a positive integer.")
    if not isinstance(warm_start_previous_direction, (bool, np.bool_)):                         ### CHANGED
        raise ValueError("warm_start_previous_direction must be a boolean.")                    ### CHANGED
    if not isinstance(warm_start_previous_coefficients, (bool, np.bool_)):                      ### CHANGED
        raise ValueError("warm_start_previous_coefficients must be a boolean.")                 ### CHANGED
    if warm_start_previous_direction and warm_start_previous_coefficients:                      ### CHANGED
        raise ValueError("Only one warm-start strategy can be enabled.")                        ### CHANGED
    ### end input checks ###

    first_start_widths, best_widths, best_projections = [], [], []
    best_directions = []
    for k in range(1, int(max_k) + 1): # k=1, ..., K

        # sliding window point cloud of dimension k from the time series
        cloud_k = sliding_window_point_cloud(series=series, embedding_dimension=int(k))

        initial_generator = np.random.default_rng(
            seed=np.random.SeedSequence([seed, replication, int(k), 20])
        )

        # Random initial directions for every start.
        directions = initial_generator.normal(size=(starts, int(k)))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)

        if k >= 2 and warm_start_previous_direction:                                            ### CHANGED
            directions[0] = np.r_[best_directions[-1], 0.0]                                     ### CHANGED
        elif k >= 3 and warm_start_previous_coefficients:                                       ### CHANGED
            previous_direction = best_directions[-1]                                            ### CHANGED
            if abs(previous_direction[0]) <= np.finfo(float).eps:                               ### CHANGED
                raise ValueError(                                                               ### CHANGED
                    "Cannot derive previous coefficients when the first "                       ### CHANGED
                    "direction coordinate is zero."                                             ### CHANGED
                )                                                                               ### CHANGED
            previous_coefficients = -previous_direction[1:] / previous_direction[0]             ### CHANGED
            coefficient_direction = np.r_[1.0, -previous_coefficients, 0.0]                     ### CHANGED
            directions[0] = coefficient_direction / np.linalg.norm(coefficient_direction)       ### CHANGED

        optimization_cloud = cloud_k

        # optional: reduce the optimization cloud to the convex hull of the sliding window point cloud
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

        # Run the single-direction optimizer once for each initial direction
        for start_index, initial_direction in enumerate(directions):
            result = projected_subgradient_empirical_width(
                points=optimization_cloud,
                iterations=iterations,
                initial_step=initial_step,
                patience=iterations + 1,
                rng=np.random.SeedSequence([seed, replication, int(k), 21, start_index]),
                initial_direction=initial_direction,
                step_schedule="geometric",
                geometric_decay=geometric_decay,
            )
            best_start_widths[start_index] = result["minimum_width"]
            best_start_directions[start_index] = result["direction"]

        first_start_widths.append(best_start_widths[0])
        best_start = int(np.argmin(best_start_widths))
        best_widths.append(best_start_widths[best_start])
        best_projections.append(cloud_k @ best_start_directions[best_start])
        best_directions.append(best_start_directions[best_start].copy())

    standard_result = (
        np.asarray(first_start_widths),
        np.asarray(best_widths),
        best_projections,
    )
    if return_directions:
        return (*standard_result, best_directions)
    return standard_result


def fit_minimum_width_known_order(series, point_cloud, order, *, seed, replication,
                                  starts, iterations, step_scale, decay):
    diameter = float(pdist(point_cloud).max())
    if not np.isfinite(diameter) or diameter <= 0.0:
        raise RuntimeError("Cannot normalize the minimum-width step: invalid diameter.")
    _, _, _, directions = batched_geometric_width_curves(
        series,
        order + 1,
        seed=seed,
        replication=replication,
        starts=starts,
        iterations=iterations,
        initial_step=step_scale / diameter,
        geometric_decay=decay,
        return_directions=True,
    )
    direction = directions[order]
    if abs(direction[0]) <= np.finfo(float).eps:
        raise RuntimeError("Cannot recover coefficients: first direction component is zero.")
    return -direction[1:] / direction[0]


def fit_ar3_mse_prefix(series: np.ndarray, replication_index: int, sample_size: int,
                       *, configuration) -> dict:
    """Fit only k=4, with the same random starts and schedule as the 50-run study."""
    CONFIG = configuration
    cloud = sliding_window_point_cloud(series[:sample_size], embedding_dimension=4)
    optimization_cloud = cloud
    try:
        hull = ConvexHull(cloud)
        optimization_cloud = cloud[np.sort(hull.vertices)]
    except QhullError:
        pass

    # A point-cloud diameter is attained by hull vertices. Thus the step-size
    # normalization and every directional width use the original cloud values.
    diameter = float(pdist(optimization_cloud).max())
    if not np.isfinite(diameter) or diameter <= 0:
        raise ValueError("The point-cloud diameter is invalid.")
    initial_step = CONFIG["initial_step_scale"] / diameter

    generator = np.random.default_rng(
        np.random.SeedSequence([CONFIG["optimizer_seed"], replication_index, 4, 20])
    )
    initial_directions = generator.normal(size=(CONFIG["starts"], 4))
    initial_directions /= np.linalg.norm(initial_directions, axis=1, keepdims=True)

    results = [
        projected_subgradient_empirical_width(
            optimization_cloud,
            iterations=CONFIG["iterations"],
            initial_step=initial_step,
            patience=CONFIG["iterations"] + 1,
            rng=np.random.SeedSequence(
                [CONFIG["optimizer_seed"], replication_index, 4, 21, start_index]
            ),
            initial_direction=direction,
            step_schedule="geometric",
            geometric_decay=CONFIG["geometric_decay"],
        )
        for start_index, direction in enumerate(initial_directions)
    ]
    best_start = int(np.argmin([result["minimum_width"] for result in results]))
    result = results[best_start]
    direction = result["direction"]
    if abs(direction[0]) <= np.finfo(float).eps:
        raise ValueError("The first direction component is numerically zero.")
    estimate = -direction[1:] / direction[0]
    if not np.all(np.isfinite(estimate)):
        raise ValueError("The coefficient estimate is not finite.")
    if not np.isclose(np.ptp(cloud @ direction), result["minimum_width"], atol=1e-9):
        raise ValueError("The selected direction does not reproduce its width.")
    return {
        "estimate": estimate,
        "minimum_width": float(result["minimum_width"]),
        "abs_u_0": float(abs(direction[0])),
        "best_start": best_start + 1,
        "diameter": diameter,
        "initial_step": initial_step,
        "hull_vertices": len(optimization_cloud),
    }


def fit_ar3_warm_start_rows(series, phi, settings, replication, *, warm,
                            variants=WARM_START_VARIANTS):
    VARIANTS = variants
    start = perf_counter()
    _, widths, _, directions = batched_geometric_width_curves(
        series, max(settings["k_values"]), seed=settings["seeds"]["3"],
        replication=replication - 1, starts=settings["starts"],
        iterations=settings["iterations"], initial_step=settings["initial_step"],
        geometric_decay=settings["geometric_decay"], return_directions=True,
        warm_start_previous_direction=warm,
    )
    elapsed = perf_counter() - start
    variant = VARIANTS[2] if warm else VARIANTS[1]
    rows = []
    for k, (width, direction) in enumerate(zip(widths, directions), start=1):
        cloud = sliding_window_point_cloud(series, k)
        actual_width = float(np.ptp(cloud @ direction))
        assert abs(actual_width - width) < 1e-10
        assert abs(np.linalg.norm(direction) - 1) < 1e-12
        candidate_width = np.nan
        if warm and k >= 2:
            padded = np.r_[directions[k - 2], 0.0]
            candidate_width = float(np.ptp(cloud @ padded))
            assert width <= candidate_width + 1e-10
            assert candidate_width <= widths[k - 2] + 1e-10
        row = {
            "variant": variant, "true_order": 3, "replication": replication,
            "k": k, "minimum_width": float(width),
            "coefficients": json.dumps(phi.tolist()),
            "padded_candidate_width": candidate_width,
            "curve_runtime_seconds": elapsed,
        }
        row.update({f"u_{j}": float(direction[j]) if j < k else np.nan
                    for j in range(max(settings["k_values"]))})
        rows.append(row)
    if warm:
        tolerance = 32 * np.finfo(float).eps * max(1.0, float(np.abs(widths).max()))
        assert np.all(np.diff(widths) <= tolerance), (replication, np.diff(widths))
    return rows
