"""Geometric step-schedule comparisons on a fixed empirical-width cloud."""

from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from . import descent

__all__ = ["tune_geometric_width_schedule"]


def tune_geometric_width_schedule(
    points,
    initial_directions,
    gamma0_grid,
    beta_grid,
    *,
    iterations,
    patience,
    improvement_tolerance,
    optimizer_seed,
    progress=False,
):
    """Select gamma_m = gamma0 * beta**m by attained empirical width.

    Every candidate uses the supplied starting directions and the same
    per-start fallback seeds as the single-process notebook. Ties are broken
    by total evaluations, then gamma0 and beta. True coefficients are not
    inputs to this search. Histories retain the best width across starts at
    each iteration, holding each stopped run at its last recorded best width.
    """
    points = np.asarray(points, dtype=float)
    directions = np.array(initial_directions, dtype=float, copy=True)
    if (
        points.ndim != 2 or len(points) < 2 or points.shape[1] < 1
        or not np.all(np.isfinite(points))
    ):
        raise ValueError("points must be a finite (n, k) cloud with n >= 2 and k >= 1.")
    if (
        directions.ndim != 2 or len(directions) < 1
        or directions.shape[1] != points.shape[1]
        or not np.all(np.isfinite(directions))
        or np.any(np.linalg.norm(directions, axis=1) <= np.finfo(float).eps)
    ):
        raise ValueError("initial_directions must contain finite nonzero vectors matching k.")
    grids = []
    for values, name in ((gamma0_grid, "gamma0_grid"), (beta_grid, "beta_grid")):
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be a nonempty finite one-dimensional grid.")
        if np.any(values <= 0) or (name == "beta_grid" and np.any(values >= 1)):
            raise ValueError("Require gamma0 > 0 and 0 < beta < 1.")
        grids.append(np.unique(values))
    for value, name in ((iterations, "iterations"), (patience, "patience")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not np.isfinite(improvement_tolerance) or improvement_tolerance < 0:
        raise ValueError("improvement_tolerance must be finite and nonnegative.")
    if (
        isinstance(optimizer_seed, (bool, np.bool_))
        or not isinstance(optimizer_seed, (int, np.integer)) or optimizer_seed < 0
    ):
        raise ValueError("optimizer_seed must be a nonnegative integer.")

    rows, per_start, histories = [], [], {}
    winning_key, winning_row, winning_results = None, None, None
    dimension = points.shape[1]
    total_candidates = len(grids[0]) * len(grids[1])
    with threadpool_limits(limits=1):
        for gamma0 in grids[0]:
            for beta in grids[1]:
                candidate_id = f"grid_{len(rows):03d}"
                started = perf_counter()
                runs = [
                    descent.projected_subgradient_empirical_width(
                        points,
                        iterations=iterations,
                        initial_step=float(gamma0),
                        geometric_decay=float(beta),
                        patience=patience,
                        improvement_tolerance=improvement_tolerance,
                        initial_direction=direction,
                        rng=np.random.SeedSequence([
                            optimizer_seed, 0, dimension, 21, start_index,
                        ]),
                        step_schedule="geometric",
                    )
                    for start_index, direction in enumerate(directions)
                ]
                selected_index = int(np.argmin([run["minimum_width"] for run in runs]))
                selected = runs[selected_index]
                direction = selected["direction"].copy()
                if direction[0] < 0:
                    direction *= -1
                width = float(selected["minimum_width"])
                if (
                    not np.isfinite(width) or not np.all(np.isfinite(direction))
                    or not np.isclose(np.linalg.norm(direction), 1.0)
                    or not np.isclose(np.ptp(points @ direction), width)
                ):
                    raise RuntimeError(f"Invalid attained width for {candidate_id}.")
                row = {
                    "candidate_id": candidate_id,
                    "gamma0": float(gamma0), "beta": float(beta),
                    "minimum_width": width, "selected_start": selected_index + 1,
                    "total_evaluations": sum(len(run["width_history"]) for run in runs),
                    "patience_stops": sum(run["termination_reason"] == "patience" for run in runs),
                    "runtime_seconds": perf_counter() - started,
                }
                row.update({f"u_{j}": float(value) for j, value in enumerate(direction)})
                rows.append(row)
                longest = max(len(run["best_width_history"]) for run in runs)
                histories[candidate_id] = np.minimum.reduce([
                    np.pad(
                        run["best_width_history"],
                        (0, longest - len(run["best_width_history"])), mode="edge",
                    )
                    for run in runs
                ])
                for start_index, run in enumerate(runs):
                    per_start.append({
                        "candidate_id": candidate_id,
                        "gamma0": float(gamma0), "beta": float(beta),
                        "start": start_index + 1,
                        "minimum_width": float(run["minimum_width"]),
                        "evaluations": len(run["width_history"]),
                        "termination_reason": run["termination_reason"],
                        "selected": start_index == selected_index,
                    })
                key = (width, row["total_evaluations"], float(gamma0), float(beta))
                if winning_key is None or key < winning_key:
                    winning_key, winning_row, winning_results = key, row, runs
                if progress:
                    print(
                        f"Pair {len(rows)}/{total_candidates}: gamma0={gamma0:.6g}, "
                        f"beta={beta:g}, best width={width:.10f}", flush=True,
                    )
    return {
        "candidates": rows, "per_start": per_start, "histories": histories,
        "selected": dict(winning_row), "best_results": winning_results,
        "direction": np.array([winning_row[f"u_{j}"] for j in range(dimension)]),
    }
