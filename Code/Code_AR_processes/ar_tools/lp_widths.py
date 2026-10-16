from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

from .functions import sliding_window_point_cloud


def lp_width_descent(points, initial_direction, *, tolerance=1e-9, max_iterations=50):
    points = np.asarray(points, dtype=float)
    direction = np.asarray(initial_direction, dtype=float)
    direction /= np.linalg.norm(direction)
    n, k = points.shape
    constraints = np.vstack((np.column_stack((points, np.zeros(n), -np.ones(n))),
                              np.column_stack((-points, np.ones(n), np.zeros(n)))))
    objective = np.r_[np.zeros(k), -1.0, 1.0]
    width = float(np.ptp(points @ direction))
    history = [width]
    statuses = []
    for _ in range(max_iterations):
        fit = linprog(objective, A_ub=constraints, b_ub=np.zeros(2 * n),
                      A_eq=np.r_[direction, 0.0, 0.0][None, :], b_eq=[1.0],
                      bounds=[(None, None)] * (k + 2), method="highs",
                      options={"primal_feasibility_tolerance": 1e-9,
                               "dual_feasibility_tolerance": 1e-9})
        if not fit.success:
            raise RuntimeError(f"Width LP failed: {fit.message}")
        candidate = fit.x[:k]
        candidate /= np.linalg.norm(candidate)
        candidate_width = float(np.ptp(points @ candidate))
        statuses.append(fit.status)
        improvement = width - candidate_width
        if candidate_width < width:
            direction, width = candidate, candidate_width
        history.append(width)
        if improvement <= tolerance * max(1.0, width):
            break
    return {"direction": direction.copy(), "width": width,
            "history": np.asarray(history), "lp_solves": len(statuses),
            "hit_iteration_limit": len(statuses) == max_iterations}


def lp_width_curve(series, max_k=10, *, tolerance=1e-9, max_iterations=50):
    common_cloud = sliding_window_point_cloud(series, max_k)
    widths, directions, diagnostics = [], [], []
    for k in range(1, max_k + 1):
        points = common_cloud[:, :k]
        if k == 1:
            widths.append(float(np.ptp(points[:, 0])))
            directions.append(np.ones(1))
            diagnostics.append({"k": 1, "lp_solves": 0, "runtime_seconds": 0.0,
                                "iteration_limit_starts": 0})
            continue
        starts = [np.r_[directions[-1], 0.0], np.r_[0.0, directions[-1]], np.eye(k)[0]]
        started = perf_counter()
        fits = [lp_width_descent(points, start, tolerance=tolerance,
                                max_iterations=max_iterations) for start in starts]
        winner = min(fits, key=lambda fit: fit["width"])
        widths.append(winner["width"])
        directions.append(winner["direction"])
        diagnostics.append({"k": k, "lp_solves": sum(fit["lp_solves"] for fit in fits),
                            "runtime_seconds": perf_counter() - started,
                            "iteration_limit_starts": sum(fit["hit_iteration_limit"] for fit in fits)})
        eps = 32 * np.finfo(float).eps * max(1.0, abs(widths[-2]))
        assert widths[-1] <= widths[-2] + eps
    return np.asarray(widths), directions, diagnostics


def exact_empirical_width_low_dimension(points):
    """Global floating-point hull calculation for k<=4.

    Width of P is the support of P-P. The smallest facet distance of the
    difference body equals its minimum support on the unit sphere.
    """
    points = np.asarray(points, dtype=float)
    k = points.shape[1]
    if k == 1:
        return float(np.ptp(points[:, 0])), np.ones(1), len(points)
    if not 2 <= k <= 4:
        raise ValueError("The audit is restricted to dimensions two through four.")
    hull = ConvexHull(points)
    vertices = points[hull.vertices]
    differences = (vertices[:, None, :] - vertices[None, :, :]).reshape(-1, k)
    difference_hull = ConvexHull(differences)
    normals = difference_hull.equations[:, :-1]
    norms = np.linalg.norm(normals, axis=1)
    distances = -difference_hull.equations[:, -1] / norms
    best = int(np.argmin(distances))
    direction = normals[best] / norms[best]
    width = float(np.ptp(points @ direction))
    assert abs(width - distances[best]) < 1e-8 * max(1.0, width)
    return width, direction, len(vertices)


def chebyshev_ar_fit(point_cloud):
    """Fit AR coefficients by minimizing the maximum absolute residual."""
    order = point_cloud.shape[1] - 1
    response = point_cloud[:, 0]
    lagged_values = point_cloud[:, 1:]
    observation_count = len(response)
    ones = np.ones(observation_count)

    # Variables are (phi_1, phi_2, phi_3, residual_center, radius).
    inequality_matrix = np.vstack((
        np.column_stack((-lagged_values, -ones, -ones)),
        np.column_stack((lagged_values, ones, -ones)),
    ))
    inequality_bounds = np.concatenate((-response, response))
    objective = np.zeros(order + 2)
    objective[-1] = 1.0

    result = linprog(
        objective,
        A_ub=inequality_matrix,
        b_ub=inequality_bounds,
        bounds=[(None, None)] * (order + 1) + [(0.0, None)],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Chebyshev LP failed: {result.message}")
    return result.x[:order], result.x[order], result.x[-1]


def unshifted_residual_widths(series, max_dimension):
    """LP-optimized residual spans for dimensions 1,...,D on common rows."""
    n = len(series) - max_dimension + 1
    y = series[max_dimension - 1:]
    lag_matrix = np.column_stack([
        series[max_dimension - 1 - j:len(series) - j]
        for j in range(1, max_dimension)
    ])
    ones = np.ones(n)
    zeros = np.zeros(n)
    widths = np.empty(max_dimension)
    widths[0] = np.ptp(y)
    for q in range(1, max_dimension):
        lags = lag_matrix[:, :q]
        constraints = np.vstack([
            np.column_stack([-lags, -ones, zeros]),
            np.column_stack([lags, zeros, ones]),
        ])
        result = linprog(
            np.r_[np.zeros(q), 1.0, -1.0],
            A_ub=constraints,
            b_ub=np.r_[-y, y],
            bounds=[(None, None)] * (q + 2),
            method="highs",
        )
        if not result.success:
            raise RuntimeError(f"LP failed at dimension {q + 1}: {result.message}")
        widths[q] = result.fun
    return widths, y, lag_matrix
