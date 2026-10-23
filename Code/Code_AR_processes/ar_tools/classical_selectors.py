from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import toeplitz
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, QhullError

from .functions import sliding_window_point_cloud

__all__ = [
    "aic_bic_ar_order",
    "yule_walker_pacf",
    "pacf_cutoff_order",
    "fit_ols_known_order",
    "fit_yule_walker_known_order",
    "fit_horvath_known_order",
    "yule_walker_coefficients",
    "conditional_gaussian_qml",
    "horvath_linf_coefficients",
]


def aic_bic_ar_order(series, maximum_order):
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
    pacf_values = yule_walker_pacf(series, maximum_order)
    cutoff = confidence_multiplier / np.sqrt(len(series))
    significant_lags = np.flatnonzero(np.abs(pacf_values[1:]) > cutoff) + 1
    selected_order = int(significant_lags[-1]) if len(significant_lags) else 0
    return selected_order, pacf_values, cutoff


def fit_ols_known_order(point_cloud):
    response = point_cloud[:, 0]
    lagged_values = point_cloud[:, 1:]
    coefficients, _, _, _ = np.linalg.lstsq(lagged_values, response, rcond=None)
    return coefficients


def fit_yule_walker_known_order(series, order):
    centered = np.asarray(series, dtype=float) - np.mean(series)
    observation_count = len(centered)
    autocovariances = np.array([
        centered[lag:] @ centered[:observation_count - lag] / observation_count
        for lag in range(order + 1)
    ])
    lag_indices = np.arange(order)
    autocovariance_matrix = autocovariances[
        np.abs(lag_indices[:, None] - lag_indices[None, :])
    ]
    return np.linalg.solve(autocovariance_matrix, autocovariances[1:])


def fit_horvath_known_order(point_cloud):
    response = point_cloud[:, 0]
    lagged_values = point_cloud[:, 1:]
    order = lagged_values.shape[1]
    minus_radius = -np.ones((len(response), 1))
    inequalities = np.vstack((
        np.column_stack((-lagged_values, minus_radius)),
        np.column_stack((lagged_values, minus_radius)),
    ))
    bounds = np.concatenate((-response, response))
    objective = np.r_[np.zeros(order), 1.0]
    solution = linprog(
        objective,
        A_ub=inequalities,
        b_ub=bounds,
        bounds=[(None, None)] * order + [(0.0, None)],
        method="highs",
    )
    if not solution.success:
        raise RuntimeError(f"Horváth LP failed: {solution.message}")
    return solution.x[:order]


def yule_walker_coefficients(series, order):
    """Solve the centered Yule-Walker equations with denominator T."""
    series = np.asarray(series, dtype=float)
    if series.ndim != 1 or not np.all(np.isfinite(series)):
        raise ValueError("Use a finite one-dimensional time series.")
    if not 1 <= order < len(series):
        raise ValueError("Require 1 <= order < the sample size.")
    centered = series - np.mean(series)
    sample_size = len(centered)
    autocovariance = np.array([
        centered[lag:] @ centered[:sample_size - lag] / sample_size
        for lag in range(order + 1)
    ])
    if autocovariance[0] <= np.finfo(float).eps:
        raise ValueError("Cannot fit Yule-Walker coefficients to a constant series.")
    return np.linalg.solve(toeplitz(autocovariance[:-1]), autocovariance[1:])


def conditional_gaussian_qml(series, order):
    """Maximize the Gaussian conditional quasi-log-likelihood, intercept zero."""
    series = np.asarray(series, dtype=float)
    if series.ndim != 1 or not np.all(np.isfinite(series)):
        raise ValueError("Use a finite one-dimensional time series.")
    if not 1 <= order < len(series):
        raise ValueError("Require 1 <= order < the sample size.")
    windows = sliding_window_point_cloud(series, order + 1)
    response = windows[:, 0]
    lagged_values = windows[:, 1:]
    coefficients, _, rank, _ = np.linalg.lstsq(lagged_values, response, rcond=None)
    if rank < order:
        raise ValueError("The conditional regression design is rank deficient.")
    residuals = response - lagged_values @ coefficients
    residual_variance = float(residuals @ residuals / len(response))
    if residual_variance <= 0 or not np.isfinite(residual_variance):
        raise ValueError("A positive residual variance is required for the quasi-likelihood.")
    maximized_log_likelihood = float(
        -0.5 * len(response) * (np.log(2.0 * np.pi) + np.log(residual_variance) + 1.0)
    )
    return coefficients, residual_variance, maximized_log_likelihood


def horvath_linf_coefficients(series, order, use_convex_hull=True):
    """Estimate AR coefficients and symmetric noise half-width by an L-infinity LP."""
    series = np.asarray(series, dtype=float)
    if series.ndim != 1 or not np.all(np.isfinite(series)):
        raise ValueError("Use a finite one-dimensional time series.")
    if not 1 <= order < len(series):
        raise ValueError("Require 1 <= order < the sample size.")
    full_cloud = sliding_window_point_cloud(series, order + 1)
    constraint_cloud = full_cloud
    if use_convex_hull and len(full_cloud) > order + 1:
        try:
            hull = ConvexHull(full_cloud)
            constraint_cloud = full_cloud[np.sort(hull.vertices)]
        except QhullError:
            pass
    response = constraint_cloud[:, 0]
    lags = constraint_cloud[:, 1:]
    minus_ones = -np.ones(len(response))
    # Variables: (phi_1, ..., phi_p, h). Bound both signs of the residual.
    inequalities = np.vstack((
        np.column_stack((-lags, minus_ones)),
        np.column_stack((lags, minus_ones)),
    ))
    inequality_bounds = np.r_[-response, response]
    objective = np.r_[np.zeros(order), 1.0]
    result = linprog(
        objective, A_ub=inequalities, b_ub=inequality_bounds,
        bounds=[(None, None)] * order + [(0.0, None)], method="highs",
        options={"primal_feasibility_tolerance": 1e-9,
                 "dual_feasibility_tolerance": 1e-9},
    )
    if not result.success:
        raise RuntimeError(f"Horvath L-infinity LP failed: {result.message}")
    coefficients = result.x[:order]
    radius = float(result.x[-1])
    full_residuals = full_cloud[:, 0] - full_cloud[:, 1:] @ coefficients
    maximum_absolute_residual = float(np.max(np.abs(full_residuals)))
    dual_objective = float(inequality_bounds @ result.ineqlin.marginals)
    duality_gap = float(abs(radius - dual_objective))
    check_tolerance = 1e-8 * (1.0 + abs(radius))
    if (not np.all(np.isfinite(coefficients)) or not np.isfinite(radius)
            or not np.isfinite(duality_gap) or radius < 0
            or abs(maximum_absolute_residual - radius) > check_tolerance
            or duality_gap > check_tolerance):
        raise RuntimeError("Horvath LP failed its full-data feasibility or optimality check.")
    return coefficients, radius, duality_gap
