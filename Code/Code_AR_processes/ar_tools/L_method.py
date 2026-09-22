from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog

__all__ = ["l_method_selector"]


def _validated_widths(minimum_widths):
    widths = np.asarray(minimum_widths, dtype=float)
    if widths.ndim != 1:
        raise ValueError("minimum_widths must be one-dimensional.")
    if len(widths) < 4:
        raise ValueError("minimum_widths must contain at least four values.")
    if not np.all(np.isfinite(widths)):
        raise ValueError("minimum_widths must contain only finite values.")
    return widths


def _validated_boolean(name, value):
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean.")
    return bool(value)


def _least_squares_fit(k_values, widths, *, affine):
    if affine:
        design = np.column_stack((np.ones(len(k_values)), k_values))
    else:
        design = np.ones((len(k_values), 1))
    coefficients, *_ = np.linalg.lstsq(design, widths, rcond=None)
    fitted_values = design @ coefficients
    intercept = float(coefficients[0])
    slope = float(coefficients[1]) if affine else 0.0
    return intercept, slope, fitted_values


def _least_absolute_deviations_fit(k_values, widths, *, affine):
    if not affine:
        intercept = float(np.median(widths))
        return intercept, 0.0, np.full(len(widths), intercept)

    design = np.column_stack((np.ones(len(k_values)), k_values))
    observation_count = len(widths)
    identity = np.eye(observation_count)
    # Variables are (intercept, slope, u_1, ..., u_n), where
    # u_i >= |width_i - intercept - slope * k_i|.
    objective = np.r_[0.0, 0.0, np.ones(observation_count)]
    constraints = np.vstack((
        np.column_stack((design, -identity)),
        np.column_stack((-design, -identity)),
    ))
    bounds = [(None, None), (None, None)] + [
        (0.0, None)
    ] * observation_count
    fit = linprog(
        objective,
        A_ub=constraints,
        b_ub=np.r_[widths, -widths],
        bounds=bounds,
        method="highs",
    )
    if not fit.success:
        raise RuntimeError(f"LAD affine fit failed: {fit.message}")

    intercept, slope = fit.x[:2]
    fitted_values = design @ fit.x[:2]
    return float(intercept), float(slope), fitted_values


def _segment_fit(k_values, widths, *, affine, squared_loss):
    if squared_loss:
        intercept, slope, fitted_values = _least_squares_fit(
            k_values, widths, affine=affine
        )
    else:
        intercept, slope, fitted_values = _least_absolute_deviations_fit(
            k_values, widths, affine=affine
        )

    residuals = widths - fitted_values
    if squared_loss:
        loss_sum = float(residuals @ residuals)
        average_error = float(np.sqrt(loss_sum / len(widths)))
    else:
        loss_sum = float(np.sum(np.abs(residuals)))
        average_error = float(loss_sum / len(widths))
    return {
        "intercept": intercept,
        "slope": slope,
        "fitted_values": fitted_values,
        "loss_sum": loss_sum,
        "average_error": average_error,
    }


def l_method_selector(
    minimum_widths,
    *,
    weighted,
    common_breakpoint,
    affine_affine,
    squared_loss,
):
    """Select an elbow from ``(m_1, ..., m_K)``.

    The four booleans select one of sixteen variants:

    ``weighted``
        Use the segment-size-weighted RMSE for squared loss or the
        segment-size-weighted MAE for absolute loss. If false, use total
        squared error or total absolute error, respectively.
    ``common_breakpoint``
        If true, the left segment is ``1, ..., q`` and the right segment is
        ``q, ..., K``. The observation at ``q`` is used in both independent
        fits; equality of their fitted values at ``q`` is not imposed. If
        false, the right segment starts at ``q + 1``.
    ``affine_affine``
        Fit affine models to both segments. If false, fit an affine left
        segment and a constant right segment.
    ``squared_loss``
        Fit segments by least squares. If false, fit them by least absolute
        deviations.

    Both segments contain at least two observations. Exact score ties favor
    the smaller selected AR order. With a common point, the selected order is
    ``q - 1``; with disjoint segments, it is ``q``. For absolute loss, the
    weighted score is the total absolute error divided by the constant total
    segment size, so weighting changes the score's scale but not the selected
    order.
    """
    widths = _validated_widths(minimum_widths)
    weighted = _validated_boolean("weighted", weighted)
    common_breakpoint = _validated_boolean(
        "common_breakpoint", common_breakpoint
    )
    affine_affine = _validated_boolean("affine_affine", affine_affine)
    squared_loss = _validated_boolean("squared_loss", squared_loss)

    embedding_dimensions = np.arange(1, len(widths) + 1, dtype=float)
    maximum_breakpoint = (
        len(widths) - 1 if common_breakpoint else len(widths) - 2
    )
    candidate_rows = []
    candidate_fits = []

    for breakpoint in range(2, maximum_breakpoint + 1):
        left_stop = breakpoint
        right_start = breakpoint - 1 if common_breakpoint else breakpoint
        left_k = embedding_dimensions[:left_stop]
        right_k = embedding_dimensions[right_start:]
        left_widths = widths[:left_stop]
        right_widths = widths[right_start:]

        left_fit = _segment_fit(
            left_k,
            left_widths,
            affine=True,
            squared_loss=squared_loss,
        )
        right_fit = _segment_fit(
            right_k,
            right_widths,
            affine=affine_affine,
            squared_loss=squared_loss,
        )

        left_count = len(left_widths)
        right_count = len(right_widths)
        if weighted:
            score = (
                left_count * left_fit["average_error"]
                + right_count * right_fit["average_error"]
            ) / (left_count + right_count)
        else:
            score = left_fit["loss_sum"] + right_fit["loss_sum"]

        selected_order = breakpoint - 1 if common_breakpoint else breakpoint
        plateau_onset = selected_order + 1
        candidate_rows.append({
            "breakpoint": breakpoint,
            "selected_order": selected_order,
            "plateau_onset": plateau_onset,
            "score": float(score),
            "left_count": left_count,
            "right_count": right_count,
            "left_loss_sum": left_fit["loss_sum"],
            "right_loss_sum": right_fit["loss_sum"],
            "left_average_error": left_fit["average_error"],
            "right_average_error": right_fit["average_error"],
            "left_intercept": left_fit["intercept"],
            "left_slope": left_fit["slope"],
            "right_intercept": right_fit["intercept"],
            "right_slope": right_fit["slope"],
        })
        if common_breakpoint:
            fitted_values = np.r_[
                left_fit["fitted_values"][:-1],
                right_fit["fitted_values"],
            ]
        else:
            fitted_values = np.r_[
                left_fit["fitted_values"],
                right_fit["fitted_values"],
            ]
        candidate_fits.append({
            "left_k": left_k.copy(),
            "left_fitted_values": left_fit["fitted_values"],
            "right_k": right_k.copy(),
            "right_fitted_values": right_fit["fitted_values"],
            "fitted_values": fitted_values,
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
        "weighted": weighted,
        "common_breakpoint": common_breakpoint,
        "affine_affine": affine_affine,
        "squared_loss": squared_loss,
        "candidate_scores": pd.DataFrame(candidate_rows),
    }
