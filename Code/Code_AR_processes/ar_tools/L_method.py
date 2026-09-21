"""L-method elbow selectors and their fitting helpers for AR width curves."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog

__all__ = [
    "horizontal_tail_lad_selector",
    "two_affine_l_method_selector",
    "affine_constant_l_method_selector",
    "shared_breakpoint_l_method_selector",
]


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


def horizontal_tail_lad_selector(k_values, widths):
    """Fit c + s(q-k)_+ by LAD for each q, subject to s >= 0."""
    k_values = np.asarray(k_values, dtype=float)
    widths = np.asarray(widths, dtype=float)
    candidate_rows = []

    for selected_q in k_values[1:-1].astype(int):
        hinge = np.maximum(selected_q - k_values, 0.0)
        sample_count = len(widths)
        identity = np.eye(sample_count)
        # Variables are (c, s, u_1, ..., u_K), with u_k >= |residual_k|.
        objective = np.concatenate(([0.0, 0.0], np.ones(sample_count)))
        constraints = np.vstack((
            np.column_stack((-np.ones(sample_count), -hinge, -identity)),
            np.column_stack(( np.ones(sample_count),  hinge, -identity)),
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
            raise RuntimeError(f"LAD fit failed for q={selected_q}: {fit.message}")
        level, slope = fit.x[:2]
        fitted_values = level + slope * hinge
        candidate_rows.append({
            "selected_q": selected_q,
            "score": float(np.sum(np.abs(widths - fitted_values))),
            "level": float(level),
            "slope": float(slope),
            "fitted_values": fitted_values,
        })

    best = min(candidate_rows, key=lambda row: (row["score"], row["selected_q"]))
    return {
        **best,
        "selected_order": best["selected_q"] - 1,
        "candidate_scores": pd.DataFrame(candidate_rows).drop(
            columns="fitted_values"
        ),
    }


def shared_breakpoint_l_method_selector(
    k_values, widths, *, tail_model="affine", score_criterion="weighted_rmse"
):
    """Compare independent least-squares fits sharing the point k=q+1.

    Candidate AR orders are q=1,...,K-2, giving at least two observations
    in each segment. Both segments include k=q+1; their fitted values there
    need not agree. No slope, continuity, or order penalty is imposed.
    ``weighted_rmse`` weights segment RMSEs by their observation counts.
    ``squared_error`` sums segment squared errors, counting the shared point
    in each fit. Exact score ties favor the smaller order.
    """
    k_values, widths = _validated_elbow_data(k_values, widths)
    if tail_model not in {"affine", "constant"}:
        raise ValueError("tail_model must be 'affine' or 'constant'.")
    if score_criterion not in {"weighted_rmse", "squared_error"}:
        raise ValueError("score_criterion must be 'weighted_rmse' or 'squared_error'.")

    rows, fits = [], []
    for breakpoint_index in range(1, len(k_values) - 1):
        head_k, tail_k = k_values[:breakpoint_index + 1], k_values[breakpoint_index:]
        head_y, tail_y = widths[:breakpoint_index + 1], widths[breakpoint_index:]
        head_intercept, head_slope, head_fit, head_rmse = _affine_fit(head_k, head_y)
        if tail_model == "affine":
            tail_intercept, tail_slope, tail_fit, tail_rmse = _affine_fit(tail_k, tail_y)
        else:
            tail_intercept, tail_slope = float(np.mean(tail_y)), 0.0
            tail_fit = np.full(len(tail_k), tail_intercept)
            tail_rmse = float(np.sqrt(np.mean((tail_y - tail_fit) ** 2)))
        weighted_rmse = (
            len(head_k) * head_rmse + len(tail_k) * tail_rmse
        ) / (len(head_k) + len(tail_k))
        squared_error = float(
            np.sum((head_y - head_fit) ** 2) + np.sum((tail_y - tail_fit) ** 2)
        )
        plateau_onset = int(k_values[breakpoint_index])
        row = {
            "selected_q": plateau_onset,
            "selected_order": plateau_onset - 1,
            "weighted_rmse": float(weighted_rmse),
            "squared_error": squared_error,
            "head_intercept": head_intercept,
            "head_slope": head_slope,
            "tail_intercept": tail_intercept,
            "tail_slope": tail_slope,
        }
        row["score"] = row[score_criterion]
        rows.append(row)
        fits.append({
            "head_k": head_k.copy(), "head_fitted_values": head_fit,
            "tail_k": tail_k.copy(), "tail_fitted_values": tail_fit,
        })
    best_index = min(range(len(rows)), key=lambda i: (rows[i]["score"], rows[i]["selected_order"]))
    return {
        **rows[best_index], **fits[best_index],
        "tail_model": tail_model, "score_criterion": score_criterion,
        "candidate_scores": pd.DataFrame(rows),
    }


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
