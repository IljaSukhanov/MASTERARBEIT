"""Experimental coefficient coordinates for the original spherical width.

For newest-first points (y_i, x_i), set r_i(phi) = y_i - x_i @ phi,
s = sqrt(1 + ||phi||^2), and F(phi) = (max(r) - min(r)) / s.
Thus F(phi) = w((1, -phi) / s); the denominator is never dropped.

The formulas below are derived in ar3_coefficient_space_descent.ipynb.
The convex max/affine subgradient rules used before normalization are in
Boyd et al., Subgradients (2022), sections 3.3 and 3.4:
https://stanford.edu/class/ee364b/lectures/subgradients_notes.pdf
F is generally nonconvex. At ties the selected active branch gives a
generalized gradient, not a global convex subgradient of F.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "direction_from_coefficients",
    "normalized_width_value_gradient",
    "coefficient_width_descent",
]


def direction_from_coefficients(coefficients):
    """Return the unit representative (1, -phi)/sqrt(1 + ||phi||^2)."""
    phi = np.asarray(coefficients, dtype=float)
    if phi.ndim != 1 or phi.size < 1 or not np.all(np.isfinite(phi)):
        raise ValueError("coefficients must be a nonempty finite vector.")
    scale = float(np.hypot.reduce(np.r_[1.0, phi]))
    if not np.isfinite(scale):
        raise ValueError("The coefficient norm is not representable.")
    return np.r_[1.0, -phi] / scale


def _validated_inputs(points, coefficients):
    points = np.asarray(points, dtype=float)
    if (points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2
            or not np.all(np.isfinite(points))):
        raise ValueError("points must be a finite (n, p+1) array, n >= 2, p >= 1.")
    phi = np.asarray(coefficients, dtype=float)
    if phi.shape != (points.shape[1] - 1,) or not np.all(np.isfinite(phi)):
        raise ValueError("coefficients must be finite and match the number of lags.")
    return points, phi


def _evaluate(points, phi):
    direction = direction_from_coefficients(phi)
    projections = points @ direction
    imax, imin = int(np.argmax(projections)), int(np.argmin(projections))
    value = float(projections[imax] - projections[imin])
    difference = points[imax] - points[imin]
    # u_0 = 1/s; F = R/s. This is -g_lag/s - R*phi/s^3.
    gradient = -difference[1:] * direction[0] - value * phi * direction[0]**2
    return value, gradient, direction, difference


def normalized_width_value_gradient(points, coefficients):
    """Evaluate the original unit-direction width and its active-branch gradient.

    At unique extrema this is the ordinary gradient. At ties it is a
    selected generalized gradient of the locally Lipschitz max expression.
    """
    points, phi = _validated_inputs(points, coefficients)
    value, gradient, _, _ = _evaluate(points, phi)
    return value, gradient


def coefficient_width_descent(
    points,
    initial_coefficients,
    *,
    iterations=5_000,
    initial_step=0.1,
    geometric_decay=0.997,
    geometry="euclidean",
    patience=None,
    improvement_tolerance=0.0,
    record_history=True,
):
    """Take each step in phi while retaining F(phi)=w((1,-phi)/||...||).

    geometry='euclidean': phi <- phi - eta * grad_phi F.
    geometry='sphere': apply the existing sphere tangent step, then invert
    radial projection at every step. This is only a change of coordinates.
    Histories evaluate states before updates, like the original optimizer.
    No unrecorded final update is returned. Best states are selected by F.
    A pole of the coefficient chart stops the 'sphere' version explicitly.
    """
    points, phi = _validated_inputs(points, initial_coefficients)
    phi = phi.copy()
    if isinstance(iterations, (bool, np.bool_)) or not isinstance(iterations, (int, np.integer)) or iterations < 1:
        raise ValueError("iterations must be a positive integer.")
    if not np.isfinite(initial_step) or initial_step <= 0:
        raise ValueError("initial_step must be finite and positive.")
    if not np.isfinite(geometric_decay) or not 0 < geometric_decay < 1:
        raise ValueError("geometric_decay must lie strictly between zero and one.")
    if geometry not in {"euclidean", "sphere"}:
        raise ValueError("geometry must be 'euclidean' or 'sphere'.")
    if patience is None:
        patience = iterations + 1
    if isinstance(patience, (bool, np.bool_)) or not isinstance(patience, (int, np.integer)) or patience < 1:
        raise ValueError("patience must be a positive integer or None.")
    if not np.isfinite(improvement_tolerance) or improvement_tolerance < 0:
        raise ValueError("improvement_tolerance must be finite and nonnegative.")
    if not isinstance(record_history, (bool, np.bool_)):
        raise ValueError("record_history must be a boolean.")

    history = []
    best_value, best_phi = np.inf, phi.copy()
    stale = 0
    reason = "iteration_limit"
    minimum_abs_u0 = np.inf
    evaluations = 0
    for iteration in range(iterations):
        value, gradient, direction, difference = _evaluate(points, phi)
        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            if evaluations == 0:
                raise FloatingPointError("The initial width or gradient is not finite.")
            reason = "nonfinite_evaluation"
            break
        evaluations += 1
        minimum_abs_u0 = min(minimum_abs_u0, float(direction[0]))
        if value < best_value - improvement_tolerance:
            best_value, best_phi = value, phi.copy()
            stale = 0
        else:
            stale += 1
        step = initial_step * geometric_decay**iteration
        if record_history:
            history.append((value, best_value, step, *phi))
        if stale >= patience:
            reason = "patience"
            break
        if iteration == iterations - 1:
            break
        if geometry == "euclidean":
            candidate_phi = phi - step * gradient
        else:
            # v=(1+eta*F)u-eta*g; phi_next=-v_lag/v_0. Normalizing
            # v to unit length cancels exactly in these coordinate ratios.
            candidate = direction - step * (difference - (direction @ difference) * direction)
            candidate_norm = float(np.linalg.norm(candidate))
            # Cancellation in the tangent step can leave a few ulps instead
            # of an exact zero at a chart pole. Stop rather than divide it.
            if abs(candidate[0]) <= 32 * np.finfo(float).eps * candidate_norm:
                reason = "chart_pole"
                break
            candidate_phi = -candidate[1:] / candidate[0]
        if not np.all(np.isfinite(candidate_phi)):
            reason = "nonfinite_update"
            break
        phi = candidate_phi

    result = {
        "minimum_width": float(best_value),
        "coefficients": best_phi,
        "direction": direction_from_coefficients(best_phi),
        "geometry": geometry,
        "termination_reason": reason,
        "evaluations": evaluations,
        "minimum_abs_u0": float(minimum_abs_u0),
    }
    if record_history:
        trace = np.asarray(history, dtype=float)
        result.update(
            width_history=trace[:, 0], best_width_history=trace[:, 1],
            step_history=trace[:, 2], coefficient_history=trace[:, 3:],
        )
    return result
