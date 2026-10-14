"""Globally enumerate support-facet normals in dimensions one to three.

For a finite full-dimensional centrally symmetric zonotope, width is a positive
linear function on each spherical sign cell. Its restriction to a great-circle
arc has second derivative minus its value, so its minimum lies at a cell vertex.
In dimension three these vertices are normals to pairs of generators. For each
generator, project onto its perpendicular plane and perform a two-dimensional
angular sweep, evaluating all pairs in O(N^2 log N) rather than O(N^3).

This is a derivation used here. Floating-point operations are not interval
certified; the separate AR tail bound controls truncation, not rounding error.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import residuez

from .support_geometry import ar_impulse_response


def minimum_zonotope_width_2d(generators):
    g = np.asarray(generators, dtype=float)
    if g.ndim != 2 or g.shape[1] != 2:
        raise ValueError("Expected a matrix of two-dimensional generator rows.")
    norms = np.linalg.norm(g, axis=1)
    g = g[norms > 1e-250].copy()
    if not len(g):
        raise ValueError("At least one nonzero generator is required.")
    flip = (g[:, 1] < 0) | ((g[:, 1] == 0) & (g[:, 0] < 0))
    g[flip] *= -1
    g = g[np.argsort(np.arctan2(g[:, 1], g[:, 0]), kind="stable")]
    normals = np.column_stack((-g[:, 1], g[:, 0]))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    before = np.vstack((np.zeros(2), np.cumsum(g, axis=0)[:-1]))
    widths = 2 * np.sum(normals * (g.sum(axis=0) - 2 * before), axis=1)
    best = int(np.argmin(widths))
    direction = normals[best]
    actual = float(2 * np.sum(np.abs(np.asarray(generators) @ direction)))
    residual = abs(actual - float(widths[best]))
    return actual, direction, residual


def minimum_zonotope_width_3d(generators):
    g = np.asarray(generators, dtype=float)
    if g.ndim != 2 or g.shape[1] != 3:
        raise ValueError("Expected a matrix of three-dimensional generator rows.")
    norm = np.linalg.norm(g, axis=1)
    if np.any(norm == 0):
        g = g[norm > 0]
        norm = norm[norm > 0]
    best, best_direction, max_residual = np.inf, None, 0.0
    for i, anchor in enumerate(g / norm[:, None]):
        reference = np.eye(3)[int(np.argmin(np.abs(anchor)))]
        e = np.cross(anchor, reference)
        e /= np.linalg.norm(e)
        f = np.cross(anchor, e)
        basis = np.column_stack((e, f))
        projected = g @ basis
        projected[i] = 0.0
        _, plane_direction, residual = minimum_zonotope_width_2d(projected)
        direction = basis @ plane_direction
        direction /= np.linalg.norm(direction)
        width = float(2 * np.sum(np.abs(g @ direction)))
        max_residual = max(max_residual, residual)
        if width < best:
            best, best_direction = width, direction
    return best, best_direction, max_residual


def population_ar3_widths(phi, *, max_k=10, tail_target=1e-8):
    phi = np.asarray(phi, dtype=float)
    if phi.shape != (3,) or phi[-1] == 0 or max_k < 5:
        raise ValueError("Expected a minimal AR(3) vector and max_k >= 5.")
    residues, poles, direct = residuez([1.0], np.r_[1.0, -phi])
    rho = float(np.max(np.abs(poles)))
    if not 0 < rho < 1 or len(direct):
        raise ValueError("Expected a stable proper AR transfer function.")
    pair_distance = np.abs(poles[:, None] - poles[None, :])
    np.fill_diagonal(pair_distance, np.inf)
    if pair_distance.min() < 1e-6:
        raise ValueError("Repeated or nearly repeated poles require a different tail bound.")
    constant = float(np.sum(np.abs(residues)))
    n = max(4, int(np.ceil(np.log(tail_target * (1 - rho)
                                  / (2 * np.sqrt(3) * constant)) / np.log(rho))) + 2)
    psi = ar_impulse_response(phi, n - 1)
    formula = np.real(np.sum(residues[None, :] * poles[None, :]
                            ** np.arange(n)[:, None], axis=1))
    formula_error = float(np.max(np.abs(psi - formula)))
    np.testing.assert_allclose(psi, formula, rtol=1e-7, atol=1e-10)

    values, directions, tail_bounds, sweep_residuals = [], [], [], []
    for k in (1, 2, 3):
        padded = np.r_[np.zeros(k - 1), psi]
        g = np.column_stack([padded[k - 1 - j:k - 1 - j + n] for j in range(k)])
        if k == 1:
            value, direction, residual = float(2 * np.abs(psi).sum()), np.ones(1), 0.0
        elif k == 2:
            value, direction, residual = minimum_zonotope_width_2d(g)
        else:
            value, direction, residual = minimum_zonotope_width_3d(g)
        tail = float(2 * np.sqrt(k) * constant * rho ** (n - k + 1) / (1 - rho))
        assert tail <= tail_target * (1 + 1e-10)
        values.append(value)
        directions.append(direction)
        tail_bounds.append(tail)
        sweep_residuals.append(residual)

    plateau = float(2 / np.linalg.norm(np.r_[1.0, -phi]))
    assert np.all(np.diff(np.r_[values, plateau]) <= tail_target * 2)
    assert values[-1] > plateau
    return {
        "widths": np.r_[values, np.repeat(plateau, max_k - 3)],
        "directions": directions,
        "tail_bounds": np.r_[tail_bounds, np.zeros(max_k - 3)],
        "generator_count": n, "spectral_radius": rho,
        "residue_constant": constant, "pole_formula_max_error": formula_error,
        "maximum_sweep_roundoff_residual": max(sweep_residuals),
    }
