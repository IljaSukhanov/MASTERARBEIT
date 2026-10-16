from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import minimize
from scipy.signal import residuez

from .functions import companion_matrix
from .support_geometry import (
    ar_impulse_response, ar_window_zonotope_generators,
    theoretical_ar_width_in_direction,
)


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


def population_minimum_low_dimension(phi, k, history, half_width):
    """Enumerate facet-normal candidates of a finite AR support zonotope."""
    if k not in (2, 3):
        raise ValueError("This population calculation supports k=2 or k=3.")
    generators = ar_window_zonotope_generators(
        phi, embedding_dimension=k, history=history, half_width=half_width,
    )
    unit_generators = generators / np.linalg.norm(generators, axis=0)
    if k == 2:
        candidates = np.column_stack((-unit_generators[1], unit_generators[0]))
    else:
        first, second = np.triu_indices(generators.shape[1], k=1)
        candidates = np.cross(
            unit_generators[:, first].T, unit_generators[:, second].T,
        )
    candidate_norms = np.linalg.norm(candidates, axis=1)
    independent = candidate_norms > 1e-12
    candidates = candidates[independent] / candidate_norms[independent, None]
    if len(candidates) == 0:
        raise RuntimeError("No nondegenerate population normal was found.")
    candidate_widths = 2.0 * np.abs(generators.T @ candidates.T).sum(axis=0)
    best = int(np.argmin(candidate_widths))
    direction = candidates[best]
    # Reevaluate through the shared population-width function so that the
    # reference retains its full impulse-response history.
    minimum = theoretical_ar_width_in_direction(
        phi, direction, half_width=half_width, history=history,
    )
    return minimum, direction


def _population_generator_matrix(impulse_response, dimension):
    """Rows are g_j=(psi_j, psi_{j-1}, ..., psi_{j-k+1})."""
    padded = np.r_[np.zeros(dimension - 1), impulse_response]
    return np.column_stack([
        padded[dimension - 1 - lag:dimension - 1 - lag + len(impulse_response)]
        for lag in range(dimension)
    ])


def _normalized_rows(vectors):
    vectors = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(vectors, axis=1)
    return vectors[norms > 1e-12] / norms[norms > 1e-12, None]


def _pole_factor_starts(coefficients, dimension):
    if dimension == 1:
        return []
    poles = np.linalg.eigvals(companion_matrix(coefficients))
    starts = []
    for indices in itertools.combinations(range(len(coefficients)), dimension - 1):
        factor = np.poly(poles[list(indices)])
        if np.max(np.abs(factor.imag)) < 1e-8:
            starts.append(factor.real / np.linalg.norm(factor.real))
    return starts


def _smooth_population_refinement(generator_matrix, start):
    direction = np.asarray(start, dtype=float)
    direction /= np.linalg.norm(direction)
    scale = max(float(np.max(np.linalg.norm(generator_matrix, axis=1))), 1.0)

    for epsilon_factor in (1e-3, 1e-5, 1e-7):
        epsilon = epsilon_factor * scale

        def objective_and_gradient(vector):
            vector_norm = np.linalg.norm(vector)
            unit_vector = vector / vector_norm
            projections = generator_matrix @ unit_vector
            smoothed_absolute = np.sqrt(projections**2 + epsilon**2)
            objective = 2.0 * np.sum(smoothed_absolute - epsilon)
            sphere_gradient = 2.0 * (
                generator_matrix.T @ (projections / smoothed_absolute)
            )
            gradient = (
                sphere_gradient
                - unit_vector * (unit_vector @ sphere_gradient)
            ) / vector_norm
            return objective, gradient

        result = minimize(
            objective_and_gradient,
            direction,
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": 120, "ftol": 1e-13,
                "gtol": 1e-9, "maxls": 40,
            },
        )
        direction = result.x / np.linalg.norm(result.x)
    return direction


def _facet_normal_refinement(generator_matrix, directions, nearest=12):
    row_norms = np.linalg.norm(generator_matrix, axis=1)
    active = row_norms > 1e-11 * row_norms.max()
    rows = generator_matrix[active]
    unit_rows = rows / row_norms[active, None]
    candidates = [direction / np.linalg.norm(direction) for direction in directions]

    for direction in directions:
        distances = np.abs(unit_rows @ direction)
        count = min(nearest, len(rows))
        nearest_rows = rows[np.argpartition(distances, count - 1)[:count]]
        for indices in itertools.combinations(
            range(len(nearest_rows)), generator_matrix.shape[1] - 1
        ):
            _, singular_values, right_vectors = np.linalg.svd(
                nearest_rows[list(indices)], full_matrices=True
            )
            if (
                len(singular_values)
                and singular_values[-1] < 1e-10 * singular_values[0]
            ):
                continue
            normal = right_vectors[-1]
            normal_norm = np.linalg.norm(normal)
            if normal_norm > 1e-12:
                candidates.append(normal / normal_norm)

    candidate_matrix = np.asarray(candidates)
    widths = 2.0 * np.sum(
        np.abs(generator_matrix @ candidate_matrix.T), axis=0
    )
    best_indices = np.argsort(widths)[:8]
    return [candidate_matrix[index] for index in best_indices]


def approximate_population_width_curve(coefficients, seed, *, k_values):
    """Approximate m_k for k<=p and use the exact plateau for k>=p+1."""
    coefficients = np.asarray(coefficients, dtype=float)
    order = len(coefficients)
    spectral_radius = float(max(abs(np.linalg.eigvals(companion_matrix(coefficients)))))
    evaluation_history = min(
        250_000,
        max(20_000, int(np.ceil(np.log(1e-10) / np.log(spectral_radius)))),
    )
    optimization_history = min(3_000, evaluation_history)
    evaluation_impulse = ar_impulse_response(coefficients, evaluation_history)
    optimization_impulse = evaluation_impulse[:optimization_history + 1]
    plateau = 2.0 / np.linalg.norm(np.r_[1.0, -coefficients])
    generator = np.random.default_rng(seed)

    widths = []
    directions = []
    half_history_increments = []
    previous_direction = None

    for dimension in range(1, order + 1):
        optimization_generators = _population_generator_matrix(
            optimization_impulse, dimension
        )
        evaluation_generators = _population_generator_matrix(
            evaluation_impulse, dimension
        )

        starts = []
        if previous_direction is not None:
            starts.append(np.r_[previous_direction, 0.0])
        starts.extend(np.eye(dimension))
        starts.append(np.r_[1.0, -coefficients[:dimension - 1]])
        starts.extend(_pole_factor_starts(coefficients, dimension))
        starts.extend(_normalized_rows(generator.normal(size=(48, dimension))))
        starts = _normalized_rows(starts)

        start_widths = 2.0 * np.sum(
            np.abs(optimization_generators @ starts.T), axis=0
        )
        best_starts = starts[np.argsort(start_widths)[:8]]
        smooth_directions = [
            _smooth_population_refinement(optimization_generators, start)
            for start in best_starts
        ]
        candidate_directions = _facet_normal_refinement(
            optimization_generators, smooth_directions, nearest=12
        ) + smooth_directions
        candidate_matrix = np.asarray(candidate_directions)
        evaluation_widths = 2.0 * np.sum(
            np.abs(evaluation_generators @ candidate_matrix.T), axis=0
        )
        best_index = int(np.argmin(evaluation_widths))
        direction = candidate_matrix[best_index]
        width = float(evaluation_widths[best_index])
        half_width = float(2.0 * np.sum(np.abs(
            evaluation_generators[:len(evaluation_generators) // 2] @ direction
        )))

        widths.append(width)
        directions.append(direction)
        half_history_increments.append(width - half_width)
        previous_direction = direction

    widths.extend([plateau] * (len(k_values) - order))
    return {
        "widths": np.asarray(widths),
        "directions": directions,
        "half_history_increments": np.asarray(half_history_increments),
        "evaluation_history": evaluation_history,
        "spectral_radius": spectral_radius,
        "plateau": plateau,
    }
