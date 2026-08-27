"""Archived Rényi and Amari alpha-divergence experiments.

This module is intentionally kept outside the active ``kde_tools`` package.
It preserves the parametrized-divergence implementations for possible future
experiments without exposing them in the current KDE workflow.
"""

from __future__ import annotations

import numpy as np


__all__ = [
    "alpha_divergence_matrix",
    "inverse_power_similarity_matrix",
    "renyi_divergence_matrix",
    "renyi_similarity_matrix",
    "symmetrized_renyi_divergence_matrix",
    "symmetrized_renyi_similarity_fft",
]


def _trapezoidal_integral(
    values: np.ndarray,
    x_grid: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    """Integrate sampled values along one axis with the trapezoidal rule."""
    return np.trapezoid(np.asarray(values, dtype=float), x=np.asarray(x_grid), axis=axis)


def _trapezoidal_weights(x_grid: np.ndarray) -> np.ndarray:
    """Return composite-trapezoidal weights for a one-dimensional grid."""
    x_grid = np.asarray(x_grid, dtype=float)
    if x_grid.ndim != 1 or len(x_grid) < 2:
        raise ValueError("x_grid must be one-dimensional with at least two nodes.")

    steps = np.diff(x_grid)
    if np.any(steps <= 0.0):
        raise ValueError("x_grid must be strictly increasing.")

    weights = np.empty_like(x_grid)
    weights[0] = steps[0] / 2.0
    weights[-1] = steps[-1] / 2.0
    weights[1:-1] = (steps[:-1] + steps[1:]) / 2.0
    return weights


def _log_trapezoidal_integral(
    log_values: np.ndarray,
    x_grid: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    """Evaluate a trapezoidal integral stably from logarithmic values."""
    log_values = np.asarray(log_values, dtype=float)
    axis = int(axis)
    if axis < 0:
        axis += log_values.ndim
    if axis < 0 or axis >= log_values.ndim:
        raise ValueError("axis is out of bounds for log_values.")

    weights = _trapezoidal_weights(x_grid)
    if log_values.shape[axis] != len(weights):
        raise ValueError("x_grid must match the selected integration axis.")

    weight_shape = [1] * log_values.ndim
    weight_shape[axis] = len(weights)
    weighted_log_values = log_values + np.log(weights).reshape(weight_shape)
    maximum = np.max(weighted_log_values, axis=axis, keepdims=True)
    return np.squeeze(
        maximum
        + np.log(
            np.sum(
                np.exp(weighted_log_values - maximum),
                axis=axis,
                keepdims=True,
            )
        ),
        axis=axis,
    )


def renyi_divergence_matrix(
    density_values: np.ndarray,
    x_grid: np.ndarray,
    alpha: float,
    bandwidths: np.ndarray | None = None,
    eps: float = 1e-300,
) -> np.ndarray:
    """Compute all directed Rényi divergences by stable quadrature."""
    alpha = float(alpha)
    if alpha <= 0.0:
        raise ValueError("alpha must be strictly positive.")
    if eps <= 0.0:
        raise ValueError("eps must be strictly positive.")

    values = np.asarray(density_values, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    if values.ndim != 2:
        raise ValueError("density_values must be a two-dimensional array.")
    if values.shape[1] != len(x_grid):
        raise ValueError("x_grid must match the density grid axis.")
    if np.any(values < 0.0):
        raise ValueError("Density values must be nonnegative.")

    values = np.maximum(values, eps)
    log_values = np.log(values)
    number_of_densities = len(values)
    divergence = np.empty((number_of_densities, number_of_densities), dtype=float)

    if np.isclose(alpha, 1.0, rtol=0.0, atol=1e-12):
        for i in range(number_of_densities):
            divergence[i] = _trapezoidal_integral(
                values=values[i][None, :]
                * (log_values[i][None, :] - log_values),
                x_grid=x_grid,
                axis=1,
            )
    else:
        for i in range(number_of_densities):
            log_integrands = (
                alpha * log_values[i][None, :]
                + (1.0 - alpha) * log_values
            )
            log_integrals = _log_trapezoidal_integral(
                log_values=log_integrands,
                x_grid=x_grid,
                axis=1,
            )
            divergence[i] = log_integrals / (alpha - 1.0)

    finite = np.isfinite(divergence)
    divergence[finite] = np.maximum(divergence[finite], 0.0)
    np.fill_diagonal(divergence, 0.0)

    if alpha > 1.0 and bandwidths is not None:
        bandwidths = np.asarray(bandwidths, dtype=float)
        if bandwidths.ndim != 1 or len(bandwidths) != number_of_densities:
            raise ValueError("bandwidths must contain one value per density.")
        if np.any(bandwidths <= 0.0):
            raise ValueError("Bandwidths must be strictly positive.")

        h_i = bandwidths[:, None]
        h_j = bandwidths[None, :]
        divergent_tails = alpha / h_i**2 + (1.0 - alpha) / h_j**2 <= 0.0
        divergence[divergent_tails] = np.inf

    return divergence


def alpha_divergence_matrix(
    density_values: np.ndarray,
    x_grid: np.ndarray,
    alpha: float,
    bandwidths: np.ndarray | None = None,
    eps: float = 1e-300,
) -> np.ndarray:
    """Compute the directed Amari alpha-divergence matrix by quadrature.

    The convention is

        A_alpha(p, q) = (integral(p**alpha * q**(1-alpha)) - 1)
                        / (alpha * (alpha - 1)).

    At ``alpha=1`` the continuous KL limit is returned.
    """
    alpha = float(alpha)
    if alpha <= 0.0:
        raise ValueError("alpha must be strictly positive.")

    renyi = renyi_divergence_matrix(
        density_values=density_values,
        x_grid=x_grid,
        alpha=alpha,
        bandwidths=bandwidths,
        eps=eps,
    )
    if np.isclose(alpha, 1.0, rtol=0.0, atol=1e-12):
        return renyi

    with np.errstate(over="ignore", invalid="ignore"):
        divergence = np.expm1((alpha - 1.0) * renyi)
        divergence /= alpha * (alpha - 1.0)

    finite = np.isfinite(divergence)
    divergence[finite] = np.maximum(divergence[finite], 0.0)
    np.fill_diagonal(divergence, 0.0)
    return divergence


def inverse_power_similarity_matrix(
    divergence_matrix: np.ndarray,
    offset: float = 1.0,
    slope: float = 1.0,
    power: float = 1.0,
) -> np.ndarray:
    """Transform divergences using a normalized inverse-power similarity."""
    divergence = np.asarray(divergence_matrix, dtype=float)
    if divergence.ndim != 2 or divergence.shape[0] != divergence.shape[1]:
        raise ValueError("divergence_matrix must be square.")
    if np.any(np.isnan(divergence)):
        raise ValueError("divergence_matrix must not contain NaN values.")
    if np.any(divergence < 0.0):
        raise ValueError("divergence_matrix must be nonnegative.")
    for name, value in (
        ("offset", offset),
        ("slope", slope),
        ("power", power),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")

    with np.errstate(over="ignore", invalid="ignore"):
        similarity = (
            float(offset)
            / (float(offset) + float(slope) * divergence)
        ) ** float(power)
    similarity[np.isposinf(divergence)] = 0.0
    np.fill_diagonal(similarity, 1.0)
    return similarity


def symmetrized_renyi_divergence_matrix(
    density_values: np.ndarray,
    x_grid: np.ndarray,
    alpha: float,
    bandwidths: np.ndarray | None = None,
    eps: float = 1e-300,
) -> np.ndarray:
    """Return the arithmetic symmetrization of directed Rényi divergence."""
    divergence = renyi_divergence_matrix(
        density_values=density_values,
        x_grid=x_grid,
        alpha=alpha,
        bandwidths=bandwidths,
        eps=eps,
    )
    return 0.5 * (divergence + divergence.T)


def renyi_similarity_matrix(
    density_values: np.ndarray,
    x_grid: np.ndarray,
    alpha: float,
    gamma: float = 1.0,
    symmetrized: bool = False,
    bandwidths: np.ndarray | None = None,
    eps: float = 1e-300,
) -> np.ndarray:
    """Transform directed or symmetrized Rényi divergence using an exponential."""
    gamma = float(gamma)
    if gamma <= 0.0:
        raise ValueError("gamma must be strictly positive.")

    divergence = renyi_divergence_matrix(
        density_values=density_values,
        x_grid=x_grid,
        alpha=alpha,
        bandwidths=bandwidths,
        eps=eps,
    )
    if symmetrized:
        divergence = 0.5 * (divergence + divergence.T)

    similarity = np.exp(-gamma * divergence)
    np.fill_diagonal(similarity, 1.0)
    return similarity


def _closed_periodic_grid(
    densities: np.ndarray,
    dx: float,
) -> tuple[np.ndarray, np.ndarray]:
    densities = np.asarray(densities, dtype=float)
    if densities.ndim != 2:
        raise ValueError("densities must be a two-dimensional array.")
    if dx <= 0.0:
        raise ValueError("dx must be strictly positive.")
    x_grid = np.arange(densities.shape[1] + 1, dtype=float) * float(dx)
    periodic_densities = np.concatenate((densities, densities[:, :1]), axis=1)
    return periodic_densities, x_grid


def symmetrized_renyi_similarity_fft(
    densities: np.ndarray,
    dx: float,
    alpha: float,
    gamma: float = 1.0,
    bandwidths: np.ndarray | None = None,
    eps: float = 1e-300,
) -> np.ndarray:
    """Compute symmetrized Rényi similarity on a periodic FFT grid."""
    values, x_grid = _closed_periodic_grid(densities, dx)
    return renyi_similarity_matrix(
        density_values=values,
        x_grid=x_grid,
        alpha=alpha,
        gamma=gamma,
        symmetrized=True,
        bandwidths=bandwidths,
        eps=eps,
    )
