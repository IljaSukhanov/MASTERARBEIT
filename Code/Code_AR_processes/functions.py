"""Shared simulation utilities for the autoregressive-process experiments."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "jones_ar_coefficients",
    "simulate_uniform_ar",
    "sliding_window_point_cloud",
]


def jones_ar_coefficients(
    p: int,
    rng: np.random.Generator | int | None = None,
    *,
    return_pacf: bool = False,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Sample a stationary AR(p) coefficient vector using Jones' method."""
    if not isinstance(p, (int, np.integer)) or p < 1:
        raise ValueError(f"p must be a positive integer, got {p}")

    generator = np.random.default_rng(rng)
    lag_indices = np.arange(1, p + 1)
    beta_shape_a = (lag_indices + 1) // 2
    beta_shape_b = lag_indices // 2 + 1
    beta_draws = generator.beta(beta_shape_a, beta_shape_b)
    pacf = 2.0 * beta_draws - 1.0

    phi = np.array([pacf[0]], dtype=float)
    for order in range(2, p + 1):
        reflection_coefficient = pacf[order - 1]
        previous_phi = phi
        phi = np.empty(order, dtype=float)
        phi[:-1] = (
            previous_phi - reflection_coefficient * previous_phi[::-1]
        )
        phi[-1] = reflection_coefficient

    if return_pacf:
        return phi, pacf
    return phi


def simulate_uniform_ar(
    phi: ArrayLike,
    sample_size: int,
    burnin: int = 500,
    half_width: float = 1.0,
    rng: np.random.Generator | int | None = None,
    *,
    intercept: float = 0.0,
) -> NDArray[np.float64]:
    """Simulate a stationary AR process with Uniform[-a, a] innovations."""
    phi = np.asarray(phi, dtype=float)
    if phi.ndim != 1 or len(phi) == 0 or not np.all(np.isfinite(phi)):
        raise ValueError("phi must be a nonempty finite one-dimensional vector")
    if not isinstance(sample_size, (int, np.integer)) or sample_size < 1:
        raise ValueError("sample_size must be a positive integer")
    if not isinstance(burnin, (int, np.integer)) or burnin < 0:
        raise ValueError("burnin must be a non-negative integer")
    if not np.isfinite(half_width) or half_width <= 0:
        raise ValueError("half_width must be a positive finite number")
    if not np.isfinite(intercept):
        raise ValueError("intercept must be finite")

    generator = np.random.default_rng(rng)
    order = len(phi)
    total_length = burnin + order + sample_size
    innovations = generator.uniform(-half_width, half_width, size=total_length)
    series = np.empty(total_length, dtype=float)
    series[:order] = intercept / (1.0 - np.sum(phi))

    for time_index in range(order, total_length):
        recent_values = series[time_index - np.arange(1, order + 1)]
        series[time_index] = intercept + phi @ recent_values + innovations[time_index]

    return series[burnin + order:]


def sliding_window_point_cloud(
    series: ArrayLike,
    embedding_dimension: int,
    *,
    newest_first: bool = True,
) -> NDArray[np.float64]:
    """Construct consecutive sliding-window vectors from a time series.

    By default, each row is ordered from newest to oldest, matching the
    minimum-width calculations. Set ``newest_first=False`` for the forward
    ordering ``(x_t, ..., x_{t+k-1})`` used by the interactive embeddings.
    """
    series = np.asarray(series, dtype=float)
    if series.ndim != 1:
        raise ValueError("series must be a one-dimensional array")
    if (
        not isinstance(embedding_dimension, (int, np.integer))
        or not 1 <= embedding_dimension <= len(series)
    ):
        raise ValueError(
            "embedding_dimension must be a positive integer less than or "
            "equal to the length of the series"
        )

    dimension = int(embedding_dimension)
    offsets = (
        range(dimension - 1, -1, -1)
        if newest_first
        else range(dimension)
    )
    window_count = len(series) - dimension + 1
    return np.column_stack(
        [series[offset:window_count + offset] for offset in offsets]
    )
