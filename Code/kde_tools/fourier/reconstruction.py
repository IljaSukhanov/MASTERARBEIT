"""Periodic Fourier reconstruction of Gaussian KDE families."""

from __future__ import annotations

import numpy as np


def kde_family_fft(
    samples: np.ndarray,
    bandwidths: np.ndarray,
    half_width: float,
    spatial_points: int,
) -> tuple[np.ndarray, float]:
    """Evaluate Gaussian KDEs on an equispaced periodic grid using ``irfft``.

    This optional approximation is intentionally isolated from the default
    exact/quadrature implementation.  The returned grid is ``[-H, H)`` with
    spacing ``dx = 2H / spatial_points``.
    """
    samples = np.asarray(samples, dtype=float)
    bandwidths = np.asarray(bandwidths, dtype=float)
    half_width = float(half_width)
    spatial_points = int(spatial_points)
    if samples.ndim != 1 or len(samples) == 0:
        raise ValueError("samples must be a nonempty one-dimensional array.")
    if bandwidths.ndim != 1 or len(bandwidths) == 0:
        raise ValueError("bandwidths must be a nonempty one-dimensional array.")
    if np.any(bandwidths <= 0.0):
        raise ValueError("bandwidths must be strictly positive.")
    if half_width <= 0.0:
        raise ValueError("half_width must be strictly positive.")
    if spatial_points < 2 or spatial_points % 2:
        raise ValueError("spatial_points must be an even integer of at least two.")

    dx = 2.0 * half_width / spatial_points
    frequencies = 2.0 * np.pi * np.fft.rfftfreq(spatial_points, d=dx)
    shifted_samples = samples + half_width
    empirical_cf = np.exp(
        -1j * frequencies[:, None] * shifted_samples[None, :]
    ).mean(axis=1)
    smoothers = np.exp(
        -0.5 * bandwidths[:, None] ** 2 * frequencies[None, :] ** 2
    )
    coefficients = smoothers * empirical_cf[None, :] / dx
    densities = np.fft.irfft(coefficients, n=spatial_points, axis=1)

    # Truncation and roundoff can create tiny negative values.  These
    # corrections are useful for logarithmic divergences, but mean that exact
    # discrete Parseval equivalence no longer holds after this point.
    densities = np.maximum(densities, np.finfo(float).tiny)
    densities /= densities.sum(axis=1, keepdims=True) * dx
    return densities, dx
