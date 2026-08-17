"""FFT-based evaluation used by the large periodic-grid experiments."""

from __future__ import annotations

import numpy as np


def kde_family_fft(
    samples: np.ndarray,
    bandwidths: np.ndarray,
    half_width: float,
    spatial_points: int,
) -> tuple[np.ndarray, float]:
    """Evaluate Gaussian KDEs on an equispaced periodic grid by Fourier inversion."""
    samples = np.asarray(samples, dtype=float)
    bandwidths = np.asarray(bandwidths, dtype=float)
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
    densities = np.maximum(densities, np.finfo(float).tiny)
    densities /= densities.sum(axis=1, keepdims=True) * dx
    return densities, dx
