"""Continuous minimization of the sample Gaussian KDE log-bandwidth energy.

Q_hat(h) = h**2 / 2 * ||partial_h f_hat_h||_2**2.
The analytic Gaussian-product expression is also used in cell 4 of
experiment_1_joint_N_lambda_limit.ipynb. The two log-h derivatives below
are original algebraic differentiations of that expression.
"""

from __future__ import annotations

from functools import lru_cache
import math

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.spatial.distance import pdist

try:
    from numba import njit
except ImportError:  # Optional acceleration; the NumPy calculation is identical.
    njit = None


QHAT_VERSION = "exact_gaussian_continuous_log_minimum_v1"


def _pair_sums_numpy(d2, h):
    u = d2 / (4.0 * h * h)
    e = np.exp(-u)
    p = u * u - 3.0 * u + 0.75
    g = ((2.0 * u - 11.0) * u + 10.5) * u - 0.75
    k = (((4.0 * u - 36.0) * u + 76.0) * u - 33.0) * u + 0.75
    return float(e @ p), float(e @ g), float(e @ k)


if njit is not None:
    @njit(cache=True, nogil=True)
    def _pair_sums(d2, h):
        s = g = k = 0.0
        for distance in d2:
            u = distance / (4.0 * h * h)
            e = math.exp(-u)
            s += e * (u * u - 3.0 * u + 0.75)
            g += e * (((2.0 * u - 11.0) * u + 10.5) * u - 0.75)
            k += e * ((((4.0 * u - 36.0) * u + 76.0) * u - 33.0) * u + 0.75)
        return s, g, k
else:
    _pair_sums = _pair_sums_numpy


def _energy(d2, n, log_h):
    h = float(np.exp(log_h))
    s, g, k = _pair_sums(d2, h)
    factor = 1.0 / (4.0 * math.sqrt(math.pi) * h * n * n)
    return np.array([0.75 * n + 2 * s, -0.75 * n + 2 * g,
                     0.75 * n + 2 * k]) * factor


def _samples(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("samples must be a nonempty finite one-dimensional array.")
    return values


def gaussian_qhat(samples, bandwidths):
    """Analytic Q_hat values for arbitrary positive, continuous bandwidths."""
    samples = _samples(samples)
    h = np.asarray(bandwidths, dtype=float)
    if np.any(~np.isfinite(h)) or np.any(h <= 0):
        raise ValueError("bandwidths must be finite and positive.")
    d2 = pdist(samples[:, None], metric="sqeuclidean")
    values = np.array([_energy(d2, len(samples), z)[0] for z in np.log(h).flat])
    return values.reshape(h.shape)


@lru_cache(maxsize=8)
def _scan_matrix(h_min, h_max, scan_points, half_width, spatial_points):
    log_h = np.linspace(np.log(h_min), np.log(h_max), scan_points)
    omega = 2 * np.pi * np.fft.rfftfreq(spatial_points, 2 * half_width / spatial_points)
    omega = omega[omega * h_min <= 10.0]
    weights = np.full(len(omega), 2.0)
    weights[0] = 1.0
    if len(omega) == spatial_points // 2 + 1:
        weights[-1] = 1.0
    h = np.exp(log_h)
    matrix = (h[:, None]**4 * omega[None, :]**4
              * np.exp(-(h[:, None] * omega[None, :])**2)
              * weights[None, :] / (4 * half_width))
    return log_h, omega, weights, matrix


def continuous_qhat_minimum(samples, h_min=0.01, h_max=2.0, *,
                            scan_points=257, half_width=12.0, spatial_points=32768):
    """Scan for minima, refine each analytically off-grid, and compare endpoints.

    A finely binned Fourier surrogate supplies starting brackets only. Every
    final candidate and both endpoints use the exact Gaussian pair formula.
    This is a numerical search over the stated finite interval, not a proof
    that every possible stationary point has been enumerated.
    """
    samples = _samples(samples)
    if not 0 < h_min < h_max or not np.isfinite(h_max):
        raise ValueError("Require finite 0 < h_min < h_max.")
    if scan_points < 5 or spatial_points < 16 or spatial_points % 2:
        raise ValueError("Use at least five scan points and an even spatial grid.")
    if half_width <= np.max(np.abs(samples)) + 5 * h_max:
        half_width = float(math.ceil(np.max(np.abs(samples)) + 6 * h_max))
    log_h, omega, weights, matrix = _scan_matrix(
        float(h_min), float(h_max), int(scan_points), float(half_width), int(spatial_points)
    )
    dx = 2 * half_width / spatial_points
    position = (samples + half_width) / dx
    left = np.floor(position).astype(int)
    fraction = position - left
    mass = (np.bincount(left % spatial_points, weights=1 - fraction, minlength=spatial_points)
            + np.bincount((left + 1) % spatial_points, weights=fraction, minlength=spatial_points))
    power = np.abs(np.fft.rfft(mass / len(samples))[:len(omega)])**2
    coarse = matrix @ power
    local = np.flatnonzero((coarse[1:-1] <= coarse[:-2])
                          & (coarse[1:-1] <= coarse[2:])) + 1
    d2 = pdist(samples[:, None], metric="sqeuclidean")
    candidates = [(float(log_h[0]), _energy(d2, len(samples), log_h[0])),
                  (float(log_h[-1]), _energy(d2, len(samples), log_h[-1]))]
    for index in local:
        lo, hi = float(log_h[index - 1]), float(log_h[index + 1])

        def surrogate(z):
            h = np.exp(z)
            return h**4 * np.sum(weights * omega**4 * power * np.exp(-(h * omega)**2)) / (4 * half_width)

        initial = minimize_scalar(surrogate, bounds=(lo, hi), method="bounded",
                                  options={"xatol": 1e-11})
        z = float(initial.x)
        refined = False
        for _ in range(8):
            value = _energy(d2, len(samples), z)
            if value[2] <= 0:
                break
            step = value[1] / value[2]
            if abs(step) < 2e-10:
                refined = True
                break
            updated = z - step
            if not lo <= updated <= hi:
                break
            z = float(updated)
        if not refined:
            left_gradient = _energy(d2, len(samples), lo)[1]
            right_gradient = _energy(d2, len(samples), hi)[1]
            if left_gradient < 0 < right_gradient:
                z = float(brentq(lambda t: _energy(d2, len(samples), t)[1], lo, hi,
                                 xtol=2e-12))
            else:
                result = minimize_scalar(lambda t: _energy(d2, len(samples), t)[0],
                                         bounds=(lo, hi), method="bounded",
                                         options={"xatol": 1e-11})
                if not result.success:
                    raise ArithmeticError("Continuous Q_hat refinement failed.")
                z = float(result.x)
        candidates.append((z, _energy(d2, len(samples), z)))
    best_z, best = min(candidates, key=lambda candidate: (candidate[1][0], candidate[0]))
    is_boundary = best_z in (log_h[0], log_h[-1])
    h = float(np.exp(best_z))
    if is_boundary:
        h = float(h_min if best_z == log_h[0] else h_max)
    elif best[2] <= 0 or abs(best[1]) > 1e-7 * max(best[0], 1e-12):
        raise ArithmeticError("Q_hat minimum failed its derivative/curvature check.")
    return {"h": h, "qhat": float(best[0]), "log_derivative": float(best[1]),
            "log_curvature": float(best[2]), "boundary": bool(is_boundary),
            "interior_candidates": int(len(local))}
