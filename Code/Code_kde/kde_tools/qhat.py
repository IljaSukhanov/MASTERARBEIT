"""Gaussian KDE scale energies and the shared interval/local search entry point.

Q_hat(h) = h**2 / 2 * ||partial_h f_hat_h||_2**2.
The analytic Gaussian-product expression is also used in cell 4 of
experiment_1_joint_N_lambda_limit.ipynb. The two log-h derivatives below
are original algebraic differentiations of that expression.
"""

from __future__ import annotations

from functools import lru_cache
import math

import numpy as np
from scipy.spatial.distance import pdist

try:
    from numba import njit
except ImportError:  # Optional acceleration; the NumPy calculation is identical.
    njit = None


from .energy_minimization import (
    energy_search_id, minimize_bandwidth_energy, search_config,
)

QHAT_VERSION = "exact_gaussian_unified_minimum_v2_" + energy_search_id()


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


class GaussianQhatObjective:
    """Share pair distances, exact evaluations and sample FFT across r values."""

    def __init__(self, samples):
        self.samples = _samples(samples)
        self.d2 = pdist(self.samples[:, None], metric="sqeuclidean")
        self._exact = {}
        self._coefficients = {}

    def log_energy(self, z, r=0.0):
        z = float(z)
        if z not in self._exact:
            self._exact[z] = _energy(self.d2, len(self.samples), z)
        value, gradient, curvature = self._exact[z]
        return np.exp(-2*r*z)*np.array([
            value, gradient-2*r*value, curvature-4*r*gradient+4*r*r*value])

    def fourier_coefficients(self, half_width, spatial_points):
        key = (half_width, spatial_points)
        if key not in self._coefficients:
            position = (self.samples+half_width)/(2*half_width/spatial_points)
            left = np.floor(position).astype(int)
            fraction = position-left
            mass = (np.bincount(left % spatial_points, weights=1-fraction, minlength=spatial_points)
                    + np.bincount((left+1) % spatial_points, weights=fraction, minlength=spatial_points))
            self._coefficients[key] = np.fft.rfft(mass/len(self.samples))
        return self._coefficients[key]

    def scan(self, z, r, config):
        h_min, h_max = float(np.exp(z[0])), float(np.exp(z[-1]))
        half_width = max(config.half_width,
                         float(np.ceil(np.max(np.abs(self.samples))+6*h_max)))
        _, omega, _, matrix = _scan_matrix(
            h_min, h_max, len(z), half_width, config.spatial_points)
        power = np.abs(self.fourier_coefficients(half_width, config.spatial_points)[:len(omega)])**2
        moments = matrix @ np.column_stack((power, omega**2*power, omega**4*power))
        h2 = np.exp(2*z)
        value = moments[:, 0]
        gradient = 4*value-2*h2*moments[:, 1]
        curvature = 16*value-20*h2*moments[:, 1]+4*h2*h2*moments[:, 2]
        factor = np.exp(-2*r*z)
        return factor[:, None]*np.column_stack((
            value, gradient-2*r*value, curvature-4*r*gradient+4*r*r*value))


def minimize_qhat(samples, h_min=0.01, h_max=2.0, *, r=0.0, config=None,
                  scan_points=None, half_width=None, spatial_points=None,
                  scan_backend=None):
    """Analyze all detected interior Q_r minima AND the closed-interval minimum.

    Numerical controls live in energy_minimization.EnergySearchConfig. Pass a
    config (or explicit overrides) to control every experiment consistently.
    samples can also be a GaussianQhatObjective to reuse work across r values.
    Return the complete minimize_bandwidth_energy report, never discard local
    minima when an endpoint wins. See that function for the numerical limits.
    """
    if not np.isfinite(r):
        raise ValueError("r must be finite.")
    cfg = search_config(config, scan_points=scan_points, half_width=half_width,
                        spatial_points=spatial_points, scan_backend=scan_backend)
    objective = samples if isinstance(samples, GaussianQhatObjective) else GaussianQhatObjective(samples)
    scan = ((lambda z: objective.scan(z, r, cfg)) if cfg.scan_backend == "fourier" else None)
    report = minimize_bandwidth_energy(
        lambda z: objective.log_energy(z, r), h_min, h_max, scan=scan, config=cfg)
    report["r"] = float(r)
    report["diagnostics"]["scan_backend"] = cfg.scan_backend
    report["diagnostics"]["effective_half_width"] = (
        max(cfg.half_width, float(np.ceil(np.max(np.abs(objective.samples))+6*h_max)))
        if cfg.scan_backend == "fourier" else None)
    return report


def expected_gaussian_qhat_log_energy(log_h, n, r=0.0, sigma2=1.0):
    """Exact E[Q_r] and two log-h derivatives for iid N(mu,sigma2) samples.

    Original Gaussian convolution calculation; no unknown-density estimate is
    used. This objective uses the SAME minimization engine as sample energies.
    """
    if isinstance(n, (bool, np.bool_)) or int(n) != n or n < 1:
        raise ValueError("n must be a positive integer.")
    if not np.isfinite(sigma2) or sigma2 <= 0 or not np.isfinite(r):
        raise ValueError("Require sigma2 > 0 and finite r.")
    h = float(np.exp(log_h))
    a, b = 1+2*r, 4-2*r
    diagonal = 3/(16*np.sqrt(np.pi)*n)*h**(-a)
    population = 3*(n-1)/(16*np.sqrt(np.pi)*n)*h**b/(sigma2+h*h)**2.5
    t = h*h/(sigma2+h*h)
    slope = b-5*t
    return np.array([diagonal+population, -a*diagonal+slope*population,
                     a*a*diagonal+(slope*slope-10*t*(1-t))*population])


def continuous_qhat_minimum(samples, h_min=0.01, h_max=2.0, *,
                            scan_points=None, half_width=None, spatial_points=None,
                            config=None):
    """Compatibility view of minimize_qhat(..., r=0), retaining full analysis."""
    report = minimize_qhat(samples, h_min, h_max, r=0.0, config=config,
                           scan_points=scan_points, half_width=half_width,
                           spatial_points=spatial_points)
    best = report["global_minimum"]
    return dict(h=best["h"], qhat=best["energy"],
                log_derivative=best["log_derivative"], log_curvature=best["log_curvature"],
                boundary=bool(best["boundary"]), interior_candidates=len(report["local_minima"]),
                analysis=report)
