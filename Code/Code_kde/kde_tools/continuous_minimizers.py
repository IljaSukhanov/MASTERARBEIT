"""Continuous ISE and Q_r selectors for a Gaussian KDE and Gaussian truth.

Q_r(h) = h**(-2*r) Q_0(h), using the thesis's r-linear-grid notation
(5_bandwidth_selection.tex, prop:main_thm). The Gaussian ISE formula is
the one in density.exact_oracle_ise_values. Its log-bandwidth derivatives
and the derivative transformation for Q_r are direct algebraic derivations.
Fourier approximations provide search brackets only; final candidates and
endpoints use the analytic Gaussian-pair objectives.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.distance import pdist

from .qhat import _energy, _samples, _scan_matrix, njit, GaussianQhatObjective, minimize_qhat
from .energy_minimization import energy_search_id, minimize_bandwidth_energy, search_config


CONTINUOUS_SELECTOR_VERSION = "unified_ise_and_qr_continuous_v2_" + energy_search_id()
METHODS = ("ISE", "Q0", "Q1", "Q13_14")
R_VALUES = {"Q0": 0.0, "Q1": 1.0, "Q13_14": 13.0 / 14.0}


def _ise_pair_sums_numpy(d2, h):
    u = d2 / (4*h*h)
    e = np.exp(-u)
    return float(e.sum()), float(e @ (2*u-1)), float(e @ (4*u*u-8*u+1))


if njit is not None:
    @njit(cache=True, nogil=True)
    def _ise_pair_sums(d2, h):
        s = g = k = 0.0
        for distance in d2:
            u = distance / (4*h*h)
            e = math.exp(-u)
            s += e
            g += e*(2*u-1)
            k += e*(4*u*u-8*u+1)
        return s, g, k
else:
    _ise_pair_sums = _ise_pair_sums_numpy


def qr_log_energy(d2, n, log_h, r):
    """Q_r and its first two derivatives with respect to log h."""
    value, gradient, curvature = _energy(d2, n, log_h)
    factor = np.exp(-2*r*log_h)
    return factor * np.array([value, gradient-2*r*value,
                              curvature-4*r*gradient+4*r*r*value])


def ise_log_energy(d2, samples, log_h, sigma2=1.0):
    """Exact Gaussian-truth ISE and its first two log-h derivatives."""
    h = float(np.exp(log_h))
    n = len(samples)
    s, g, k = _ise_pair_sums(d2, h)
    kde = np.array([n+2*s, -n+2*g, n+2*k]) / (2*np.sqrt(np.pi)*h*n*n)
    variance = sigma2+h*h
    v = samples*samples/variance
    t = h*h/variance
    cross = np.exp(-0.5*v)/np.sqrt(2*np.pi*variance)
    a = t*(v-1)
    a_prime = 2*t*(1-t)*(v-1)-2*t*t*v
    result = kde-2*np.array([cross.mean(), np.mean(cross*a),
                             np.mean(cross*(a*a+a_prime))])
    result[0] += 1/(2*np.sqrt(np.pi*sigma2))
    return result


def continuous_ise_qr_minima(samples, h_min=0.01, h_max=2.0, *, sigma2=1.0,
                             scan_points=None, half_width=None, spatial_points=None,
                             methods=METHODS, config=None, scan_backend=None):
    """Compatibility views of the shared interval/local minimization engine.

    Each returned method retains legacy fields plus the complete `analysis`
    report: all detected local minima, both endpoints and global status.
    """
    methods = tuple(methods)
    if not methods or len(set(methods)) != len(methods) or set(methods)-set(METHODS):
        raise ValueError("Choose unique methods from ISE, Q0, Q1, Q13_14.")
    samples = _samples(samples)
    if not np.isfinite(sigma2) or sigma2 <= 0:
        raise ValueError("sigma2 must be finite and positive.")
    cfg = search_config(config, scan_points=scan_points, half_width=half_width,
                        spatial_points=spatial_points, scan_backend=scan_backend)
    objective = GaussianQhatObjective(samples)
    reports = {}
    for method in methods:
        if method != "ISE":
            reports[method] = minimize_qhat(objective, h_min, h_max,
                                            r=R_VALUES[method], config=cfg)
    if "ISE" in methods:
        d2 = objective.d2
        truth_norm = 1/(2*np.sqrt(np.pi*sigma2))

        def ise_scan(z):
            # Reuse the same sample FFT and spatial padding as Q_r. This
            # approximation only locates brackets; final ISE values are exact.
            width = max(cfg.half_width, float(np.ceil(np.max(np.abs(samples))+6*np.exp(z[-1]))))
            _, omega, weights, _ = _scan_matrix(
                float(np.exp(z[0])), float(np.exp(z[-1])), len(z), width, cfg.spatial_points)
            coefficients = objective.fourier_coefficients(width, cfg.spatial_points)[:len(omega)]
            power = np.abs(coefficients)**2
            real = (coefficients*np.exp(1j*omega*width)).real
            u = (np.exp(z[:, None])*omega[None, :])**2
            kde = np.exp(-u)*power[None, :]
            cross = np.exp(-.5*(u+sigma2*omega[None, :]**2))*real[None, :]
            factor = weights/(2*width)
            return np.column_stack((
                (kde-2*cross) @ factor+truth_norm,
                (-2*u*kde+2*u*cross) @ factor,
                ((4*u*u-4*u)*kde+(4*u-2*u*u)*cross) @ factor))

        reports["ISE"] = minimize_bandwidth_energy(
            lambda z: ise_log_energy(d2, samples, z, sigma2), h_min, h_max,
            scan=ise_scan if cfg.scan_backend == "fourier" else None, config=cfg)
    result = {}
    for method in methods:
        report = reports[method]
        best = report["global_minimum"]
        result[method] = dict(h=best["h"], objective=best["energy"],
                              log_derivative=best["log_derivative"],
                              log_curvature=best["log_curvature"], boundary=best["boundary"],
                              interior_candidates=len(report["local_minima"]), analysis=report)
    return result
