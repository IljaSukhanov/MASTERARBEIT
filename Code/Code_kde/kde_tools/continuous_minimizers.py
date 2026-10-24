"""Continuous ISE and Q_r selectors for a Gaussian KDE and Gaussian truth.

Q_r(h) = h**(-2*r) Q_0(h), using the thesis's r-linear-grid notation
(5_bandwidth_selection.tex, prop:main_thm). The Gaussian ISE formula is
the one in density.exact_oracle_ise_values. Its log-bandwidth derivatives
and the derivative transformation for Q_r are direct algebraic derivations.
Fourier approximations provide search brackets only; final candidates and
endpoints use the analytic Gaussian-pair objectives.
"""

from __future__ import annotations

from functools import lru_cache
import math

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.spatial.distance import pdist

from .qhat import _energy, _samples, _scan_matrix, njit


CONTINUOUS_SELECTOR_VERSION = "exact_ise_and_qr_continuous_v1"
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


@lru_cache(maxsize=4)
def _ise_scan_matrices(h_min, h_max, scan_points, half_width, spatial_points, sigma2):
    log_h, omega, weights, _ = _scan_matrix(
        h_min, h_max, scan_points, half_width, spatial_points)
    h = np.exp(log_h)
    kde = weights[None, :] * np.exp(-(h[:, None]*omega[None, :])**2)/(2*half_width)
    cross = weights[None, :] * np.exp(
        -0.5*(h[:, None]**2+sigma2)*omega[None, :]**2)/(2*half_width)
    return kde, cross


def _minimum_from_scan(log_h, coarse_values, surrogate, exact):
    """Refine all scanned interior minima and compare both exact endpoints."""
    local = np.flatnonzero((coarse_values[1:-1] <= coarse_values[:-2])
                          & (coarse_values[1:-1] <= coarse_values[2:]))+1
    candidates = [(float(log_h[0]), exact(log_h[0])),
                  (float(log_h[-1]), exact(log_h[-1]))]
    for index in local:
        lo, hi = float(log_h[index-1]), float(log_h[index+1])
        initial = minimize_scalar(surrogate, bounds=(lo, hi), method="bounded",
                                  options={"xatol": 1e-11})
        if not initial.success:
            raise ArithmeticError("Continuous selector bracket refinement failed.")
        z, refined = float(initial.x), False
        for _ in range(8):
            value, gradient, curvature = exact(z)
            if curvature <= 0:
                break
            step = gradient/curvature
            if abs(step) < 2e-10:
                refined = True
                break
            if not lo <= z-step <= hi:
                break
            z = float(z-step)
        if not refined:
            left, right = exact(lo)[1], exact(hi)[1]
            if left < 0 < right:
                z = float(brentq(lambda t: exact(t)[1], lo, hi, xtol=2e-12))
            else:
                result = minimize_scalar(lambda t: exact(t)[0], bounds=(lo, hi),
                                         method="bounded", options={"xatol": 1e-11})
                if not result.success:
                    raise ArithmeticError("Continuous selector exact refinement failed.")
                z = float(result.x)
        candidates.append((z, exact(z)))
    z, values = min(candidates, key=lambda candidate: (candidate[1][0], candidate[0]))
    boundary = -1 if z == log_h[0] else (1 if z == log_h[-1] else 0)
    if not boundary and (values[2] <= 0 or abs(values[1]/values[2]) > 1e-7):
        raise ArithmeticError("Continuous selector failed its stationarity/curvature check.")
    return {"h": float(np.exp(z)), "objective": float(values[0]),
            "log_derivative": float(values[1]), "log_curvature": float(values[2]),
            "boundary": boundary, "interior_candidates": int(len(local))}


def continuous_ise_qr_minima(samples, h_min=0.01, h_max=2.0, *, sigma2=1.0,
                             scan_points=257, half_width=16.0, spatial_points=32768,
                             methods=METHODS):
    """Return continuous interval minimizers of ISE, Q_0, Q_1, Q_(13/14).

    Every method uses the same Gaussian KDE sample and the same closed
    bandwidth interval. The finite scan locates candidate basins, rather
    than restricting the selected bandwidth to a grid. This numerical search
    does not constitute a certificate that all stationary points were found.
    """
    methods = tuple(methods)
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(METHODS):
        raise ValueError("Choose unique methods from ISE, Q0, Q1, Q13_14.")
    samples = _samples(samples)
    if not 0 < h_min < h_max or not np.isfinite(h_max) or not 0 < sigma2 < np.inf:
        raise ValueError("Require finite 0 < h_min < h_max and positive variance.")
    if scan_points < 5 or spatial_points < 16 or spatial_points % 2:
        raise ValueError("Use at least five scan points and an even Fourier grid.")
    if half_width <= np.max(np.abs(samples))+5*h_max:
        half_width = float(math.ceil(np.max(np.abs(samples))+6*h_max))
    log_h, omega, weights, q_matrix = _scan_matrix(
        float(h_min), float(h_max), int(scan_points), float(half_width), int(spatial_points))
    dx = 2*half_width/spatial_points
    position = (samples+half_width)/dx
    left = np.floor(position).astype(int)
    fraction = position-left
    mass = (np.bincount(left % spatial_points, weights=1-fraction, minlength=spatial_points)
            + np.bincount((left+1) % spatial_points, weights=fraction, minlength=spatial_points))
    coefficients = np.fft.rfft(mass/len(samples))[:len(omega)]
    power = np.abs(coefficients)**2
    empirical_real = (coefficients*np.exp(1j*omega*half_width)).real
    q_coarse = q_matrix @ power
    d2 = pdist(samples[:, None], metric="sqeuclidean")
    results = {}
    for name, r in R_VALUES.items():
        if name not in methods:
            continue
        def surrogate(z):
            h = np.exp(z)
            return h**(4-2*r)*np.sum(weights*omega**4*power*np.exp(-(h*omega)**2))/(4*half_width)

        results[name] = _minimum_from_scan(
            log_h, q_coarse*np.exp(-2*r*log_h), surrogate,
            lambda z: qr_log_energy(d2, len(samples), z, r))
    if "ISE" not in methods:
        for result in results.values():
            if result["boundary"]:
                result["h"] = float(h_min if result["boundary"] == -1 else h_max)
        return {name: results[name] for name in methods}
    kde_matrix, cross_matrix = _ise_scan_matrices(
        float(h_min), float(h_max), int(scan_points), float(half_width),
        int(spatial_points), float(sigma2))
    truth_norm = 1/(2*np.sqrt(np.pi*sigma2))
    ise_coarse = kde_matrix @ power-2*(cross_matrix @ empirical_real)+truth_norm

    def ise_surrogate(z):
        h = np.exp(z)
        terms = (power*np.exp(-(h*omega)**2)
                 - 2*empirical_real*np.exp(-0.5*(h*h+sigma2)*omega**2))
        return float(weights @ terms)/(2*half_width)+truth_norm

    results["ISE"] = _minimum_from_scan(
        log_h, ise_coarse, ise_surrogate,
        lambda z: ise_log_energy(d2, samples, z, sigma2))
    for result in results.values():
        if result["boundary"]:
            result["h"] = float(h_min if result["boundary"] == -1 else h_max)
    return {name: results[name] for name in methods}
