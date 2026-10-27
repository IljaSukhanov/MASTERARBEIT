"""Numerical local Q_r minima on (0.01,2), with exact Gaussian refinement.

Fourier derivative splines locate candidates. Accepted roots use exact
Gaussian-pair gradients and curvatures. Ambiguous scans use exact curvature
augmentation. Finite scans do not certify exhaustive stationary-point counts.
"""
from functools import lru_cache
import runpy

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import brentq
from scipy.spatial.distance import pdist

from kde_tools.qhat import _energy, _scan_matrix

H_MIN, H_MAX, HALF_WIDTH = 0.01, 2.0, 16.0


def transform(base, z, r):
    value, gradient, curvature = base
    factor = np.exp(-2*r*z)
    return factor*np.array((value, gradient-2*r*value,
                            curvature-4*r*gradient+4*r*r*value))


def same_minima(first, second, tolerance=1e-7):
    return len(first) == len(second) and all(
        abs(np.log(float(a['h'])/float(b['h']))) < tolerance for a, b in zip(first, second))


class GridScanner:
    def __init__(self, points=1025, spatial_points=65536):
        self.spatial_points = spatial_points
        self.z, self.omega, _, q0 = _scan_matrix(H_MIN, H_MAX, points, HALF_WIDTH, spatial_points)
        u = (np.exp(self.z[:, None])*self.omega[None, :])**2
        a = 4-2*u
        self.matrices = np.stack((q0, q0*a, q0*(a*a-4*u)))

    def approximate(self, samples):
        position = (samples+HALF_WIDTH)/(2*HALF_WIDTH/self.spatial_points)
        left = np.floor(position).astype(int)
        fraction = position-left
        mass = (np.bincount(left % self.spatial_points, weights=1-fraction, minlength=self.spatial_points)
                + np.bincount((left+1) % self.spatial_points, weights=fraction, minlength=self.spatial_points))
        power = np.abs(np.fft.rfft(mass/len(samples))[:len(self.omega)])**2
        return self.matrices @ power


def spline_candidates(z, values):
    spline = CubicHermiteSpline(z, values[1], values[2], extrapolate=False)
    roots = spline.roots(extrapolate=False)
    roots = roots[np.isfinite(roots) & (roots > z[0]) & (roots < z[-1])]
    roots = np.array([root for i, root in enumerate(roots)
                      if i == 0 or abs(root-roots[i-1]) > 1e-8])
    minima = roots[spline.derivative()(roots) > 0]
    extrema = spline.derivative().roots(extrapolate=False)
    extrema = extrema[np.isfinite(extrema) & (extrema > z[0]) & (extrema < z[-1])]
    near_tangency = any(abs(float(spline(t)))/max(float(np.interp(t, z, values[0])), 1e-20) < .001
                       for t in extrema)
    return minima, near_tangency


def point_from_root(z, values):
    value, gradient, curvature = values
    h = float(np.exp(z))
    if not H_MIN < h < H_MAX or value <= 0 or curvature <= 0 or abs(gradient/curvature) > 1e-8:
        raise ArithmeticError('Local minimum failed interval, stationarity, or curvature checks.')
    return dict(h=h, energy=float(value), log_derivative=float(gradient),
                log_curvature=float(curvature), newton_correction=float(abs(gradient/curvature)))


def curvature_minima(z, approximate, exact):
    """Exact curvature augmentation, following the validated Q1 experiment."""
    nodes = {float(z[0]), float(z[-1])}
    for index in np.flatnonzero(approximate[2, :-1]*approximate[2, 1:] < 0):
        lo, hi = int(index), int(index+1)
        for _ in range(9):
            if exact(z[lo])[2]*exact(z[hi])[2] < 0:
                break
            lo, hi = max(0, lo-1), min(len(z)-1, hi+1)
        else:
            raise ArithmeticError('Could not verify an exact curvature bracket.')
        nodes.add(float(brentq(lambda t: exact(t)[2], z[lo], z[hi], xtol=2e-13)))
    for index in np.flatnonzero(approximate[1, :-1]*approximate[1, 1:] < 0):
        nodes.update((float(z[index]), float(z[index+1])))
    nodes = sorted(nodes)
    found = []
    for lo, hi in zip(nodes[:-1], nodes[1:]):
        if exact(lo)[1] < 0 < exact(hi)[1]:
            root = float(brentq(lambda t: exact(t)[1], lo, hi, xtol=2e-13))
            if found and abs(np.log(found[-1]['h'])-root) < 1e-8:
                continue
            found.append(point_from_root(root, exact(root)))
    return found


def sample_minima(samples, methods, scanner, refined_scanner):
    """Share the sample FFT, pair distances, and exact base-energy cache."""
    approximate = scanner.approximate(samples)
    d2 = pdist(samples[:, None], metric='sqeuclidean')
    cache = {}
    refined_base = None

    def base(z):
        z = float(z)
        if z not in cache:
            cache[z] = _energy(d2, len(samples), z)
        return cache[z]

    results, diagnostics = {}, {}
    for method, r in methods.items():
        values = transform(approximate, scanner.z, r)
        starts, near_tangency = spline_candidates(scanner.z, values)
        coarse, coarse_tangency = spline_candidates(scanner.z[::2], values[:, ::2])
        signatures_agree = len(starts) == len(coarse) and (
            not len(starts) or np.max(np.abs(starts-coarse)) < 1e-3)
        exact = lambda z: transform(base(z), float(z), r)
        fallback = near_tangency or coarse_tangency or not signatures_agree
        found = []
        if not fallback:
            for start in starts:
                root = float(start)
                index = int(np.searchsorted(scanner.z, start))
                lo, hi = scanner.z[max(0, index-2)], scanner.z[min(len(scanner.z)-1, index+1)]
                for _ in range(8):
                    value = exact(root)
                    if value[2] <= 0:
                        fallback = True
                        break
                    step = value[1]/value[2]
                    if abs(step) < 2e-10:
                        found.append(point_from_root(root, value))
                        break
                    if not lo < root-step < hi:
                        fallback = True
                        break
                    root = float(root-step)
                else:
                    fallback = True
                if fallback:
                    break
        if fallback:
            if refined_base is None:
                refined_base = refined_scanner.approximate(samples)
            refined_values = transform(refined_base, refined_scanner.z, r)
            found = curvature_minima(refined_scanner.z, refined_values, exact)
        results[method] = found
        diagnostics[method] = dict(curvature_fallback=fallback,
                                   coarse_fine_signature_agrees=signatures_agree)
    return results, diagnostics


def exact_dense_minima(samples, methods, old_workflow, points=1025):
    """Independent full exact derivative scan shared across all r values."""
    old = runpy.run_path(str(old_workflow))
    d2 = pdist(samples[:, None], metric='sqeuclidean')
    z = np.linspace(np.log(H_MIN), np.log(H_MAX), points)
    base = np.array([_energy(d2, len(samples), t) for t in z])
    return {method: old['roots_from_scan'](d2, len(samples), z, base, r)
            for method, r in methods.items()}
