"""One continuous search for interval and interior bandwidth-energy minima.

The objective supplies value and two derivatives in the selected coordinate.
An optional inexpensive scan supplies candidate brackets only. All reported
points and endpoint comparisons use the exact objective. SciPy's brentq
refines sign-changing brackets (official reference:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html).
Resolution agreement is numerical evidence, not exhaustive root isolation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import brentq


ENERGY_SEARCH_VERSION = "bandwidth_energy_stationary_search_v2"


@dataclass(frozen=True)
class EnergySearchConfig:
    """Central numerical controls; callers can explicitly override any field."""

    scan_points: int = 257
    check_resolution: bool = True
    max_refinements: int = 2
    root_xtol: float = 2e-12
    root_rtol: float = 1e-14
    root_match_tolerance: float = 1e-7
    bracket_expansion: int = 8
    comparison_rtol: float = 1e-10
    comparison_atol: float = 0.0
    scan_backend: str = "fourier"
    half_width: float = 16.0
    spatial_points: int = 32768

    def __post_init__(self):
        for name, minimum in (("scan_points", 5), ("max_refinements", 0),
                              ("bracket_expansion", 0)):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}.")
            object.__setattr__(self, name, int(value))
        for name in ("root_xtol", "root_match_tolerance"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not np.isfinite(self.root_rtol) or self.root_rtol < 4*np.finfo(float).eps:
            raise ValueError("root_rtol must be at least four machine epsilons.")
        for name in ("comparison_rtol", "comparison_atol"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if not isinstance(self.check_resolution, (bool, np.bool_)):
            raise ValueError("check_resolution must be boolean.")
        object.__setattr__(self, "check_resolution", bool(self.check_resolution))
        if self.scan_backend not in ("exact", "fourier"):
            raise ValueError("scan_backend must be 'exact' or 'fourier'.")
        if not np.isfinite(self.half_width) or self.half_width <= 0:
            raise ValueError("half_width must be finite and positive.")
        if (isinstance(self.spatial_points, (bool, np.bool_)) or
                not isinstance(self.spatial_points, (int, np.integer)) or
                self.spatial_points < 16 or self.spatial_points % 2):
            raise ValueError("spatial_points must be an even integer of at least 16.")
        object.__setattr__(self, "spatial_points", int(self.spatial_points))


DEFAULT_ENERGY_SEARCH = EnergySearchConfig()


def search_config(config=None, **overrides):
    """Resolve defaults in one place; None overrides leave defaults intact."""
    if config is None:
        config = DEFAULT_ENERGY_SEARCH
    elif isinstance(config, dict):
        config = EnergySearchConfig(**config)
    if not isinstance(config, EnergySearchConfig):
        raise TypeError("config must be an EnergySearchConfig or a settings dict.")
    return replace(config, **{key: value for key, value in overrides.items() if value is not None})


def energy_search_settings(config=None, **overrides):
    return {"version": ENERGY_SEARCH_VERSION, **asdict(search_config(config, **overrides))}


def energy_search_id(config=None, **overrides):
    payload = json.dumps(energy_search_settings(config, **overrides), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def unified_energy_run_directory(directory):
    """Put new runs in a namespace determined by the shared search settings."""
    return Path(directory) / ("unified_" + energy_search_id())


def same_stationary_points(first, second, tolerance=1e-7):
    return len(first) == len(second) and all(
        a["kind"] == b["kind"] and abs(np.log(a["h"]/b["h"])) <= tolerance
        for a, b in zip(first, second))


def minimize_bandwidth_energy(log_energy, h_min, h_max, *, scan=None, config=None,
                              scan_points=None, bandwidth_coordinates=False):
    """Return BOTH the interval minimum and detected strict interior minima.

    By default the callback returns energy and two derivatives in log h.
    With bandwidth_coordinates=True it receives h and returns derivatives
    in h. The optional scan uses the same coordinate and returns an (n, 3)
    array of approximations. Reported log derivatives remain available for
    existing callers.
    This function owns root isolation, refinement, classification, endpoint
    comparison and resolution checks for every caller, including Q_r and ISE.

    Result keys: global_minimum, global_minimizers (ties within configured
    tolerance), local_minima, local_maxima, stationary_points, endpoints,
    best_local_minimum (None if absent), and diagnostics. 'Global' means the
    smallest exact energy among detected interior minima and both endpoints;
    diagnostics explicitly record that the finite search is not a certificate.
    """
    cfg = search_config(config, scan_points=scan_points)
    if not np.isfinite(h_min) or not np.isfinite(h_max) or not 0 < h_min < h_max:
        raise ValueError("Require finite 0 < h_min < h_max.")
    lower, upper = ((float(h_min), float(h_max)) if bandwidth_coordinates
                    else (float(np.log(h_min)), float(np.log(h_max))))
    cache = {}

    def exact(z):
        z = float(z)
        if z not in cache:
            value = np.asarray(log_energy(z), dtype=float)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError("The energy and both log-bandwidth derivatives must be finite.")
            cache[z] = value
        return cache[z]

    def point(z, kind, boundary=0):
        value, gradient, curvature = map(float, exact(z))
        h = float(h_min if boundary < 0 else h_max if boundary > 0 else
                  z if bandwidth_coordinates else np.exp(z))
        if bandwidth_coordinates:
            gradient, curvature = h*gradient, h*h*curvature+h*gradient
        return dict(h=h, energy=value, objective=value, log_derivative=gradient,
                    log_curvature=curvature, boundary=boundary, kind=kind,
                    newton_correction=(abs(gradient/curvature) if curvature else None))

    def root(lo, hi, component):
        return float(brentq(lambda z: exact(z)[component], lo, hi,
                            xtol=cfg.root_xtol, rtol=cfg.root_rtol))

    def search(points):
        z = (np.geomspace(lower, upper, points) if bandwidth_coordinates
             else np.linspace(lower, upper, points))
        values = (np.asarray(scan(z), dtype=float) if scan is not None
                  else np.array([exact(t) for t in z]))
        if values.shape != (points, 3) or not np.all(np.isfinite(values)):
            raise ValueError("scan must return a finite (scan_points, 3) array.")
        # Include curvature extrema: a shallow min/max pair can lie between
        # two grid nodes whose first derivatives have the same sign.
        nodes = {lower, upper}
        if scan is None:
            nodes.update(map(float, z))
        candidates = set(map(int, np.flatnonzero(values[:-1, 1]*values[1:, 1] <= 0)))
        curvature_brackets = set(map(int, np.flatnonzero(values[:-1, 2]*values[1:, 2] < 0)))
        if np.any(values[:, 1]) or np.any(values[:, 2]):
            spline = CubicHermiteSpline(z, values[:, 1], values[:, 2], extrapolate=False)
            for guess in spline.roots(extrapolate=False):
                if np.isfinite(guess) and lower < guess < upper:
                    candidates.add(min(points-2, max(0, int(np.searchsorted(z, guess)-1))))
        for index in candidates:
            nodes.update((float(z[index]), float(z[index+1])))
        for index in curvature_brackets:
            lo, hi = index, index+1
            for _ in range(cfg.bracket_expansion+1):
                if exact(z[lo])[2]*exact(z[hi])[2] < 0:
                    nodes.add(root(z[lo], z[hi], 2))
                    break
                if lo == 0 and hi == points-1:
                    break
                lo, hi = max(0, lo-1), min(points-1, hi+1)
            nodes.update((float(z[lo]), float(z[hi])))
        nodes = sorted(nodes)
        roots = []
        for lo, hi in zip(nodes[:-1], nodes[1:]):
            gl, gr = exact(lo)[1], exact(hi)[1]
            if gl*gr < 0:
                roots.append((root(lo, hi, 1), "minimum" if gl < gr else "maximum"))
        # A root can coincide exactly with a scan/curvature node.
        for i, z0 in enumerate(nodes[1:-1], 1):
            if exact(z0)[1] == 0:
                gl = exact((nodes[i-1]+z0)/2)[1]
                gr = exact((z0+nodes[i+1])/2)[1]
                if gl < 0 < gr:
                    roots.append((z0, "minimum"))
                elif gl > 0 > gr:
                    roots.append((z0, "maximum"))
        result = []
        for z0, kind in sorted(roots):
            if not lower < z0 < upper:
                continue
            # A wide bracket can contain more than one root. Classify the
            # refined root itself, rather than assigning the bracket's sign.
            curvature = exact(z0)[2]
            if curvature > 0:
                kind = "minimum"
            elif curvature < 0:
                kind = "maximum"
            duplicate_tolerance = 8*max(cfg.root_xtol, cfg.root_rtol*max(1., abs(z0)))
            previous = (result[-1]["h"] if bandwidth_coordinates else
                        np.log(result[-1]["h"])) if result else None
            if (result and result[-1]["kind"] == kind and
                    abs(previous-z0) <= duplicate_tolerance):
                continue
            result.append(point(z0, kind))
        return result

    points = cfg.scan_points
    stationary = search(points)
    stable = None
    refinements = 0
    resolution_error = None
    if cfg.check_resolution:
        for refinements in range(1, cfg.max_refinements+1):
            points = 2*points-1
            refined = search(points)
            stable = same_stationary_points(stationary, refined, cfg.root_match_tolerance)
            resolution_error = (max((abs(np.log(a["h"]/b["h"]))
                                     for a, b in zip(stationary, refined)), default=0.0)
                                if len(stationary) == len(refined) else None)
            stationary = refined
            if stable:
                break
    minima = [p for p in stationary if p["kind"] == "minimum"]
    maxima = [p for p in stationary if p["kind"] == "maximum"]
    endpoints = [point(lower, "endpoint", -1), point(upper, "endpoint", 1)]
    candidates = endpoints+minima
    best = min(candidates, key=lambda p: (p["energy"], p["h"]))
    for candidate in candidates:
        candidate["is_global_minimum"] = bool(np.isclose(
            candidate["energy"], best["energy"], rtol=cfg.comparison_rtol, atol=cfg.comparison_atol))
    for candidate in maxima:
        candidate["is_global_minimum"] = False
    return dict(global_minimum=best, global_minimizers=[p for p in candidates if p["is_global_minimum"]],
                local_minima=minima, local_maxima=maxima, stationary_points=stationary,
                endpoints=endpoints, best_local_minimum=(min(minima, key=lambda p: (p["energy"], p["h"]))
                                                       if minima else None),
                local_minimum_count=len(minima),
                diagnostics=dict(settings=energy_search_settings(cfg), scan_points=points,
                                 search_coordinate="bandwidth" if bandwidth_coordinates else "log_bandwidth",
                                 refinements=refinements, resolution_stable=stable,
                                 maximum_log_h_discrepancy=resolution_error,
                                 exact_evaluations=len(cache), approximate_scan=scan is not None,
                                 certified_global=False))
