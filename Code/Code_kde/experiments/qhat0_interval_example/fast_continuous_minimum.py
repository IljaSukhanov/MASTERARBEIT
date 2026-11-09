"""Benchmark a coarse analytic derivative scan followed by Brent refinement.

This is an experimental alternative to the shared diagnostic search,
not a replacement for it. It refines detected negative-to-positive derivative
crossings only, and does not check a denser scan or isolate every stationary
point. Narrow or degenerate minima can be missed. The returned best_candidate
is therefore not a certified global minimum.

Objective: kde_tools.qhat.GaussianQhatObjective.
Root solver: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html
"""

from __future__ import annotations

import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits

KDE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KDE_ROOT))

from kde_tools.bandwidths import bandwidth_grid_linear_in_psi_beta
from kde_tools.energy_minimization import EnergySearchConfig
from kde_tools.experiment_inputs import load_reference_sample
from kde_tools.fourier.similarities import beta2_divergence_matrix_fourier
from kde_tools.perron import perron_eigenpair_from_divergence
from kde_tools.qhat import GaussianQhatObjective, minimize_qhat


def coarse_qhat_minima(samples, h_min=0.01, h_max=2.0, *, scan_points=33):
    """Refine detected nondegenerate Q_0 minima on a coarse geometrically spaced bandwidth grid.

    A fresh objective constructs sample distances inside every call. All scan
    values, derivatives, refined roots and endpoint values are analytic.
    The bandwidth root tolerances match the shared search's default tolerances.
    """
    if not np.isfinite(h_min) or not np.isfinite(h_max) or not 0 < h_min < h_max:
        raise ValueError("Require finite 0 < h_min < h_max.")
    if isinstance(scan_points, bool) or not isinstance(scan_points, int) or scan_points < 3:
        raise ValueError("scan_points must be an integer of at least 3.")
    objective = GaussianQhatObjective(samples)
    h_grid = np.geomspace(h_min, h_max, scan_points)
    values = np.array([objective.derivatives(h) for h in h_grid])
    gradient = values[:, 1]
    roots = []
    brackets = []
    for i in np.flatnonzero((gradient[:-1] < 0) & (gradient[1:] > 0)):
        root = brentq(lambda h: objective.derivatives(h)[1], h_grid[i], h_grid[i+1],
                      xtol=2e-12, rtol=1e-14)
        roots.append(float(root))
        brackets.append([float(h_grid[i]), float(h_grid[i+1])])
    for i in range(1, scan_points-1):
        if gradient[i] == 0 and gradient[i-1] < 0 < gradient[i+1]:
            roots.append(float(h_grid[i]))

    def point(h, kind):
        value, slope, curvature = map(float, objective.derivatives(h))
        return dict(h=float(h), energy=value, log_derivative=h*slope,
                    log_curvature=h*h*curvature+h*slope, kind=kind)

    stationary_candidates = [point(t, "minimum") for t in sorted(set(roots))]
    minima = [p for p in stationary_candidates if p["log_curvature"] > 0]
    unresolved = [p for p in stationary_candidates if p["log_curvature"] <= 0]
    endpoints = [point(h_grid[0], "endpoint"), point(h_grid[-1], "endpoint")]
    endpoints[0]["h"], endpoints[1]["h"] = float(h_min), float(h_max)
    return {
        "local_minima": minima, "endpoints": endpoints,
        "best_local_minimum": min(minima, key=lambda p: p["energy"]) if minima else None,
        "best_candidate": min(endpoints+minima, key=lambda p: p["energy"]),
        "scan_points": scan_points, "minimum_brackets": brackets,
        "unresolved_stationary_candidates": unresolved,
        "exact_evaluations": len(objective._bandwidth_exact),
        "resolution_check_performed": False, "certified_global": False,
    }


def main():
    samples, metadata = load_reference_sample()
    full_config = EnergySearchConfig(scan_backend="exact")

    def full_search():
        result = minimize_qhat(samples, 0.01, 2., config=full_config)
        return dict(h=result["best_local_minimum"]["h"],
                    energy=result["best_local_minimum"]["energy"],
                    exact_evaluations=result["diagnostics"]["exact_evaluations"])

    def quick_search(points):
        result = coarse_qhat_minima(samples, scan_points=points)
        if result["best_local_minimum"] is None:
            raise ArithmeticError("The coarse scan did not find the reference-sample valley.")
        return dict(h=result["best_local_minimum"]["h"],
                    energy=result["best_local_minimum"]["energy"],
                    exact_evaluations=result["exact_evaluations"],
                    minimum_brackets=result["minimum_brackets"],
                    log_curvature=result["best_local_minimum"]["log_curvature"],
                    local_minimum_count=len(result["local_minima"]))

    def spectral():
        grid = bandwidth_grid_linear_in_psi_beta(0.01, 2., 2., 100)
        divergence = beta2_divergence_matrix_fourier(samples, grid)
        perron = perron_eigenpair_from_divergence(divergence, 1000.)
        return dict(h=float(grid[perron.selected_index]))

    methods = {
        "coarse_17_then_brent": lambda: quick_search(17),
        "coarse_33_then_brent": lambda: quick_search(33),
        "coarse_65_then_brent": lambda: quick_search(65),
        "shared_full_analytic_search": full_search,
        "spectral_fourier_N100": spectral,
    }
    runs = {name: [] for name in methods}
    first_calls = {}
    rng = np.random.default_rng(20261030)
    with threadpool_limits(limits=1):
        for name, method in methods.items():
            started = perf_counter()
            result = method()
            first_calls[name] = dict(seconds=perf_counter()-started, result=result)
        for _ in range(9):
            for name in rng.permutation(list(methods)):
                started = perf_counter()
                result = methods[name]()
                runs[name].append(dict(seconds=perf_counter()-started, result=result))
    reference_h = first_calls["shared_full_analytic_search"]["result"]["h"]
    summary = {}
    for name, measurements in runs.items():
        times = [m["seconds"] for m in measurements]
        result = measurements[-1]["result"]
        if name.startswith("coarse_") and any(
                abs(m["result"]["h"]-reference_h) > 1e-10 for m in measurements):
            raise ArithmeticError("A refined local minimum disagrees with the reference search.")
        summary[name] = dict(median_ms=1000*float(np.median(times)),
                             minimum_ms=1000*min(times), maximum_ms=1000*max(times),
                             result=result, absolute_h_difference=abs(result["h"]-reference_h))
    output = {
        "sample": metadata, "h_interval": [0.01, 2.], "r": 0.,
        "repetitions_after_initialization": 9, "numerical_library_threads": 1,
        "python_version": platform.python_version(), "machine": platform.machine(),
        "scope": "Fresh sample-specific computation in every call, including pair distances "
                 "or spectral matrix construction. Excludes imports, sample loading, "
                 "validation, plotting and saving. Methods interleaved in seeded random order.",
        "limitation": "Coarse derivative scanning can miss narrow or degenerate minima. "
                      "Matching this sample does not establish completeness on other samples.",
        "summary": summary, "first_calls": first_calls, "runs": runs,
    }
    directory = KDE_ROOT / "outputs" / "qhat0_interval_example_perron_n100"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "fast_continuous_comparison.json"
    path.write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps({"summary": summary, "output": str(path)}, indent=2))


if __name__ == "__main__":
    main()
