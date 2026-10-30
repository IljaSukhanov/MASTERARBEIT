"""Compare bounded Brent minimization with the coarse derivative-root scan.

SciPy distinguishes brentq (a sign-bracketed root finder) from minimize_scalar
with method='bounded' (a local minimizer that stays inside given bounds).
https://docs.scipy.org/doc/scipy-1.13.1/reference/generated/scipy.optimize.minimize_scalar.html
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from threadpoolctl import threadpool_limits

KDE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KDE_ROOT))

from kde_tools.experiment_inputs import load_reference_sample
from kde_tools.qhat import GaussianQhatObjective
from fast_continuous_minimum import coarse_qhat_minima


def bounded_minimum(samples, coordinate):
    objective = GaussianQhatObjective(samples)
    if coordinate == "h":
        bounds = (0.01, 2.)
        function = lambda h: objective.log_energy(np.log(h))[0]
    elif coordinate == "log_h":
        bounds = tuple(np.log([0.01, 2.]))
        function = lambda z: objective.log_energy(z)[0]
    else:
        raise ValueError("Unknown coordinate")
    solution = minimize_scalar(function, bounds=bounds, method="bounded",
                               options={"xatol": 2e-12, "maxiter": 1000})
    if not solution.success:
        raise ArithmeticError(solution.message)
    z = float(np.log(solution.x) if coordinate == "h" else solution.x)

    def point(t):
        energy, gradient, curvature = map(float, objective.log_energy(t))
        return dict(h=float(np.exp(t)), energy=energy,
                    log_derivative=gradient, log_curvature=curvature)

    interior = point(z)
    endpoints = [point(float(t)) for t in np.log([0.01, 2.])]
    endpoints[0]["h"], endpoints[1]["h"] = 0.01, 2.
    best = min([interior, *endpoints], key=lambda p: p["energy"])
    return dict(**best, exact_evaluations=len(objective._exact),
                minimizer_function_evaluations=solution.nfev,
                interior_candidate=interior, endpoints=endpoints)


def main():
    samples, metadata = load_reference_sample()
    output = KDE_ROOT / "outputs" / "qhat0_interval_example_perron_n100"
    reference = json.loads((KDE_ROOT / "outputs/qhat0_interval_example/search_report.json").read_text())
    h_star = reference["analysis"]["global_minimum"]["h"]
    objective = GaussianQhatObjective(samples)
    lower, upper = map(float, np.log([0.01, 2.]))
    endpoint_gradients = [float(objective.log_energy(t)[1]) for t in (lower, upper)]
    try:
        brentq(lambda z: objective.log_energy(z)[1], lower, upper)
        root_attempt = "Unexpectedly accepted"
    except ValueError as error:
        root_attempt = str(error)

    def coarse():
        result = coarse_qhat_minima(samples, scan_points=33)
        return dict(**result["best_candidate"], exact_evaluations=result["exact_evaluations"])

    methods = {"bounded_brent_in_h": lambda: bounded_minimum(samples, "h"),
               "bounded_brent_in_log_h": lambda: bounded_minimum(samples, "log_h"),
               "coarse_33_then_brentq": coarse}
    runs = {name: [] for name in methods}
    rng = np.random.default_rng(20261030)
    with threadpool_limits(limits=1):
        for method in methods.values():
            method()  # Initialization; every measured call gets a fresh objective.
        for _ in range(9):
            for name in rng.permutation(list(methods)):
                started = perf_counter()
                result = methods[name]()
                runs[name].append(dict(seconds=perf_counter()-started, result=result))
        # Existing independent samples and checked references provide an
        # empirical check of how often the whole-interval local solve agrees.
        bank_root = KDE_ROOT / "outputs" / "alpha_repeated_sample_experiment"
        bank = np.load(bank_root / "samples.npz")["samples"]
        independent_checks = []
        for index, sample in enumerate(bank):
            ref = json.loads((bank_root / "checkpoints" / f"sample_{index:03d}.json").read_text())
            target = ref["reference"]["global_minimum"]
            row = dict(sample_id=index, reference_h=target["h"], reference_energy=target["energy"])
            for coordinate in ("h", "log_h"):
                result = bounded_minimum(sample, coordinate)
                row[coordinate] = dict(
                    h=result["h"], energy=result["energy"],
                    absolute_log_h_error=abs(float(np.log(result["h"]/target["h"]))),
                    relative_energy_excess=result["energy"]/target["energy"]-1,
                )
            independent_checks.append(row)
    summary = {}
    for name, measurements in runs.items():
        summary[name] = dict(median_ms=1000*float(np.median([r["seconds"] for r in measurements])),
                             result=measurements[-1]["result"],
                             absolute_h_error=abs(measurements[-1]["result"]["h"]-h_star))
    validation = {}
    for coordinate in ("h", "log_h"):
        validation[coordinate] = dict(
            samples=len(independent_checks),
            bandwidth_matches_within_log_tolerance_1e_minus6=sum(
                r[coordinate]["absolute_log_h_error"] <= 1e-6 for r in independent_checks),
            energy_matches_within_relative_tolerance_1e_minus8=sum(
                abs(r[coordinate]["relative_energy_excess"]) <= 1e-8 for r in independent_checks),
            maximum_relative_energy_excess=max(r[coordinate]["relative_energy_excess"] for r in independent_checks),
        )
    result = dict(sample=metadata, h_interval=[0.01, 2.], reference_h=h_star,
                  endpoint_log_derivatives=endpoint_gradients,
                  full_interval_brentq_attempt=root_attempt,
                  timing_scope="Nine initialized runs, one numerical thread; includes fresh pair distances, "
                               "bounded minimization and endpoint comparison; excludes imports, loading and saving.",
                  summary=summary, runs=runs, independent_sample_validation=validation,
                  independent_sample_checks=independent_checks)
    path = output / "whole_interval_brent_comparison.json"
    path.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({key:result[key] for key in
                      ("reference_h", "endpoint_log_derivatives", "full_interval_brentq_attempt",
                       "summary", "independent_sample_validation")}, indent=2))


if __name__ == "__main__":
    main()
