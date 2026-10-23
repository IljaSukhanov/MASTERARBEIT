"""Compare the direct thesis quadrature with the previous reconstruction route.

No figure is plotted and no historical checkpoint is changed. Run from any
working directory; results are written beside this script.
"""

from __future__ import annotations

import os
for variable in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import json
from pathlib import Path
import statistics
import sys
import time

CODE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE))
import numpy as np

from kde_tools.fourier import (
    FOURIER_L2_METHOD_VERSION, beta2_divergence_matrix_fourier, kde_family_fft,
)
from kde_tools.perron import perron_eigenpair_from_divergence
from kde_tools.similarities import beta2_divergence_matrix, squared_l2_distance_from_gram


def previous_route(samples, bandwidths, half_width=12.0, spatial_points=8192):
    values, dx = kde_family_fft(samples, bandwidths, half_width, spatial_points)
    return 0.5 * squared_l2_distance_from_gram(dx * (values @ values.T))


def main():
    report = {"l2_method": FOURIER_L2_METHOD_VERSION, "numpy": np.__version__,
              "exact_checks": [], "selection_checks": [], "timings": []}
    for seed in (20260817, 20260823, 20260828):
        sample = np.random.default_rng(seed).normal(size=100)
        h = np.geomspace(0.01, 2.0, 20)
        reference = beta2_divergence_matrix(sample, h)
        current = beta2_divergence_matrix_fourier(sample, h)
        previous = previous_route(sample, h)
        np.testing.assert_allclose(current, reference, atol=1e-12, rtol=1e-10)
        report["exact_checks"].append({
            "seed": seed, "n": 100, "N": 20,
            "new_maximum_absolute_error": float(np.max(np.abs(current-reference))),
            "previous_maximum_absolute_error": float(np.max(np.abs(previous-reference))),
        })

    for n, N, seeds in (
        (200, 201, (20260818, 20260823, 20260828, 2026102301)),
        (1000, 201, (20260818, 20260823, 2026102301)),
        (200, 1000, (20260828,)),
    ):
        h = np.geomspace(0.01, 2.0, N)
        for seed in seeds:
            sample = np.random.default_rng(seed).normal(size=n)
            current = beta2_divergence_matrix_fourier(sample, h)
            previous = previous_route(sample, h)
            refined = beta2_divergence_matrix_fourier(
                sample, h, half_width=24.0, spatial_points=16384
            )
            row = {"n": n, "N": N, "seed": seed,
                   "matrix_difference_previous": float(np.max(np.abs(current-previous))),
                   "matrix_difference_refined_spacing": float(np.max(np.abs(current-refined))),
                   "selections": []}
            for lam in (1.0, 10.0, 100.0, 1000.0, n**1.5):
                choices = {}
                for name, d in (("new", current), ("previous", previous), ("refined", refined)):
                    eigenpair = perron_eigenpair_from_divergence(d, lam)
                    choices[name] = eigenpair.selected_index
                row["selections"].append({"lambda": lam, **choices})
                if len(set(choices.values())) != 1:
                    raise AssertionError(f"Selection changed: {n=}, {N=}, {seed=}, {lam=}, {choices}")
            report["selection_checks"].append(row)
            print(json.dumps({"n": n, "N": N, "seed": seed, "choices": row["selections"]}), flush=True)

    for n, N in ((200, 201), (1000, 201), (200, 1000)):
        sample = np.random.default_rng(20261022+n).normal(size=n)
        h = np.geomspace(0.01, 2.0, N)
        functions = {"previous": previous_route, "new": beta2_divergence_matrix_fourier}
        for function in functions.values():
            function(sample, h)
        times = {name: [] for name in functions}
        for i in range(5):
            names = list(functions)
            if i % 2:
                names.reverse()
            for name in names:
                started = time.perf_counter()
                functions[name](sample, h)
                times[name].append(time.perf_counter()-started)
        medians = {name: statistics.median(t) for name, t in times.items()}
        report["timings"].append({"n": n, "N": N, "seconds": medians,
                                  "speedup": medians["previous"]/medians["new"]})
    report["scope"] = (
        "Eight sample/grid cases, five lambdas per case, comparing current, previous "
        "and refined-grid choices. Exact checks use three smaller samples. "
        "This is a migration check, not a rerun of every saved Monte Carlo experiment."
    )
    path = Path(__file__).with_name("migration_report.json")
    path.write_text(json.dumps(report, indent=2)+"\n")
    print("Report: "+str(path), flush=True)


if __name__ == "__main__":
    main()
