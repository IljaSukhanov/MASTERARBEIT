"""Compare the default frequency grid with cutoff 450 on the saved sample.

Run with one numerical-library thread for repeatable timings:
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/g_rs_mpl_cache python3 benchmark.py
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter

CODE_DIRECTORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE_DIRECTORY))

import numpy as np
from scipy.special import erfc

from kde_tools.density import exact_gaussian_kde_l2_gram
from kde_tools.fourier import (
    FOURIER_L2_METHOD_VERSION,
    gaussian_kde_l2_gram_fourier,
)


def main() -> None:
    sample_path = (
        CODE_DIRECTORY / "outputs" / "data" / "g_rs_approximation_vs_omega"
        / "sample.csv"
    )
    samples = np.loadtxt(sample_path, skiprows=1)
    bandwidths = np.geomspace(0.01, 2.0, 1000)
    default_spacing = math.pi / 12.0
    candidate_intervals = math.ceil(450.0 / default_spacing)
    grids = {
        "default": {
            "frequency_cutoff": math.pi * 8192 / 24.0,
            "frequency_points": 4097,
        },
        "cutoff_450": {
            "frequency_cutoff": 450.0,
            "frequency_points": candidate_intervals + 1,
        },
    }
    # These full evaluations warm up both grids and supply a matrix comparison.
    gram_matrices = {
        name: gaussian_kde_l2_gram_fourier(samples, bandwidths, **grid)
        for name, grid in grids.items()
    }

    durations = {name: [] for name in grids}
    for repetition in range(7):
        order = list(grids) if repetition % 2 == 0 else list(reversed(grids))
        for name in order:
            start = perf_counter()
            gaussian_kde_l2_gram_fourier(samples, bandwidths, **grids[name])
            durations[name].append(perf_counter() - start)

    # Use a small grid to make the independent analytic calculation inexpensive.
    validation_bandwidths = np.array(
        [0.01, 0.011, 0.02, 0.05, 0.1, 0.2, 0.4, 1.0, 2.0]
    )
    exact = exact_gaussian_kde_l2_gram(samples, validation_bandwidths)
    validation = {}
    for name, grid in grids.items():
        approximate = gaussian_kde_l2_gram_fourier(
            samples, validation_bandwidths, **grid
        )
        validation[name] = {
            "maximum_absolute_difference_from_analytic": float(
                np.max(np.abs(approximate - exact))
            ),
            "h_min_diagonal_difference_from_analytic": float(
                abs(approximate[0, 0] - exact[0, 0])
            ),
        }

    medians = {name: statistics.median(values) for name, values in durations.items()}
    report = {
        "method_version": FOURIER_L2_METHOD_VERSION,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"
            )
        },
        "scope": "Gram matrix construction only; excludes divergence and eigenproblem",
        "sample_path": str(sample_path),
        "n": len(samples),
        "N": len(bandwidths),
        "h_min": float(bandwidths.min()),
        "h_max": float(bandwidths.max()),
        "repetitions": 7,
        "grids": grids,
        "median_seconds": medians,
        "timing_seconds": durations,
        "measured_speedup": medians["default"] / medians["cutoff_450"],
        "node_reduction_fraction": 1.0 - (
            grids["cutoff_450"]["frequency_points"]
            / grids["default"]["frequency_points"]
        ),
        "maximum_absolute_difference_between_full_grams": float(
            np.max(np.abs(gram_matrices["default"] - gram_matrices["cutoff_450"]))
        ),
        "validation_bandwidths": validation_bandwidths.tolist(),
        "validation": validation,
        "uniform_truncation_bound_at_450": float(
            erfc(0.01 * 450.0) / (2 * math.sqrt(math.pi) * 0.01)
        ),
    }
    output_path = Path(__file__).with_name("benchmark.json")
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
