"""Compare the existing FFT route with direct thesis-frequency quadrature.

This is an isolated research prototype. It does not modify the KDE library,
notebooks, figures, or protected thesis. No figures are displayed or created.
"""

from __future__ import annotations

import os

for variable in (
    "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

import importlib.util
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np
from threadpoolctl import threadpool_info

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reconstruction = load_module(
    "benchmark_reconstruction", "Code/Code_kde/kde_tools/fourier/reconstruction.py"
)
density = load_module("benchmark_density", "Code/Code_kde/kde_tools/density.py")


def divergence_from_gram(gram: np.ndarray) -> np.ndarray:
    diagonal = np.diag(gram)
    return 0.5 * np.maximum(
        diagonal[:, None] + diagonal[None, :] - 2.0 * gram, 0.0
    )


def existing_fft(samples, bandwidths, half_width=12.0, spatial_points=8192):
    """Same calls and operations as beta2_divergence_matrix_fft."""
    values, dx = reconstruction.kde_family_fft(
        samples, bandwidths, half_width, spatial_points
    )
    return divergence_from_gram(dx * (values @ values.T))


def thesis_frequency_quadrature(
    samples, bandwidths, half_width=12.0, spatial_points=8192
):
    """Direct trapezoidal spectral integral, factored as a real Gram product."""
    samples = np.asarray(samples, dtype=float)
    bandwidths = np.asarray(bandwidths, dtype=float)
    length = 2.0 * half_width
    dx = length / spatial_points
    omega = 2.0 * np.pi * np.fft.rfftfreq(spatial_points, d=dx)
    delta_omega = 2.0 * np.pi / length
    empirical_cf = np.exp(-1j * omega[:, None] * samples[None, :]).mean(axis=1)
    weights = np.full(len(omega), delta_omega / np.pi)
    weights[[0, -1]] *= 0.5
    feature_weights = np.sqrt(weights * np.abs(empirical_cf)**2)
    features = np.exp(-0.5 * (bandwidths[:, None] * omega[None, :])**2)
    features *= feature_weights[None, :]
    return divergence_from_gram(features @ features.T)


def main():
    methods = {"existing_fft": existing_fft,
               "thesis_frequency_quadrature": thesis_frequency_quadrature}
    validation = []
    for seed in (20260817, 20260828, 20260830):
        samples = np.random.default_rng(seed).normal(size=100)
        bandwidths = np.geomspace(0.01, 2.0, 20)
        exact = divergence_from_gram(
            density.exact_gaussian_kde_l2_gram(samples, bandwidths)
        )
        actual = {name: method(samples, bandwidths)
                  for name, method in methods.items()}
        errors = {name: float(np.max(np.abs(value-exact)))
                  for name, value in actual.items()}
        validation.append({
            "seed": seed, "n": 100, "N": 20,
            "maximum_absolute_error_against_exact": errors,
            "maximum_absolute_difference_between_routes": float(np.max(np.abs(
                actual["existing_fft"]-actual["thesis_frequency_quadrature"]
            ))),
        })
        if max(errors.values()) > 1e-10:
            raise ArithmeticError(f"Unexpected nominal-grid validation error: {errors}")

    timings = []
    for n, number_of_bandwidths in ((200, 201), (1000, 201), (200, 1000)):
        samples = np.random.default_rng(20261022+n).normal(size=n)
        bandwidths = np.geomspace(0.01, 2.0, number_of_bandwidths)
        for _ in range(2):
            for method in methods.values():
                method(samples, bandwidths)
        measurements = {name: [] for name in methods}
        last_results = {}
        for repetition in range(7):
            names = list(methods)
            if repetition % 2:
                names.reverse()
            for name in names:
                start = time.perf_counter()
                last_results[name] = methods[name](samples, bandwidths)
                measurements[name].append(time.perf_counter()-start)
        medians = {name: statistics.median(values)
                   for name, values in measurements.items()}
        entry = {
            "n": n, "N": number_of_bandwidths,
            "median_seconds": medians,
            "measurements_seconds": measurements,
            "speedup_existing_over_thesis": medians["existing_fft"] /
                medians["thesis_frequency_quadrature"],
            "maximum_absolute_difference_between_routes": float(np.max(np.abs(
                last_results["existing_fft"]-
                last_results["thesis_frequency_quadrature"]
            ))),
        }
        timings.append(entry)
        print(json.dumps(entry), flush=True)

    report = {
        "date": "2026-10-22",
        "scope": "Divergence-matrix construction only; no eigensolver, plotting, I/O or full experiment rerun in measured functions.",
        "settings": {"half_width": 12.0, "spatial_points": 8192,
                     "frequency_nodes": 4097, "h_min": 0.01, "h_max": 2.0,
                     "bandwidth_grid": "geometric", "warmups_per_method": 2,
                     "timed_repetitions": 7, "method_order": "alternating"},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform(), "machine": platform.machine(),
                        "requested_blas_threads_via_environment": 1,
                        "threadpools_reported": threadpool_info(),
                        "thread_note": "Thread limits requested before NumPy import; an empty threadpool list does not independently confirm the backend's runtime thread count."},
        "accuracy_validation": validation,
        "timings": timings,
        "limits": [
            "Same frequency cutoff and spacing for both routes, not a separate optimization of either grid.",
            "Exact Gaussian-pair validation uses three n=100 samples and 20 bandwidths; large timed matrices are compared against the existing implementation.",
            "The prototype does not reproduce spatial clipping/renormalization or the irfft real-Nyquist convention; their effects are negligible in these nominal-grid comparisons.",
            "No claim about agreement of every selected bandwidth or all existing experiments.",
        ],
    }
    (OUT / "results.json").write_text(json.dumps(report, indent=2)+"\n")
    print("Results saved to " + str(OUT / "results.json"), flush=True)


if __name__ == "__main__":
    main()
