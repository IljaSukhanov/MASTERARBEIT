"""Check the Figure 3.3 jump against independent numerical refinements.

This checks eigensolver agreement, analytic Gaussian KDE integration,
FFT refinement, and refinement of the full logarithmic bandwidth grid.
All checks use the same original data; they do not establish a continuum
limit theorem or explain which sample features predict a crossing.
"""

from pathlib import Path
import json
import os
import sys
import time

CODE_DIRECTORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_DIRECTORY))
os.environ.setdefault("MPLCONFIGDIR", str(CODE_DIRECTORY / ".matplotlib-cache"))

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq
from scipy.signal import find_peaks
from threadpoolctl import threadpool_limits

from kde_tools.bandwidths import bandwidth_grid_linear_in_psi_beta
from kde_tools.fourier import DEFAULT_FOURIER_GRID_TAG, beta2_divergence_matrix_fft
from kde_tools.perron import perron_eigenpair_from_divergence
from kde_tools.similarities import beta2_divergence_matrix

OUTPUT_DIRECTORY = (
    CODE_DIRECTORY / "outputs/data/spectral_lambda_jump_study/numerical_artefact_checks"
    / DEFAULT_FOURIER_GRID_TAG
)


def peak_record(result, h_grid):
    peaks = find_peaks(result.vector, prominence=1e-8 * result.vector.max())[0]
    return {
        "selected_bandwidth": float(h_grid[result.selected_index]),
        "selected_index": result.selected_index,
        "resolved_peak_bandwidths": h_grid[peaks].tolist(),
        "resolved_peak_weights": result.vector[peaks].tolist(),
        "relative_residual": result.relative_residual,
        "relative_eigenvalue_gap": result.relative_gap,
    }


def refine_crossing(divergence, h_grid):
    cut = int(np.searchsorted(h_grid, 0.16))

    def gap(value):
        result = perron_eigenpair_from_divergence(divergence, value)
        return float(result.vector[:cut].max() - result.vector[cut:].max())

    crossing = brentq(gap, 10.0, 30.0, xtol=1e-9, rtol=1e-12)
    below = perron_eigenpair_from_divergence(divergence, crossing - 0.02)
    at = perron_eigenpair_from_divergence(divergence, crossing)
    above = perron_eigenpair_from_divergence(divergence, crossing + 0.02)
    lo = int(np.argmax(at.vector[:cut]))
    hi = cut + int(np.argmax(at.vector[cut:]))
    valley = lo + int(np.argmin(at.vector[lo:hi + 1]))
    assert lo < valley < hi
    assert below.selected_index < cut <= above.selected_index
    return {
        "lambda_crossing": crossing,
        "lower_peak_h_at_crossing": float(h_grid[lo]),
        "upper_peak_h_at_crossing": float(h_grid[hi]),
        "relative_valley_depth_at_crossing": float(
            (min(at.vector[lo], at.vector[hi]) - at.vector[valley]) / at.vector.max()
        ),
        "gap_below": gap(crossing - 0.02),
        "gap_above": gap(crossing + 0.02),
        "below": peak_record(below, h_grid),
        "at": peak_record(at, h_grid),
        "above": peak_record(above, h_grid),
    }


def run_checks():
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    original = np.random.default_rng(20260823).normal(size=200)
    original_grid = bandwidth_grid_linear_in_psi_beta(grid_size=201)
    report = {
        "sample_seed": 20260823, "sample_size": 200,
        "bandwidth_min": 0.01, "bandwidth_max": 2.0,
        "peak_relative_prominence": 1e-8,
        "scope": "Numerical checks on Figure 3.3; no continuum-limit proof.",
        "independent_dense_solver": [], "fft_refinement": [],
        "bandwidth_grid_refinement": [],
    }
    with threadpool_limits(limits=1):
        baseline = beta2_divergence_matrix_fft(original, original_grid)
        for value in (17.86, 17.87, 17.88):
            shared = perron_eigenpair_from_divergence(baseline, value)
            # Solve the unshifted original similarity with an independent dense routine.
            similarity = np.exp(-value * baseline)
            eigenvalues, vectors = eigh(similarity, subset_by_index=[199, 200])
            vector = vectors[:, -1]
            vector *= np.sign(vector.sum())
            vector /= vector.sum()
            report["independent_dense_solver"].append({
                "lambda": value, "shared_selected_h": float(original_grid[shared.selected_index]),
                "dense_selected_h": float(original_grid[np.argmax(vector)]),
                "maximum_component_discrepancy": float(np.abs(vector - shared.vector).max()),
                "relative_eigenvalue_gap_original_similarity": float(
                    (eigenvalues[-1] - eigenvalues[-2]) / eigenvalues[-1]
                ),
                "shared_relative_residual": shared.relative_residual,
            })
        started = time.perf_counter()
        exact = beta2_divergence_matrix(original, original_grid)
        report["analytic_integration"] = {
            "maximum_divergence_discrepancy_from_original_fft": float(np.abs(exact - baseline).max()),
            **refine_crossing(exact, original_grid),
        }
        print(f"Analytic integration checked in {time.perf_counter() - started:.1f}s", flush=True)
        for half_width, spatial_points in ((12.0, 8192), (16.0, 16384), (16.0, 32768)):
            divergence = beta2_divergence_matrix_fft(
                original, original_grid, half_width=half_width, spatial_points=spatial_points
            )
            row = {"half_width": half_width, "spatial_points": spatial_points,
                   "maximum_divergence_discrepancy_from_analytic": float(np.abs(divergence - exact).max()),
                   **refine_crossing(divergence, original_grid)}
            report["fft_refinement"].append(row)
            print(f"FFT {half_width:g}/{spatial_points}: lambda crossing {row['lambda_crossing']:.10f}", flush=True)
        for count in (201, 401, 801, 1601):
            grid = bandwidth_grid_linear_in_psi_beta(grid_size=count)
            divergence = baseline if count == 201 else beta2_divergence_matrix_fft(original, grid)
            row = {"bandwidth_grid_size": count, **refine_crossing(divergence, grid)}
            report["bandwidth_grid_refinement"].append(row)
            print(f"Bandwidth grid {count}: lambda crossing {row['lambda_crossing']:.10f}; "
                  f"peaks {row['lower_peak_h_at_crossing']:.6f}, {row['upper_peak_h_at_crossing']:.6f}", flush=True)
        source = CODE_DIRECTORY / "outputs/data/spectral_lambda_jump_study/2ce04db2a16e/checkpoint.npz"
        with np.load(source, allow_pickle=False) as checkpoint:
            no_switch_sample = checkpoint["samples"][835].copy()
        divergence = beta2_divergence_matrix_fft(no_switch_sample, original_grid)
        lambdas = np.geomspace(0.01, 200.0, 501)
        rows = []
        for value in lambdas:
            result = perron_eigenpair_from_divergence(divergence, value)
            row = {"lambda": float(value), **peak_record(result, original_grid)}
            if len(row["resolved_peak_weights"]) >= 2:
                ranked = np.sort(row["resolved_peak_weights"])
                row["relative_gap_between_tallest_two_peaks"] = float(
                    (ranked[-1] - ranked[-2]) / ranked[-1]
                )
            rows.append(row)
        report["no_switch_example"] = {
            "repetition_zero_based": 835, "lambda_grid": "geomspace(0.01, 200, 501)",
            "profiles_with_multiple_resolved_peaks": sum(len(row["resolved_peak_bandwidths"]) >= 2 for row in rows),
            "lower_bandwidth_peak_always_selected_during_coexistence": all(
                row["selected_bandwidth"] == row["resolved_peak_bandwidths"][0]
                for row in rows if len(row["resolved_peak_bandwidths"]) >= 2
            ),
            "last_sampled_lambda_with_multiple_peaks": max(
                row["lambda"] for row in rows if len(row["resolved_peak_bandwidths"]) >= 2
            ),
            "minimum_relative_height_gap_while_multiple_peaks_resolved": min(
                row["relative_gap_between_tallest_two_peaks"] for row in rows
                if "relative_gap_between_tallest_two_peaks" in row
            ),
            "maximum_adjacent_selected_index_change": int(np.max(np.abs(np.diff([row["selected_index"] for row in rows])))),
            "profiles": rows,
        }
    path = OUTPUT_DIRECTORY / "results.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Results: {path}", flush=True)
    return report


if __name__ == "__main__":
    run_checks()
