"""Plot the reference-sample Q_0, with a local-minimum close-up.

Pass --h-max 100 to show the eventual decrease towards zero.
Pass --timing-runs 9 to measure the complete search repeatedly after its first
call. Each run builds a fresh objective; plotting and file operations are not
included in the measured time.
Pass --spectral-n 100 to add the Perron-peak choice on a geometric grid with
lambda_N = N**1.5. The spectral timing rebuilds the Fourier divergence matrix
and solves the eigenproblem on every run; analytic overlaps check the choice
outside the measured interval.

The Gaussian objective and analytic derivatives are implemented in
kde_tools.qhat; classification and endpoint comparisons use the shared search
in kde_tools.energy_minimization. The result is numerical, not certified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from threadpoolctl import threadpool_limits


KDE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KDE_ROOT))

from kde_tools.energy_minimization import EnergySearchConfig
from kde_tools.bandwidths import bandwidth_grid_linear_in_psi_beta
from kde_tools.experiment_inputs import load_reference_sample
from kde_tools.fourier.similarities import (
    DEFAULT_FREQUENCY_CUTOFF,
    DEFAULT_FREQUENCY_INTERVALS,
    beta2_divergence_matrix_fourier,
)
from kde_tools.pdf_export import enable_automatic_pdf_saving
from kde_tools.perron import PERRON_SOLVER_VERSION, perron_eigenpair_from_divergence
from kde_tools.qhat import gaussian_qhat, minimize_qhat
from kde_tools.similarities import beta2_divergence_matrix


def benchmark_spectral(samples, sample_metadata, h_min, h_max, n_grid, repetitions):
    """Measure a fresh production spectral selection and check exact overlaps."""
    lambda_value = float(n_grid**1.5)

    def run_once():
        started = perf_counter()
        grid = bandwidth_grid_linear_in_psi_beta(h_min, h_max, 2.0, n_grid)
        grid_done = perf_counter()
        divergence = beta2_divergence_matrix_fourier(samples, grid)
        matrix_done = perf_counter()
        perron = perron_eigenpair_from_divergence(divergence, lambda_value)
        selected_index = perron.selected_index
        selected_h = float(grid[selected_index])
        finished = perf_counter()
        return grid, divergence, perron, {
            "total_seconds": finished-started,
            "grid_seconds": grid_done-started,
            "divergence_seconds": matrix_done-grid_done,
            "eigensolve_and_peak_seconds": finished-matrix_done,
            "selected_index_zero_based": selected_index,
            "selected_h": selected_h,
        }

    with threadpool_limits(limits=1):
        grid, divergence, perron, first_run = run_once()
        runs = []
        for _ in range(repetitions):
            repeated_grid, repeated_divergence, repeated_perron, run = run_once()
            if run["selected_index_zero_based"] != first_run["selected_index_zero_based"]:
                raise ArithmeticError("Repeated spectral choices disagree.")
            runs.append(run)
        # Independent analytic Gaussian overlaps check the approximation and
        # peak selection. Validation is deliberately outside benchmark timing.
        exact_divergence = beta2_divergence_matrix(samples, grid)
        exact_perron = perron_eigenpair_from_divergence(exact_divergence, lambda_value)
    if exact_perron.selected_index != perron.selected_index:
        raise ArithmeticError("Fourier and analytic-overlap Perron peaks disagree.")

    selected_h = float(grid[perron.selected_index])
    result = {
        "sample": sample_metadata, "h_interval": [h_min, h_max],
        "N": n_grid, "grid": "geometric (r=0), including both endpoints",
        "lambda_N": lambda_value, "lambda_rule": "N**(3/2)",
        "similarity": "S_ij = exp(-lambda_N * D_ij), D_ij = 0.5 * ||f_hi-f_hj||_2**2",
        "divergence_method": "shared direct Fourier trapezoidal quadrature",
        "frequency_cutoff": DEFAULT_FREQUENCY_CUTOFF,
        "frequency_intervals": DEFAULT_FREQUENCY_INTERVALS,
        "solver": PERRON_SOLVER_VERSION,
        "selected_index_zero_based": perron.selected_index,
        "selected_index_one_based": perron.selected_index+1,
        "selected_h": selected_h,
        "peak_weight_sum_normalized": float(perron.vector[perron.selected_index]),
        "energy_at_selected_h": float(gaussian_qhat(samples, [selected_h])[0]),
        "relative_residual": perron.relative_residual,
        "relative_gap": perron.relative_gap,
        "validation": {
            "analytic_overlap_selected_h": float(grid[exact_perron.selected_index]),
            "maximum_absolute_divergence_error": float(np.max(np.abs(divergence-exact_divergence))),
            "maximum_absolute_vector_error": float(np.max(np.abs(perron.vector-exact_perron.vector))),
            "analytic_overlap_relative_residual": exact_perron.relative_residual,
        },
        "timing": {
            "scope": "Geometric grid creation, fresh Fourier divergence matrix, "
                     "exponential transformation, stabilized Perron eigensolve and peak selection. "
                     "Excludes imports, sample loading, validation, energy evaluation, plotting and saving.",
            "timer": "time.perf_counter", "numerical_library_threads": 1,
            "python_version": platform.python_version(), "machine": platform.machine(),
            "first_call": first_run,
            "repetitions_after_initialization": repetitions,
            "median_total_seconds": float(np.median([r["total_seconds"] for r in runs])),
            "minimum_total_seconds": min(r["total_seconds"] for r in runs),
            "maximum_total_seconds": max(r["total_seconds"] for r in runs),
            "median_divergence_seconds": float(np.median([r["divergence_seconds"] for r in runs])),
            "median_eigensolve_and_peak_seconds": float(np.median([r["eigensolve_and_peak_seconds"] for r in runs])),
            "runs": runs,
        },
    }
    return result, grid, divergence, perron.vector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h-max", type=float, default=2.0)
    parser.add_argument("--timing-runs", type=int, default=0)
    parser.add_argument("--spectral-n", type=int)
    args = parser.parse_args()
    if args.timing_runs < 0:
        parser.error("--timing-runs must be nonnegative")
    if args.spectral_n is not None and args.spectral_n < 2:
        parser.error("--spectral-n must be at least two")
    h_min, h_max = 0.01, args.h_max
    wide = h_max > 2.0
    example_name = "qhat0_interval_example" + (f"_hmax{h_max:g}" if h_max != 2 else "")
    if args.spectral_n is not None:
        example_name += f"_perron_n{args.spectral_n}"
    output = KDE_ROOT / "outputs" / example_name
    output.mkdir(parents=True, exist_ok=True)
    pdf_directory = enable_automatic_pdf_saving(example_name)
    samples, sample_metadata = load_reference_sample()
    config = EnergySearchConfig(scan_backend="exact")
    started = perf_counter()
    report = minimize_qhat(samples, h_min, h_max, r=0.0, config=config)
    first_call_seconds = perf_counter() - started
    timing = None
    if args.timing_runs:
        runs = []
        for _ in range(args.timing_runs):
            started = perf_counter()
            repeated = minimize_qhat(samples, h_min, h_max, r=0.0, config=config)
            seconds = perf_counter() - started
            if repeated["diagnostics"]["resolution_stable"] is not True:
                raise ArithmeticError("A timed search did not stabilize.")
            runs.append({
                "seconds": seconds,
                "local_minimum_h": repeated["best_local_minimum"]["h"],
                "exact_evaluations": repeated["diagnostics"]["exact_evaluations"],
            })
        durations = np.array([run["seconds"] for run in runs])
        timing = {
            "scope": "Complete minimize_qhat call with a fresh objective: pair distances, "
                     "stationary-point scan, root refinement, classification, resolution "
                     "check and endpoint comparison. Excludes imports, sample loading, "
                     "plotting and saving. Repeated runs follow one initialization call.",
            "timer": "time.perf_counter",
            "sample": sample_metadata,
            "h_interval": [h_min, h_max], "r": 0.0,
            "settings": report["diagnostics"]["settings"],
            "python_version": platform.python_version(),
            "machine": platform.machine(),
            "first_call_seconds": first_call_seconds,
            "repetitions_after_initialization": args.timing_runs,
            "median_seconds": float(np.median(durations)),
            "minimum_seconds": float(durations.min()),
            "maximum_seconds": float(durations.max()),
            "runs": runs,
        }
        (output / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")
    if report["diagnostics"]["resolution_stable"] is not True:
        raise ArithmeticError("The stationary-point search did not stabilize.")
    best = report["global_minimum"]
    local = report["best_local_minimum"]
    if local is None or local["log_curvature"] <= 0:
        raise ArithmeticError("Expected an interior minimum for the reference sample.")
    spectral = None
    if args.spectral_n is not None:
        spectral, spectral_grid, divergence, vector = benchmark_spectral(
            samples, sample_metadata, h_min, h_max, args.spectral_n,
            repetitions=args.timing_runs or 9,
        )
        spectral["continuous_local_minimum_h"] = local["h"]
        spectral["absolute_bandwidth_difference"] = abs(spectral["selected_h"]-local["h"])
        (output / "spectral_selection.json").write_text(json.dumps(spectral, indent=2) + "\n")
        np.savetxt(output / "perron_vector.csv", np.column_stack((spectral_grid, vector)),
                   delimiter=",", header="h,perron_weight_sum_normalized", comments="")
        np.savez(output / "spectral_matrices.npz", h_grid=spectral_grid,
                 divergence=divergence, similarity=np.exp(-spectral["lambda_N"]*divergence),
                 perron_vector=vector)

    h = np.unique(np.r_[np.geomspace(h_min, h_max, 1200),
                           np.linspace(h_min, h_max, 1200),
                           [p["h"] for p in report["stationary_points"]]])
    energy = gaussian_qhat(samples, h)
    close_h = np.linspace(0.07, 0.40, 700)
    close_energy = gaussian_qhat(samples, close_h)
    np.savetxt(output / "qhat0_curve.csv", np.column_stack((h, energy)),
               delimiter=",", header="h,qhat0", comments="")
    (output / "search_report.json").write_text(
        json.dumps({"sample": sample_metadata, "h_interval": [h_min, h_max],
                    "analysis": report, "timing": timing, "spectral": spectral}, indent=2) + "\n"
    )

    blue, red, amber, dark = "#245b90", "#bf3548", "#ba781b", "#253347"
    green = "#267653"
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelsize": 11, "axes.titlesize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#aeb8c4", "text.color": dark,
        "axes.labelcolor": dark, "xtick.color": dark, "ytick.color": dark,
        "pdf.fonttype": 42,
    })
    fig, (full, close) = plt.subplots(
        1, 2, figsize=(12.3, 5.5), gridspec_kw={"width_ratios": [1.35, 1]},
    )
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.23 if spectral else 0.19 if timing else 0.15,
                        top=0.76, wspace=0.27)
    fig.suptitle(
        r"$\widehat{Q}_0(h)=\frac{h^2}{2}\,\|\partial_h\widehat{f}_h\|_2^2$",
        x=0.075, y=0.96, ha="left", fontsize=20,
    )
    fig.text(0.075, 0.825,
             "Gaussian kernel KDE  |  fixed reference sample, n = 200  |  "
             + (f'N = {spectral["N"]}, geometric grid, '
                + r'$\lambda_N=N^{3/2}=$' + f'{spectral["lambda_N"]:g}' if spectral else
                "left: logarithmic h-axis; right: linear axes" if wide else "linear axes"),
             fontsize=10.5, color="#59677a")

    full.plot(h, energy, color=blue, lw=2.0, label=r"$\widehat{Q}_0(h)$")
    if wide:
        full.set_xscale("log")
    full.scatter([p["h"] for p in report["endpoints"]],
                 [p["energy"] for p in report["endpoints"]],
                 marker="s", s=32, color=dark, zorder=6,
                 label="Interval endpoints", clip_on=False)
    for maximum in report["local_maxima"]:
        full.scatter(maximum["h"], maximum["energy"], marker="^", s=54,
                     color=amber, edgecolors="white", linewidths=0.6, zorder=7)
        full.annotate(
            f'Local maximum\nh = {maximum["h"]:.4f}',
            xy=(maximum["h"], maximum["energy"]),
            xytext=(0.6 if wide else 1.15, 0.045),
            color=amber, arrowprops={"arrowstyle": "->", "color": amber},
        )
    full.scatter(local["h"], local["energy"], s=55, color=red, zorder=8,
                 edgecolors="white", linewidths=0.8, label="Interior local minimum")
    full.annotate(
        f'Local minimum\nh = {local["h"]:.5f}',
        xy=(local["h"], local["energy"]), xytext=(0.027 if wide else 0.42, 0.014),
        color=red, arrowprops={"arrowstyle": "->", "color": red},
    )
    if best["boundary"]:
        full.scatter(best["h"], best["energy"], s=65, color=green, marker="D",
                     edgecolors="white", linewidths=0.7, zorder=9, clip_on=False)
        full.annotate(
            f'Interval minimum: h = {best["h"]:g}\n'
            + r'$\widehat{Q}_0$' + f' = {best["energy"]:.6f}',
            xy=(best["h"], best["energy"]), xytext=(0.99, 0.29),
            textcoords="axes fraction", ha="right", color=green,
            arrowprops={"arrowstyle": "->", "color": green},
        )
    full.set(xlim=(h_min, h_max), ylim=(0, 0.067),
             title=f"Full interval [0.01, {h_max:g}]", xlabel="Bandwidth h",
             ylabel=r"Energy $\widehat{Q}_0(h)$")
    if wide:
        ticks = [value for value in (0.01, 0.1, 1, 10, 100, 1000) if value <= h_max]
        full.set_xticks(ticks, [f"{value:g}" for value in ticks])
    else:
        full.set_xticks([0.01, 0.5, 1, 1.5, 2], ["0.01", "0.5", "1", "1.5", "2"])
    full.legend(loc="upper right" if wide else "lower right", frameon=False, fontsize=9)

    close.plot(close_h, close_energy, color=blue, lw=2.0)
    close.axvline(local["h"], color=red, lw=1, ls="--", alpha=0.65)
    close.scatter(local["h"], local["energy"], s=65, color=red, zorder=8,
                  edgecolors="white", linewidths=0.8)
    close.annotate(
        r'$h_{\mathrm{loc}}$' + f' = {local["h"]:.5f}\n'
        + r'$\widehat{Q}_0(h_{\mathrm{loc}})$' + f' = {local["energy"]:.7f}',
        xy=(local["h"], local["energy"]),
        xytext=(0.105, 0.0045) if spectral else (0.212, 0.0051),
        color=red, arrowprops={"arrowstyle": "->", "color": red},
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.5},
    )
    close.set(xlim=(0.07, 0.40), ylim=(0.0021, 0.0065),
              title="Close-up around the local minimum", xlabel="Bandwidth h",
              ylabel=r"Energy $\widehat{Q}_0(h)$")
    close.xaxis.set_major_locator(MultipleLocator(0.1))
    if spectral:
        for ax in (full, close):
            ax.axvline(spectral["selected_h"], color=green, ls="--", lw=1.2, alpha=0.75)
            ax.scatter(spectral["selected_h"], spectral["energy_at_selected_h"],
                       marker="D", s=53, color=green, zorder=7,
                       edgecolors="white", linewidths=0.7,
                       label=f'Spectral choice h = {spectral["selected_h"]:.5f}')
        close.annotate(
            r'$h_{\mathrm{spectral}}$' + f' = {spectral["selected_h"]:.5f}',
            xy=(spectral["selected_h"], spectral["energy_at_selected_h"]),
            xytext=(0.215, 0.0058), color=green,
            arrowprops={"arrowstyle": "->", "color": green},
        )
        full.legend(loc="lower right", frameon=False, fontsize=8.5)
    for ax in (full, close):
        ax.grid(axis="y", color="#e5eaf0", lw=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(length=3)
    fig.text(0.075, 0.105 if spectral else 0.065 if timing else 0.035,
             (f"On [0.01, {h_max:g}], the right endpoint is lower than the interior local minimum."
              if best["boundary"] > 0 else
              f"The marked local minimum is the lowest detected energy on [0.01, {h_max:g}], including both endpoints."),
             fontsize=9, color="#59677a")
    if timing:
        fig.text(0.075, 0.067 if spectral else 0.027,
                 f'Complete energy search: {1000*timing["median_seconds"]:.1f} ms '
                 f'(median of {args.timing_runs} runs after initialization; '
                 'includes refinement and endpoint comparison).',
                 fontsize=9, color="#59677a")
    if spectral:
        fig.text(0.075, 0.029,
                 f'Spectral selection: {1000*spectral["timing"]["median_total_seconds"]:.1f} ms '
                 f'(median of {spectral["timing"]["repetitions_after_initialization"]} runs; '
                 'includes Fourier matrix construction and eigensolve; one numerical thread).',
                 fontsize=9, color="#59677a")

    png_path = output / f"{example_name}.png"
    fig.savefig(png_path, dpi=180, facecolor="white")
    plt.show()  # The shared exporter automatically saves the matching PDF.
    plt.close(fig)
    print(json.dumps({"png": str(png_path), "pdf_directory": str(pdf_directory),
                      "minimum": best, "local_minimum": local,
                      "local_minimum_count": len(report["local_minima"]),
                      "timing": timing,
                      "spectral": spectral,
                      "endpoints": report["endpoints"]}, indent=2))


if __name__ == "__main__":
    main()
