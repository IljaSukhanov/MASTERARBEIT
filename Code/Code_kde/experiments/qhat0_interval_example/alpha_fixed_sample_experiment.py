"""Compare six lambda_N=N**alpha rules on the canonical fixed KDE sample.

All selectors use the shared Fourier divergence and stabilized Perron solver.
The continuous reference is the shared analytic Q_0 interval search. This is
a fixed-sample approximation experiment, not a statistical tuning study.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from threadpoolctl import threadpool_limits


KDE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KDE_ROOT))

from kde_tools.bandwidths import bandwidth_grid_linear_in_psi_beta
from kde_tools.energy_minimization import EnergySearchConfig
from kde_tools.experiment_inputs import load_reference_sample
from kde_tools.fourier.similarities import (
    DEFAULT_FREQUENCY_CUTOFF,
    DEFAULT_FREQUENCY_INTERVALS,
    beta2_divergence_matrix_fourier,
    gaussian_kde_l2_gram_fourier,
)
from kde_tools.pdf_export import enable_automatic_pdf_saving
from kde_tools.perron import PERRON_SOLVER_VERSION, perron_eigenpair_from_divergence
from kde_tools.qhat import GaussianQhatObjective, minimize_qhat
from kde_tools.similarities import beta2_divergence_matrix, squared_l2_distance_from_gram


NAME = "alpha_fixed_sample_experiment"
ALPHAS = (0.5, 1.0, 1.25, 1.5, 1.75, 1.9)
GRID_SIZES = (100, 200, 400)
H_MIN, H_MAX = 0.01, 2.0


def main():
    output = KDE_ROOT / "outputs" / NAME
    output.mkdir(parents=True, exist_ok=True)
    pdf_directory = enable_automatic_pdf_saving(NAME)
    samples, sample_metadata = load_reference_sample()
    objective = GaussianQhatObjective(samples)
    reference = minimize_qhat(objective, H_MIN, H_MAX, r=0.0,
                              config=EnergySearchConfig(scan_backend="exact"))
    if reference["diagnostics"]["resolution_stable"] is not True:
        raise ArithmeticError("The continuous reference search did not stabilize.")
    h_star = reference["global_minimum"]["h"]
    q_star = reference["global_minimum"]["energy"]
    rows, validations, grid_baselines = [], [], []
    with threadpool_limits(limits=1):
        for n_grid in GRID_SIZES:
            grid = bandwidth_grid_linear_in_psi_beta(H_MIN, H_MAX, 2., n_grid)
            nearest_h = float(grid[np.argmin(np.abs(grid-h_star))])
            nearest_error = abs(nearest_h-h_star)
            grid_baselines.append(dict(N=n_grid, h=nearest_h,
                                       relative_error_percent=100*nearest_error/h_star))
            started = perf_counter()
            divergence = beta2_divergence_matrix_fourier(samples, grid)
            matrix_seconds = perf_counter()-started
            vectors, selected_indices = [], []
            for alpha in ALPHAS:
                lambda_value = float(n_grid**alpha)
                started = perf_counter()
                result = perron_eigenpair_from_divergence(divergence, lambda_value)
                index = result.selected_index
                eigen_seconds = perf_counter()-started
                selected_h = float(grid[index])
                selected_q = objective.energy(selected_h)
                rows.append(dict(
                    N=n_grid, alpha=alpha, lambda_N=lambda_value,
                    selected_index_zero_based=index, selected_index_one_based=index+1,
                    selected_h=selected_h, reference_h=h_star,
                    absolute_h_error=abs(selected_h-h_star),
                    signed_relative_h_error_percent=100*(selected_h/h_star-1),
                    relative_h_error_percent=100*abs(selected_h-h_star)/h_star,
                    energy_at_selected_h=selected_q,
                    relative_energy_excess_percent=100*(selected_q/q_star-1),
                    nearest_grid_h=nearest_h,
                    nearest_grid_relative_error_percent=100*nearest_error/h_star,
                    relative_residual=result.relative_residual,
                    relative_gap=result.relative_gap,
                    divergence_construction_seconds=matrix_seconds,
                    eigensolve_and_peak_seconds=eigen_seconds,
                ))
                vectors.append(result.vector)
                selected_indices.append(index)
            np.savez(output / f"profiles_n{n_grid}.npz", h_grid=grid,
                     alphas=np.array(ALPHAS), divergence=divergence,
                     perron_vectors=np.array(vectors))

            # Keep the cutoff fixed and halve the frequency spacing.
            refined_gram = gaussian_kde_l2_gram_fourier(
                samples, grid, frequency_cutoff=DEFAULT_FREQUENCY_CUTOFF,
                frequency_points=2*DEFAULT_FREQUENCY_INTERVALS+1,
            )
            refined_divergence = 0.5*squared_l2_distance_from_gram(refined_gram)
            refined_indices = [perron_eigenpair_from_divergence(
                refined_divergence, float(n_grid**alpha)).selected_index for alpha in ALPHAS]
            if refined_indices != selected_indices:
                raise ArithmeticError(f"Frequency refinement changed a peak at N={n_grid}.")
            check = dict(
                N=n_grid, frequency_intervals_baseline=DEFAULT_FREQUENCY_INTERVALS,
                frequency_intervals_refined=2*DEFAULT_FREQUENCY_INTERVALS,
                maximum_absolute_divergence_refinement_change=float(
                    np.max(np.abs(divergence-refined_divergence))),
                all_refined_indices_match=True,
            )
            if n_grid == GRID_SIZES[0]:
                exact = beta2_divergence_matrix(samples, grid)
                exact_indices = [perron_eigenpair_from_divergence(
                    exact, float(n_grid**alpha)).selected_index for alpha in ALPHAS]
                if exact_indices != selected_indices:
                    raise ArithmeticError("Analytic overlaps changed an N=100 peak.")
                check.update(all_analytic_overlap_indices_match=True,
                             maximum_absolute_analytic_divergence_error=float(
                                 np.max(np.abs(divergence-exact))))
            validations.append(check)
            print(f"Finished N={n_grid}; all six selections passed frequency refinement.", flush=True)

    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    winners = []
    for n_grid in GRID_SIZES:
        subset = [row for row in rows if row["N"] == n_grid]
        lowest = min(row["absolute_h_error"] for row in subset)
        winners.append(dict(N=n_grid,
                            alphas=[row["alpha"] for row in subset
                                    if np.isclose(row["absolute_h_error"], lowest, rtol=0., atol=1e-14)],
                            minimum_relative_error_percent=100*lowest/h_star))
    summary = dict(
        sample=sample_metadata, h_interval=[H_MIN, H_MAX], grid="geometric (r=0)",
        alphas=ALPHAS, grid_sizes=GRID_SIZES, lambda_rule="N**alpha; multiplier c=1",
        divergence="half squared L2 distance between Gaussian KDEs",
        frequency_cutoff=DEFAULT_FREQUENCY_CUTOFF,
        frequency_intervals=DEFAULT_FREQUENCY_INTERVALS,
        perron_solver=PERRON_SOLVER_VERSION, numerical_library_threads=1,
        reference_analysis=reference, grid_baselines=grid_baselines,
        winners_by_grid_size=winners, results=rows, validation=validations,
        timing_note="Each divergence matrix was built once per N and reused across alpha. "
                    "Recorded component timings are single observations, not repeated benchmarks.",
        scope="One fixed sample; numerical approximation of the Q_0 interval minimum.",
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")

    colors = ("#245b90", "#2a9d8f", "#c79225", "#8b5ca8", "#c34c5e", "#344254")
    markers = ("o", "s", "^", "D", "v", "P")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#aeb8c4", "text.color": "#253347",
        "axes.labelcolor": "#253347", "xtick.color": "#253347",
        "ytick.color": "#253347", "pdf.fonttype": 42,
    })
    fig, (choices, errors) = plt.subplots(1, 2, figsize=(12.8, 6.2))
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.16, top=0.71, wspace=0.24)
    fig.suptitle(r"Choosing $\alpha$ on the fixed reference sample", x=0.075,
                 y=0.97, ha="left", fontsize=19)
    fig.text(0.075, 0.89,
             r"$n=200$  |  geometric grid on $[0.01,2]$  |  $\lambda_N=N^\alpha$"
             + f"  |  continuous reference h* = {h_star:.6f}",
             color="#59677a", fontsize=11)
    # Combine identical paths so a later curve cannot hide an earlier one.
    path_groups = {}
    for alpha in ALPHAS:
        subset = [row for row in rows if row["alpha"] == alpha]
        key = tuple(row["selected_h"] for row in subset)
        if key not in path_groups:
            path_groups[key] = dict(alphas=[], rows=subset)
        path_groups[key]["alphas"].append(alpha)
    for group, color, marker in zip(path_groups.values(), colors, markers):
        subset = group["rows"]
        alpha_labels = ", ".join(f"{alpha:g}" for alpha in group["alphas"])
        label = (rf"$\alpha={alpha_labels}$" if len(group["alphas"]) == 1 else
                 r"$\alpha\in\{" + alpha_labels + r"\}$")
        x = [row["N"] for row in subset]
        choices.plot(x, [row["selected_h"] for row in subset], lw=1.8,
                     marker=marker, ms=6, color=color, label=label)
        errors.plot(x, [row["relative_h_error_percent"] for row in subset], lw=1.8,
                    marker=marker, ms=6, color=color)
    choices.axhline(h_star, color="#17202e", ls="--", lw=1.4,
                    label="Continuous reference")
    choices.plot(GRID_SIZES, [p["h"] for p in grid_baselines], color="#8c929a",
                 lw=1.1, ls=":", marker="x", ms=5, label="Nearest available grid point")
    errors.plot(GRID_SIZES, [p["relative_error_percent"] for p in grid_baselines],
                color="#8c929a", lw=1.1, ls=":", marker="x", ms=5)
    fig.legend(*choices.get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.53, 0.835), ncol=4, frameon=False,
               fontsize=10, columnspacing=1.8, handlelength=2.5)
    choices.set(title="Selected bandwidth", ylabel="Bandwidth h")
    errors.set(title="Distance from the continuous reference",
               ylabel="Absolute relative bandwidth error (%)", yscale="log")
    errors.yaxis.set_major_formatter(ScalarFormatter())
    for axis in (choices, errors):
        axis.set(xlabel="Number of bandwidth candidates N", xlim=(85, 415))
        axis.set_xticks(GRID_SIZES)
        axis.grid(axis="y", color="#e5eaf0", lw=0.7)
        axis.set_axisbelow(True)
    fig.text(0.075, 0.07,
             "Identical paths for alpha = 1.75 and 1.9 are combined. The dotted line is the nearest-grid benchmark.",
             color="#59677a", fontsize=9)
    fig.text(0.075, 0.032,
             "All 18 peak selections agree after doubling the Fourier frequency resolution. One fixed sample.",
             color="#59677a", fontsize=9)
    png = output / f"{NAME}.png"
    fig.savefig(png, dpi=180, facecolor="white")
    plt.show()  # Shared automatic PDF exporter saves the same figure.
    plt.close(fig)
    print(json.dumps(dict(png=str(png), pdf_directory=str(pdf_directory),
                          reference_h=h_star, winners=winners,
                          rows=[{key:row[key] for key in
                                 ("N", "alpha", "lambda_N", "selected_h", "relative_h_error_percent")}
                                for row in rows]), indent=2))


if __name__ == "__main__":
    main()
