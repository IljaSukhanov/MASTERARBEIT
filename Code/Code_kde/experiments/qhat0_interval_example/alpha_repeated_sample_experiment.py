"""Paired comparison of alpha across 200 independent standard-Gaussian samples.

Each sample has its own analytic, resolution-checked Q_0 interval reference.
All six exponents and three grids see identical observations. Frequency
refinement validates every spectral selection; checkpoints allow resuming.
This evaluates approximation of Q_0 minimizers, not density-estimation risk.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

KDE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KDE_ROOT))

from kde_tools.bandwidths import bandwidth_grid_linear_in_psi_beta
from kde_tools.energy_minimization import EnergySearchConfig
from kde_tools.experiment_inputs import paired_gaussian_sample_bank, sampling_inputs
from kde_tools.fourier.similarities import (
    DEFAULT_FREQUENCY_CUTOFF, DEFAULT_FREQUENCY_INTERVALS, FOURIER_L2_METHOD_VERSION,
    gaussian_kde_l2_gram_fourier,
)
from kde_tools.pdf_export import enable_automatic_pdf_saving
from kde_tools.perron import PERRON_SOLVER_VERSION, perron_eigenpair_from_divergence
from kde_tools.qhat import GaussianQhatObjective, minimize_qhat
from kde_tools.similarities import squared_l2_distance_from_gram


NAME = "alpha_repeated_sample_experiment"
ALPHAS = (0.5, 1.0, 1.25, 1.5, 1.75, 1.9)
GRID_SIZES = (100, 200, 400)
REPETITIONS, SAMPLE_SIZE = 200, 200
H_MIN, H_MAX = 0.01, 2.0


def divergence_at_resolution(samples, grid, intervals):
    gram = gaussian_kde_l2_gram_fourier(
        samples, grid, frequency_cutoff=DEFAULT_FREQUENCY_CUTOFF,
        frequency_points=intervals+1,
    )
    return 0.5*squared_l2_distance_from_gram(gram)


def process_sample(index, sample):
    objective = GaussianQhatObjective(sample)
    started = perf_counter()
    reference = minimize_qhat(objective, H_MIN, H_MAX, r=0.,
                              config=EnergySearchConfig(scan_backend="exact"))
    if reference["diagnostics"]["resolution_stable"] is not True:
        reference = minimize_qhat(objective, H_MIN, H_MAX, r=0., config=
                                  EnergySearchConfig(scan_backend="exact", scan_points=1025,
                                                     max_refinements=3))
    if reference["diagnostics"]["resolution_stable"] is not True:
        raise ArithmeticError(f"Sample {index}: reference did not stabilize.")
    reference_seconds = perf_counter()-started
    target = reference["global_minimum"]
    h_star, q_star = target["h"], target["energy"]
    rows, checks = [], []
    order_rng = np.random.default_rng(np.random.SeedSequence([20261030, index]))
    for n_grid in GRID_SIZES:
        grid = bandwidth_grid_linear_in_psi_beta(H_MIN, H_MAX, 2., n_grid)
        nearest_h = float(grid[np.argmin(np.abs(grid-h_star))])
        started = perf_counter()
        divergence = divergence_at_resolution(sample, grid, DEFAULT_FREQUENCY_INTERVALS)
        matrix_seconds = perf_counter()-started
        results, solve_seconds = {}, {}
        for alpha in order_rng.permutation(ALPHAS):
            alpha = float(alpha)
            started = perf_counter()
            result = perron_eigenpair_from_divergence(divergence, float(n_grid**alpha))
            _ = result.selected_index
            solve_seconds[alpha] = perf_counter()-started
            results[alpha] = result

        refined = divergence_at_resolution(sample, grid, 2*DEFAULT_FREQUENCY_INTERVALS)
        refined_results = {alpha: perron_eigenpair_from_divergence(refined, float(n_grid**alpha))
                           for alpha in ALPHAS}
        changed = [alpha for alpha in ALPHAS
                   if results[alpha].selected_index != refined_results[alpha].selected_index]
        intervals_used = DEFAULT_FREQUENCY_INTERVALS
        if changed:
            # A changed peak is not silently ignored. Confirm the finer result
            # at another doubling, then use the confirmed fine result for this block.
            finest = divergence_at_resolution(sample, grid, 4*DEFAULT_FREQUENCY_INTERVALS)
            finest_results = {alpha: perron_eigenpair_from_divergence(finest, float(n_grid**alpha))
                              for alpha in ALPHAS}
            if any(refined_results[a].selected_index != finest_results[a].selected_index
                   for a in ALPHAS):
                raise ArithmeticError(f"Sample {index}, N={n_grid}: Fourier peaks do not stabilize.")
            results = refined_results
            intervals_used = 2*DEFAULT_FREQUENCY_INTERVALS
        checks.append(dict(
            N=n_grid, baseline_intervals=DEFAULT_FREQUENCY_INTERVALS,
            refinement_intervals=2*DEFAULT_FREQUENCY_INTERVALS,
            maximum_absolute_divergence_change=float(np.max(np.abs(divergence-refined))),
            changed_alphas=changed, intervals_used=intervals_used,
            all_final_peaks_resolution_confirmed=True,
        ))
        for alpha in ALPHAS:
            result = results[alpha]
            selected_h = float(grid[result.selected_index])
            error = 100*abs(selected_h-h_star)/h_star
            selected_q = objective.energy(selected_h)
            rows.append(dict(
                sample_id=index, N=n_grid, alpha=alpha, lambda_N=float(n_grid**alpha),
                selected_h=selected_h, reference_h=h_star, reference_boundary=target["boundary"],
                reference_local_minimum_count=reference["local_minimum_count"],
                relative_h_error_percent=error, within_5_percent=bool(error <= 5.),
                absolute_log_h_error=abs(float(np.log(selected_h/h_star))),
                relative_energy_excess_percent=100*(selected_q/q_star-1),
                nearest_grid_h=nearest_h,
                nearest_grid_relative_error_percent=100*abs(nearest_h-h_star)/h_star,
                reference_seconds=reference_seconds,
                divergence_construction_seconds=matrix_seconds,
                eigensolve_and_peak_seconds=solve_seconds[alpha],
                build_plus_solve_seconds=matrix_seconds+solve_seconds[alpha],
                timing_baseline_peak_changed=alpha in changed,
                frequency_intervals_used=intervals_used,
                relative_residual=result.relative_residual, relative_gap=result.relative_gap,
            ))
    return dict(sample_id=index, sample_sha256=hashlib.sha256(sample.tobytes()).hexdigest(),
                reference=reference, rows=rows, validation=checks)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_summary(records):
    rows = [row for record in records for row in record["rows"]]
    aggregates = []
    for n_grid in GRID_SIZES:
        for alpha in ALPHAS:
            subset = [r for r in rows if r["N"] == n_grid and r["alpha"] == alpha]
            errors = np.array([r["relative_h_error_percent"] for r in subset])
            elapsed = np.array([r["build_plus_solve_seconds"] for r in subset])
            hits = sum(r["within_5_percent"] for r in subset)
            aggregates.append(dict(
                N=n_grid, alpha=alpha, samples=len(subset),
                mean_relative_h_error_percent=float(errors.mean()),
                standard_error_of_mean_error_percent=float(errors.std(ddof=1)/np.sqrt(len(errors))),
                median_relative_h_error_percent=float(np.median(errors)),
                p90_relative_h_error_percent=float(np.percentile(errors, 90)),
                within_5_percent_count=hits, within_5_percent_rate=100*hits/len(subset),
                median_build_plus_solve_ms=1000*float(np.median(elapsed)),
                mean_build_plus_solve_ms=1000*float(elapsed.mean()),
                median_eigensolve_ms=1000*float(np.median(
                    [r["eigensolve_and_peak_seconds"] for r in subset])),
                mean_energy_excess_percent=float(np.mean(
                    [r["relative_energy_excess_percent"] for r in subset])),
                mean_nearest_grid_error_percent=float(np.mean(
                    [r["nearest_grid_relative_error_percent"] for r in subset])),
            ))
    rankings = []
    for n_grid in GRID_SIZES:
        subset = [r for r in aggregates if r["N"] == n_grid]
        best_mean = min(subset, key=lambda r: r["mean_relative_h_error_percent"])
        top_hits = max(r["within_5_percent_count"] for r in subset)
        # Comparisons are paired: every exponent sees the same sample.
        paired = []
        winner_errors = np.array([r["relative_h_error_percent"] for r in rows
                                  if r["N"] == n_grid and r["alpha"] == best_mean["alpha"]])
        for alpha in ALPHAS:
            other = np.array([r["relative_h_error_percent"] for r in rows
                              if r["N"] == n_grid and r["alpha"] == alpha])
            difference = other-winner_errors
            paired.append(dict(alpha=alpha, mean_error_difference_percentage_points=float(difference.mean()),
                               paired_standard_error=float(difference.std(ddof=1)/np.sqrt(len(difference)))))
        rankings.append(dict(N=n_grid, minimum_mean_error_alpha=best_mean["alpha"],
                             maximum_within_5_percent_alphas=[r["alpha"] for r in subset
                                                             if r["within_5_percent_count"] == top_hits],
                             paired_mean_error_differences_against_lowest_mean=paired))
    return rows, aggregates, rankings


def plot_results(summary, output):
    enable_automatic_pdf_saving(NAME)
    data = summary["aggregates"]
    colors = ("#245b90", "#2a9d8f", "#c79225", "#8b5ca8", "#c34c5e", "#344254")
    markers = ("o", "s", "^", "D", "v", "P")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.labelsize": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#aeb8c4", "text.color": "#253347",
                         "axes.labelcolor": "#253347", "xtick.color": "#253347",
                         "ytick.color": "#253347", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.3))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.20, top=0.72, wspace=0.30)
    fig.suptitle(r"Choosing $\alpha$ across 200 independent samples", x=0.065,
                 y=0.97, ha="left", fontsize=19)
    fig.text(0.065, 0.89,
             r"Gaussian samples, $n=200$  |  geometric grids on $[0.01,2]$  |  $\lambda_N=N^\alpha$"
             + "  |  each sample has its own continuous reference",
             color="#59677a", fontsize=10.5)
    for alpha, color, marker in zip(ALPHAS, colors, markers):
        subset = [r for r in data if r["alpha"] == alpha]
        x = [r["N"] for r in subset]
        axes[0].plot(x, [r["mean_relative_h_error_percent"] for r in subset],
                     color=color, marker=marker, ms=6, lw=1.8, label=rf"$\alpha={alpha:g}$")
        axes[1].plot(x, [r["within_5_percent_rate"] for r in subset],
                     color=color, marker=marker, ms=6, lw=1.8)
        axes[2].plot(x, [r["median_build_plus_solve_ms"] for r in subset],
                     color=color, marker=marker, ms=6, lw=1.8)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.53, 0.825), ncol=6, frameon=False, fontsize=10,
               columnspacing=2.0)
    axes[0].set(title="Mean bandwidth error", ylabel="Mean absolute relative error (%)")
    axes[1].set(title="Selections within 5% of the reference", ylabel="Samples (%)", ylim=(-2, 102))
    axes[2].set(title="Median computation time", ylabel="Matrix construction + solve (ms)")
    for axis in axes:
        axis.set(xlabel="Number of bandwidth candidates N", xlim=(85, 415))
        axis.set_xticks(GRID_SIZES)
        axis.grid(axis="y", color="#e5eaf0", lw=0.7)
        axis.set_axisbelow(True)
    fig.text(0.065, 0.10,
             "Paired comparison: all exponents use the same 200 samples. References include both interval endpoints.",
             color="#59677a", fontsize=9)
    fig.text(0.065, 0.059,
             "Every spectral peak is checked at a finer Fourier resolution. Timings exclude reference searches, validation, plotting and saving.",
             color="#59677a", fontsize=9)
    fig.text(0.065, 0.021,
             "This experiment measures agreement with the energy minimum; all results concern standard-Gaussian samples of size 200.",
             color="#59677a", fontsize=9)
    fig.savefig(output / f"{NAME}.png", dpi=180, facecolor="white")
    plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Compute only this many of the 200 checkpoints; no final summary.")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    output = KDE_ROOT / "outputs" / NAME
    output.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        plot_results(json.loads((output / "summary.json").read_text()), output)
        return
    samples = paired_gaussian_sample_bank(repetitions=REPETITIONS, maximum_n=SAMPLE_SIZE)
    configuration = dict(
        version="alpha_repeated_samples_v1", repetitions=REPETITIONS, sample_size=SAMPLE_SIZE,
        distribution="N(0,1)", sample_bank_seed=sampling_inputs()["paired_sample_bank"]["seed"],
        sample_bank_sha256=hashlib.sha256(samples.tobytes()).hexdigest(),
        alphas=list(ALPHAS), grid_sizes=list(GRID_SIZES), h_interval=[H_MIN, H_MAX],
        grid="geometric (r=0)", lambda_rule="N**alpha; c=1",
        frequency_cutoff=DEFAULT_FREQUENCY_CUTOFF, frequency_intervals=DEFAULT_FREQUENCY_INTERVALS,
        fourier_method=FOURIER_L2_METHOD_VERSION, perron_solver=PERRON_SOLVER_VERSION,
        numerical_library_threads=1,
        reference="Shared analytic Q_0 interval search, with resolution checks and endpoints",
        timing="One measured matrix construction per sample and N, reused across exponents; "
               "per-row timing is matrix-construction plus eigensolve duration, excluding validation.",
    )
    config_path = output / "configuration.json"
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise ValueError("Saved experiment configuration differs; preserve the existing run.")
    config_path.write_text(json.dumps(configuration, indent=2)+"\n")
    np.savez(output / "samples.npz", samples=samples)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    count = REPETITIONS if args.limit is None else min(REPETITIONS, args.limit)
    started = perf_counter()
    records = []
    with threadpool_limits(limits=1):
        for index in range(count):
            path = checkpoints / f"sample_{index:03d}.json"
            if path.exists():
                record = json.loads(path.read_text())
                if record["sample_sha256"] != hashlib.sha256(samples[index].tobytes()).hexdigest():
                    raise ValueError("Checkpoint sample identity mismatch.")
            else:
                record = process_sample(index, samples[index])
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(record, indent=2)+"\n")
                temporary.replace(path)
            records.append(record)
            if (index+1) % 10 == 0 or index+1 == count:
                print(f"Completed {index+1}/{REPETITIONS} samples ({perf_counter()-started:.1f} s this run)", flush=True)
    if count < REPETITIONS:
        print("Partial checkpoint run complete; rerun without --limit to finish.")
        return
    rows, aggregates, rankings = make_summary(records)
    if len(rows) != REPETITIONS*len(GRID_SIZES)*len(ALPHAS):
        raise ArithmeticError("The results table is incomplete.")
    references = [r["reference"]["global_minimum"] for r in records]
    validation = dict(
        samples=REPETITIONS, spectral_selections=len(rows),
        all_continuous_references_resolution_stable=all(
            r["reference"]["diagnostics"]["resolution_stable"] is True for r in records),
        reference_lower_endpoint_count=sum(p["boundary"] < 0 for p in references),
        reference_upper_endpoint_count=sum(p["boundary"] > 0 for p in references),
        samples_without_detected_local_minimum=sum(r["reference"]["local_minimum_count"] == 0 for r in records),
        all_final_spectral_peaks_resolution_confirmed=True,
        refined_frequency_promotions=sum(bool(c["changed_alphas"]) for r in records for c in r["validation"]),
        baseline_peak_changes=sum(len(c["changed_alphas"]) for r in records for c in r["validation"]),
        maximum_relative_eigenpair_residual=max(r["relative_residual"] for r in rows),
        maximum_absolute_divergence_refinement_change=max(
            c["maximum_absolute_divergence_change"] for r in records for c in r["validation"]),
    )
    summary = dict(configuration=configuration, aggregates=aggregates, rankings=rankings,
                   validation=validation, elapsed_seconds_this_invocation=perf_counter()-started,
                   scope="Exploratory paired comparison on 200 standard-Gaussian samples. "
                         "No independent post-selection confirmation batch is included.")
    write_csv(output / "results.csv", rows)
    write_csv(output / "aggregate_results.csv", aggregates)
    (output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    plot_results(summary, output)
    print(json.dumps(dict(output=str(output), rankings=rankings, validation=validation), indent=2))


if __name__ == "__main__":
    main()
