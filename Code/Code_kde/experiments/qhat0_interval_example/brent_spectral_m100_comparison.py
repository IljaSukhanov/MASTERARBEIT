"""Mean runtime and paired bandwidth comparison on 100 N(0,1) samples.

Bounded Brent is applied to Q_0 in h and log h. Spectral selection uses
geometric grids with N=100,200,400 and lambda=N**1.9. Each measured call
starts from the raw sample and rebuilds its sample-specific quantities.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy
from threadpoolctl import threadpool_info, threadpool_limits

KDE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KDE_ROOT))

from kde_tools.bandwidths import bandwidth_grid_linear_in_psi_beta
from kde_tools.energy_minimization import EnergySearchConfig
from kde_tools.experiment_inputs import paired_gaussian_sample_bank, sampling_inputs
from kde_tools.fourier.similarities import (
    DEFAULT_FREQUENCY_CUTOFF, DEFAULT_FREQUENCY_INTERVALS,
    beta2_divergence_matrix_fourier, gaussian_kde_l2_gram_fourier,
)
from kde_tools.pdf_export import enable_automatic_pdf_saving
from kde_tools.perron import PERRON_SOLVER_VERSION, perron_eigenpair_from_divergence
from kde_tools.qhat import GaussianQhatObjective, minimize_qhat
from kde_tools.similarities import beta2_divergence_matrix, squared_l2_distance_from_gram
from whole_interval_brent_comparison import bounded_minimum


NAME = "brent_spectral_m100_comparison"
M, SAMPLE_SIZE, ALPHA = 100, 200, 1.9
GRID_SIZES = (100, 200, 400)
METHODS = ("brent_h", "brent_log_h", "spectral_N100", "spectral_N200", "spectral_N400")
LABELS = ("Brent in h", "Brent in log h", "Spectral N=100", "Spectral N=200", "Spectral N=400")


def spectral_selection(sample, n_grid):
    grid = bandwidth_grid_linear_in_psi_beta(0.01, 2., 2., n_grid)
    lambda_value = float(n_grid**ALPHA)
    divergence = beta2_divergence_matrix_fourier(sample, grid)
    perron = perron_eigenpair_from_divergence(divergence, lambda_value)
    index = perron.selected_index
    return dict(h=float(grid[index]), selected_index=index, grid=grid,
                divergence=divergence, perron=perron)


def execute_method(method, sample):
    if method == "brent_h":
        return bounded_minimum(sample, "h")
    if method == "brent_log_h":
        return bounded_minimum(sample, "log_h")
    return spectral_selection(sample, int(method.removeprefix("spectral_N")))


def reference_for_sample(index, sample):
    previous = KDE_ROOT / "outputs/alpha_repeated_sample_experiment/checkpoints" / f"sample_{index:03d}.json"
    if previous.exists():
        record = json.loads(previous.read_text())
        if record["sample_sha256"] != hashlib.sha256(sample.tobytes()).hexdigest():
            raise ValueError("Reference sample identity mismatch.")
        reference = record["reference"]
    else:
        reference = minimize_qhat(sample, .01, 2., r=0.,
                                  config=EnergySearchConfig(scan_backend="exact"))
    if reference["diagnostics"]["resolution_stable"] is not True:
        raise ArithmeticError("Reference did not pass its resolution check.")
    return reference


def process_sample(index, sample):
    reference = reference_for_sample(index, sample)
    target = reference["global_minimum"]
    measured = {}
    order = np.random.default_rng(np.random.SeedSequence([20261030, 100, index])).permutation(METHODS)
    for method in order:
        started = perf_counter()
        result = execute_method(method, sample)
        seconds = perf_counter()-started
        measured[str(method)] = (result, seconds)
    # All diagnostics below are outside the timed calls.
    objective = GaussianQhatObjective(sample)
    rows, checks = [], []
    brent_log_h = measured["brent_log_h"][0]["h"]
    for method in METHODS:
        result, seconds = measured[method]
        h = result["h"]
        energy = float(objective.log_energy(np.log(h))[0])
        error = 100*abs(h-target["h"])/target["h"]
        rows.append(dict(
            sample_id=index, method=method,
            N=int(method.removeprefix("spectral_N")) if method.startswith("spectral") else "",
            alpha=ALPHA if method.startswith("spectral") else "",
            selected_h=h, seconds=seconds, time_ms=1000*seconds,
            brent_log_h=brent_log_h, reference_h=target["h"], reference_boundary=target["boundary"],
            absolute_relative_difference_to_brent_log_percent=100*abs(h-brent_log_h)/brent_log_h,
            absolute_relative_reference_error_percent=error,
            within_5_percent_of_reference=bool(error <= 5.),
            matches_reference_within_log_tolerance_1e_minus6=bool(abs(np.log(h/target["h"])) <= 1e-6),
            energy_at_selected_h=energy,
            relative_energy_excess_percent=100*(energy/target["energy"]-1),
            function_evaluations=result.get("minimizer_function_evaluations", ""),
        ))
        if not method.startswith("spectral"):
            continue
        n_grid = int(method.removeprefix("spectral_N"))
        gram = gaussian_kde_l2_gram_fourier(
            sample, result["grid"], frequency_cutoff=DEFAULT_FREQUENCY_CUTOFF,
            frequency_points=2*DEFAULT_FREQUENCY_INTERVALS+1,
        )
        fine_divergence = 0.5*squared_l2_distance_from_gram(gram)
        fine = perron_eigenpair_from_divergence(fine_divergence, float(n_grid**ALPHA))
        if fine.selected_index != result["selected_index"]:
            raise ArithmeticError(f"Sample {index}, N={n_grid}: frequency refinement changed the peak.")
        check = dict(N=n_grid, refined_peak_matches=True,
                     relative_residual=result["perron"].relative_residual,
                     maximum_absolute_divergence_refinement_change=float(
                         np.max(np.abs(result["divergence"]-fine_divergence))))
        if n_grid == 100 and index in (0, 54, 98):
            exact_divergence = beta2_divergence_matrix(sample, result["grid"])
            exact = perron_eigenpair_from_divergence(exact_divergence, float(n_grid**ALPHA))
            if exact.selected_index != result["selected_index"]:
                raise ArithmeticError("Analytic-overlap selection disagrees.")
            check.update(analytic_overlap_peak_matches=True,
                         maximum_absolute_analytic_divergence_error=float(
                             np.max(np.abs(result["divergence"]-exact_divergence))))
        checks.append(check)
    return dict(sample_id=index, sample_sha256=hashlib.sha256(sample.tobytes()).hexdigest(),
                reference=reference, measured_order=list(map(str, order)), rows=rows, validation=checks)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(records):
    rows = [row for record in records for row in record["rows"]]
    summaries = []
    for method, label in zip(METHODS, LABELS):
        subset = [r for r in rows if r["method"] == method]
        times = np.array([r["time_ms"] for r in subset])
        bandwidths = np.array([r["selected_h"] for r in subset])
        summaries.append(dict(
            method=method, label=label, samples=len(subset),
            mean_time_ms=float(times.mean()),
            standard_deviation_time_ms=float(times.std(ddof=1)),
            standard_error_mean_time_ms=float(times.std(ddof=1)/np.sqrt(len(times))),
            median_time_ms=float(np.median(times)),
            mean_selected_h=float(bandwidths.mean()), sd_selected_h=float(bandwidths.std(ddof=1)),
            minimum_selected_h=float(bandwidths.min()), maximum_selected_h=float(bandwidths.max()),
            mean_absolute_relative_reference_error_percent=float(np.mean(
                [r["absolute_relative_reference_error_percent"] for r in subset])),
            mean_absolute_relative_difference_to_brent_log_percent=float(np.mean(
                [r["absolute_relative_difference_to_brent_log_percent"] for r in subset])),
            within_5_percent_of_reference_count=sum(r["within_5_percent_of_reference"] for r in subset),
            matches_reference_within_log_tolerance_1e_minus6_count=sum(
                r["matches_reference_within_log_tolerance_1e_minus6"] for r in subset),
        ))
    baseline = next(r["mean_time_ms"] for r in summaries if r["method"] == "brent_log_h")
    for row in summaries:
        row["mean_time_ratio_to_brent_log"] = row["mean_time_ms"]/baseline
    return rows, summaries


def plot_results(summary, output):
    enable_automatic_pdf_saving(NAME)
    rows, aggregates = summary["results"], summary["aggregates"]
    colors = ("#c79225", "#8b5ca8", "#245b90", "#2a9d8f", "#c34c5e")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.labelsize": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#aeb8c4", "text.color": "#253347",
                         "axes.labelcolor": "#253347", "xtick.color": "#253347",
                         "ytick.color": "#253347", "pdf.fonttype": 42})
    fig, (timing, selections) = plt.subplots(1, 2, figsize=(13, 6.2))
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.22, top=0.76, wspace=0.27)
    fig.suptitle("Bounded Brent versus spectral selection", x=0.075, y=0.96,
                 ha="left", fontsize=20)
    fig.text(0.075, 0.86,
             r"100 paired Gaussian samples, $n=200$  |  $h\in[0.01,2]$  |  geometric spectral grids, $\lambda_N=N^{1.9}$",
             fontsize=10.5, color="#59677a")
    means = [r["mean_time_ms"] for r in aggregates]
    errors = [r["standard_error_mean_time_ms"] for r in aggregates]
    bars = timing.bar(np.arange(5), means, color=colors, width=0.65,
                      yerr=errors, capsize=3, error_kw={"elinewidth": 1, "ecolor": "#253347"})
    timing.set_xticks(np.arange(5), ("Brent\nh", "Brent\nlog h", "Spectral\nN=100",
                                      "Spectral\nN=200", "Spectral\nN=400"))
    timing.set(title="Mean computation time", ylabel="Mean wall time per sample (ms)",
               ylim=(0, 1.22*max(means)))
    timing.bar_label(bars, labels=[f"{v:.2f}" for v in means], padding=5, fontsize=10)
    by_method = {method: sorted([r for r in rows if r["method"] == method],
                                key=lambda r: r["sample_id"]) for method in METHODS}
    x = np.array([r["selected_h"] for r in by_method["brent_log_h"]])
    all_h = np.array([r["selected_h"] for r in rows])
    lo, hi = max(0., float(all_h.min())-.025), float(all_h.max())+.025
    selections.plot([lo, hi], [lo, hi], ls="--", lw=1.1, color="#59677a", label="Same bandwidth")
    selections.scatter(x, [r["selected_h"] for r in by_method["brent_h"]],
                       marker="x", color=colors[0], s=24, alpha=.6, label="Brent in h")
    for method, color, marker in zip(METHODS[2:], colors[2:], ("o", "s", "^")):
        selections.scatter(x, [r["selected_h"] for r in by_method[method]],
                           color=color, marker=marker, s=23, alpha=.65,
                           label=method.replace("spectral_N", "Spectral N="))
    selections.set(title="Bandwidths selected on the same samples",
                   xlabel="Bandwidth selected by Brent in log h",
                   ylabel="Bandwidth selected by comparison method", xlim=(lo, hi), ylim=(lo, hi))
    selections.legend(frameon=False, fontsize=8.5, loc="lower right")
    for axis in (timing, selections):
        axis.grid(axis="y", color="#e5eaf0", lw=.7)
        axis.set_axisbelow(True)
    fig.text(0.075, 0.11,
             "Arithmetic means over 100 timed calls per method. Error bars: one standard error of the mean time. One numerical thread.",
             color="#59677a", fontsize=9)
    fig.text(0.075, 0.068,
             "Each call starts from raw observations; spectral times include fresh matrix construction and the eigensolve.",
             color="#59677a", fontsize=9)
    fig.text(0.075, 0.026,
             "Imports, sample generation, reference searches, validation, plotting and saving are outside the timed calls.",
             color="#59677a", fontsize=9)
    fig.savefig(output / f"{NAME}.png", dpi=180, facecolor="white")
    plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    output = KDE_ROOT / "outputs" / NAME
    output.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        plot_results(json.loads((output / "summary.json").read_text()), output)
        return
    samples = paired_gaussian_sample_bank(repetitions=M, maximum_n=SAMPLE_SIZE)
    configuration = dict(
        version="brent_spectral_mean_m100_v1", samples=M, sample_size=SAMPLE_SIZE,
        distribution="N(0,1)", seed=sampling_inputs()["paired_sample_bank"]["seed"],
        sample_source="First 100 rows of the existing paired Gaussian sample bank, first 200 observations per row",
        sample_sha256=hashlib.sha256(samples.tobytes()).hexdigest(), h_interval=[.01, 2.],
        N_values=list(GRID_SIZES), alpha=ALPHA, lambda_rule="N**1.9", geometric_grid_r=0.,
        methods=list(METHODS), brent_xatol=2e-12, brent_maxiter=1000,
        brent_objective="Shared analytic Gaussian Q_0 objective; pair distances prepared inside each call; endpoints compared",
        frequency_cutoff=DEFAULT_FREQUENCY_CUTOFF, frequency_intervals=DEFAULT_FREQUENCY_INTERVALS,
        spectral_solver=PERRON_SOLVER_VERSION, numerical_library_threads=1,
        timing="One fresh call per method and sample, 100 timed calls per method; mean across all 100. "
               "Method order randomized separately for each sample. All methods initialized once before timing. "
               "No sample-specific matrix or objective cache is shared between timed calls.",
        excluded_from_timing=["imports", "sample generation/loading", "reference searches",
                              "validation", "plotting", "saving"],
        python_version=platform.python_version(), numpy_version=np.__version__, scipy_version=scipy.__version__,
        machine=platform.machine(), platform=platform.platform(),
    )
    try:
        configuration["cpu"] = subprocess.check_output(
            ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        configuration["cpu"] = platform.processor()
    config_path = output / "configuration.json"
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise ValueError("Existing run configuration differs.")
    config_path.write_text(json.dumps(configuration, indent=2)+"\n")
    np.savez(output / "samples.npz", samples=samples)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    records = []
    started = perf_counter()
    with threadpool_limits(limits=1):
        configuration["active_threadpools"] = [
            {key:info.get(key) for key in ("internal_api", "prefix", "num_threads", "version", "architecture")}
            for info in threadpool_info()]
        for method in METHODS:
            _ = execute_method(method, samples[0])
        for index, sample in enumerate(samples):
            path = checkpoints / f"sample_{index:03d}.json"
            if path.exists():
                record = json.loads(path.read_text())
                if record["sample_sha256"] != hashlib.sha256(sample.tobytes()).hexdigest():
                    raise ValueError("Checkpoint sample mismatch.")
            else:
                record = process_sample(index, sample)
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(record, indent=2)+"\n")
                temporary.replace(path)
            records.append(record)
            if (index+1) % 10 == 0:
                print(f"Completed {index+1}/{M} paired samples ({perf_counter()-started:.1f} s)", flush=True)
    rows, aggregates = aggregate(records)
    if len(rows) != M*len(METHODS) or len({(r["sample_id"], r["method"]) for r in rows}) != M*len(METHODS):
        raise ArithmeticError("Incomplete or duplicate results.")
    validation = dict(
        paired_samples=M, timed_calls_per_method=M,
        all_continuous_references_resolution_stable=all(
            r["reference"]["diagnostics"]["resolution_stable"] is True for r in records),
        spectral_selections_checked=sum(len(r["validation"]) for r in records),
        all_spectral_selections_match_frequency_refinement=True,
        analytic_overlap_checks=sum("analytic_overlap_peak_matches" in c for r in records for c in r["validation"]),
        maximum_spectral_relative_residual=max(c["relative_residual"] for r in records for c in r["validation"]),
    )
    summary = dict(configuration=configuration, aggregates=aggregates, validation=validation,
                   elapsed_seconds_this_invocation=perf_counter()-started, results=rows)
    write_csv(output / "results.csv", rows)
    write_csv(output / "aggregate_results.csv", aggregates)
    (output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    plot_results(summary, output)
    print(json.dumps(dict(aggregates=aggregates, validation=validation, output=str(output)), indent=2))


if __name__ == "__main__":
    main()
