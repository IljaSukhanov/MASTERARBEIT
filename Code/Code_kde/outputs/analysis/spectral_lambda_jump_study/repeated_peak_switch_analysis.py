"""Study coexisting Perron peaks and dominance switches in saved samples.

Reuse Experiment 1's samples and grids. Detect numerical interior peaks with
SciPy find_peaks and a relative prominence filter. Refine every adjacent
lambda interval with a nonadjacent selected-index change, including downward
changes. Accept a dominance switch only when two separated global peaks at
the crossing have a resolved intervening valley and their order reverses.

This is a descriptive finite-grid detector, not a statistical significance
test or a certified enumeration of every possible transition in continuous
lambda. References:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html
"""

from pathlib import Path
import argparse
import csv
import json
import os
import sys
import time

CODE_DIRECTORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE_DIRECTORY))
os.environ.setdefault("MPLCONFIGDIR", str(CODE_DIRECTORY / ".matplotlib-cache"))

import numpy as np
from scipy.optimize import brentq
from scipy.signal import find_peaks
from threadpoolctl import threadpool_limits

from kde_tools.bandwidths import bandwidth_grid_linear_in_psi_beta
from kde_tools.fourier import beta2_divergence_matrix_fft
from kde_tools.perron import PERRON_SOLVER_VERSION, perron_eigenpair_from_divergence

METHOD_VERSION = "resolved_peak_crossing_v1"
RELATIVE_PROMINENCE = 1e-8
SENSITIVITY_PROMINENCES = (1e-10, 1e-8, 1e-6)
DEFAULT_SOURCE = (CODE_DIRECTORY / "outputs/data/spectral_lambda_jump_study"
                  / "2ce04db2a16e/checkpoint.npz")


def resolved_peaks(vector, relative_prominence=RELATIVE_PROMINENCE):
    """Interior maxima only; a flat top counts as one peak."""
    return find_peaks(vector, prominence=relative_prominence * vector.max())[0]


def roundoff_window(vector):
    return 256 * np.finfo(float).eps * float(vector.max())


def crossing_event(divergence, h_grid, lambda_left, lambda_right,
                   left_result, right_result, interval_index):
    before, after = left_result.selected_index, right_result.selected_index
    cut = (before + after) // 2
    low_log, high_log = np.log(lambda_left), np.log(lambda_right)
    cache = {float(low_log): left_result, float(high_log): right_result}

    def evaluate(log_lambda):
        key = float(log_lambda)
        if key not in cache:
            cache[key] = perron_eigenpair_from_divergence(divergence, np.exp(key))
        result = cache[key]
        vector = result.vector
        lo = int(np.argmax(vector[:cut + 1]))
        hi = cut + 1 + int(np.argmax(vector[cut + 1:]))
        return result, lo, hi, float(vector[lo] - vector[hi])

    g_left = evaluate(low_log)[3]
    g_right = evaluate(high_log)[3]
    row = {"interval_index": int(interval_index),
           "lambda_left": float(lambda_left), "lambda_right": float(lambda_right),
           "selected_index_before": before, "selected_index_after": after,
           "direction": "upward" if after > before else "downward",
           "status": "rejected", "reason": "", "accepted_sensitivity": {}}
    if (g_left * g_right >= 0 or abs(g_left) <= roundoff_window(left_result.vector)
            or abs(g_right) <= roundoff_window(right_result.vector)):
        row["reason"] = "endpoint_order_not_resolved"
        return row
    try:
        log_root = brentq(lambda value: evaluate(value)[3], low_log, high_log,
                          xtol=1e-10, rtol=1e-12, maxiter=80)
    except (RuntimeError, ValueError) as error:
        row["status"] = "refinement_failure"
        row["reason"] = str(error)
        return row
    result, lo, hi, height_gap = evaluate(log_root)
    vector = result.vector
    valley = lo + int(np.argmin(vector[lo:hi + 1]))
    depth = float(min(vector[lo], vector[hi]) - vector[valley])
    relative_depth = depth / float(vector.max())
    separated = hi - lo >= 2 and lo < valley < hi and 0 < lo < hi < len(vector) - 1
    step = min(1e-4, 0.05 * min(log_root - low_log, high_log - log_root))
    below, _, _, g_below = evaluate(log_root - step)
    above, _, _, g_above = evaluate(log_root + step)
    order_reverses = (
        g_below * g_above < 0
        and abs(g_below) > roundoff_window(below.vector)
        and abs(g_above) > roundoff_window(above.vector)
        and (g_below > 0) == (after > before)
    )
    for threshold in SENSITIVITY_PROMINENCES:
        row["accepted_sensitivity"][str(threshold)] = bool(
            separated and order_reverses and relative_depth > threshold
            and len(resolved_peaks(vector, threshold)) >= 2
        )
    row.update({
        "lambda_crossing": float(np.exp(log_root)),
        "lambda_just_below": float(np.exp(log_root - step)),
        "lambda_just_above": float(np.exp(log_root + step)),
        "lower_peak_index": lo, "upper_peak_index": hi,
        "lower_peak_h": float(h_grid[lo]), "upper_peak_h": float(h_grid[hi]),
        "valley_index": valley, "valley_h": float(h_grid[valley]),
        "valley_depth": depth, "relative_valley_depth": relative_depth,
        "peak_height_gap_at_root": height_gap,
        "peak_count_at_crossing": int(len(resolved_peaks(vector))),
        "height_gap_just_below": g_below, "height_gap_just_above": g_above,
        "relative_residual_at_crossing": result.relative_residual,
    })
    if row["accepted_sensitivity"][str(RELATIVE_PROMINENCE)]:
        row["status"] = "confirmed"
    elif not separated:
        row["reason"] = "one_peak_moving_across_grid_or_no_interior_valley"
    elif not order_reverses:
        row["reason"] = "local_order_reversal_not_resolved"
    else:
        row["reason"] = "valley_below_prominence_filter"
    return row


def analyze_sample(samples, h_grid, lambdas, configuration, expected_indices=None):
    divergence = beta2_divergence_matrix_fft(
        samples, h_grid, half_width=configuration["fft_half_width"],
        spatial_points=configuration["fft_spatial_points"],
    )
    results = [perron_eigenpair_from_divergence(divergence, value) for value in lambdas]
    indices = np.array([result.selected_index for result in results], dtype=int)
    if expected_indices is not None and not np.array_equal(indices, expected_indices):
        raise RuntimeError("Recomputed trajectory differs from the saved Experiment 1 path.")
    peak_counts = {str(threshold): [int(len(resolved_peaks(result.vector, threshold)))
                                   for result in results]
                   for threshold in SENSITIVITY_PROMINENCES}
    candidates = np.flatnonzero(np.abs(np.diff(indices)) >= 2)
    events = [crossing_event(divergence, h_grid, lambdas[k], lambdas[k + 1],
                             results[k], results[k + 1], int(k)) for k in candidates]
    return {"status": "success", "peak_counts": peak_counts,
            "max_relative_residual": float(max(result.relative_residual for result in results)),
            "candidate_intervals": len(candidates), "events": events}


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_analysis(source_checkpoint=DEFAULT_SOURCE, output_directory=None,
                 max_samples=None, checkpoint_every=25):
    source_checkpoint = Path(source_checkpoint)
    with np.load(source_checkpoint, allow_pickle=False) as source:
        source_configuration = json.loads(str(source["configuration_json"].item()))
        samples = source["samples"].copy()
        h_grid = source["bandwidth_grid"].copy()
        lambdas = source["lambda_values"].copy()
        expected_indices = source["selected_indices"].copy()
        if not np.all(source["finished"] & source["numerical_success"]):
            raise ValueError("Finish Experiment 1 successfully before running this analysis.")
    if source_configuration["perron_solver"] != PERRON_SOLVER_VERSION:
        raise ValueError("Source and current Perron solver versions differ.")
    count = len(samples) if max_samples is None else min(int(max_samples), len(samples))
    if count < 1:
        raise ValueError("Analyze at least one sample.")
    output_directory = (Path(output_directory) if output_directory is not None
                        else source_checkpoint.parent / "peak_switches")
    output_directory.mkdir(parents=True, exist_ok=True)
    settings = {"method_version": METHOD_VERSION, "relative_prominence": RELATIVE_PROMINENCE,
                "sensitivity_prominences": list(SENSITIVITY_PROMINENCES), "samples": count,
                "source_checkpoint": str(source_checkpoint.resolve()),
                "source_configuration": source_configuration,
                "scope": "Sampled lambda range, plus numerical refinement of nonadjacent selected-index changes."}
    checkpoint_path = output_directory / "checkpoint.json"
    records = []
    if checkpoint_path.is_file():
        cached = json.loads(checkpoint_path.read_text())
        if cached["settings"] != settings:
            raise ValueError("Peak-analysis checkpoint settings differ; choose another output directory.")
        records = cached["records"]
    print(f"Peak analysis: {len(records)}/{count} samples cached.", flush=True)
    # Positive control: reproduce the two-peak crossing in Figure 3.3.
    with threadpool_limits(limits=1):
        reference = np.random.default_rng(20260823).normal(size=200)
        reference_h_grid = bandwidth_grid_linear_in_psi_beta(grid_size=201)
        reference_divergence = beta2_divergence_matrix_fft(reference, reference_h_grid,
                                                        half_width=12.0, spatial_points=8192)
        reference_event = crossing_event(
            reference_divergence, reference_h_grid, 17.86, 17.88,
            perron_eigenpair_from_divergence(reference_divergence, 17.86),
            perron_eigenpair_from_divergence(reference_divergence, 17.88), -1,
        )
        assert reference_event["status"] == "confirmed"
        assert reference_event["peak_count_at_crossing"] == 2
        assert abs(reference_event["lambda_crossing"] - 17.8737506447) < 1e-5
        atomic_json(output_directory / "figure_3_3_positive_control.json", reference_event)
        started = time.perf_counter()
        initial_count = len(records)
        try:
            for repetition in range(initial_count, count):
                result = analyze_sample(samples[repetition], h_grid, lambdas,
                                        source_configuration, expected_indices[repetition])
                result["repetition"] = repetition
                records.append(result)
                if len(records) % checkpoint_every == 0:
                    atomic_json(checkpoint_path, {"settings": settings, "records": records})
                    elapsed = time.perf_counter() - started
                    computed = len(records) - initial_count
                    eta = elapsed / max(1, computed) * (count - len(records))
                    switches = sum(any(event["status"] == "confirmed" for event in row["events"])
                                   for row in records)
                    print(f"Checked {len(records)}/{count}; samples with confirmed switch: {switches}; "
                          f"elapsed {elapsed:.1f}s; remaining about {eta:.1f}s", flush=True)
        finally:
            atomic_json(checkpoint_path, {"settings": settings, "records": records})
    return export_summary(settings, records, output_directory)


def export_summary(settings, records, output_directory):
    rows, event_rows = [], []
    for record in records:
        accepted = [event for event in record["events"] if event["status"] == "confirmed"]
        counts = record["peak_counts"][str(RELATIVE_PROMINENCE)]
        max_including_refinement = max(
            max(counts), max((event.get("peak_count_at_crossing", 0)
                              for event in record["events"]), default=0)
        )
        first_up = next((event for event in accepted if event["direction"] == "upward"), None)
        rows.append({
            "repetition": record["repetition"], "max_peak_count_on_original_lambda_grid": max(counts),
            "has_multiple_peaks_on_original_lambda_grid": max(counts) >= 2,
            "has_exactly_two_peaks_on_original_lambda_grid": 2 in counts,
            "max_peak_count_including_refined_crossings": max_including_refinement,
            "has_multiple_peaks_including_refined_crossings": max_including_refinement >= 2,
            "confirmed_switches": len(accepted),
            "confirmed_upward_switches": sum(event["direction"] == "upward" for event in accepted),
            "confirmed_downward_switches": sum(event["direction"] == "downward" for event in accepted),
            "exactly_two_peak_switches": sum(event["peak_count_at_crossing"] == 2 for event in accepted),
            "exactly_two_peak_upward_switches": sum(event["peak_count_at_crossing"] == 2
                                                    and event["direction"] == "upward" for event in accepted),
            "first_upward_switch_lambda": first_up["lambda_crossing"] if first_up else "",
            "max_relative_residual": record["max_relative_residual"],
        })
        for event in record["events"]:
            event_rows.append({"repetition": record["repetition"], **event})
    with (output_directory / "sample_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    atomic_json(output_directory / "candidate_crossings.json", event_rows)
    accepted_rows = [event for event in event_rows if event["status"] == "confirmed"]
    if accepted_rows:
        with (output_directory / "confirmed_switches.csv").open("w", newline="") as stream:
            fields = [key for key in accepted_rows[0] if key != "accepted_sensitivity"]
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(accepted_rows)
    switch_samples = sum(row["confirmed_switches"] > 0 for row in rows)
    multimodal_samples = sum(row["has_multiple_peaks_on_original_lambda_grid"] for row in rows)
    no_switch_multimodal = sum(row["has_multiple_peaks_including_refined_crossings"]
                              and row["confirmed_switches"] == 0 for row in rows)
    summary = {
        "samples": len(records), "relative_prominence_filter": RELATIVE_PROMINENCE,
        "samples_with_multiple_peaks_on_sampled_grid": multimodal_samples,
        "samples_with_multiple_peaks_including_refined_crossings": sum(row["has_multiple_peaks_including_refined_crossings"] for row in rows),
        "samples_with_exactly_two_peaks_somewhere_on_sampled_grid": sum(row["has_exactly_two_peaks_on_original_lambda_grid"] for row in rows),
        "samples_with_confirmed_peak_switch": switch_samples,
        "samples_with_confirmed_upward_peak_switch": sum(row["confirmed_upward_switches"] > 0 for row in rows),
        "samples_with_confirmed_downward_peak_switch": sum(row["confirmed_downward_switches"] > 0 for row in rows),
        "samples_with_exactly_two_peaks_at_a_confirmed_switch": sum(row["exactly_two_peak_switches"] > 0 for row in rows),
        "samples_with_exactly_two_peaks_at_an_upward_switch": sum(row["exactly_two_peak_upward_switches"] > 0 for row in rows),
        "samples_with_more_than_two_peaks_somewhere_on_sampled_grid": sum(row["max_peak_count_on_original_lambda_grid"] > 2 for row in rows),
        "samples_with_more_than_two_peaks_including_refined_crossings": sum(row["max_peak_count_including_refined_crossings"] > 2 for row in rows),
        "samples_with_multiple_peaks_but_no_confirmed_switch": no_switch_multimodal,
        "confirmed_events": len(accepted_rows),
        "candidate_intervals": len(event_rows),
        "refinement_failures": sum(event["status"] == "refinement_failure" for event in event_rows),
        "max_relative_residual": max(row["max_relative_residual"] for row in rows),
        "sensitivity": {}, "method": settings,
    }
    for threshold in SENSITIVITY_PROMINENCES:
        summary["sensitivity"][str(threshold)] = {
            "samples_with_multiple_peaks_on_sampled_grid": sum(max(row["peak_counts"][str(threshold)]) >= 2 for row in records),
            "samples_with_confirmed_switch": sum(any(event["accepted_sensitivity"].get(str(threshold), False)
                                                     for event in row["events"]) for row in records),
        }
    atomic_json(output_directory / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "method"}, indent=2))
    return {"output_directory": str(output_directory), "summary": summary,
            "source_checkpoint": settings["source_checkpoint"]}


def plot_analysis(analysis):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator
    from kde_tools.pdf_export import enable_automatic_pdf_saving

    output_directory = Path(analysis["output_directory"])
    summary = analysis["summary"]
    records = json.loads((output_directory / "checkpoint.json").read_text())["records"]
    enable_automatic_pdf_saving("spectral_lambda_jump_study")
    image_directory = CODE_DIRECTORY / "outputs/images/spectral_lambda_jump_study" / "peak_switches"
    image_directory.mkdir(parents=True, exist_ok=True)
    first_upward = []
    for record in records:
        events = [event for event in record["events"] if event["status"] == "confirmed"
                  and event["direction"] == "upward"]
        if events:
            first_upward.append({"repetition": record["repetition"], **events[0]})
    with np.load(analysis["source_checkpoint"], allow_pickle=False) as source:
        lambdas = source["lambda_values"].copy()
        h_grid = source["bandwidth_grid"].copy()
        samples = source["samples"].copy()
        config = json.loads(str(source["configuration_json"].item()))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    no_switch_multi = summary["samples_with_multiple_peaks_but_no_confirmed_switch"]
    categories = ["No multiple peaks\nor confirmed switch", "Multiple peaks;\nno confirmed switch", "Confirmed\npeak switch"]
    values = [summary["samples"] - no_switch_multi - summary["samples_with_confirmed_peak_switch"],
              no_switch_multi, summary["samples_with_confirmed_peak_switch"]]
    bars = axes[0].bar(categories, values, color=["#aeb6c0", "#85b7d9", "#2a6fbb"])
    for bar, count in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                     f"{count} ({100 * count / summary['samples']:.1f}%)", ha="center", fontsize=10)
    axes[0].set_ylim(0, max(values) * 1.2 + 10)
    axes[0].set_ylabel("Number of samples")
    axes[0].set_title(f"Coexisting peaks and switches (M={summary['samples']})")
    axes[0].grid(axis="y", alpha=0.18)
    roots = [event["lambda_crossing"] for event in first_upward]
    axes[1].hist(roots, bins=lambdas, color="#32866a")
    axes[1].set_xscale("log")
    axes[1].set_xlim(lambdas[0], lambdas[-1])
    axes[1].set_xlabel(r"Refined crossing parameter $\lambda$")
    axes[1].set_ylabel("Number of samples")
    axes[1].set_title(f"First confirmed upward switch ({len(roots)} samples)")
    axes[1].grid(axis="y", alpha=0.18)
    fig.suptitle("Repeated sampling: two competing Perron peaks and a switch", fontsize=14)
    fig.tight_layout()
    fig.savefig(image_directory / "peak_switch_frequency_and_lambda.png", dpi=150, bbox_inches="tight")
    plt.show(); plt.close(fig)

    # Deterministic examples spanning the quartiles of confirmed crossings.
    examples = [event for event in first_upward if event["peak_count_at_crossing"] == 2]
    examples.sort(key=lambda event: event["lambda_crossing"])
    if not examples:
        print("No exactly-two-peak upward-switch examples in this run.")
        return str(image_directory)
    number_examples = min(4, len(examples))
    quantiles = np.linspace(0.125, 0.875, number_examples) if number_examples > 1 else [0.5]
    chosen = [examples[int(round(q * (len(examples) - 1)))] for q in quantiles]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))
    example_data = []
    with threadpool_limits(limits=1):
        for ax, event in zip(axes.ravel(), chosen):
            repetition = event["repetition"]
            divergence = beta2_divergence_matrix_fft(
                samples[repetition], h_grid, half_width=config["fft_half_width"],
                spatial_points=config["fft_spatial_points"],
            )
            low, high = event["lower_peak_index"], event["upper_peak_index"]
            window = np.arange(max(0, low - 3), min(len(h_grid), high + 4))
            local_deficits = []
            for label, value, color, linestyle in (
                ("Before", event["lambda_just_below"], "#2a6fbb", "-"),
                ("After", event["lambda_just_above"], "#d18425", "--"),
            ):
                result = perron_eigenpair_from_divergence(divergence, value)
                vector = result.vector
                deficit = (vector - vector.max()) / vector.max()
                local_deficits.append(deficit[window])
                ax.plot(h_grid[window], deficit[window], color=color, linestyle=linestyle,
                        linewidth=1.5, marker="o", markersize=2,
                        label=f"{label} crossing")
                ax.scatter(h_grid[result.selected_index], deficit[result.selected_index],
                           color=color, marker="*", s=80, zorder=5)
                example_data.append({"repetition": repetition, "label": label, "lambda": value,
                                     "selected_h": float(h_grid[result.selected_index]),
                                     "h_grid": h_grid.tolist(), "perron_vector": vector.tolist()})
            ax.set_xlim(h_grid[window[0]], h_grid[window[-1]])
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
            floor = float(np.min(local_deficits))
            ax.set_ylim(floor * 1.15, max(1e-6, abs(floor) * 0.06))
            ax.axhline(0, color="#888888", linewidth=0.7, linestyle=":")
            ax.set_xlabel("Bandwidth h")
            ax.set_ylabel(r"$(v_i-\max_j v_j)/\max_j v_j$")
            ax.set_title(f"Sample {repetition}; crossing lambda={event['lambda_crossing']:.5g}", fontsize=11)
            ax.grid(alpha=0.16)
    for unused_ax in axes.ravel()[number_examples:]:
        unused_ax.set_visible(False)
    fig.suptitle("Independent samples: the selected peak changes sides", fontsize=14)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, 0.01), frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    atomic_json(output_directory / "example_profiles.json", example_data)
    fig.savefig(image_directory / "independent_sample_peak_switch_examples.png", dpi=150, bbox_inches="tight")
    plt.show(); plt.close(fig)
    return str(image_directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--compute-only", action="store_true")
    args = parser.parse_args()
    output = run_analysis(max_samples=args.max_samples, output_directory=args.output_directory)
    if not args.compute_only:
        plot_analysis(output)
