"""Fresh Figure 3.3 samples: coarse lambda scans, fine grids, and peak roots.

This numerical detector preserves the original bandwidth grid. It refines
every coarse interval with a selected-index displacement of at least two.
Only separated global peaks with a resolved valley and a height-order
reversal are reported as switches. A coarse scan can miss a short switch
and switch-back inside one interval. No continuum-h claim is made.
"""

from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
import sys
import time

CODE_DIRECTORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE_DIRECTORY))
sys.path.insert(0, str(CODE_DIRECTORY / "outputs/analysis/spectral_lambda_jump_study"))
os.environ.setdefault("MPLCONFIGDIR", str(CODE_DIRECTORY / ".matplotlib-cache"))

import numpy as np
import pandas as pd
import scipy
from threadpoolctl import threadpool_limits

from kde_tools.fourier import beta2_divergence_matrix_fft
from kde_tools.perron import (PERRON_SOLVER_VERSION, PerronConvergenceError,
                              PerronDegeneracyError, perron_eigenpair_from_divergence)
from repeated_peak_switch_analysis import crossing_event, resolved_peaks, roundoff_window

NOTEBOOK_NAME = "spectral_lambda_switch_distribution"
METHOD_VERSION = "fresh_samples_coarse_fine_peak_roots_v1"


def settings_for(samples=100, sample_size=200, seed=2026101306, fine_points=201):
    if samples < 1 or sample_size < 2 or seed < 0 or fine_points < 3:
        raise ValueError("Require samples>=1, sample_size>=2, seed>=0, fine_points>=3.")
    source_paths = [CODE_DIRECTORY / "kde_tools/perron.py",
                    CODE_DIRECTORY / "outputs/analysis/spectral_lambda_jump_study/repeated_peak_switch_analysis.py",
                    *sorted((CODE_DIRECTORY / "kde_tools/fourier").glob("*.py"))]
    return {"method_version": METHOD_VERSION, "samples": int(samples),
            "sample_size": int(sample_size), "seed": int(seed),
            "sampling": "default_rng(SeedSequence([seed, repetition])).normal(size=n)",
            "bandwidth_grid": np.geomspace(0.01, 2.0, 201).tolist(),
            "coarse_lambdas": np.geomspace(0.01, 10000.0, 101).tolist(),
            "fine_points_per_candidate_interval": int(fine_points),
            "fine_grid": "linspace(coarse_left, coarse_right, fine_points)",
            "relative_peak_prominence": 1e-8, "minimum_candidate_index_displacement": 2,
            "numerical_bracket_target_relative_width": 1e-6,
            "one_event_per_sample": "largest absolute log bandwidth ratio; earliest lambda breaks ties",
            "fft_half_width": 12.0, "fft_spatial_points": 8192,
            "perron_solver": PERRON_SOLVER_VERSION,
            "numpy_version": np.__version__, "scipy_version": scipy.__version__,
            "shared_source_sha256": {str(p.relative_to(CODE_DIRECTORY)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in source_paths}}


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":")) + "\n")
    temporary.replace(path)


def narrow_numerical_bracket(event, evaluate, target):
    """Bisection until the lambda bracket is narrow or signs hit roundoff.

    This is a bracket for the computed peak-height difference, not an
    interval-arithmetic enclosure of the exact mathematical crossing.
    """
    cut = (event["lower_peak_index"] + event["upper_peak_index"]) // 2

    def gap(log_lambda):
        result = evaluate(np.exp(log_lambda))
        v = result.vector
        return float(v[:cut + 1].max() - v[cut + 1:].max()), roundoff_window(v)

    left, right = np.log(event["lambda_left"]), np.log(event["lambda_right"])
    g_left, _ = gap(left)
    g_right, _ = gap(right)
    assert g_left * g_right < 0
    limited = False
    for _ in range(40):
        if np.expm1(right - left) <= target:
            break
        midpoint = (left + right) / 2
        g_mid, tolerance = gap(midpoint)
        if abs(g_mid) <= tolerance:
            limited = True
            break
        if (g_mid > 0) == (g_left > 0):
            left, g_left = midpoint, g_mid
        else:
            right, g_right = midpoint, g_mid
    root = event["lambda_crossing"]
    lo, hi = float(np.exp(left)), float(np.exp(right))
    assert lo * (1 - 1e-9) <= root <= hi * (1 + 1e-9)
    return {"refined_lambda_left": lo, "refined_lambda_right": hi,
            "refined_relative_bracket_width": (hi - lo) / root,
            "bracket_precision_limited_by_roundoff": limited,
            "refined_height_gap_left": g_left, "refined_height_gap_right": g_right}


def analyze_sample(sample, settings, repetition):
    h_grid = np.array(settings["bandwidth_grid"])
    coarse_lambdas = np.array(settings["coarse_lambdas"])
    divergence = beta2_divergence_matrix_fft(sample, h_grid,
                                           half_width=settings["fft_half_width"],
                                           spatial_points=settings["fft_spatial_points"])
    cache = {}

    def evaluate(lambda_value):
        key = float(lambda_value)
        if key not in cache:
            cache[key] = perron_eigenpair_from_divergence(divergence, key)
        return cache[key]

    coarse_results = [evaluate(value) for value in coarse_lambdas]
    indices = np.array([r.selected_index for r in coarse_results], dtype=int)
    candidates = np.flatnonzero(np.abs(np.diff(indices)) >= 2)
    fine_scans, events, rejected = [], [], []
    for coarse_index in candidates:
        left, right = coarse_lambdas[coarse_index:coarse_index + 2]
        fine_lambdas = np.linspace(left, right, settings["fine_points_per_candidate_interval"])
        fine_results = [evaluate(value) for value in fine_lambdas]
        fine_indices = np.array([r.selected_index for r in fine_results], dtype=int)
        fine_candidates = np.flatnonzero(np.abs(np.diff(fine_indices)) >= 2)
        fine_scans.append({"coarse_interval_index": int(coarse_index),
                           "lambda_values": fine_lambdas.tolist(), "selected_indices": fine_indices.tolist(),
                           "peak_counts": [len(resolved_peaks(r.vector)) for r in fine_results]})
        for fine_index in fine_candidates:
            event = crossing_event(divergence, h_grid, fine_lambdas[fine_index], fine_lambdas[fine_index + 1],
                                   fine_results[fine_index], fine_results[fine_index + 1], int(fine_index))
            # Near a fine-grid node, widen by one neighbor to resolve signs.
            if event["status"] != "confirmed" and event["reason"] in (
                    "endpoint_order_not_resolved", "local_order_reversal_not_resolved"):
                a, b = max(0, fine_index - 1), min(len(fine_lambdas) - 1, fine_index + 2)
                event = crossing_event(divergence, h_grid, fine_lambdas[a], fine_lambdas[b],
                                       fine_results[a], fine_results[b], int(fine_index))
                event["fine_validation_interval_expanded"] = True
            event.update({"repetition": repetition, "coarse_interval_index": int(coarse_index),
                          "coarse_lambda_left": float(left), "coarse_lambda_right": float(right),
                          "coarse_h_before": float(h_grid[indices[coarse_index]]),
                          "coarse_h_after": float(h_grid[indices[coarse_index + 1]]),
                          "fine_grid_lambda_left": float(fine_lambdas[fine_index]),
                          "fine_grid_lambda_right": float(fine_lambdas[fine_index + 1])})
            if event["status"] != "confirmed":
                rejected.append(event)
                continue
            if any(abs(np.log(event["lambda_crossing"] / saved["lambda_crossing"])) < 1e-7
                   and event["lower_peak_index"] == saved["lower_peak_index"]
                   and event["upper_peak_index"] == saved["upper_peak_index"] for saved in events):
                continue
            before = evaluate(event["lambda_just_below"]).selected_index
            after = evaluate(event["lambda_just_above"]).selected_index
            h_before, h_after = float(h_grid[before]), float(h_grid[after])
            event.update({"h_before": h_before, "h_after": h_after,
                          "signed_delta_h": h_after - h_before, "absolute_delta_h": abs(h_after - h_before),
                          "h_after_over_before": h_after / h_before,
                          "severity_percent": 100 * (max(h_before, h_after) / min(h_before, h_after) - 1),
                          "absolute_log_h_ratio": abs(np.log(h_after / h_before)),
                          "jump_grid_steps": abs(after - before)})
            event.update(narrow_numerical_bracket(
                event, evaluate, settings["numerical_bracket_target_relative_width"]))
            events.append(event)
    events.sort(key=lambda row: row["lambda_crossing"])
    return {"repetition": repetition, "status": "success", "coarse_selected_indices": indices.tolist(),
            "coarse_peak_counts": [len(resolved_peaks(r.vector)) for r in coarse_results],
            "candidate_coarse_intervals": len(candidates), "fine_scans": fine_scans,
            "events": events, "rejected_fine_candidates": rejected,
            "eigenpair_evaluations": len(cache),
            "maximum_relative_residual": float(max(r.relative_residual for r in cache.values()))}


def analyze_task(task):
    repetition, settings = task
    sample = np.random.default_rng(np.random.SeedSequence([settings["seed"], repetition])).normal(
        size=settings["sample_size"])
    with threadpool_limits(limits=1):
        try:
            return analyze_sample(sample, settings, repetition)
        except (PerronConvergenceError, PerronDegeneracyError) as error:
            return {"repetition": repetition, "status": "numerical_failure", "reason": str(error), "events": []}


def representative_event(events):
    return min(events, key=lambda row: (-row["absolute_log_h_ratio"], row["lambda_crossing"])) if events else None


def describe(values):
    values = np.asarray(values, dtype=float)
    return (dict(zip(("minimum", "q25", "median", "q75", "maximum"),
                     np.quantile(values, [0, 0.25, 0.5, 0.75, 1]).tolist())) if len(values) else None)


def export_results(settings, records, directory):
    sample_rows, event_rows = [], []
    for record in sorted(records, key=lambda row: row["repetition"]):
        events = record["events"]
        chosen = representative_event(events)
        row = {"repetition": record["repetition"],
               "status": "numerical_failure" if record["status"] != "success" else
                         ("confirmed_switch" if chosen else "no_detected_switch"),
               "confirmed_events": len(events), "candidate_coarse_intervals": record.get("candidate_coarse_intervals"),
               "maximum_relative_residual": record.get("maximum_relative_residual")}
        fields = ("lambda_crossing", "coarse_lambda_left", "coarse_lambda_right", "fine_grid_lambda_left",
                  "fine_grid_lambda_right", "refined_lambda_left", "refined_lambda_right", "direction",
                  "h_before", "h_after", "signed_delta_h", "absolute_delta_h", "h_after_over_before",
                  "severity_percent", "jump_grid_steps", "relative_valley_depth",
                  "refined_relative_bracket_width", "bracket_precision_limited_by_roundoff")
        row.update({key: chosen[key] if chosen else None for key in fields})
        sample_rows.append(row)
        event_rows.extend({key: value for key, value in event.items() if key != "accepted_sensitivity"}
                          for event in events)
    pd.DataFrame(sample_rows).to_csv(directory / "sample_results.csv", index=False)
    pd.DataFrame(event_rows, columns=list(event_rows[0]) if event_rows else ["repetition", "lambda_crossing"]).to_csv(
        directory / "all_confirmed_switches.csv", index=False)
    chosen = [representative_event(row["events"]) for row in records if row["events"]]
    all_events = [event for row in records for event in row["events"]]
    summary = {"settings": settings, "output_directory": str(directory), "completed_samples": len(records),
               "successful_samples": sum(row["status"] == "success" for row in records),
               "samples_with_confirmed_switch": len(chosen),
               "samples_without_detected_switch": sum(row["status"] == "success" and not row["events"] for row in records),
               "samples_with_multiple_switches": sum(len(row["events"]) > 1 for row in records),
               "confirmed_switch_events": len(all_events),
               "upward_events": sum(row["direction"] == "upward" for row in all_events),
               "downward_events": sum(row["direction"] == "downward" for row in all_events),
               "largest_switch_per_sample_lambda": describe([row["lambda_crossing"] for row in chosen]),
               "largest_switch_per_sample_severity_percent": describe([row["severity_percent"] for row in chosen]),
               "largest_switch_per_sample_absolute_delta_h": describe([row["absolute_delta_h"] for row in chosen]),
               "maximum_refined_relative_bracket_width": max((r["refined_relative_bracket_width"] for r in all_events), default=None),
               "roundoff_limited_brackets": sum(r["bracket_precision_limited_by_roundoff"] for r in all_events),
               "refinement_failures": sum(event["status"] == "refinement_failure" for row in records
                                           for event in row.get("rejected_fine_candidates", [])),
               "maximum_relative_residual": max((r.get("maximum_relative_residual", 0) for r in records), default=0),
               "scope": "Confirmed separated-peak switches conditional on the original 201-node h grid and lambda range. Short switch-and-switch-back events inside a noncandidate coarse interval may be missed. Numerical brackets are not interval-certified bounds."}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def run_analysis(samples=100, sample_size=200, seed=2026101306, fine_points=201, workers=3):
    settings = settings_for(samples, sample_size, seed, fine_points)
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
    directory = CODE_DIRECTORY / "outputs/data" / NOTEBOOK_NAME / fingerprint
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / "checkpoint.json"
    records = []
    if checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text())
        if checkpoint["settings"] != settings:
            raise ValueError("Checkpoint settings do not match this experiment.")
        records = checkpoint["records"]
    done = {row["repetition"] for row in records}
    tasks = [(repetition, settings) for repetition in range(samples) if repetition not in done]
    raw_samples = np.array([np.random.default_rng(np.random.SeedSequence([seed, r])).normal(size=sample_size)
                            for r in range(samples)])
    np.savez_compressed(directory / "samples_and_grids.npz", samples=raw_samples,
                        bandwidth_grid=settings["bandwidth_grid"], coarse_lambdas=settings["coarse_lambdas"])
    print(f"Coarse-to-fine experiment: {len(records)}/{samples} samples cached; output {directory}", flush=True)
    started = time.perf_counter()
    try:
        if tasks:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                for future in as_completed([pool.submit(analyze_task, task) for task in tasks]):
                    records.append(future.result())
                    if len(records) % 5 == 0:
                        atomic_json(checkpoint_path, {"settings": settings, "records": records})
                        print(f"Completed {len(records)}/{samples}; switch samples "
                              f"{sum(bool(row['events']) for row in records)}; elapsed {time.perf_counter()-started:.1f}s",
                              flush=True)
    finally:
        atomic_json(checkpoint_path, {"settings": settings, "records": records})
    # Fixed positive control is separate from the 100 fresh random samples.
    control_path = directory / "figure_3_3_control.json"
    if not control_path.is_file():
        with threadpool_limits(limits=1):
            control = analyze_sample(np.random.default_rng(20260823).normal(size=200),
                                     settings_for(fine_points=fine_points), -1)
        largest = representative_event(control["events"])
        assert largest is not None and abs(largest["lambda_crossing"] - 17.8737506447) < 1e-5
        atomic_json(control_path, control)
    summary = export_results(settings, records, directory)
    print(json.dumps({key: value for key, value in summary.items() if key not in ("settings", "scope")}, indent=2), flush=True)
    return summary


def plot_distributions(summary):
    import matplotlib.pyplot as plt
    from kde_tools.pdf_export import enable_automatic_pdf_saving
    enable_automatic_pdf_saving(NOTEBOOK_NAME)
    directory = Path(summary["output_directory"])
    image_directory = CODE_DIRECTORY / "outputs/images" / NOTEBOOK_NAME / directory.name
    image_directory.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(directory / "sample_results.csv")
    selected = rows.loc[rows["status"] == "confirmed_switch"]
    coarse = np.array(summary["settings"]["coarse_lambdas"])
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.8), constrained_layout=True)
    fig.suptitle(f"{len(rows)} independent samples: switch locations and severity\n"
                 f"{len(selected)} samples with a confirmed switch; {summary['samples_without_detected_switch']} without one detected",
                 fontsize=14)
    if len(selected):
        counts, _ = np.histogram(selected["lambda_crossing"], bins=coarse)
        axes[0, 0].bar(coarse[:-1], counts, width=np.diff(coarse), align="edge", color="#6a8fac", edgecolor="white")
        values = selected["lambda_crossing"].to_numpy()
        log_edges = np.linspace(np.log10(values.min()) - 0.10, np.log10(values.max()) + 0.10, 11)
        axes[0, 1].hist(values, bins=10 ** log_edges, color="#2563a6", edgecolor="white")
        for axis in axes[0]:
            axis.set_xscale("log")
            axis.set_xlim(10 ** log_edges[0], 10 ** log_edges[-1])
            axis.set_ylabel("Sample count")
        axes[1, 0].hist(selected["severity_percent"], bins=10, color="#d96927", edgecolor="white")
        axes[1, 0].set_ylabel("Sample count")
        for direction, color, marker in (("upward", "#2563a6", "o"), ("downward", "#d96927", "v")):
            subset = selected.loc[selected["direction"] == direction]
            if len(subset):
                axes[1, 1].scatter(subset["lambda_crossing"], subset["severity_percent"],
                                   color=color, marker=marker, s=35, alpha=0.85, label=direction)
        axes[1, 1].set_xscale("log")
        axes[1, 1].legend(frameon=False)
    else:
        for axis in axes.flat:
            axis.text(0.5, 0.5, "No confirmed switches", transform=axis.transAxes, ha="center")
    axes[0, 0].set(title="Original coarse lambda intervals", xlabel=r"Coarse $\lambda$ interval")
    axes[0, 1].set(title="Refined crossing locations", xlabel=r"Refined crossing $\lambda_c$")
    axes[1, 0].set(title="Size of the switch", xlabel="Bandwidth separation (%)")
    axes[1, 1].set(title="Location versus severity", xlabel=r"Refined crossing $\lambda_c$",
                   ylabel="Bandwidth separation (%)")
    for axis in axes.flat:
        axis.grid(alpha=0.16)
        axis.spines[["top", "right"]].set_visible(False)
    path = image_directory / "switch_location_and_severity_distributions.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return path


def plot_refinement_examples(summary):
    import matplotlib.pyplot as plt
    from kde_tools.pdf_export import enable_automatic_pdf_saving
    enable_automatic_pdf_saving(NOTEBOOK_NAME)
    directory = Path(summary["output_directory"])
    records = json.loads((directory / "checkpoint.json").read_text())["records"]
    control = json.loads((directory / "figure_3_3_control.json").read_text())
    chosen_records = sorted([r for r in records if r["events"]],
                            key=lambda r: representative_event(r["events"])["lambda_crossing"])
    examples = [("Figure 3.3 reference", control)]
    if chosen_records:
        record = chosen_records[len(chosen_records) // 2]
        examples.append((f"Independent sample {record['repetition']}", record))
    with np.load(directory / "samples_and_grids.npz", allow_pickle=False) as saved:
        samples, h_grid, coarse = saved["samples"], saved["bandwidth_grid"], saved["coarse_lambdas"]
    fig, axes = plt.subplots(len(examples), 3, figsize=(13.5, 3.3 * len(examples)),
                             squeeze=False, constrained_layout=True)
    fig.suptitle("Locate a switch: coarse scan, fine grid, equal peak heights", fontsize=15)
    with threadpool_limits(limits=1):
        for row_index, (label, record) in enumerate(examples):
            event = representative_event(record["events"])
            fine = next(scan for scan in record["fine_scans"]
                        if scan["coarse_interval_index"] == event["coarse_interval_index"])
            lambdas = np.array(fine["lambda_values"])
            root = event["lambda_crossing"]
            left, right = event["coarse_lambda_left"], event["coarse_lambda_right"]
            axes[row_index, 0].plot(coarse, h_grid[record["coarse_selected_indices"]], "o-", color="#6a8fac", ms=4)
            axes[row_index, 0].axvspan(left, right, color="#d96927", alpha=0.13)
            axes[row_index, 0].set(xlim=(left / 1.4, right * 1.4), title=f"{label}: coarse interval", ylabel="Selected bandwidth h")
            axes[row_index, 1].plot(lambdas, h_grid[fine["selected_indices"]], ".-", color="#2563a6", ms=3)
            axes[row_index, 1].set(title=f"201 fine lambda values; separation {event['severity_percent']:.1f}%")
            sample = np.random.default_rng(20260823).normal(size=200) if record["repetition"] == -1 else samples[record["repetition"]]
            divergence = beta2_divergence_matrix_fft(sample, h_grid, half_width=12, spatial_points=8192)
            cut = (event["lower_peak_index"] + event["upper_peak_index"]) // 2
            gaps = []
            for value in lambdas:
                vector = perron_eigenpair_from_divergence(divergence, value).vector
                gaps.append(1e6 * (vector[:cut + 1].max() - vector[cut + 1:].max()) / vector.max())
            axes[row_index, 2].plot(lambdas, gaps, color="#3d536c")
            axes[row_index, 2].axhline(0, color="black", lw=0.8)
            axes[row_index, 2].set(title=rf"Peak-height crossing: $\lambda_c={root:.6g}$", ylabel="Relative peak-height difference (ppm)")
            for axis in axes[row_index]:
                axis.axvline(root, color="#d96927", ls="--", lw=1)
                axis.set_xlabel(r"$\lambda$")
                axis.grid(alpha=0.16)
                axis.spines[["top", "right"]].set_visible(False)
    image_directory = CODE_DIRECTORY / "outputs/images" / NOTEBOOK_NAME / directory.name
    image_directory.mkdir(parents=True, exist_ok=True)
    path = image_directory / "coarse_to_fine_switch_examples.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return path


def validate_independent_eigensolver(summary):
    """Check reference and representative crossings on the original S."""
    from scipy.linalg import eigh
    directory = Path(summary["output_directory"])
    records = json.loads((directory / "checkpoint.json").read_text())["records"]
    control = json.loads((directory / "figure_3_3_control.json").read_text())
    selected = sorted([record for record in records if record["events"]],
                      key=lambda row: representative_event(row["events"])["lambda_crossing"])
    positions = np.unique(np.linspace(0, len(selected) - 1, min(5, len(selected))).astype(int)) if selected else []
    examples = [control] + [selected[index] for index in positions]
    with np.load(directory / "samples_and_grids.npz", allow_pickle=False) as saved:
        samples, h_grid = saved["samples"], saved["bandwidth_grid"]
    checks = []
    with threadpool_limits(limits=1):
        for record in examples:
            event = representative_event(record["events"])
            sample = np.random.default_rng(20260823).normal(size=200) if record["repetition"] == -1 else samples[record["repetition"]]
            divergence = beta2_divergence_matrix_fft(sample, h_grid, half_width=12, spatial_points=8192)
            for side, value, expected_h in (("below", event["lambda_just_below"], event["h_before"]),
                                            ("above", event["lambda_just_above"], event["h_after"])):
                shared = perron_eigenpair_from_divergence(divergence, value)
                eigenvalues, eigenvectors = eigh(np.exp(-value * divergence),
                                                 subset_by_index=[len(h_grid) - 1, len(h_grid) - 1])
                dense = eigenvectors[:, 0]
                if dense.sum() < 0:
                    dense = -dense
                dense /= dense.sum()
                discrepancy = float(np.abs(dense - shared.vector).max())
                assert discrepancy < 1e-12
                assert np.isclose(h_grid[np.argmax(dense)], expected_h, rtol=1e-12)
                checks.append({"repetition": record["repetition"], "side": side, "lambda": value,
                               "dense_selected_h": float(h_grid[np.argmax(dense)]),
                               "expected_selected_h": expected_h,
                               "maximum_vector_component_discrepancy": discrepancy})
    report = {"checks": checks,
              "maximum_vector_component_discrepancy": max(row["maximum_vector_component_discrepancy"] for row in checks),
              "scope": "Independent dense eigensolver below/above the reference crossing and five representative random-sample crossings."}
    atomic_json(directory / "validation.json", report)
    return report


def plot_ten_paths(summary, *, include_closest_pair=False):
    """Ten Figure 3.3-type paths, selected to show the range of behaviors.

    Choose seven switch samples at spaced ranks of their largest-switch
    lambda, including the multiple-switch example when present. Choose
    three no-detection samples at spaced ranks of their final bandwidth.
    These are illustrative selections, not a frequency estimate.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, NullFormatter
    from kde_tools.pdf_export import enable_automatic_pdf_saving

    directory = Path(summary["output_directory"])
    checkpoint = json.loads((directory / "checkpoint.json").read_text())
    records = sorted(checkpoint["records"], key=lambda r: r["repetition"])
    settings = checkpoint["settings"]
    h_grid, coarse = np.array(settings["bandwidth_grid"]), np.array(settings["coarse_lambdas"])
    switches = sorted([r for r in records if r["status"] == "success" and r["events"]],
                      key=lambda r: representative_event(r["events"])["lambda_crossing"])
    no_switch = sorted([r for r in records if r["status"] == "success" and not r["events"]],
                       key=lambda r: (h_grid[r["coarse_selected_indices"][-1]], r["repetition"]))
    if len(switches) < 7 or len(no_switch) < 3:
        raise ValueError("This illustrative selection requires seven switch and three no-detection samples.")
    chosen_switches = [switches[i] for i in np.rint(np.linspace(0, len(switches) - 1, 7)).astype(int)]
    multiple = [r for r in switches if len(r["events"]) > 1]
    if multiple and multiple[0]["repetition"] not in [r["repetition"] for r in chosen_switches]:
        target = representative_event(multiple[0]["events"])["lambda_crossing"]
        # Keep both endpoint examples; replace the closest interior lambda.
        replace = min(range(1, 6), key=lambda i: abs(np.log(
            representative_event(chosen_switches[i]["events"])["lambda_crossing"] / target)))
        chosen_switches[replace] = multiple[0]
    chosen_switches.sort(key=lambda r: representative_event(r["events"])["lambda_crossing"])
    chosen_no = [no_switch[i] for i in np.rint(np.linspace(0, len(no_switch) - 1, 3)).astype(int)]
    # Interleave no-detection cases on the first page; second page includes
    # the double-switch and late-switch examples.
    chosen = chosen_switches[:3] + chosen_no + chosen_switches[3:]
    assert len({r["repetition"] for r in chosen}) == 10
    selected_metadata = {
        "selection": "Seven spaced ranks of largest-switch lambda, including the multiple-switch sample; three spaced ranks of final h among no-detection samples. Illustrative, not random or proportional.",
        "samples": [{"repetition": r["repetition"], "confirmed_events": len(r["events"]),
                     "crossing_lambdas": [e["lambda_crossing"] for e in r["events"]],
                     "severity_percent": [e["severity_percent"] for e in r["events"]]}
                    for r in chosen],
    }
    closest_pairs = None
    if include_closest_pair:
        base_metadata_path = directory / "ten_example_paths.json"
        if not base_metadata_path.is_file():
            atomic_json(base_metadata_path, selected_metadata)
        pair_path = directory / "closest_pair_limit/sample_results.csv"
        if not pair_path.is_file():
            from check_closest_pair_limit import run_checks
            run_checks(directory)
        closest_pairs = pd.read_csv(pair_path).set_index("repetition")
        for row in selected_metadata["samples"]:
            pair = closest_pairs.loc[row["repetition"]]
            row.update({"closest_pair_lower_h": float(pair["lower_pair_h"]),
                        "closest_pair_upper_h": float(pair["upper_pair_h"]),
                        "selected_at_10000_in_closest_pair": bool(pair["selected_at_10000_in_pair"])})
    metadata_name = "ten_example_paths_with_closest_pair.json" if include_closest_pair else "ten_example_paths.json"
    atomic_json(directory / metadata_name, selected_metadata)
    enable_automatic_pdf_saving(NOTEBOOK_NAME)
    image_directory = CODE_DIRECTORY / "outputs/images" / NOTEBOOK_NAME / directory.name
    image_directory.mkdir(parents=True, exist_ok=True)
    maximum_h = max(float(h_grid[r["coarse_selected_indices"]].max()) for r in chosen)
    y_max = np.ceil((maximum_h + 0.025) / 0.05) * 0.05
    colors = {"curve": "#2563a6", "coarse": "#a2b8cc", "switch": "#d96927", "pair": "#aa354b"}
    legend = [Line2D([0], [0], color=colors["curve"], lw=1.7, label="Selected h: coarse and local fine scans"),
              Line2D([0], [0], color=colors["coarse"], marker="o", ls="none", ms=3,
                     label="Original coarse lambda evaluations"),
              Line2D([0], [0], color=colors["switch"], ls="--", label="Confirmed peak switch")]
    if include_closest_pair:
        legend.append(Line2D([0], [0], color=colors["pair"], ls=":", lw=1.4,
                             label="Closest bandwidth pair: horizontal lines"))
    paths = []
    for page, examples in enumerate((chosen[:6], chosen[6:]), start=1):
        columns = 3 if len(examples) == 6 else 2
        fig, axes = plt.subplots(2, columns, figsize=(12.8, 6.8), sharex=True, sharey=True,
                                 constrained_layout=True)
        title_suffix = " with closest pairs" if include_closest_pair else ""
        fig.suptitle(f"Ten Figure 3.3-type sample paths - examples {1 if page == 1 else 7}-{6 if page == 1 else 10}{title_suffix}",
                     fontsize=14)
        for axis, record in zip(axes.flat, examples):
            evaluated = {float(lam): int(index) for lam, index in zip(coarse, record["coarse_selected_indices"])}
            for scan in record["fine_scans"]:
                evaluated.update({float(lam): int(index) for lam, index in zip(scan["lambda_values"], scan["selected_indices"])})
            lambdas = np.array(sorted(evaluated))
            indices = np.array([evaluated[value] for value in lambdas])
            axis.plot(lambdas, h_grid[indices], color=colors["curve"], lw=1.8)
            axis.scatter(coarse, h_grid[record["coarse_selected_indices"]], color=colors["coarse"],
                         s=7, zorder=3)
            panel_title = f"Sample {record['repetition']} (n={settings['sample_size']})"
            if include_closest_pair:
                pair = closest_pairs.loc[record["repetition"]]
                lower_h, upper_h = float(pair["lower_pair_h"]), float(pair["upper_pair_h"])
                axis.axhspan(lower_h, upper_h, color=colors["pair"], alpha=0.065, zorder=0)
                for value in (lower_h, upper_h):
                    axis.axhline(value, color=colors["pair"], ls=":", lw=1.2, alpha=0.95)
                panel_title += f"\nClosest pair h = {lower_h:.4f}, {upper_h:.4f}"
            notes = []
            for event in record["events"]:
                root = event["lambda_crossing"]
                axis.axvline(root, color=colors["switch"], ls="--", lw=1, alpha=0.8)
                axis.plot([root, root], [event["h_before"], event["h_after"]], color=colors["switch"], lw=2)
                axis.scatter([root, root], [event["h_before"], event["h_after"]],
                             color=colors["switch"], s=18, zorder=4)
                notes.append(rf"$\lambda_c={root:.4g}$; separation {event['severity_percent']:.1f}% ({event['direction']})")
            if not notes:
                notes = ["No separated-peak switch detected"]
            axis.text(0.035, 0.035, "\n".join(notes), transform=axis.transAxes, fontsize=8.4,
                      va="bottom", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.90})
            axis.set(title=panel_title,
                     xscale="log", xlim=(coarse[0], coarse[-1]), ylim=(0, y_max))
            if include_closest_pair:
                axis.set_title(panel_title, fontsize=10.5)
            axis.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
            axis.xaxis.set_minor_formatter(NullFormatter())
            axis.grid(alpha=0.16)
            axis.spines[["top", "right"]].set_visible(False)
        for axis in axes[-1]:
            axis.set_xlabel(r"Similarity parameter $\lambda$")
        for axis in axes[:, 0]:
            axis.set_ylabel("Selected bandwidth h")
        fig.legend(handles=legend, loc="outside lower center", ncol=2 if include_closest_pair else 3,
                   frameon=False, fontsize=9)
        file_suffix = "_with_closest_pair" if include_closest_pair else ""
        path = image_directory / f"ten_figure_3_3_type_paths_{page}{file_suffix}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        paths.append(path)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    run_analysis(samples=args.samples, workers=args.workers)
