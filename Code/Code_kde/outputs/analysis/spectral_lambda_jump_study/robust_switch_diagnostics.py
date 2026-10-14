"""Separate numerical-grid effects from sample-dependent sharp peak switches.

Compare a fixed cohort (first 200 saved independent samples) across grid
resolutions and grid-node placements. Reuse the original 201-node records.
A sharp event requires separated global peaks, a resolved valley, a local
height-order reversal, and a bandwidth ratio at the crossing >= 1.2.
Candidate intervals have an endpoint bandwidth ratio >= 1.05, independently
of bandwidth-grid resolution. Sampling in lambda can miss short events.

Also scan deterministic population-density and expected-empirical-divergence
Gaussian references. The Perron vector of exp(-lambda * E[D]) is NOT E[v]
or the Perron vector of E[exp(-lambda * D)].
"""

from pathlib import Path
import argparse
import importlib.util
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

CODE_DIRECTORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE_DIRECTORY))
os.environ.setdefault("MPLCONFIGDIR", str(CODE_DIRECTORY / ".matplotlib-cache"))

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import norm
from threadpoolctl import threadpool_limits

from kde_tools.fourier import beta2_divergence_matrix_fft
from kde_tools.perron import PERRON_SOLVER_VERSION, perron_eigenpair_from_divergence
from kde_tools.similarities import population_gaussian_beta2_divergence_matrix, squared_l2_distance_from_gram

_spec = importlib.util.spec_from_file_location(
    "peak_crossing_detector", Path(__file__).with_name("repeated_peak_switch_analysis.py")
)
_peak_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_peak_module)

SOURCE_DIRECTORY = CODE_DIRECTORY / "outputs/data/spectral_lambda_jump_study/2ce04db2a16e"
OUTPUT_DIRECTORY = SOURCE_DIRECTORY / "numerical_vs_sampling"
COHORT_SIZE = 200
METHOD_VERSION = "paired_sharp_switch_refinement_v1"
SCENARIOS = {"endpoint_401": (401, "endpoint"),
             "endpoint_801": (801, "endpoint"),
             "midpoint_401": (401, "midpoint")}
SHARP_RATIO = 1.2
MINIMUM_CANDIDATE_RATIO = 1.05
RELATIVE_PROMINENCE = 1e-8
_SAMPLES = _LAMBDAS = _CONFIGURATION = None


def bandwidth_grid(size, placement="endpoint"):
    a, b = np.log(0.01), np.log(2.0)
    if placement == "endpoint":
        return np.exp(np.linspace(a, b, size))
    if placement == "midpoint":
        return np.exp(a + (np.arange(size) + 0.5) * (b - a) / size)
    raise ValueError(placement)


def wilson_interval(successes, count):
    """Wilson's nominal 95% interval for independent Bernoulli outcomes."""
    z = float(norm.ppf(0.975))
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * np.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [float(center - radius), float(center + radius)]


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.replace(path)


def initialize_worker():
    global _SAMPLES, _LAMBDAS, _CONFIGURATION
    with np.load(SOURCE_DIRECTORY / "checkpoint.npz", allow_pickle=False) as source:
        _SAMPLES = source["samples"].copy()
        _LAMBDAS = source["lambda_values"].copy()
        _CONFIGURATION = json.loads(str(source["configuration_json"].item()))


def analyze_task(task):
    scenario, repetition = task
    size, placement = SCENARIOS[scenario]
    h_grid = bandwidth_grid(size, placement)
    with threadpool_limits(limits=1):
        divergence = beta2_divergence_matrix_fft(
            _SAMPLES[repetition], h_grid, half_width=_CONFIGURATION["fft_half_width"],
            spatial_points=_CONFIGURATION["fft_spatial_points"]
        )
        results = [perron_eigenpair_from_divergence(divergence, value) for value in _LAMBDAS]
        indices = np.array([result.selected_index for result in results])
        counts = [int(len(find_peaks(result.vector, prominence=RELATIVE_PROMINENCE * result.vector.max())[0]))
                  for result in results]
        candidates = np.flatnonzero(np.abs(np.diff(np.log(h_grid[indices]))) >= np.log(MINIMUM_CANDIDATE_RATIO))
        events = [_peak_module.crossing_event(divergence, h_grid, _LAMBDAS[k], _LAMBDAS[k + 1],
                                              results[k], results[k + 1], int(k)) for k in candidates]
    confirmed = [event for event in events if event["status"] == "confirmed"]
    for event in confirmed:
        event["bandwidth_ratio_at_crossing"] = event["upper_peak_h"] / event["lower_peak_h"]
    return {
        "scenario": scenario, "repetition": repetition, "bandwidth_grid_size": size,
        "placement": placement, "selected_indices": indices.tolist(), "peak_counts": counts,
        "candidate_intervals": len(candidates), "confirmed_events": confirmed,
        "refinement_failures": sum(event["status"] == "refinement_failure" for event in events),
        "maximum_relative_residual": float(max(result.relative_residual for result in results)),
    }


def deterministic_controls():
    """Original analytic Gaussian references, with 601 lambda values each."""
    rows = []
    with threadpool_limits(limits=1):
        for size in (201, 801):
            h_grid = bandwidth_grid(size)
            variance_sum = h_grid[:, None] ** 2 + h_grid[None, :] ** 2
            # Diagonal sample pairs have distance zero; distinct independent
            # N(0,1) pairs have a N(0,2) difference. This is our derivation.
            expected_gram = (1 / 200) / np.sqrt(2 * np.pi * variance_sum)
            expected_gram += (1 - 1 / 200) / np.sqrt(2 * np.pi * (variance_sum + 2))
            matrices = {
                "population_densities": population_gaussian_beta2_divergence_matrix(h_grid),
                "expected_empirical_divergence_n200": 0.5 * squared_l2_distance_from_gram(expected_gram),
            }
            lambdas = np.geomspace(0.01, 10000, 601)
            for name, divergence in matrices.items():
                results = [perron_eigenpair_from_divergence(divergence, value) for value in lambdas]
                indices = np.array([result.selected_index for result in results])
                counts = [int(len(find_peaks(result.vector, prominence=RELATIVE_PROMINENCE * result.vector.max())[0]))
                          for result in results]
                candidates = np.flatnonzero(np.abs(np.diff(np.log(h_grid[indices]))) >= np.log(MINIMUM_CANDIDATE_RATIO))
                events = [_peak_module.crossing_event(divergence, h_grid, lambdas[k], lambdas[k + 1],
                                                      results[k], results[k + 1], int(k)) for k in candidates]
                confirmed = [event for event in events if event["status"] == "confirmed"
                             and event["upper_peak_h"] / event["lower_peak_h"] >= SHARP_RATIO]
                rows.append({"reference": name, "bandwidth_grid_size": size,
                             "lambda_grid": "geomspace(0.01, 10000, 601)",
                             "maximum_resolved_peak_count": max(counts),
                             "lambda_points_with_multiple_peaks": sum(count >= 2 for count in counts),
                             "confirmed_sharp_switches": len(confirmed),
                             "maximum_adjacent_log_bandwidth_change": float(np.abs(np.diff(np.log(h_grid[indices]))).max()),
                             "lambdas": lambdas.tolist(), "selected_bandwidths": h_grid[indices].tolist()})
                print(f"Control {name}, N={size}: max peaks {max(counts)}, sharp switches {len(confirmed)}", flush=True)
    atomic_json(OUTPUT_DIRECTORY / "deterministic_controls.json", rows)
    return rows


def export_results(settings, new_records, controls):
    original = json.loads((SOURCE_DIRECTORY / "peak_switches/checkpoint.json").read_text())["records"]
    original_cohort = []
    for record in original[:settings["cohort_size"]]:
        confirmed = [dict(event) for event in record["events"] if event["status"] == "confirmed"]
        for event in confirmed:
            event["bandwidth_ratio_at_crossing"] = event["upper_peak_h"] / event["lower_peak_h"]
        original_cohort.append({"scenario": "endpoint_201", "repetition": record["repetition"],
                                "bandwidth_grid_size": 201, "placement": "endpoint",
                                "peak_counts": record["peak_counts"]["1e-08"],
                                "confirmed_events": confirmed, "refinement_failures": 0,
                                "maximum_relative_residual": record["max_relative_residual"]})
    all_records = original_cohort + new_records
    table_rows = []
    scenario_summary = {}
    original_flags = np.array([any(event["bandwidth_ratio_at_crossing"] >= SHARP_RATIO
                                  for event in record["confirmed_events"]) for record in original_cohort])
    for scenario in ("endpoint_201", *SCENARIOS):
        records = sorted([record for record in all_records if record["scenario"] == scenario],
                         key=lambda record: record["repetition"])
        assert len(records) == settings["cohort_size"]
        flags = []
        for record in records:
            sharp = [event for event in record["confirmed_events"] if event["bandwidth_ratio_at_crossing"] >= SHARP_RATIO]
            flag = bool(sharp)
            flags.append(flag)
            table_rows.append({"scenario": scenario, "repetition": record["repetition"],
                               "has_sharp_switch": flag, "sharp_switch_events": len(sharp),
                               "first_sharp_switch_lambda": sharp[0]["lambda_crossing"] if sharp else None,
                               "maximum_peak_count": max(record["peak_counts"]),
                               "maximum_relative_residual": record["maximum_relative_residual"]})
        flags = np.array(flags)
        scenario_summary[scenario] = {
            "samples": len(records), "samples_with_sharp_switch": int(flags.sum()),
            "frequency": float(flags.mean()), "wilson_95_percent_interval": wilson_interval(int(flags.sum()), len(records)),
            "same_classification_as_original_201": int(np.sum(flags == original_flags)),
            "original_switch_persists": int(np.sum(flags & original_flags)),
            "original_switch_not_detected": int(np.sum(~flags & original_flags)),
            "new_switch_detected": int(np.sum(flags & ~original_flags)),
            "refinement_failures": sum(record["refinement_failures"] for record in records),
            "sensitivity_by_minimum_bandwidth_ratio": {
                str(ratio): int(sum(any(event["bandwidth_ratio_at_crossing"] >= ratio for event in record["confirmed_events"])
                                    for record in records)) for ratio in (1.1, 1.2, 1.5)
            },
        }
    old_count = sum(any(event["status"] == "confirmed" and event["upper_peak_h"] / event["lower_peak_h"] >= SHARP_RATIO
                        for event in record["events"]) for record in original)
    summary = {
        "method": settings, "paired_scenarios": scenario_summary,
        "original_1000_sample_sharp_switch_frequency": {"samples": len(original), "samples_with_sharp_switch": old_count,
                                                        "frequency": old_count / len(original),
                                                        "wilson_95_percent_interval": wilson_interval(old_count, len(original))},
        "deterministic_controls": [{key: value for key, value in row.items() if key not in ("lambdas", "selected_bandwidths")}
                                    for row in controls],
        "maximum_relative_residual": max(record["maximum_relative_residual"] for record in all_records),
        "scope": "Numerical detection in the stated lambda range and grids; short switches between sampled lambda points can be missed. Expected divergence is a deterministic reference, not E[v].",
    }
    pd.DataFrame(table_rows).to_csv(OUTPUT_DIRECTORY / "paired_sample_results.csv", index=False)
    atomic_json(OUTPUT_DIRECTORY / "all_records.json", all_records)
    (OUTPUT_DIRECTORY / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def run_analysis(cohort_size=COHORT_SIZE, workers=3):
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    settings = {"version": METHOD_VERSION, "cohort_size": cohort_size,
                "cohort_selection": "first saved independent samples; same observations in every scenario",
                "scenarios": SCENARIOS, "sharp_bandwidth_ratio": SHARP_RATIO,
                "minimum_candidate_endpoint_ratio": MINIMUM_CANDIDATE_RATIO,
                "relative_prominence": RELATIVE_PROMINENCE, "perron_solver": PERRON_SOLVER_VERSION,
                "lambda_grid": "original 101 values, geomspace(0.01, 10000, 101)"}
    # JSON round-trip preserves checkpoint comparisons across tuple/list conversion.
    settings = json.loads(json.dumps(settings))
    checkpoint_path = OUTPUT_DIRECTORY / "checkpoint.json"
    records = []
    if checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text())
        if checkpoint["settings"] != settings:
            raise ValueError("Diagnostics settings differ from the existing checkpoint.")
        records = checkpoint["records"]
    done = {(record["scenario"], record["repetition"]) for record in records}
    tasks = [(scenario, repetition) for scenario in SCENARIOS for repetition in range(cohort_size)
             if (scenario, repetition) not in done]
    print(f"Paired grid checks: {len(records)}/{len(SCENARIOS) * cohort_size} tasks cached", flush=True)
    started = time.perf_counter()
    context = multiprocessing.get_context("spawn")
    try:
        if tasks:
            with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=initialize_worker) as pool:
                futures = {pool.submit(analyze_task, task): task for task in tasks}
                for future in as_completed(futures):
                    records.append(future.result())
                    if len(records) % 20 == 0:
                        atomic_json(checkpoint_path, {"settings": settings, "records": records})
                        print(f"Completed {len(records)}/{len(SCENARIOS) * cohort_size} paired tasks; "
                              f"elapsed {time.perf_counter() - started:.1f}s", flush=True)
    finally:
        atomic_json(checkpoint_path, {"settings": settings, "records": records})
    control_path = OUTPUT_DIRECTORY / "deterministic_controls.json"
    controls = json.loads(control_path.read_text()) if control_path.is_file() else deterministic_controls()
    return export_results(settings, records, controls)


def audit_fragile_case(repetition=68):
    """Inspect the single discordant sample in the paired 200-sample cohort.

    A targeted lambda scan compares the two competing bandwidth regions.
    This checks whether a short switch was missed near the original root;
    it does not certify the absence of a switch for every continuous lambda.
    """
    with np.load(SOURCE_DIRECTORY / "checkpoint.npz", allow_pickle=False) as source:
        sample = source["samples"][repetition].copy()
    rows = []
    lambdas = np.r_[1e-4, 1e-3, np.geomspace(0.01, 1.0, 81)]
    with threadpool_limits(limits=1):
        for size, placement in ((201, "endpoint"), (401, "endpoint"),
                                (801, "endpoint"), (401, "midpoint")):
            h_grid = bandwidth_grid(size, placement)
            divergence = beta2_divergence_matrix_fft(sample, h_grid)
            lower = np.flatnonzero((h_grid > 0.055) & (h_grid < 0.083))
            upper = np.flatnonzero((h_grid > 0.085) & (h_grid < 0.13))
            values = []
            for lambda_value in lambdas:
                result = perron_eigenpair_from_divergence(divergence, lambda_value)
                lower_index = lower[np.argmax(result.vector[lower])]
                upper_index = upper[np.argmax(result.vector[upper])]
                values.append({
                    "lambda": float(lambda_value), "selected_h": float(h_grid[result.selected_index]),
                    "relative_lower_minus_upper_height": float(
                        (result.vector[lower_index] - result.vector[upper_index]) / result.vector.max()),
                    "resolved_peak_count": int(len(find_peaks(
                        result.vector, prominence=RELATIVE_PROMINENCE * result.vector.max())[0])),
                })
            rows.append({"grid_size": size, "placement": placement, "values": values})
    report = {"repetition": repetition,
              "scope": "Targeted dense lambda scan from 0.01 to 1, plus 0.0001 and 0.001; compares competing bandwidth regions, not a continuum proof.",
              "scenarios": rows}
    atomic_json(OUTPUT_DIRECTORY / f"fragile_case_{repetition}.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-size", type=int, default=COHORT_SIZE)
    parser.add_argument("--workers", type=int, default=3)
    arguments = parser.parse_args()
    run_analysis(arguments.cohort_size, arguments.workers)
