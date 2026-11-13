"""Reusable functions for 02_kde_numerics. Experiment settings stay in the notebook."""

from __future__ import annotations
from kde_tools.fourier.similarities import FOURIER_L2_METHOD_VERSION
from pathlib import Path
import hashlib
import inspect
import scipy
import uuid
from IPython.display import Markdown
from IPython.display import display
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from datetime import datetime
from datetime import timezone
from functools import partial
from kde_tools.density import exact_gaussian_kde_l2_gram
from kde_tools.fourier.similarities import gaussian_kde_l2_gram_fourier
from threadpoolctl import threadpool_limits
from time import get_clock_info
from time import perf_counter
import csv
import io
import json
import numpy as np
import os
import platform
import subprocess
import sys


def calculate_one_sample(sample_index, *, K_candidates, bandwidths, frequency_cutoff, samples):
    current_sample = samples[sample_index]
    reference = exact_gaussian_kde_l2_gram(current_sample, bandwidths)
    local_errors = np.empty(len(K_candidates))
    local_worst_indices = np.empty((len(K_candidates), 2), dtype=int)
    for k_index, K in enumerate(K_candidates):
        approximate = gaussian_kde_l2_gram_fourier(
            current_sample,
            bandwidths,
            frequency_cutoff=frequency_cutoff,
            frequency_points=int(K) + 1,
        )
        difference = np.abs(approximate - reference)
        r, s = np.unravel_index(np.argmax(difference), difference.shape)
        local_errors[k_index] = difference[r, s]
        local_worst_indices[k_index] = (r, s)
    return (sample_index, reference, local_errors, local_worst_indices)


def timing_statistics(values):
    return {
        "mean_seconds": float(values.mean()),
        "standard_deviation_seconds": float(values.std(ddof=1)),
        "median_seconds": float(np.median(values)),
        "minimum_seconds": float(values.min()),
        "maximum_seconds": float(values.max()),
        "total_seconds": float(values.sum()),
    }


def run_fourier_error_experiment(
    *,
    K_candidates,
    N,
    bandwidths,
    data_directory,
    frequency_cutoff,
    realizations,
    samples,
    settings,
    workers,
):
    """Compute the error curves, or reuse a verified matching calculation."""
    calculation_settings = {
        key: value for key, value in settings.items() if key != "summary_curves"
    }
    cached_directory = None
    for candidate in sorted(data_directory.parent.iterdir()):
        settings_path = candidate / "settings.json"
        required_paths = [
            candidate / name
            for name in (
                "per_sample_results.npz",
                "exact_grams.npy",
                "samples.npy",
                "bandwidths.npy",
            )
        ]
        if not settings_path.is_file() or not all((p.is_file() for p in required_paths)):
            continue
        candidate_settings = json.loads(settings_path.read_text())
        candidate_calculation_settings = {
            key: value for key, value in candidate_settings.items() if key != "summary_curves"
        }
        if candidate_calculation_settings != calculation_settings:
            continue
        if not np.array_equal(np.load(candidate / "samples.npy"), samples) or not np.array_equal(
            np.load(candidate / "bandwidths.npy"), bandwidths
        ):
            continue
        with np.load(candidate / "per_sample_results.npz") as cached:
            if (
                not np.array_equal(cached["K"], K_candidates)
                or cached["errors"].shape != (realizations, len(K_candidates))
                or cached["worst_indices"].shape != (realizations, len(K_candidates), 2)
                or (not np.isfinite(cached["errors"]).all())
                or np.any(cached["errors"] < 0)
            ):
                continue
            cached_errors = cached["errors"].copy()
            cached_worst_indices = cached["worst_indices"].copy()
        cached_references = np.load(candidate / "exact_grams.npy")
        if (
            cached_references.shape != (realizations, N, N)
            or not np.isfinite(cached_references).all()
        ):
            continue
        cached_directory = candidate
        break
    if cached_directory is not None:
        per_sample_errors = cached_errors
        worst_indices = cached_worst_indices
        exact_grams = cached_references
        elapsed_seconds = 0.0
        calculation_directory = cached_directory
        print(
            f"Reused all {realizations} verified samples from {cached_directory.name}; no simulations rerun."
        )
    else:
        calculation_directory = data_directory
        np.save(calculation_directory / "samples.npy", samples)
        np.save(calculation_directory / "bandwidths.npy", bandwidths)
        per_sample_errors = np.empty((realizations, len(K_candidates)))
        worst_indices = np.empty((realizations, len(K_candidates), 2), dtype=int)
        exact_grams = np.empty((realizations, N, N))
        start = perf_counter()
        with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    partial(
                        calculate_one_sample,
                        K_candidates=K_candidates,
                        bandwidths=bandwidths,
                        frequency_cutoff=frequency_cutoff,
                        samples=samples,
                    ),
                    m,
                )
                for m in range(realizations)
            ]
            for completed, future in enumerate(as_completed(futures), start=1):
                sample_index, reference, local_errors, local_worst = future.result()
                exact_grams[sample_index] = reference
                per_sample_errors[sample_index] = local_errors
                worst_indices[sample_index] = local_worst
                if completed % 10 == 0 or completed == realizations:
                    print(
                        f"Completed {completed}/{realizations} samples ({perf_counter() - start:.1f} s).",
                        flush=True,
                    )
        elapsed_seconds = perf_counter() - start
        np.save(data_directory / "exact_grams.npy", exact_grams)
        np.savez_compressed(
            data_directory / "per_sample_results.npz",
            K=K_candidates,
            errors=per_sample_errors,
            worst_indices=worst_indices,
        )
    (data_directory / "cache_reference.json").write_text(
        json.dumps(
            {"relative_directory": os.path.relpath(calculation_directory, data_directory)}, indent=2
        )
        + "\n"
    )
    return (
        per_sample_errors,
        worst_indices,
        exact_grams,
        elapsed_seconds,
        cached_directory,
        calculation_directory,
    )


def summarize_fourier_error(
    *,
    K_candidates,
    N,
    absolute_tolerance,
    cached_directory,
    configuration_id,
    data_directory,
    elapsed_seconds,
    frequency_cutoff,
    highlight_K,
    n,
    per_sample_errors,
    realizations,
    workers,
):
    maximum_errors = per_sample_errors.max(axis=0)
    worst_sample_indices = per_sample_errors.argmax(axis=0)
    exceeding_counts = (per_sample_errors > absolute_tolerance).sum(axis=0)
    summary_table = np.column_stack(
        [K_candidates, maximum_errors, worst_sample_indices, exceeding_counts]
    )
    np.savetxt(
        data_directory / "g_rs_error_vs_k_maximum_summary.csv",
        summary_table,
        delimiter=",",
        comments="",
        header="K,max_E_K,worst_sample_zero_based,samples_exceeding_tolerance",
        fmt=["%d", "%.17g", "%d", "%d"],
    )
    highlight_index = int(np.flatnonzero(K_candidates == highlight_K)[0])
    passing_maximum = np.flatnonzero(maximum_errors <= absolute_tolerance)
    report = {
        "configuration_id": configuration_id,
        "n": n,
        "realizations": realizations,
        "N": N,
        "frequency_cutoff": frequency_cutoff,
        "absolute_tolerance": absolute_tolerance,
        "first_tested_K_max_below_tolerance": (
            int(K_candidates[passing_maximum[0]]) if len(passing_maximum) else None
        ),
        "highlight_K": highlight_K,
        "max_error_at_highlight_K": float(maximum_errors[highlight_index]),
        "samples_exceeding_tolerance_at_highlight_K": int(exceeding_counts[highlight_index]),
        "largest_tested_K": int(K_candidates[-1]),
        "max_error_at_largest_K": float(maximum_errors[-1]),
        "calculation_seconds_this_run": elapsed_seconds,
        "workers": workers,
        "reused_cache_configuration": (
            cached_directory.name if cached_directory is not None else None
        ),
        "empirical_scope": "maximum over 200 samples of size 200 and all pairs of 50 bandwidths including diagonal",
    }
    (data_directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"First tested K with maximum E_K <= {absolute_tolerance:g}: {report['first_tested_K_max_below_tolerance']}"
    )
    print(f"At K={highlight_K}: maximum error={maximum_errors[highlight_index]:.8e}.")
    print(
        f"Samples above the target at K={highlight_K}: {exceeding_counts[highlight_index]}/{realizations}."
    )
    return (maximum_errors, report)


def save_runtime_comparison(
    *,
    K,
    N,
    absolute_tolerance,
    active_threadpools,
    bandwidths,
    code_directory,
    configuration_id,
    data_directory,
    exact_grams,
    exact_seconds,
    experiment_seconds,
    fourier_grams,
    fourier_seconds,
    frequency_cutoff,
    n,
    notebook_name,
    per_sample_max_errors,
    realizations,
    run_id,
    seed,
    settings,
    thread_environment_keys,
):
    assert exact_grams.shape == fourier_grams.shape == (realizations, N, N)
    assert np.isfinite(exact_grams).all() and np.isfinite(fourier_grams).all()
    assert np.isfinite(exact_seconds).all() and np.all(exact_seconds > 0)
    assert np.isfinite(fourier_seconds).all() and np.all(fourier_seconds > 0)
    assert np.array_equal(exact_grams, exact_grams.transpose(0, 2, 1))
    assert np.allclose(fourier_grams, fourier_grams.transpose(0, 2, 1), rtol=0, atol=1e-14)
    all_errors = np.abs(fourier_grams - exact_grams)
    assert np.array_equal(per_sample_max_errors, all_errors.max(axis=(1, 2)))
    worst_l, worst_r, worst_s = np.unravel_index(np.argmax(all_errors), all_errors.shape)
    maximum_error = float(all_errors[worst_l, worst_r, worst_s])
    mean_exact_seconds = float(exact_seconds.mean())
    mean_fourier_seconds = float(fourier_seconds.mean())
    speedup = mean_exact_seconds / mean_fourier_seconds
    cpu_model = platform.processor()
    if sys.platform == "darwin":
        cpu_query = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        )
        if cpu_query.returncode == 0:
            cpu_model = cpu_query.stdout.strip()
    report = {
        **settings,
        "configuration_id": configuration_id,
        "run_id": run_id,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "clock_info": vars(get_clock_info("perf_counter")),
        "thread_environment": {key: os.environ.get(key) for key in thread_environment_keys},
        "detected_threadpools_during_timing": active_threadpools,
        "exact": timing_statistics(exact_seconds),
        "fourier": timing_statistics(fourier_seconds),
        "speedup_ratio_of_mean_times": speedup,
        "maximum_absolute_error": maximum_error,
        "samples_exceeding_tolerance": int(
            np.count_nonzero(per_sample_max_errors > absolute_tolerance)
        ),
        "worst_case": {
            "sample_l_one_based": int(worst_l) + 1,
            "seed": seed + int(worst_l),
            "r_zero_based": int(worst_r),
            "s_zero_based": int(worst_s),
            "h_r": float(bandwidths[worst_r]),
            "h_s": float(bandwidths[worst_s]),
            "exact_value": float(exact_grams[worst_l, worst_r, worst_s]),
            "fourier_value": float(fourier_grams[worst_l, worst_r, worst_s]),
        },
        "ordered_pairs_per_sample_including_diagonal": N * N,
        "unique_symmetric_pairs_per_sample_including_diagonal": N * (N + 1) // 2,
        "checked_entries_total": realizations * N * N,
        "experiment_elapsed_seconds_excluding_warmup_and_exports": experiment_seconds,
        "timing_order_counts": {
            "exact_first": (realizations + 1) // 2,
            "fourier_first": realizations // 2,
        },
        "validation": {
            "all_entries_finite": True,
            "all_times_finite_and_positive": True,
            "exact_matrices_symmetric": True,
            "fourier_matrices_symmetric_to_1e_minus_14": True,
            "stored_per_sample_maxima_match_full_matrices": True,
        },
    }
    (data_directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    np_config_buffer = io.StringIO()
    from contextlib import redirect_stdout

    with redirect_stdout(np_config_buffer):
        np.show_config()
    (data_directory / "numpy_configuration.txt").write_text(np_config_buffer.getvalue())
    with (data_directory / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "mean_seconds",
                "mean_milliseconds",
                "standard_deviation_seconds",
                "maximum_absolute_error_vs_exact",
            ]
        )
        writer.writerow(
            [
                "Exact Gaussian formula",
                mean_exact_seconds,
                1000 * mean_exact_seconds,
                report["exact"]["standard_deviation_seconds"],
                "",
            ]
        )
        writer.writerow(
            [
                "Fourier approximation",
                mean_fourier_seconds,
                1000 * mean_fourier_seconds,
                report["fourier"]["standard_deviation_seconds"],
                maximum_error,
            ]
        )
    summary_markdown = f"| Method | Mean time per complete matrix (ms) | Maximum absolute error |\n| --- | ---: | ---: |\n| Exact Gaussian formula | {1000 * mean_exact_seconds:.3f} | Reference |\n| Fourier approximation | {1000 * mean_fourier_seconds:.3f} | {maximum_error:.3e} |\n"
    summary_note = f"\n100 standard-normal samples, n=200; N=50 geometric bandwidths in [0.01, 2]; Omega={frequency_cutoff:g}, K={K} intervals. The Fourier implementation is {speedup:.2f} times faster by the ratio of mean times. The error is the maximum over all {realizations * N * N:,} matrix entries, including diagonals. Measured on {cpu_model} ({platform.system()} {platform.release()}); timings are machine dependent.\n"
    (data_directory / "summary.md").write_text(summary_markdown + summary_note)
    mantissa, exponent = f"{maximum_error:.3e}".split("e")
    latex_error = f"${mantissa}\\times10^{{{int(exponent)}}}$"
    latex_table = "\n".join(
        [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\begin{tabular}{lrr}",
            "\\hline",
            "Method & Mean time (ms) & Maximum absolute error \\\\",
            "\\hline",
            f"Exact Gaussian formula & {1000 * mean_exact_seconds:.3f} & --- " + "\\\\",
            f"Fourier approximation & {1000 * mean_fourier_seconds:.3f} & {latex_error} " + "\\\\",
            "\\hline",
            "\\end{tabular}",
            f"\\caption{{Runtime comparison over $M={realizations}$ standard-normal samples of size $n={n}$, using $N={N}$ geometrically spaced bandwidths in $[0.01,2]$, $\\Omega={frequency_cutoff:g}$ and $K={K}$ trapezoidal intervals. Times are arithmetic means for complete Gram-matrix calculations. The maximum absolute error is taken over all samples and bandwidth pairs, including diagonal entries, with the float64 analytic calculation as reference.}}",
            "\\label{tab:exact_fourier_gram_runtime}",
            "\\end{table}",
            "",
        ]
    )
    (data_directory / "runtime_comparison_table.tex").write_text(latex_table)
    (code_directory / "outputs" / "data" / notebook_name / "latest_run.json").write_text(
        json.dumps(
            {
                "relative_directory": str(data_directory.relative_to(code_directory)),
                "configuration_id": configuration_id,
                "run_id": run_id,
            },
            indent=2,
        )
        + "\n"
    )
    display(Markdown(summary_markdown + summary_note))
    print(
        f"Worst error: {maximum_error:.12e}; sample l={worst_l + 1}, h={bandwidths[worst_r]:g}, s={bandwidths[worst_s]:g}."
    )
    print(
        f"Samples above {absolute_tolerance:g}: {report['samples_exceeding_tolerance']}/{realizations}."
    )
    print(f"LaTeX table: {data_directory / 'runtime_comparison_table.tex'}")
    return report


def prepare_runtime_run(
    *,
    K,
    N,
    absolute_tolerance,
    bandwidths,
    code_directory,
    frequency_cutoff,
    h_max,
    h_min,
    n,
    notebook_name,
    realizations,
    samples,
    seed,
    warmup_calls_per_method,
):
    source_paths = {
        "exact": Path(inspect.getfile(exact_gaussian_kde_l2_gram)),
        "fourier": Path(inspect.getfile(gaussian_kde_l2_gram_fourier)),
    }
    settings = {
        "experiment_version": "paired_full_gram_runtime_v1",
        "sample_distribution": "standard_normal",
        "n": n,
        "realizations": realizations,
        "N": N,
        "h_min": h_min,
        "h_max": h_max,
        "bandwidth_grid": "geometric",
        "seed": seed,
        "seed_rule": "seed + zero_based_sample_index",
        "warmup_seed": seed + 1000000,
        "frequency_cutoff": float(frequency_cutoff),
        "K": int(K),
        "frequency_points": int(K) + 1,
        "K_convention": "number of trapezoidal intervals",
        "method_version": FOURIER_L2_METHOD_VERSION,
        "timer": "time.perf_counter",
        "timing_scope": "one full Gram helper call; no sampling, warm-up, validation or I/O",
        "warmup_calls_per_method": warmup_calls_per_method,
        "order": "exact first for even zero-based indices; Fourier first for odd indices",
        "parallel_samples": False,
        "requested_numerical_library_threads": 1,
        "maximum_error": "maximum absolute difference over all samples and all ordered bandwidth pairs, including diagonal",
        "absolute_tolerance": absolute_tolerance,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "source_sha256": {
            key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in source_paths.items()
        },
        "sample_sha256": hashlib.sha256(samples.tobytes()).hexdigest(),
        "bandwidth_sha256": hashlib.sha256(bandwidths.tobytes()).hexdigest(),
    }
    configuration_id = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[
        :12
    ]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:6]
    data_directory = code_directory / "outputs" / "data" / notebook_name / configuration_id / run_id
    data_directory.mkdir(parents=True, exist_ok=False)
    (data_directory / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    np.save(data_directory / "samples.npy", samples)
    np.save(data_directory / "bandwidths.npy", bandwidths)
    print(f"{realizations} samples of size {n}; {N} bandwidths; Omega={frequency_cutoff:g}; K={K}.")
    print(f"Results: {data_directory}")
    return (data_directory, settings, configuration_id, run_id)
