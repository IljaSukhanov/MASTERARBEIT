"""Shared runners and plots for continuous bandwidth-minimizer distributions."""

from __future__ import annotations
from kde_tools.continuous_minimizers import METHODS
from kde_tools.continuous_minimizers import continuous_ise_qr_minima
from kde_tools.continuous_runs import FIELDS
from kde_tools.continuous_runs import run_continuous_point
from kde_tools.density import exact_oracle_ise_values
from matplotlib.ticker import PercentFormatter
from threadpoolctl import threadpool_limits
import csv
import json
import numpy as np


def run_sample_size(
    n,
    *,
    reuse=True,
    workers,
    configuration,
    continuous_cache_root,
    continuous_import_directories,
    full_samples,
    run_directory,
    search_settings,
):
    shared = run_continuous_point(
        n,
        full_samples,
        search_settings,
        METHODS,
        run_directory,
        cache_root=continuous_cache_root,
        configuration=configuration,
        reuse=reuse,
        workers=workers,
        import_directories=continuous_import_directories,
    )
    print(
        f"n={n}: reused {shared['reused_selections']} continuous selections; computed {shared['computed_selections']}",
        flush=True,
    )
    return shared["values"]


def style_panel(ax, current_n, ticks, y_max, right_limit, *, histogram_bandwidths):
    ax.set_title(f"$n={current_n}$", fontsize=13, pad=8)
    ax.set_xticks(ticks, [f"{histogram_bandwidths[i]:.3g}" for i in ticks])
    ax.set_xlim(-0.7, right_limit)
    ax.set_ylim(0, y_max)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("continuously selected bandwidth h", fontsize=10)
    ax.set_ylabel("relative frequency", fontsize=10)
    ax.tick_params(labelsize=9, colors="#304967", labelbottom=True, labelleft=True)
    ax.grid(axis="y", color="#DDE5EC", linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#B8C5D1")


def draw_histograms(ax, distributions, *, colors, styles, histogram_edges):
    for method in METHODS:
        ax.stairs(
            distributions[method],
            histogram_edges,
            fill=True,
            color=colors[method],
            alpha=0.07,
            linewidth=0,
        )
        ax.stairs(
            distributions[method],
            histogram_edges,
            color=colors[method],
            linestyle=styles[method],
            linewidth=1.8 if method == "ISE" else 1.5,
        )


def save_continuous_distributions(
    *,
    histogram_points,
    repetitions,
    sample_sizes,
    histogram_bandwidths,
    histogram_edges,
    histogram_positions,
    results_by_n,
    run_directory,
):
    frequencies, interior_frequencies, boundary_masses = ({}, {}, {})
    selection_rows, frequency_rows, summary_rows = ([], [], [])
    for current_n in sample_sizes:
        values = results_by_n[current_n]
        frequencies[current_n], interior_frequencies[current_n], boundary_masses[current_n] = (
            {},
            {},
            {},
        )
        for method_index, method in enumerate(METHODS):
            selected = values[:, method_index, 0]
            boundary = values[:, method_index, 4].astype(int)
            positions = np.interp(
                np.log(selected), np.log(histogram_bandwidths), histogram_positions
            )
            counts = np.histogram(positions, bins=histogram_edges)[0]
            interior_counts = np.histogram(positions[boundary != 1], bins=histogram_edges)[0]
            frequencies[current_n][method] = counts / repetitions
            interior_frequencies[current_n][method] = interior_counts / repetitions
            boundary_masses[current_n][method] = float(np.mean(boundary == 1))
            assert counts.sum() == repetitions
            assert interior_counts.sum() + np.sum(boundary == 1) == repetitions
            for repetition in range(repetitions):
                selection_rows.append(
                    {
                        "n": current_n,
                        "repetition": repetition,
                        "method": method,
                        **{
                            field: float(values[repetition, method_index, i])
                            for i, field in enumerate(FIELDS)
                        },
                    }
                )
            for index in range(histogram_points):
                frequency_rows.append(
                    {
                        "n": current_n,
                        "method": method,
                        "grid_index": index,
                        "h": float(histogram_bandwidths[index]),
                        "count": int(counts[index]),
                        "relative_frequency": float(counts[index] / repetitions),
                        "interior_relative_frequency": float(interior_counts[index] / repetitions),
                    }
                )
            summary_rows.append(
                {
                    "n": current_n,
                    "method": method,
                    "M": repetitions,
                    "mean_h": float(np.mean(selected)),
                    "median_h": float(np.median(selected)),
                    "sd_h": float(np.std(selected, ddof=1)),
                    "q05_h": float(np.quantile(selected, 0.05)),
                    "q95_h": float(np.quantile(selected, 0.95)),
                    "lower_boundary_count": int(np.sum(boundary == -1)),
                    "upper_boundary_count": int(np.sum(boundary == 1)),
                    "upper_boundary_frequency": boundary_masses[current_n][method],
                    "interior_count": int(np.sum(boundary == 0)),
                    "max_interior_candidates": int(np.max(values[:, method_index, 5])),
                }
            )
    for filename, rows in (
        ("continuous_minimizers.csv", selection_rows),
        ("frequencies.csv", frequency_rows),
        ("summary.csv", summary_rows),
    ):
        with (run_directory / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(" n    selector      median h    at h=2")
    for row in summary_rows:
        print(
            f"{row['n']:4d}  {row['method']:9s}  {row['median_h']:10.5f}  {row['upper_boundary_frequency']:8.1%}"
        )
    peak_comparison_rows = []
    for current_n in sample_sizes:
        values = results_by_n[current_n]
        ise_mode = float(histogram_bandwidths[np.argmax(frequencies[current_n]["ISE"])])
        green_interior = values[:, METHODS.index("Q13_14"), 4] == 0
        if np.any(green_interior):
            green_mode = float(
                histogram_bandwidths[np.argmax(interior_frequencies[current_n]["Q13_14"])]
            )
            paired_ise = values[green_interior, METHODS.index("ISE"), 0]
            paired_green = values[green_interior, METHODS.index("Q13_14"), 0]
            mode_ratio = green_mode / ise_mode
            paired_ratio = float(np.median(paired_green / paired_ise))
            paired_difference = float(np.median(paired_green - paired_ise))
        else:
            green_mode = mode_ratio = paired_ratio = paired_difference = None
        peak_comparison_rows.append(
            {
                "n": current_n,
                "ise_mode_bin_h": ise_mode,
                "q13_14_interior_mode_bin_h": green_mode,
                "interior_mode_ratio_q13_14_over_ise": mode_ratio,
                "q13_14_interior_count": int(np.sum(green_interior)),
                "q13_14_upper_boundary_frequency": boundary_masses[current_n]["Q13_14"],
                "paired_median_h_ratio_on_q13_14_interior_samples": paired_ratio,
                "paired_median_h_difference_on_q13_14_interior_samples": paired_difference,
            }
        )
    with (run_directory / "peak_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=peak_comparison_rows[0].keys())
        writer.writeheader()
        writer.writerows(peak_comparison_rows)
    return (frequencies, boundary_masses)


def validate_continuous_distributions(
    *,
    half_width,
    h_max,
    h_min,
    repetitions,
    sample_sizes,
    scan_points,
    sigma2,
    spatial_points,
    full_samples,
    results_by_n,
    run_directory,
):
    validation_rows = []
    with threadpool_limits(limits=1):
        for current_n in sample_sizes:
            values = results_by_n[current_n]
            interior = values[:, :, 4] == 0
            assert np.all(values[:, :, 3][interior] > 0)
            assert np.all(np.abs(values[:, :, 2][interior] / values[:, :, 3][interior]) <= 1e-07)
            assert np.all(values[:, :, 2][values[:, :, 4] == 1] <= 1e-08)
            assert np.all(values[:, :, 2][values[:, :, 4] == -1] >= -1e-08)
            for repetition in sorted({0, repetitions // 2, repetitions - 1}):
                sample = full_samples[repetition, :current_n]
                refined = continuous_ise_qr_minima(
                    sample,
                    h_min,
                    h_max,
                    sigma2=sigma2,
                    scan_points=2 * scan_points - 1,
                    spatial_points=2 * spatial_points,
                    half_width=half_width,
                )
                for method_index, method in enumerate(METHODS):
                    difference = abs(refined[method]["h"] - values[repetition, method_index, 0])
                    if difference > 1e-07:
                        raise ArithmeticError(f"Refined search changed {method} at n={current_n}.")
                    validation_rows.append(
                        {
                            "n": current_n,
                            "repetition": repetition,
                            "method": method,
                            "bandwidth_discrepancy": float(difference),
                        }
                    )
                alternative_h = np.r_[values[repetition, :, 0], h_min, h_max]
                independent_ise = exact_oracle_ise_values(sample, alternative_h, sigma2)
                if independent_ise[0] > independent_ise.min() + 1e-10:
                    raise ArithmeticError(
                        "ISE selector did not minimize the independent candidate scores."
                    )
                if abs(independent_ise[0] - values[repetition, 0, 1]) > 1e-10:
                    raise ArithmeticError("Analytic ISE disagrees with the shared exact helper.")
    validation_report = {
        "completed_samples": len(sample_sizes) * repetitions,
        "continuous_minimizers": len(sample_sizes) * repetitions * len(METHODS),
        "maximum_refinement_discrepancy": max(
            (row["bandwidth_discrepancy"] for row in validation_rows)
        ),
        "checks": [
            "interval bounds",
            "probability totals",
            "interior stationarity and curvature",
            "one-sided boundary derivative signs",
            "doubled search resolution",
            "independent exact ISE scores",
        ],
        "refinement_rows": validation_rows,
    }
    (run_directory / "validation.json").write_text(json.dumps(validation_report, indent=2) + "\n")
    print(f"Validated {validation_report['continuous_minimizers']} continuous minimizers.")
    print(
        f"Maximum change after search refinement: {validation_report['maximum_refinement_discrepancy']:.3e}"
    )
    return validation_report
