"""Overlay the original Perron curves and show every refined lambda as a dot.

Retain Figure 3.3's original sample and 201-node h grid. Evaluate five
Perron profiles at consecutive local lambda values with spacing 0.01.
The Perron overlay shows the competing peak region on one axes, with a
common vertical offset and magnification; each profile's maximum is not
subtracted. Matching colored dots indicate each selected maximum.
The bandwidth plot contains every one of the 1,001 evaluated lambda points
without an inset. All displayed plots are exported automatically as PDFs.
"""
from pathlib import Path
import json
import os
import sys

CODE_DIRECTORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE_DIRECTORY))
os.environ.setdefault("MPLCONFIGDIR", str(CODE_DIRECTORY / ".matplotlib-cache"))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from kde_tools.bandwidths import bandwidth_grid_linear_in_psi_beta
from kde_tools.fourier import beta2_divergence_matrix_fft
from kde_tools.pdf_export import enable_automatic_pdf_saving
from kde_tools.perron import PERRON_SOLVER_VERSION, perron_eigenpair_from_divergence


def compute_overlay_profiles(source_directory):
    original_summary = json.loads((source_directory / "figure_3_3_perron_profiles_near_jump_summary.json").read_text())
    sweep_settings = json.loads((source_directory / "figure_3_3_fine_lambda_grid_summary.json").read_text())
    original_profiles = pd.read_csv(source_directory / "figure_3_3_perron_profiles_near_jump.csv")
    lambda_values = np.array([17.85, 17.86, 17.87, 17.88, 17.89])
    h_grid = bandwidth_grid_linear_in_psi_beta(
        h_min=sweep_settings["bandwidth_min"], h_max=sweep_settings["bandwidth_max"],
        grid_size=sweep_settings["bandwidth_grid_size"])
    samples = np.random.default_rng(original_summary["sample_seed"]).normal(size=original_summary["sample_size"])
    records, summaries, discrepancies = [], [], []
    with threadpool_limits(limits=1):
        divergence = beta2_divergence_matrix_fft(samples, h_grid,
            half_width=sweep_settings["fft_half_width"], spatial_points=sweep_settings["fft_spatial_points"])
        for value in lambda_values:
            result = perron_eigenpair_from_divergence(divergence, value)
            for i, (h, weight) in enumerate(zip(h_grid, result.vector)):
                records.append({"lambda": float(value), "bandwidth_index_zero_based": i, "bandwidth": float(h),
                                "perron_component_sum_normalized": float(weight),
                                "is_selected_maximum": i == result.selected_index})
            summaries.append({"lambda": float(value), "selected_index": result.selected_index,
                              "selected_bandwidth": float(h_grid[result.selected_index]),
                              "relative_residual": result.relative_residual})
            previous = original_profiles.loc[original_profiles["lambda"] == value].sort_values("bandwidth_index_zero_based")
            if len(previous):
                difference = float(np.abs(result.vector - previous["perron_component_sum_normalized"].to_numpy()).max())
                assert difference < 1e-12
                discrepancies.append(difference)
    profiles = pd.DataFrame(records)
    profiles.to_csv(source_directory / "figure_3_3_perron_profiles_five_lambdas.csv", index=False)
    metadata = {"sample_seed": original_summary["sample_seed"], "sample_size": original_summary["sample_size"],
                "bandwidth_grid": "geomspace(0.01, 2, 201)", "normalization": "sum(v)=1",
                "lambda_values": lambda_values.tolist(), "local_lambda_step": 0.01,
                "previous_overlay_lambdas": [r["lambda"] for r in original_summary["profiles"]],
                "refined_bandwidth_plot_lambda_step": sweep_settings["lambda_step"],
                "perron_solver": PERRON_SOLVER_VERSION, "profiles": summaries,
                "maximum_discrepancy_from_archived_three_profiles": max(discrepancies)}
    (source_directory / "figure_3_3_perron_profiles_five_lambdas_summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Five local lambda values:", lambda_values.tolist())
    print("Selected bandwidths:", [r["selected_bandwidth"] for r in summaries])
    return profiles, metadata


def main(*, perron_only=False):
    source_directory = Path(__file__).resolve().parent
    image_directory = CODE_DIRECTORY / "outputs/images/spectral_lambda_jump_study"
    image_directory.mkdir(parents=True, exist_ok=True)
    enable_automatic_pdf_saving("spectral_lambda_jump_study")

    profiles, profile_summary = compute_overlay_profiles(source_directory)
    baseline, magnification = 0.005550, 1e6
    colors = ["#d96927", "#2563a6", "#9153a3", "#23846d", "#bf3e58"]
    styles = ["-", "-", "--", "-.", "-"]
    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    handles = []
    for row, color, style in zip(profile_summary["profiles"], colors, styles):
        values = profiles.loc[profiles["lambda"] == row["lambda"]].sort_values("bandwidth_index_zero_based")
        assert len(values) == 201
        h = values["bandwidth"].to_numpy()
        v = values["perron_component_sum_normalized"].to_numpy()
        y = magnification * (v - baseline)
        shown = (h >= 0.10) & (h <= 0.22)
        curve, = ax.plot(h[shown], y[shown], color=color, linestyle=style, linewidth=2,
                        label=rf"$\lambda={row['lambda']:.2f}$; selected $h={row['selected_bandwidth']:.6f}$")
        handles.append(curve)
        winner = row["selected_index"]
        ax.scatter(h[winner], y[winner], marker="o", s=55, color=color, linewidths=0, zorder=5)
    handles.append(Line2D([0], [0], color="#444444", marker="o", linestyle="none", markersize=6,
                          label="Selected maximum (matching curve color)"))
    ax.set(xlim=(0.10, 0.22), xlabel="Bandwidth h (original grid; competing peak region)",
           ylabel=r"$10^6\,(v_i-0.005550)$",
           title="Figure 3.3: Perron curves together around the switch")
    ax.grid(alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=handles, loc="upper center", ncol=1, framealpha=0.96, fontsize=9)
    # Leave a clear strip for the legend, above every peak.
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + 0.85)
    path = image_directory / "figure_3_3_perron_curves_overlaid.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    if perron_only:
        print(f"Perron overlay: {path}")
        return [path]

    refined = pd.read_csv(source_directory / "figure_3_3_fine_lambda_grid_10_30.csv")
    assert len(refined) == 1001 and refined["lambda"].is_unique
    lambdas = refined["lambda"].to_numpy()
    bandwidths = refined["selected_bandwidth"].to_numpy()
    assert np.allclose(np.diff(lambdas), 0.02)
    crossing = json.loads((CODE_DIRECTORY / "outputs/data/spectral_lambda_switch_distribution/59e74dc1d05b/figure_3_3_control.json").read_text())
    critical = next(e["lambda_crossing"] for e in crossing["events"] if e["status"] == "confirmed")
    fig, ax = plt.subplots(figsize=(11.5, 6.2), constrained_layout=True)
    points = ax.scatter(lambdas, bandwidths, s=6, marker="o", color="#2563a6", linewidths=0,
                        label=r"One dot per evaluated $\lambda$: 1,001 points, $\Delta\lambda=0.02$")
    assert len(points.get_offsets()) == 1001
    ax.axvline(critical, color="#d96927", lw=1.2, ls="--",
               label=rf"Refined peak crossing $\lambda_c\approx{critical:.5f}$")
    ax.set(xlim=(10, 30), ylim=(0.089, 0.222), xlabel=r"Similarity parameter $\lambda$",
           ylabel="Chosen bandwidth h", title="Figure 3.3: refined lambda grid with a dot at every lambda")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.96)
    ax.grid(alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    path2 = image_directory / "figure_3_3_fine_lambda_grid_every_point.png"
    fig.savefig(path2, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Perron overlay: {path}")
    print(f"All 1,001 lambda dots: {path2}")
    return [path, path2]


if __name__ == "__main__":
    main()
