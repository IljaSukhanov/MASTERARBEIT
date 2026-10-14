"""Plot the original Figure 3.3 Perron profiles around the lambda jump.

Top: sum-normalized Perron vectors on the complete original bandwidth grid.
Bottom: three separate views of the competing peak regions, after subtracting
each vector's maximum and multiplying by 1e6 to reveal the shallow valley.
Peak-height differences are reported separately at their actual 1e-9 scale.
"""

from pathlib import Path
import csv
import json
import os
import sys

CODE_DIRECTORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE_DIRECTORY))
os.environ.setdefault("MPLCONFIGDIR", str(CODE_DIRECTORY / ".matplotlib-cache"))

import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

from kde_tools.bandwidths import bandwidth_grid_linear_in_psi_beta
from kde_tools.fourier import beta2_divergence_matrix_fft
from kde_tools.pdf_export import enable_automatic_pdf_saving
from kde_tools.perron import PERRON_SOLVER_VERSION, perron_eigenpair_from_divergence


def main():
    analysis_directory = Path(__file__).resolve().parent
    image_directory = CODE_DIRECTORY / "outputs" / "images" / "spectral_lambda_jump_study"
    image_directory.mkdir(parents=True, exist_ok=True)
    lambdas = np.array([17.86, 17.87, 17.88])
    bandwidth_grid = bandwidth_grid_linear_in_psi_beta(grid_size=201)
    samples = np.random.default_rng(20260823).normal(size=200)
    vectors, summaries = [], []
    lower_region = np.flatnonzero(bandwidth_grid < 0.16)
    upper_region = np.flatnonzero(bandwidth_grid >= 0.16)
    with threadpool_limits(limits=1):
        divergence = beta2_divergence_matrix_fft(
            samples, bandwidth_grid, half_width=12.0, spatial_points=8192
        )
        for value in lambdas:
            result = perron_eigenpair_from_divergence(divergence, value)
            vector = result.vector
            lower = int(lower_region[np.argmax(vector[lower_region])])
            upper = int(upper_region[np.argmax(vector[upper_region])])
            vectors.append(vector)
            summaries.append({
                "lambda": float(value), "selected_index": result.selected_index,
                "selected_bandwidth": float(bandwidth_grid[result.selected_index]),
                "lower_peak_index": lower, "upper_peak_index": upper,
                "lower_peak_bandwidth": float(bandwidth_grid[lower]),
                "upper_peak_bandwidth": float(bandwidth_grid[upper]),
                "lower_peak_weight": float(vector[lower]),
                "upper_peak_weight": float(vector[upper]),
                "lower_minus_upper_peak_height": float(vector[lower] - vector[upper]),
                "relative_residual": result.relative_residual,
                "relative_gap": result.relative_gap,
            })
    vectors = np.array(vectors)
    assert [row["selected_index"] for row in summaries] == [91, 91, 114]
    assert np.allclose(vectors.sum(axis=1), 1.0, rtol=0, atol=1e-14)

    data_path = analysis_directory / "figure_3_3_perron_profiles_near_jump.csv"
    with data_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["lambda", "bandwidth_index_zero_based", "bandwidth",
                         "perron_component_sum_normalized", "is_selected_maximum"])
        for value, vector, summary in zip(lambdas, vectors, summaries):
            writer.writerows(
                (float(value), i, float(h), float(weight), i == summary["selected_index"])
                for i, (h, weight) in enumerate(zip(bandwidth_grid, vector))
            )
    metadata = {
        "sample_seed": 20260823, "sample_size": 200,
        "bandwidth_grid": "geomspace(0.01, 2, 201)",
        "normalization": "sum(v)=1", "peak_region_split_h": 0.16,
        "perron_solver": PERRON_SOLVER_VERSION, "profiles": summaries,
    }
    (analysis_directory / "figure_3_3_perron_profiles_near_jump_summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    enable_automatic_pdf_saving("figure_3_3_perron_peak_profiles")
    colors = ["#2a6fbb", "#8d55a3", "#32866a"]
    styles = ["-", "--", ":"]
    fig = plt.figure(figsize=(12, 7.5))
    layout = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])
    full = fig.add_subplot(layout[0, :])
    zooms = [fig.add_subplot(layout[1, j]) for j in range(3)]
    fig.suptitle("Figure 3.3: Perron profiles across the bandwidth switch", fontsize=15)

    for value, vector, summary, color, style in zip(lambdas, vectors, summaries, colors, styles):
        full.plot(bandwidth_grid, vector, color=color, linestyle=style, linewidth=1.9,
                  label=rf"$\lambda={value:.2f}$: selected $h={summary['selected_bandwidth']:.6f}$")
        selected = summary["selected_index"]
        full.scatter(bandwidth_grid[selected], vector[selected], s=65, marker="*",
                     color=color, zorder=5)
    full.set_xscale("log")
    full.set_xlim(bandwidth_grid[0], bandwidth_grid[-1])
    full.set_ylim(float(vectors.min()) - 0.00012, float(vectors.max()) + 0.00028)
    full.set_xlabel("Bandwidth h (original 201-point grid)")
    full.set_ylabel(r"Perron component $v_i$; $\sum_i v_i=1$")
    full.set_title("Full profiles; stars mark the selected maxima", fontsize=11, loc="left")
    full.grid(alpha=0.18)
    full.legend(loc="lower center", fontsize=9, framealpha=0.97)

    zoom_mask = (bandwidth_grid >= 0.10) & (bandwidth_grid <= 0.22)
    deficits = 1e6 * (vectors - vectors.max(axis=1, keepdims=True))
    local_deficit_min = float(deficits[:, zoom_mask].min())
    common_ymin = local_deficit_min * 1.35
    for ax, value, vector, deficit, summary, color in zip(
        zooms, lambdas, vectors, deficits, summaries, colors
    ):
        ax.plot(bandwidth_grid[zoom_mask], deficit[zoom_mask], color=color,
                linewidth=1.7, marker="o", markersize=2.5)
        winner = summary["selected_index"]
        other = summary["upper_peak_index"] if winner == summary["lower_peak_index"] else summary["lower_peak_index"]
        ax.scatter(bandwidth_grid[other], deficit[other], s=58, facecolors="white",
                   edgecolors=color, linewidths=1.3, zorder=4)
        ax.scatter(bandwidth_grid[winner], deficit[winner], s=100, marker="*",
                   color=color, zorder=5)
        ax.axhline(0, color="#777777", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.set_xlim(0.10, 0.22)
        ax.set_ylim(common_ymin, 0.09)
        ax.set_xticks([0.10, 0.14, 0.18, 0.22])
        ax.set_xlabel("Bandwidth h")
        ax.set_title(rf"$\lambda={value:.2f}$; winner $h={summary['selected_bandwidth']:.6f}$", fontsize=10)
        ax.grid(alpha=0.16)
        gap = summary["lower_minus_upper_peak_height"] * 1e9
        ax.text(0.5, 0.07, f"Lower peak minus upper peak:\n{gap:+.3f} × 10⁻⁹",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9})
    zooms[0].set_ylabel(r"$10^6\,(v_i-\max_j v_j)$")
    for ax in zooms[1:]:
        ax.tick_params(labelleft=False)
    fig.text(0.5, 0.02,
             "Bottom: competing peak regions, with each profile's maximum subtracted. "
             "Star = winner; hollow circle = competing peak.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.055, 1, 0.965), h_pad=2.3, w_pad=1.8)
    image_path = image_directory / "figure_3_3_perron_profiles_near_jump.png"
    fig.savefig(image_path, dpi=160, bbox_inches="tight")
    plt.show()  # Shared automatic exporter also saves this displayed figure as PDF.
    plt.close(fig)
    print(json.dumps(metadata, indent=2))
    print(f"Image: {image_path}")
    print(f"Data: {data_path}")


if __name__ == "__main__":
    main()
