"""Recompute Figure 3.3 on 1,001 lambda values from 10 to 30.

The original normal sample (seed 20260823, n=200), 201 geometric
bandwidth candidates on [0.01, 2], FFT settings, and shared Perron
selection convention are unchanged. Every point is a new numerical solve.
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
    original_directory = analysis_directory.parent / "perron_figures_3_1_to_3_4"
    with (original_directory / "figure_3_3_comparison.csv").open() as stream:
        original_rows = list(csv.DictReader(stream))
    coarse_lambdas = np.array([float(row["position"]) for row in original_rows])
    coarse_indices = np.array([int(row["old_index"]) for row in original_rows])
    coarse_bandwidths = np.array([float(row["old_bandwidth"]) for row in original_rows])
    inside = (coarse_lambdas >= 10) & (coarse_lambdas <= 30)

    samples = np.random.default_rng(20260823).normal(size=200)
    bandwidth_grid = bandwidth_grid_linear_in_psi_beta(grid_size=201)
    lambdas = np.linspace(10.0, 30.0, 1001)
    indices = np.empty(len(lambdas), dtype=np.int32)
    residuals = np.empty(len(lambdas))
    relative_gaps = np.empty(len(lambdas))
    with threadpool_limits(limits=1):
        divergence = beta2_divergence_matrix_fft(
            samples, bandwidth_grid, half_width=12.0, spatial_points=8192
        )
        # Independently confirm the sample/settings against original grid points.
        for value, original_index in zip(coarse_lambdas[inside], coarse_indices[inside]):
            result = perron_eigenpair_from_divergence(divergence, value)
            assert result.selected_index == original_index
        for k, value in enumerate(lambdas):
            result = perron_eigenpair_from_divergence(divergence, value)
            indices[k] = result.selected_index
            residuals[k] = result.relative_residual
            relative_gaps[k] = result.relative_gap

    bandwidths = bandwidth_grid[indices]
    steps = np.diff(indices.astype(np.int64))
    jump_index = int(np.argmax(steps))
    assert steps[jump_index] > 0
    data_path = analysis_directory / "figure_3_3_fine_lambda_grid_10_30.csv"
    with data_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["lambda", "selected_index", "selected_bandwidth",
                         "relative_residual", "relative_gap"])
        writer.writerows(zip(lambdas, indices, bandwidths, residuals, relative_gaps))
    summary = {
        "source_figure": "3.3", "sample_seed": 20260823, "sample_size": 200,
        "bandwidth_grid_size": 201, "bandwidth_min": 0.01, "bandwidth_max": 2.0,
        "fft_half_width": 12.0, "fft_spatial_points": 8192,
        "lambda_grid": "linspace(10, 30, 1001)", "lambda_step": 0.02,
        "perron_solver": PERRON_SOLVER_VERSION,
        "original_grid_points_verified": int(inside.sum()),
        "largest_upward_pair": lambdas[jump_index:jump_index + 2].tolist(),
        "bandwidths_at_largest_upward_pair": bandwidths[jump_index:jump_index + 2].tolist(),
        "largest_upward_grid_steps": int(steps[jump_index]),
        "maximum_relative_residual": float(residuals.max()),
        "minimum_relative_gap": float(relative_gaps.min()),
    }
    (analysis_directory / "figure_3_3_fine_lambda_grid_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    enable_automatic_pdf_saving("figure_3_3_fine_lambda_grid")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.step(lambdas, bandwidths, where="post", color="#222222", linewidth=1.8,
            label=r"Recomputed: 1,001 lambda values, $\Delta\lambda=0.02$")
    ax.scatter(coarse_lambdas[inside], coarse_bandwidths[inside], facecolors="white",
               edgecolors="#2a6fbb", linewidths=1.5, s=55, zorder=4,
               label="Original Figure 3.3 grid points")
    ax.axvspan(lambdas[jump_index], lambdas[jump_index + 1],
               color="#e69536", alpha=0.3, zorder=2)
    ax.scatter(lambdas[jump_index:jump_index + 2],
               bandwidths[jump_index:jump_index + 2], color="#c76b00", s=45, zorder=5)
    ax.set_xlim(10, 30)
    ax.set_ylim(float(bandwidths.min()) - 0.008, float(bandwidths.max()) + 0.012)
    ax.set_xlabel(r"Similarity scale $\lambda$")
    ax.set_ylabel("Chosen bandwidth h")
    ax.set_title("Figure 3.3: finer lambda grid from 10 to 30")
    ax.grid(alpha=0.18)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.96)
    ax.text(0.98, 0.05,
            f"Largest upward pair: ({lambdas[jump_index]:.2f}, {lambdas[jump_index + 1]:.2f})\n"
            f"h: {bandwidths[jump_index]:.6f} to {bandwidths[jump_index + 1]:.6f}\n"
            f"{int(steps[jump_index])} bandwidth-grid steps",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.94})
    fig.tight_layout()
    image_directory = CODE_DIRECTORY / "outputs" / "images" / "spectral_lambda_jump_study"
    image_directory.mkdir(parents=True, exist_ok=True)
    image_path = image_directory / "figure_3_3_fine_lambda_grid_10_30.png"
    fig.savefig(image_path, dpi=160, bbox_inches="tight")
    plt.show()  # Shared automatic exporter saves the displayed figure as PDF.
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    print(f"Image: {image_path}")
    print(f"Data: {data_path}")


if __name__ == "__main__":
    main()
