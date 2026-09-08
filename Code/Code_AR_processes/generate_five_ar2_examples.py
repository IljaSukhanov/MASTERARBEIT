"""Generate five reproducible AR(2) pipeline plots using shared utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__:
    from .functions import (
        jones_ar_coefficients,
        simulate_uniform_ar,
        sliding_window_point_cloud,
        batched_geometric_width_curves,
        theoretical_ar_width_in_direction,
        population_minimizing_ar2_direction,
        horizontal_tail_lad_selector,
    )
else:
    from functions import (
        jones_ar_coefficients,
        simulate_uniform_ar,
        sliding_window_point_cloud,
        batched_geometric_width_curves,
        theoretical_ar_width_in_direction,
        population_minimizing_ar2_direction,
        horizontal_tail_lad_selector,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output" / "figures" / "five_ar2_examples"
CSV_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "outputs"
    / "csv"
    / "order_selection"
    / "ar2"
)


def generate_process(seed: int, replication: int) -> dict[str, object]:
    k_values = np.arange(1, 11)
    generator = np.random.default_rng(np.random.SeedSequence([seed, replication, 0]))
    phi = jones_ar_coefficients(2, rng=generator)
    series = simulate_uniform_ar(phi, 2_000, 1_000, 1.0, rng=generator)
    _, widths, _, empirical_directions = batched_geometric_width_curves(
        series,
        k_values,
        seed=seed,
        replication=replication,
        starts=3,
        iterations=500,
        initial_step=0.2,
        geometric_decay=0.99,
        return_directions=True,
    )
    selection = horizontal_tail_lad_selector(k_values, widths)
    population_width = 2.0 / np.linalg.norm(np.r_[1.0, -phi])
    population_at_empirical_min = np.asarray(
        [
            theoretical_ar_width_in_direction(phi, direction, half_width=1.0)
            for direction in empirical_directions
        ]
    )
    empirical_at_population_min = []
    for k in k_values:
        direction = population_minimizing_ar2_direction(phi, int(k))
        cloud = sliding_window_point_cloud(series, int(k))
        empirical_at_population_min.append(np.ptp(cloud @ direction))

    return {
        "replication": replication,
        "phi": phi,
        "k_values": k_values,
        "widths": widths,
        "selection": selection,
        "population_width": population_width,
        "population_at_empirical_min": population_at_empirical_min,
        "empirical_at_population_min": np.asarray(empirical_at_population_min),
    }


def draw_plot(axis: plt.Axes, result: dict[str, object], *, compact: bool = False) -> None:
    k_values = result["k_values"]
    widths = result["widths"]
    selection = result["selection"]
    phi = result["phi"]
    population_width = result["population_width"]

    axis.plot(
        k_values,
        widths,
        color="#1764ab",
        marker="o",
        markersize=5.8 if compact else 7.5,
        linewidth=2.0 if compact else 2.3,
        zorder=4,
        label=r"empirical $\widehat m_k$",
    )
    axis.plot(
        k_values[2:],
        np.full(len(k_values) - 2, population_width),
        color="black",
        linestyle="--",
        linewidth=2.0 if compact else 2.5,
        zorder=2,
        label=r"population $m_k=m^*$ for $k\geq3$",
    )
    axis.plot(
        k_values,
        result["population_at_empirical_min"],
        color="#e66101",
        marker="s",
        markersize=4.8 if compact else 5.8,
        linestyle="--",
        linewidth=1.8 if compact else 2.0,
        zorder=3,
        label=r"population at empirical minimizer $w_k(\widehat u_k)$",
    )
    q_hat = selection["selected_q"]
    p_hat = selection["selected_order"]
    fitted_values = selection["fitted_values"]
    affine_mask = k_values <= q_hat
    constant_mask = k_values >= q_hat
    axis.plot(
        k_values[affine_mask],
        fitted_values[affine_mask],
        color="#6a3d9a",
        linestyle="-",
        linewidth=2.4 if compact else 3.0,
        zorder=5,
        label="selected affine LAD fit",
    )
    axis.plot(
        k_values[constant_mask],
        fitted_values[constant_mask],
        color="#6a3d9a",
        linestyle=":",
        linewidth=2.4 if compact else 3.0,
        zorder=5,
        label="selected constant LAD fit",
    )
    axis.axvline(q_hat, color="#c51b2b", linestyle="-.", linewidth=1.7, zorder=1)
    axis.scatter(
        [q_hat],
        [selection["level"]],
        marker="o",
        s=105 if compact else 150,
        color="#c51b2b",
        edgecolor="none",
        zorder=7,
    )
    annotation = (
        rf"$\phi_1={phi[0]:.3f},\ \phi_2={phi[1]:.3f}$" + "\n"
        + rf"$\widehat q={q_hat},\ \widehat p={p_hat}$"
    )
    axis.text(
        0.97,
        0.95,
        annotation,
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9 if compact else 11,
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.94,
        },
    )
    axis.set(
        title=f"AR(2) process {result['replication']}",
        xlabel=r"embedding dimension $k$",
        ylabel="directional width",
        xticks=k_values,
    )
    axis.grid(axis="y", color="0.88", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed = 20260819
    results = [generate_process(seed, replication) for replication in range(1, 6)]

    summary_rows = []
    for result in results:
        figure, axis = plt.subplots(figsize=(10.5, 6.4), constrained_layout=True)
        draw_plot(axis, result)
        axis.set_title(f"AR(2) process {result['replication']}: population and empirical widths")
        axis.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=3,
            frameon=False,
            fontsize=9,
        )
        path = OUTPUT_DIR / f"ar2_process_{result['replication']}.png"
        figure.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(figure)
        selection = result["selection"]
        summary_rows.append(
            {
                "process": result["replication"],
                "phi_1": result["phi"][0],
                "phi_2": result["phi"][1],
                "selected_q": selection["selected_q"],
                "selected_p": selection["selected_order"],
                "population_width": result["population_width"],
            }
        )

    overview, axes = plt.subplots(3, 2, figsize=(14, 15), constrained_layout=True)
    for axis, result in zip(axes.flat, results):
        draw_plot(axis, result, compact=True)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    overview.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    overview.suptitle(
        r"Five stationary AR(2) processes ($T=2{,}000$, $K=10$)",
        fontsize=18,
    )
    overview.savefig(OUTPUT_DIR / "five_ar2_processes_overview.png", dpi=200, bbox_inches="tight")
    plt.close(overview)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(CSV_OUTPUT_DIR / "five_ar2_examples_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
