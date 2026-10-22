"""Shared matrix figures for AR tuning experiments."""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np

from .plotting_functions import save_figure


ITERATION_COUNT_COLOR = "#c75f00"


def annotate_tuning_cell(ax, x, y, label, iteration_limit_count=None):
    """Show the objective score above an optional orange full-budget count."""
    background = {"facecolor": "white", "alpha": .8, "edgecolor": "none", "pad": 1}
    ax.text(x, y + (.16 if iteration_limit_count is not None else 0), label,
            ha="center", va="center", fontsize=8, color="black", bbox=background)
    if iteration_limit_count is not None:
        ax.text(x, y - .16, f"MAX {int(iteration_limit_count)}",
                ha="center", va="center", fontsize=8,
                color=ITERATION_COUNT_COLOR, bbox=background)


def plot_tuning_matrices(
    ranking, scales, half_lives, *, metric, title, colorbar_label,
    filename, pdf_directory, image_directory, maximize=False, precision=7,
    iteration_limit_metric=None,
):
    """Plot one step-rule grid with optional orange full-iteration counts.

    The primary score alone controls colors and winner boxes.
    Full-iteration counts are informational and never affect selection.
    """
    rules = ranking["step_rule"].unique()
    if len(rules) != 1 or rules[0] not in {"absolute", "diameter_normalized"}:
        raise ValueError("Supply exactly one supported step rule per tuning matrix.")
    step_rule = rules[0]
    matrices = {}
    fig, ax = plt.subplots(
        figsize=(9, max(5.0, (.48 if iteration_limit_metric else .32) * len(scales) + 2)),
        constrained_layout=True,
    )
    color_min = 0 if maximize else ranking[metric].min()
    color_max = max(1, ranking[metric].max()) if maximize else ranking[metric].max()
    rows = ranking
    best_value = rows[metric].max() if maximize else rows[metric].min()
    best_rows = rows.loc[rows[metric] == best_value]
    matrix = rows.pivot(index="step_value", columns="half_life", values=metric).reindex(
        index=scales, columns=half_lives,
    )
    matrices[step_rule] = matrix
    values = matrix.to_numpy()
    if iteration_limit_metric:
        iteration_limit_values = rows.pivot(
            index="step_value", columns="half_life", values=iteration_limit_metric,
        ).reindex(index=scales, columns=half_lives).to_numpy()
    image = ax.imshow(values, origin="lower", aspect="auto",
                      cmap="Blues" if maximize else "viridis_r",
                      vmin=color_min, vmax=color_max)
    for i in range(len(scales)):
        for j in range(len(half_lives)):
            label = f"{values[i, j]:.{precision}f}"
            if iteration_limit_metric:
                label = f"W {label}"
            annotate_tuning_cell(
                ax, j, i, label,
                iteration_limit_values[i, j] if iteration_limit_metric else None,
            )
    for best in best_rows.to_dict("records"):
        x = np.flatnonzero(half_lives == best["half_life"])[0]
        y = np.flatnonzero(scales == best["step_value"])[0]
        ax.add_patch(Rectangle((x - .48, y - .48), .96, .96,
                               fill=False, edgecolor="red", linewidth=2))
    ax.set(xticks=np.arange(len(half_lives)), xticklabels=[f"{h:g}" for h in half_lives],
           yticks=np.arange(len(scales)), yticklabels=[f"{s:g}" for s in scales],
           xlabel="Half-life H",
           ylabel=("Scale c (gamma_0 = c/L)" if step_rule == "diameter_normalized" else "Initial step gamma_0"),
           title=("Diameter-normalized steps" if step_rule == "diameter_normalized" else "Absolute steps"))
    fig.suptitle(title)
    if iteration_limit_metric:
        fig.legend(handles=[
            Patch(fill=False, edgecolor="red", linewidth=2,
                  label="Highest width co-winner count (W)"),
            Patch(facecolor=ITERATION_COUNT_COLOR, edgecolor="none",
                  label="MAX: runs using all maximum iterations"),
        ],
                   loc="outside lower center", ncol=1)
    colorbar = fig.colorbar(image, ax=ax, label=colorbar_label, shrink=.8)
    if maximize:
        colorbar.set_ticks(np.unique(np.linspace(0, color_max, min(int(color_max) + 1, 8), dtype=int)))
    save_figure(fig, filename, pdf_directory=pdf_directory, image_directory=image_directory)
    return fig, matrices
