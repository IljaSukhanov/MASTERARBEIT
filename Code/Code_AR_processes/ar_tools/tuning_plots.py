"""Shared matrix figures for AR tuning experiments."""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np

from .plotting_functions import save_figure


def plot_tuning_matrices(
    ranking, scales, half_lives, *, metric, title, colorbar_label,
    filename, pdf_directory, image_directory, maximize=False, precision=7,
    secondary_metric=None,
):
    """Plot one step-rule grid, optionally adding a second score and boxes.

    When supplied, secondary_metric follows the same maximize/minimize rule.
    The primary score controls the colors. All tied best entries are boxed.
    """
    rules = ranking["step_rule"].unique()
    if len(rules) != 1 or rules[0] not in {"absolute", "diameter_normalized"}:
        raise ValueError("Supply exactly one supported step rule per tuning matrix.")
    step_rule = rules[0]
    matrices = {}
    fig, ax = plt.subplots(
        figsize=(9, max(5.0, (.48 if secondary_metric else .32) * len(scales) + 2)),
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
    if secondary_metric:
        secondary_values = rows.pivot(
            index="step_value", columns="half_life", values=secondary_metric,
        ).reindex(index=scales, columns=half_lives).to_numpy()
        secondary_best = rows[secondary_metric].max() if maximize else rows[secondary_metric].min()
        secondary_best_rows = rows.loc[rows[secondary_metric] == secondary_best]
    image = ax.imshow(values, origin="lower", aspect="auto",
                      cmap="Blues" if maximize else "viridis_r",
                      vmin=color_min, vmax=color_max)
    for i in range(len(scales)):
        for j in range(len(half_lives)):
            label = f"{values[i, j]:.{precision}f}"
            if secondary_metric:
                label = f"W {label}\nMAE {secondary_values[i, j]:.{precision}f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=8,
                    bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none", "pad": 1})
    for best in best_rows.to_dict("records"):
        x = np.flatnonzero(half_lives == best["half_life"])[0]
        y = np.flatnonzero(scales == best["step_value"])[0]
        if secondary_metric:
            ax.add_patch(Rectangle((x - .48, y - .48), .96, .96,
                                   fill=False, edgecolor="red", linewidth=2))
        else:
            ax.scatter(x, y, marker="s", s=850, facecolors="none", edgecolors="red", linewidths=2)
    if secondary_metric:
        primary_best_ids = set(best_rows["candidate_id"])
        for best in secondary_best_rows.to_dict("records"):
            x = np.flatnonzero(half_lives == best["half_life"])[0]
            y = np.flatnonzero(scales == best["step_value"])[0]
            inset = .07 if best["candidate_id"] in primary_best_ids else .02
            ax.add_patch(Rectangle((x - .5 + inset, y - .5 + inset),
                                   1 - 2 * inset, 1 - 2 * inset,
                                   fill=False, edgecolor="#198754", linewidth=2))
    ax.set(xticks=np.arange(len(half_lives)), xticklabels=[f"{h:g}" for h in half_lives],
           yticks=np.arange(len(scales)), yticklabels=[f"{s:g}" for s in scales],
           xlabel="Half-life H",
           ylabel=("Scale c (gamma_0 = c/L)" if step_rule == "diameter_normalized" else "Initial step gamma_0"),
           title=("Diameter-normalized steps" if step_rule == "diameter_normalized" else "Absolute steps"))
    fig.suptitle(title)
    if secondary_metric:
        fig.legend(handles=[Patch(fill=False, edgecolor="red", linewidth=2,
                                  label="Highest width count (W)"),
                            Patch(fill=False, edgecolor="#198754", linewidth=2,
                                  label="Highest coefficient MAE count (MAE)")],
                   loc="outside lower center", ncol=2)
    colorbar = fig.colorbar(image, ax=ax, label=colorbar_label, shrink=.8)
    if maximize:
        colorbar.set_ticks(np.unique(np.linspace(0, color_max, min(int(color_max) + 1, 8), dtype=int)))
    save_figure(fig, filename, pdf_directory=pdf_directory, image_directory=image_directory)
    return fig, matrices
