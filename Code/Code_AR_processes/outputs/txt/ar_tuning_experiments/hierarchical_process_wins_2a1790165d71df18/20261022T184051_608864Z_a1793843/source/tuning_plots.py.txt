"""Shared matrix figures for AR tuning experiments."""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np

from .plotting_functions import save_figure


ITERATION_COUNT_COLOR = "#c75f00"
COEFFICIENT_MAE_COLOR = "#198754"


def annotate_tuning_cell(
    ax, x, y, label, iteration_limit_count=None, *, mean_coefficient_mae=None,
    mae_precision=3,
):
    """Show width, orange iteration counts and green mean MAE on separate lines."""
    background = {"facecolor": "white", "alpha": .8, "edgecolor": "none", "pad": 1}
    lines = [(label, "black")]
    if iteration_limit_count is not None:
        lines.append((f"MAX {int(iteration_limit_count)}", ITERATION_COUNT_COLOR))
    if mean_coefficient_mae is not None:
        mae_label = (f"MAE {mean_coefficient_mae:.{mae_precision}g}"
                     if np.isfinite(mean_coefficient_mae) else "MAE n/a")
        lines.append((mae_label, COEFFICIENT_MAE_COLOR))
    extent = .28 if len(lines) == 3 else .16
    offsets = np.linspace(extent, -extent, len(lines)) if len(lines) > 1 else [0]
    for offset, (text, color) in zip(offsets, lines):
        ax.text(x, y + offset, text, ha="center", va="center", fontsize=8,
                color=color, bbox=background)


def plot_tuning_matrices(
    ranking, scales, half_lives, *, metric, title, colorbar_label,
    filename, pdf_directory, image_directory, maximize=False, precision=7,
    iteration_limit_metric=None, coefficient_mae_metric=None, mae_precision=3,
):
    """Plot one step-rule grid with optional iteration counts and mean MAE.

    The primary score controls background colors and red winner boxes.
    Green boxes mark every exact minimum of the unrounded finite mean MAE
    within this matrix; inset them when a cell also has a red winner box.
    Full-iteration counts and coefficient MAE never affect width selection.
    """
    rules = ranking["step_rule"].unique()
    if len(rules) != 1 or rules[0] not in {"absolute", "diameter_normalized"}:
        raise ValueError("Supply exactly one supported step rule per tuning matrix.")
    step_rule = rules[0]
    matrices = {}
    row_height = .60 if coefficient_mae_metric else (.48 if iteration_limit_metric else .32)
    fig, ax = plt.subplots(
        figsize=(9, max(5.0, row_height * len(scales) + 2.5)),
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
    if coefficient_mae_metric:
        coefficient_mae_values = rows.pivot(
            index="step_value", columns="half_life", values=coefficient_mae_metric,
        ).reindex(index=scales, columns=half_lives).to_numpy()
    image = ax.imshow(values, origin="lower", aspect="auto",
                      cmap="Blues" if maximize else "viridis_r",
                      vmin=color_min, vmax=color_max)
    for i in range(len(scales)):
        for j in range(len(half_lives)):
            label = f"{values[i, j]:.{precision}f}"
            if iteration_limit_metric or coefficient_mae_metric:
                label = f"W {label}"
            annotate_tuning_cell(
                ax, j, i, label,
                iteration_limit_values[i, j] if iteration_limit_metric else None,
                mean_coefficient_mae=(coefficient_mae_values[i, j]
                                      if coefficient_mae_metric else None),
                mae_precision=mae_precision,
            )
    for best in best_rows.to_dict("records"):
        x = np.flatnonzero(half_lives == best["half_life"])[0]
        y = np.flatnonzero(scales == best["step_value"])[0]
        ax.add_patch(Rectangle((x - .48, y - .48), .96, .96,
                               fill=False, edgecolor="red", linewidth=2))
    if coefficient_mae_metric:
        finite_mae = rows.loc[np.isfinite(rows[coefficient_mae_metric])]
        mae_best_rows = finite_mae.loc[
            finite_mae[coefficient_mae_metric].eq(finite_mae[coefficient_mae_metric].min())
        ]
        width_best_ids = set(best_rows["candidate_id"])
        for best in mae_best_rows.to_dict("records"):
            x = np.flatnonzero(half_lives == best["half_life"])[0]
            y = np.flatnonzero(scales == best["step_value"])[0]
            extent = .40 if best["candidate_id"] in width_best_ids else .48
            ax.add_patch(Rectangle((x - extent, y - extent), 2 * extent, 2 * extent,
                                   fill=False, edgecolor=COEFFICIENT_MAE_COLOR, linewidth=2))
    ax.set(xticks=np.arange(len(half_lives)), xticklabels=[f"{h:g}" for h in half_lives],
           yticks=np.arange(len(scales)), yticklabels=[f"{s:g}" for s in scales],
           xlabel="Half-life H",
           ylabel=("Scale c (gamma_0 = c/L)" if step_rule == "diameter_normalized" else "Initial step gamma_0"),
           title=("Diameter-normalized steps" if step_rule == "diameter_normalized" else "Absolute steps"))
    fig.suptitle(title)
    if iteration_limit_metric or coefficient_mae_metric:
        legend = [Patch(fill=False, edgecolor="red", linewidth=2,
                        label="Highest width co-winner count (W)")]
        if iteration_limit_metric:
            legend.append(Patch(facecolor=ITERATION_COUNT_COLOR, edgecolor="none",
                                label="MAX: runs using all maximum iterations"))
        if coefficient_mae_metric:
            legend.append(Patch(fill=False, edgecolor=COEFFICIENT_MAE_COLOR, linewidth=2,
                                label="Lowest mean coefficient MAE in this matrix"))
        fig.legend(handles=legend, loc="outside lower center", ncol=1)
    colorbar = fig.colorbar(image, ax=ax, label=colorbar_label, shrink=.8)
    if maximize:
        colorbar.set_ticks(np.unique(np.linspace(0, color_max, min(int(color_max) + 1, 8), dtype=int)))
    save_figure(fig, filename, pdf_directory=pdf_directory, image_directory=image_directory)
    return fig, matrices
