"""Shared matrix figures for paired AR tuning experiments."""

import matplotlib.pyplot as plt
import numpy as np

from .plotting_functions import save_figure


def plot_tuning_matrices(
    ranking, scales, half_lives, *, metric, title, colorbar_label,
    filename, pdf_directory, image_directory, maximize=False, precision=7,
):
    """Plot both step rules, highlighting all per-rule best matrix entries."""
    matrices = {}
    fig, axes = plt.subplots(
        1, 2, figsize=(14, max(5.0, .32 * len(scales) + 2)), constrained_layout=True,
    )
    color_min = 0 if maximize else ranking[metric].min()
    color_max = max(1, ranking[metric].max()) if maximize else ranking[metric].max()
    for ax, rule in zip(axes, ("absolute", "diameter_normalized")):
        rows = ranking.loc[ranking["step_rule"] == rule]
        best_value = rows[metric].max() if maximize else rows[metric].min()
        best_rows = rows.loc[rows[metric] == best_value]
        matrix = rows.pivot(index="step_value", columns="half_life", values=metric).reindex(
            index=scales, columns=half_lives,
        )
        matrices[rule] = matrix
        values = matrix.to_numpy()
        image = ax.imshow(values, origin="lower", aspect="auto",
                          cmap="Blues" if maximize else "viridis_r",
                          vmin=color_min, vmax=color_max)
        for i in range(len(scales)):
            for j in range(len(half_lives)):
                ax.text(j, i, f"{values[i, j]:.{precision}f}", ha="center", va="center", fontsize=8,
                        bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none", "pad": 1})
        for best in best_rows.to_dict("records"):
            ax.scatter(np.flatnonzero(half_lives == best["half_life"])[0],
                       np.flatnonzero(scales == best["step_value"])[0],
                       marker="s", s=850, facecolors="none", edgecolors="red", linewidths=2)
        ax.set(xticks=np.arange(len(half_lives)), xticklabels=[f"{h:g}" for h in half_lives],
               yticks=np.arange(len(scales)), yticklabels=[f"{s:g}" for s in scales],
               xlabel="Half-life H", ylabel="Initial step gamma_0" if rule == "absolute" else "Scale c",
               title="Absolute steps" if rule == "absolute" else "Diameter-normalized steps")
    fig.suptitle(title)
    colorbar = fig.colorbar(image, ax=axes, label=colorbar_label, shrink=.8)
    if maximize:
        colorbar.set_ticks(np.unique(np.linspace(0, color_max, min(int(color_max) + 1, 8), dtype=int)))
    save_figure(fig, filename, pdf_directory=pdf_directory, image_directory=image_directory)
    return fig, matrices
