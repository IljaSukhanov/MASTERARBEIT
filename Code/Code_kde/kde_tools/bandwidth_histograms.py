"""Grouped grid-position histograms in the thesis comparison-figure style."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np

from .bandwidths import gaussian_mise_oracle_bandwidth


DISTRIBUTION_NAMES = (
    "spectral selections", "ISE-optimal grid bandwidths", "Q-hat minimizers",
)
LOCALIZED_SPECTRAL_LABEL = r"spectral ($\lambda_N=N^{3/2}$)"


def bandwidth_distribution_names(lambda_value=1.0, include_localized=False):
    if not include_localized:
        return DISTRIBUTION_NAMES
    return (rf"spectral ($\lambda={lambda_value:g}$)",
            *DISTRIBUTION_NAMES[1:], LOCALIZED_SPECTRAL_LABEL)


def grouped_grid_counts(counts, grid_size, repetitions, positions_per_bin=5):
    """Group counts after selection, preserving every observation's mass."""
    values = np.asarray(counts)
    if (values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != grid_size
            or not np.issubdtype(values.dtype, np.integer)
            or np.any(values < 0) or np.any(values.sum(axis=1) != repetitions)):
        raise ValueError("Expected nonnegative integer count rows summing to M.")
    if (isinstance(positions_per_bin, bool)
            or not isinstance(positions_per_bin, (int, np.integer))
            or positions_per_bin < 1):
        raise ValueError("positions_per_bin must be a positive integer.")
    starts = np.arange(0, grid_size, positions_per_bin)
    stops = np.minimum(starts + positions_per_bin, grid_size)
    grouped = np.add.reduceat(values, starts, axis=1)
    assert np.all(grouped.sum(axis=1) == repetitions)
    return starts, stops, grouped


def plot_grouped_bandwidth_distributions(
    h_grid, counts, *, sample_size, repetitions, lambda_value=1.0,
    positions_per_bin=5, ax=None, show_legend=True, title=None,
    localized_counts=None,
):
    """Match the colors, grid, bin widths, and legend of Figure 3.7."""
    grid = np.asarray(h_grid, dtype=float)
    counts = np.asarray(counts)
    if counts.shape != (3, len(grid)):
        raise ValueError("Expected the baseline spectral, ISE, and Q-hat counts.")
    include_localized = localized_counts is not None
    if include_localized:
        localized_counts = np.asarray(localized_counts)
        if localized_counts.shape != (len(grid),):
            raise ValueError("Localized counts must have one entry per grid position.")
        counts = np.vstack((counts, localized_counts))
    names = bandwidth_distribution_names(lambda_value, include_localized)
    starts, stops, grouped = grouped_grid_counts(
        counts, len(grid), repetitions, positions_per_bin,
    )
    if ax is None:
        figure, ax = plt.subplots(figsize=(8.4, 5.6 if include_localized else 5.2))
        figure.subplots_adjust(left=.11, right=.985, top=.84,
                               bottom=.30 if include_localized else .26)
    else:
        figure = ax.figure
    centers = (starts + stops - 1) / 2
    for label, color, opacity, values in zip(
        names[:3], ("#C65D00", "#9467BD", "#66C2A5"),
        (.70, .70, .55), grouped[:3],
    ):
        nonzero = np.flatnonzero(values)
        ax.bar(centers[nonzero], values[nonzero] / repetitions,
               width=.94 * (stops - starts)[nonzero], color=color,
               alpha=opacity, edgecolor="white", linewidth=.3, label=label)
    if include_localized:
        edges = np.r_[starts - .5, stops[-1] - .5]
        ax.stairs(grouped[-1] / repetitions, edges, color="#264B96",
                  linewidth=1.4, label=LOCALIZED_SPECTRAL_LABEL, zorder=4)
    oracle = gaussian_mise_oracle_bandwidth(sample_size)
    position = np.interp(np.log(oracle), np.log(grid), np.arange(len(grid)))
    ax.axvline(position, color="#1f77b4", linestyle=(0, (5, 5)), linewidth=1.5,
               label=f"Gaussian MISE optimum: h={oracle:.4g}")
    ticks = np.unique(np.linspace(0, len(grid) - 1, 7, dtype=int))
    ax.set_xticks(ticks, [f"{grid[index]:.3g}" for index in ticks])
    ax.set_xlim(-.5, len(grid) - .5)
    ax.set_ylim(0, 1.08 * grouped.max() / repetitions)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7, steps=[1, 2, 5, 10]))
    ax.set_xlabel("selected bandwidth h", fontsize=12, labelpad=10, color="#304967")
    ax.set_ylabel("relative frequency", fontsize=12, labelpad=10, color="#304967")
    ax.tick_params(axis="both", labelsize=10, colors="#304967", length=0)
    ax.grid(axis="y", color="#E5ECF6", linewidth=.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#E5ECF6")
    parameters = rf"$N={len(grid)}$, $n={sample_size}$, $M={repetitions}$"
    if not include_localized:
        parameters += rf", $\lambda={lambda_value:g}$"
    ax.set_title(title or (
        f"{positions_per_bin} consecutive grid positions per bin\n"
        + parameters
    ), fontsize=12, color="#304967", pad=14)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(name) for name in names]
    order.append(next(i for i, name in enumerate(labels) if name.startswith("Gaussian MISE")))
    if show_legend:
        ax.legend([handles[i] for i in order], [labels[i] for i in order],
                  loc="upper center", bbox_to_anchor=(.5, -.23), ncol=2,
                  frameon=False, fontsize=8.5, labelcolor="#304967",
                  columnspacing=1.2, handlelength=2.3)
    return figure, ax, (starts, stops, grouped)
