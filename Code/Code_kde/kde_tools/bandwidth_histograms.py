"""Grouped grid-position histograms in the thesis comparison-figure style."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import numpy as np

from .bandwidths import gaussian_mise_oracle_bandwidth


def plot_thesis_bandwidth_distributions(h_grid, counts, *, sample_size,
                                        repetitions, positions_per_bin=5):
    """Draw four spectral distributions in the thesis comparison style."""
    grid = np.asarray(h_grid, dtype=float)
    counts = np.asarray(counts)
    if counts.shape != (4, len(grid)):
        raise ValueError("Expected counts for the four spectral choices.")
    starts, stops, grouped = grouped_grid_counts(
        counts, len(grid), repetitions, positions_per_bin,
    )
    centers = (starts + stops - 1) / 2
    fig, ax = plt.subplots(figsize=(8, 3.7))
    labels = (r'$\lambda=1$', r'$\lambda_N=N^{1/2}$',
              r'$\lambda_N=N$', r'$\lambda_N=N^{3/2}$')
    colors = ('#DF6818', '#A14AA4', '#199C70', '#2867B2')
    edges = np.r_[starts - .5, stops[-1] - .5]
    for values, color, label in zip(grouped, colors, labels):
        selected = np.flatnonzero(values)
        ax.bar(centers[selected], values[selected] / repetitions,
               width=.94 * (stops - starts)[selected], color=color, alpha=.35,
               edgecolor='white', linewidth=.3, label=label)
        ax.stairs(values / repetitions, edges, color=color, linewidth=1.1,
                  zorder=3)
    oracle = gaussian_mise_oracle_bandwidth(sample_size)
    position = np.interp(np.log(oracle), np.log(grid), np.arange(len(grid)))
    ax.axvline(position, color='tab:red', linestyle='--', linewidth=1.8, zorder=4,
               label=r'$h_{\mathrm{oracle}}$')
    ticks = np.unique(np.linspace(0, len(grid) - 1, 7, dtype=int))
    ax.set_xticks(ticks, [f'{grid[i]:.3g}' for i in ticks])
    ax.set_xlim(-.5, len(grid) - .5)
    y_max = max(.07, float(np.ceil(1.05 * grouped.max() / repetitions * 100)) / 100)
    ax.set_ylim(0, y_max)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    tick_step = .05 if y_max > .20 else (.02 if y_max > .10 else .01)
    ax.yaxis.set_major_locator(MultipleLocator(tick_step))
    ax.tick_params(axis='both', labelsize=9, colors='#304967', length=0)
    ax.grid(axis='y', color='#E5ECF6', linewidth=.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#E5ECF6')
    ax.set_xlabel(r'chosen bandwidth $h_{k_N}$', fontsize=11, color='#304967', labelpad=9)
    ax.set_ylabel('relative frequency', fontsize=11, color='#304967', labelpad=9)
    fig.tight_layout()
    return fig, ax, (starts, stops, grouped)


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
