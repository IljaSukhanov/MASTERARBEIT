"""Export the paired repeated-sample histograms and the four-panel comparison."""
from pathlib import Path
import csv
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import numpy as np

from kde_tools.bandwidths import gaussian_mise_oracle_bandwidth

CODE = Path(__file__).resolve().parents[2]
DATA = CODE / 'outputs/data/repeated_alpha_evolution'
OUT = CODE / 'outputs/pdfs/repeated_alpha_evolution'
grid_size_requested = int(os.environ.get('REPEATED_ALPHA_GRID_SIZE', '400'))
if grid_size_requested != 1000:
    DATA = DATA / f'n{grid_size_requested}'
if grid_size_requested != 400:
    OUT = OUT / f'n{grid_size_requested}'
OUT.mkdir(parents=True, exist_ok=True)
with np.load(DATA / 'paired_selections.npz') as saved:
    indices = saved['selected_indices'].copy()
    grid = saved['grid'].copy()
    exponents = saved['exponents'].copy()
    lambdas = saved['lambdas'].copy()
    completed = saved['completed'].copy()
assert completed.shape == (1000,) and np.all(completed)
assert indices.shape == (1000, 6) and np.array_equal(exponents, [0, .5, 1, 1.5])
repetitions, grid_size = indices.shape[0], len(grid)
assert grid_size == grid_size_requested
counts = np.stack([np.bincount(indices[:, i], minlength=grid_size) for i in range(6)])
positions_per_bin = 2 if grid_size == 400 else 5
starts = np.arange(0, grid_size, positions_per_bin)
stops = np.minimum(starts + positions_per_bin, grid_size)
grouped = np.add.reduceat(counts, starts, axis=1)
assert np.all(grouped.sum(axis=1) == repetitions)
centers = (starts + stops - 1) / 2
widths = .94 * (stops - starts)
y_max = max(.07, float(np.ceil(1.05 * grouped.max() / repetitions * 100)) / 100)
ticks = np.unique(np.linspace(0, grid_size - 1, 7, dtype=int))
oracle = gaussian_mise_oracle_bandwidth(200)
oracle_position = float(np.interp(np.log(oracle), np.log(grid), np.arange(grid_size)))
plt.style.use('default')


def draw(ax, column):
    for values, color, alpha in zip((grouped[column], grouped[4], grouped[5]),
                                    ('#C65D00', '#555555', '#43877E'), (.7, .7, .55)):
        selected = np.flatnonzero(values)
        ax.bar(centers[selected], values[selected] / repetitions, width=widths[selected],
               color=color, alpha=alpha, edgecolor='white', linewidth=.3)
    ax.axvline(oracle_position, color='tab:red', linestyle='--', linewidth=1.8)
    ax.set_xticks(ticks, [f'{grid[i]:.3g}' for i in ticks])
    ax.set_xlim(-.5, grid_size-.5)
    ax.set_ylim(0, y_max)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_locator(MultipleLocator(.02 if y_max > .10 else .01))
    ax.tick_params(axis='both', labelsize=9, colors='#304967', length=0)
    ax.grid(axis='y', color='#E5ECF6', linewidth=.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#E5ECF6')


suffixes = ('alpha_0', 'alpha_1_2', 'alpha_1', 'alpha_3_2')
panel_titles = (r'(a) $\lambda_N=1$', r'(b) $\lambda_N=N^{1/2}$',
                r'(c) $\lambda_N=N$', r'(d) $\lambda_N=N^{3/2}$')
for column, suffix in enumerate(suffixes):
    fig, ax = plt.subplots(figsize=(8, 3.7))
    draw(ax, column)
    ax.set_xlabel(r'chosen bandwidth $h_{k_N}$', fontsize=11, color='#304967', labelpad=9)
    ax.set_ylabel('relative frequency', fontsize=11, color='#304967', labelpad=9)
    fig.tight_layout()
    fig.savefig(OUT / f'{suffix}.pdf', bbox_inches='tight')
    plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(12, 7.8), sharex=True, sharey=True)
for column, ax in enumerate(axes.flat):
    draw(ax, column)
    ax.set_title(panel_titles[column], fontsize=11, pad=8, color='#304967')
for ax in axes[:, 0]:
    ax.set_ylabel('relative frequency', fontsize=10, color='#304967', labelpad=8)
for ax in axes[-1, :]:
    ax.set_xlabel(r'chosen bandwidth $h_{k_N}$', fontsize=10, color='#304967', labelpad=8)
fig.tight_layout(h_pad=2.2, w_pad=1.4)
fig.savefig(OUT / 'four_panel_comparison.pdf', bbox_inches='tight')
fig.savefig(DATA / 'four_panel_comparison.png', dpi=180, bbox_inches='tight')
plt.close(fig)

with (DATA / 'paired_selections.csv').open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['repetition', 'alpha_0', 'alpha_1_2', 'alpha_1', 'alpha_3_2', 'ise', 'qhat'])
    for i, row in enumerate(indices):
        writer.writerow([i, *grid[row]])
with (DATA / 'grouped_counts.csv').open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['first_grid_index', 'last_grid_index', 'h_first', 'h_last',
                     'alpha_0', 'alpha_1_2', 'alpha_1', 'alpha_3_2', 'ise', 'qhat'])
    for i, (start, stop) in enumerate(zip(starts, stops)):
        writer.writerow([start, stop-1, grid[start], grid[stop-1], *grouped[:, i]])
(DATA / 'plot_settings.json').write_text(json.dumps({
    'grid_size': grid_size,
    'positions_per_bin': positions_per_bin,
    'number_of_bins': len(starts),
    'common_upper_y_limit': y_max,
}, indent=2) + '\n')
print(f'Saved 4 panels and comparison in {OUT}; {positions_per_bin} positions per bin; common upper y limit {y_max:.2f}')
