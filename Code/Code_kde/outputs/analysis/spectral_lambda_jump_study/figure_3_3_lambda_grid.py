"""Display the saved original Figure 3.3 choices with their lambda grid.

Data source: ../perron_figures_3_1_to_3_4/figure_3_3_comparison.csv,
columns position, old_index, old_bandwidth. These are the original values
recovered from the preserved thesis figure, not Monte Carlo samples.
The curve retains the original notebook's post-step plotting convention.
"""
from pathlib import Path
import csv
import json
import os
import sys

CODE_DIRECTORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE_DIRECTORY))
os.environ.setdefault('MPLCONFIGDIR', str(CODE_DIRECTORY / '.matplotlib-cache'))

import matplotlib.pyplot as plt
import numpy as np
from kde_tools.bandwidths import bandwidth_grid_linear_in_psi_beta, gaussian_mise_oracle_bandwidth
from kde_tools.density import exact_oracle_ise_values
from kde_tools.pdf_export import enable_automatic_pdf_saving

source_directory = CODE_DIRECTORY / 'outputs' / 'analysis' / 'perron_figures_3_1_to_3_4'
with (source_directory / 'figure_3_3_comparison.csv').open() as stream:
    rows = list(csv.DictReader(stream))
lambdas = np.array([float(row['position']) for row in rows])
indices = np.array([int(row['old_index']) for row in rows])
bandwidths = np.array([float(row['old_bandwidth']) for row in rows])
previous = json.loads((source_directory / 'previous_figure_data.json').read_text())
original_h = np.asarray(previous['figure_3_3']['selected_bandwidths']).ravel()
assert np.array_equal(bandwidths, original_h)
assert np.allclose(lambdas, np.geomspace(1e-2, 1e4, 101), rtol=1e-14, atol=0)
k = int(np.argmax(np.diff(indices)))
assert k == 54 and int(indices[k + 1] - indices[k]) == 23

h_grid = bandwidth_grid_linear_in_psi_beta(grid_size=201)
assert np.array_equal(bandwidths, h_grid[indices])
# Original reference lines: use the original fixed sample and bandwidth grid.
original_sample = np.random.default_rng(20260823).normal(size=200)
oracle_h = gaussian_mise_oracle_bandwidth(200)
ise_h = h_grid[int(np.argmin(exact_oracle_ise_values(original_sample, h_grid, sigma2=1.0)))]
report = json.loads((source_directory / 'comparison_report.json').read_text())
closest_pair = report['figures']['figure_3_3']['closest_pair_bandwidths']

pdf_directory = enable_automatic_pdf_saving('figure_3_3_lambda_grid')
image_directory = CODE_DIRECTORY / 'outputs' / 'images' / 'spectral_lambda_jump_study'
image_directory.mkdir(parents=True, exist_ok=True)
fig, (full, zoom) = plt.subplots(
    2, 1, figsize=(11.5, 8.6), gridspec_kw={'height_ratios': [1.25, 1]},
)
fig.suptitle('Figure 3.3: original data with the lambda grid', fontsize=15)

for ax in (full, zoom):
    for value in lambdas:
        ax.axvline(value, color='#92979d', linewidth=0.55, alpha=0.32, zorder=0)
    ax.axvspan(lambdas[k], lambdas[k+1], color='#e69536', alpha=0.2, zorder=1)
    ax.step(lambdas, bandwidths, where='post', color='#222222', linewidth=1.8,
            marker='o', markersize=3.5, label='Original spectral selection (N=201)', zorder=3)
    ax.scatter(lambdas[k:k+2], bandwidths[k:k+2], color='#c76b00', s=48, zorder=4)
    ax.set_xscale('log')
    ax.set_ylabel('Chosen bandwidth h')
    ax.grid(axis='y', alpha=0.18)
    ax.set_axisbelow(True)

full.set_yscale('log')
full.set_xlim(lambdas[0], lambdas[-1])
full.set_ylim(0.01, 2)
full.axhline(oracle_h, color='tab:blue', linestyle='--', linewidth=1.2,
             label=f'Gaussian MISE oracle: h={oracle_h:.4g}')
full.axhline(ise_h, color='tab:purple', linestyle='--', linewidth=1.2,
             label=f'Sample ISE minimum: h={ise_h:.4g}')
for j, h in enumerate(closest_pair):
    full.axhline(h, color='tab:red', linestyle='--', linewidth=0.8,
                 label=f'Closest-pair bandwidth: h={h:.6f}')
full.plot([], [], color='#92979d', linewidth=0.8, label='Vertical lines: all 101 lambda-grid points')
full.set_xlabel(r'Similarity scale $\lambda$')
full.set_title(r'$\lambda_k=10^{-2+0.06k}$, $k=0,\ldots,100$', fontsize=11, loc='left')
full.legend(loc='upper right', fontsize=8, framealpha=0.97)

# Show every lambda-grid value around the jump as an explicitly labeled tick.
local_lambdas = lambdas[49:61]
zoom.set_xlim(local_lambdas[0], local_lambdas[-1])
zoom.set_ylim(0.085, 0.229)
zoom.set_xticks(local_lambdas)
zoom.set_xticklabels([f'{value:.4f}' for value in local_lambdas], rotation=35, ha='right', fontsize=9)
zoom.minorticks_off()
zoom.set_xlabel(r'Similarity scale $\lambda$: actual sampled grid values')
zoom.set_title('Zoom: the adjacent sampled pair bracketing the largest upward change', fontsize=11, loc='left')
zoom.annotate(f'h={bandwidths[k]:.6f}', (lambdas[k], bandwidths[k]),
              xytext=(-6, -23), textcoords='offset points', ha='right', fontsize=10,
              arrowprops={'arrowstyle':'-', 'color':'#c76b00'})
zoom.annotate(f'h={bandwidths[k+1]:.6f}', (lambdas[k+1], bandwidths[k+1]),
              xytext=(16, 12), textcoords='offset points', ha='left', fontsize=10,
              arrowprops={'arrowstyle':'-', 'color':'#c76b00'})
zoom.text(0.98, 0.05,
          f'Pair: ({lambdas[k]:.10f}, {lambdas[k+1]:.10f})\n23 bandwidth-grid steps',
          transform=zoom.transAxes, ha='right', va='bottom', fontsize=9.5,
          bbox={'facecolor':'white', 'edgecolor':'#dddddd', 'alpha':0.94})
fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.0)
image_path = image_directory / 'figure_3_3_original_with_lambda_grid.png'
fig.savefig(image_path, dpi=160, bbox_inches='tight')
plt.show()  # Automatically exports the entire displayed figure to PDF.
plt.close(fig)
print(f'Image: {image_path}')
print(f'Lambda pair: {lambdas[k]:.12f}, {lambdas[k+1]:.12f}')
