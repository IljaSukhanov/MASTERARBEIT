"""Plot detected minima against the number of linear-grid subintervals."""
from collections import Counter
from pathlib import Path
import csv
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

KDE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KDE_ROOT))
from kde_tools.pdf_export import enable_automatic_pdf_saving

SOURCE = KDE_ROOT / 'outputs/brent_minima_vs_m_extended/minima_vs_m.csv'
OUTPUT = KDE_ROOT / 'outputs/brent_minima_vs_subintervals'
with SOURCE.open(newline='') as f:
    old = list(csv.DictReader(f))
with (OUTPUT / 'additional_coarse_samples.csv').open(newline='') as f:
    small = list(csv.DictReader(f))
counts = Counter()
sample_counts = Counter()
for row in small:
    n = int(row['subintervals'])
    counts[n] += int(row['detected_minima'])
    sample_counts[n] += 1
if any(n != 100 for n in sample_counts.values()):
    raise ValueError('Missing results for a coarse grid.')
for row in old:
    n = int(row['M']) - 1
    if n in counts or int(row['samples']) != 100:
        raise ValueError('Duplicate or incomplete grid results.')
    counts[n] = int(row['detected_minima_total'])
# M=2 has no interior grid point. This is a limiting case, not an invocation
# of grid_brent_qhat_minimum, whose minimum accepted grid size is M=3.
counts[1] = 0
intervals = sorted(counts)
totals = [counts[n] for n in intervals]
with (OUTPUT / 'minima_vs_subintervals.csv').open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['subintervals', 'M', 'samples', 'detected_minima_total', 'case'])
    writer.writerows((n, n+1, 100, counts[n],
                      'no interior grid points; no search' if n == 1 else 'linear scan and bounded Brent')
                     for n in intervals)

enable_automatic_pdf_saving('brent_minima_vs_subintervals')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': True, 'axes.spines.right': True,
                     'pdf.fonttype': 42})
def included(n):
    """Use 2**k for k >= 1 and 3*2**k for k >= 0."""
    return (n >= 2 and n & (n - 1) == 0) or (
        n >= 3 and n % 3 == 0 and (n // 3) & (n // 3 - 1) == 0)

shown = [n for n in intervals if included(n)]
plateau_start = next(n for i, n in enumerate(shown)
                     if all(counts[later] == counts[n] for later in shown[i:]))
fig, ax = plt.subplots(figsize=(10.5, 5.5))
ax.axvline(plateau_start, color='#d27827', linestyle='--', linewidth=1.5,
           label=rf'$M-1={plateau_start}$', zorder=1)
ax.plot(shown, [counts[n] for n in shown], color='#245b90', marker='o',
        mfc='white', mew=1.4, ms=5.1, lw=1.7)
ax.set_xscale('log', base=2)
ax.set_axisbelow(True)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(99, 128)
ax.set_yticks([100, 105, 110, 115, 120, 125])
ax.set_xlim(1.75, 65000)
ticks = [2, 3, 4, 6, 8, 16, 32, 128, 512, 2048, 8192, 49152]
ax.set_xticks(ticks, [f'{n:,}' for n in ticks])
ax.set_ylabel('# detected local minima across 100 samples')
ax.set_xlabel('# subintervals (log-scale)')
ax.legend(fontsize=8.5)
fig.tight_layout()
fig.savefig(OUTPUT / 'minima_vs_subintervals.png', dpi=180, facecolor='white')
plt.show()
plt.close(fig)
