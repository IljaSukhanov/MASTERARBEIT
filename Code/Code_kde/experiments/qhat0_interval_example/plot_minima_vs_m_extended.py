"""Plot the extended paired linear-scan experiment."""
from pathlib import Path
import csv
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

KDE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KDE_ROOT))
from kde_tools.pdf_export import enable_automatic_pdf_saving

OUTPUT = KDE_ROOT / 'outputs/brent_minima_vs_m_extended'
with (OUTPUT / 'minima_vs_m.csv').open(newline='') as stream:
    rows = list(csv.DictReader(stream))
ms = [int(r['M']) for r in rows]
counts = [int(r['detected_minima_total']) for r in rows]

enable_automatic_pdf_saving('brent_minima_vs_m_extended')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42})
fig, ax = plt.subplots(figsize=(9, 5.3))
ax.plot(ms, counts, color='#245b90', marker='o', mfc='white', mew=1.5,
        ms=5.4, lw=1.8)
for i in (0, 2, 6, 10, 14, 18, len(ms)-1):
    ax.annotate(str(counts[i]), (ms[i], counts[i]), xytext=(0, 9),
                textcoords='offset points', ha='center', fontsize=9,
                color='#245b90')
ticks = [33, 129, 513, 2049, 8193, 49153]
ax.set_xscale('log', base=2)
ax.set_xticks(ticks, [f'{m:,}' for m in ticks])
ax.set_xlim(27, 60000)
ax.set_ylim(0, max(counts) * 1.16)
ax.set_xlabel('Number of grid points $M$ (log scale)')
ax.set_ylabel('Detected local minima across 100 samples')
ax.set_title('Local minima found versus refinement', loc='left',
             fontsize=16, fontweight='semibold', pad=15)
ax.grid(axis='y', color='#e2e9ef', linewidth=.8)
ax.set_axisbelow(True)
fig.text(.125, .02, 'Linear grid on $[0.01,2]$  |  $n=200$  |  $\\hat Q_0$',
         fontsize=10, color='#56687a')
fig.tight_layout(rect=(0, .06, 1, 1))
fig.savefig(OUTPUT / 'minima_vs_m_extended.png', dpi=180, facecolor='white')
plt.show()
plt.close(fig)
