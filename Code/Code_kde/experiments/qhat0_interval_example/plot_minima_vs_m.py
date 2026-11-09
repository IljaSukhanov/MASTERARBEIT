"""Plot the total number of detected valleys in the paired linear scans."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

KDE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KDE_ROOT))
from kde_tools.pdf_export import enable_automatic_pdf_saving

SOURCE = KDE_ROOT / 'outputs/brent_linear_refinements_spectral_alpha15/results.csv'
OUTPUT = KDE_ROOT / 'outputs/brent_minima_vs_m'


def main():
    counts = defaultdict(dict)
    with SOURCE.open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['family'] == 'brent_linear':
                m, sample = int(row['M']), int(row['sample_id'])
                if sample in counts[m]:
                    raise ValueError(f'Duplicate sample {sample} for M={m}.')
                counts[m][sample] = int(row['detected_valleys'])
    ms = sorted(counts)
    if not ms or any(set(counts[m]) != set(range(100)) for m in ms):
        raise ValueError('Expected the same 100 samples for each linear scan.')
    totals = [sum(counts[m].values()) for m in ms]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / 'minima_vs_m.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['M', 'samples', 'detected_minima_total', 'mean_per_sample'])
        writer.writerows((m, 100, total, total / 100) for m, total in zip(ms, totals))

    enable_automatic_pdf_saving('brent_minima_vs_m')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(ms, totals, color='#245b90', marker='o', mfc='white', mew=1.8,
            ms=7, lw=2)
    for m, total in zip(ms, totals):
        ax.annotate(str(total), (m, total), xytext=(0, 9),
                    textcoords='offset points', ha='center', fontsize=10,
                    color='#245b90')
    ax.set_xscale('log', base=2)
    ax.set_xticks(ms, [str(m) for m in ms])
    ax.set_xlim(27, 1260)
    ax.set_ylim(0, max(totals) * 1.16)
    ax.set_xlabel('Number of grid points $M$ (log scale)')
    ax.set_ylabel('Detected local minima across 100 samples')
    ax.set_title('Local minima found versus refinement', loc='left',
                 fontsize=16, fontweight='semibold', pad=15)
    ax.grid(axis='y', color='#e2e9ef', linewidth=.8)
    ax.set_axisbelow(True)
    fig.text(.13, .02, 'Linear grid on $[0.01,2]$  |  $n=200$  |  $\\hat Q_0$',
             fontsize=10, color='#56687a')
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(OUTPUT / 'minima_vs_m.png', dpi=180, facecolor='white')
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
