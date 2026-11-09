"""Plot selected bandwidth versus scan grid size for paired sample 54.

The six previously benchmarked grid sizes are read from the saved alpha=3/2
comparison; four coarser sizes are evaluated here to show the transition.
All energy minimization uses the same sample and Q_0(h) on [0.01,2].
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

KDE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KDE_ROOT))
from kde_tools.experiment_inputs import paired_gaussian_sample_bank
from kde_tools.pdf_export import enable_automatic_pdf_saving
from kde_tools.qhat import grid_brent_qhat_minimum

SAMPLE = 54  # Zero-based ID, as in the saved results.
ADDITIONAL_M = (3, 5, 9, 17)
BENCHMARKED_M = (33, 65, 129, 257, 513, 1025)
SPACINGS = ('linear', 'geometric')
SOURCE = KDE_ROOT/'outputs/brent_refinements_spectral_alpha15_m1025_reference/results.csv'
CHECKPOINT = KDE_ROOT/f'outputs/brent_refinements_spectral_alpha15/checkpoints/sample_{SAMPLE:03d}.json'
OUTPUT = KDE_ROOT/'outputs/brent_refinement_sample54'


def main():
    sample = paired_gaussian_sample_bank(repetitions=100, maximum_n=200)[SAMPLE]
    check = json.loads(CHECKPOINT.read_text())
    if hashlib.sha256(sample.tobytes()).hexdigest() != check['sample_sha256']:
        raise ValueError('Sample does not match the saved comparison.')
    with SOURCE.open(newline='') as stream:
        old = [row for row in csv.DictReader(stream) if int(row['sample_id']) == SAMPLE]
    by_method = {row['method']: row for row in old}
    reference = float(by_method['brent_geometric_1025']['selected_h'])
    unrefined = float(by_method['brent_unrefined']['selected_h'])
    rows = []
    with threadpool_limits(limits=1):
        for spacing in SPACINGS:
            for m in ADDITIONAL_M:
                result = grid_brent_qhat_minimum(sample, grid_points=m, grid_spacing=spacing)
                rows.append(dict(sample_id=SAMPLE, spacing=spacing, M=m,
                                 selected_h=result['h'], valleys=result['valleys_detected'],
                                 source='additional calculation'))
            for m in BENCHMARKED_M:
                saved = by_method[f'brent_{spacing}_{m}']
                rows.append(dict(sample_id=SAMPLE, spacing=spacing, M=m,
                                 selected_h=float(saved['selected_h']),
                                 valleys=int(saved['detected_valleys']),
                                 source='saved paired comparison'))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT/'data.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (OUTPUT/'configuration.json').write_text(json.dumps(dict(
        sample_id=SAMPLE, sample_id_indexing='zero-based', n=200, r=0,
        h_interval=[.01,2.], sample_sha256=check['sample_sha256'],
        M_includes_endpoints=True, additional_M=ADDITIONAL_M, benchmarked_M=BENCHMARKED_M,
        reference='geometric refined Brent M=1025, same sample', reference_h=reference,
        reference_is_certified_global=False, unrefined_h=unrefined),indent=2)+'\n')

    enable_automatic_pdf_saving('brent_refinement_sample54')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'axes.labelcolor':'#243342',
                         'xtick.color':'#34485b','ytick.color':'#34485b'})
    fig,ax=plt.subplots(figsize=(10.5,5.8))
    fig.suptitle('Selected bandwidth versus refinement',x=.11,y=.97,ha='left',
                 fontsize=19,fontweight='semibold',color='#15283b')
    ax.set_title(r'Sample ID 54  |  $n=200$  |  $\hat Q_0$ on $[0.01,2]$',
                 loc='left',fontsize=11,color='#56687a',pad=13)
    ax.axhline(unrefined,color='#9aa8b4',lw=1.2,ls=':',zorder=1,
               label=f'Unrefined Brent: {unrefined:.4f}')
    ax.axhline(reference,color='#697987',lw=1.2,ls='--',zorder=1,
               label=f'Reference (geometric M=1025): {reference:.4f}')
    styles={'linear':dict(color='#245b90',marker='o',ms=9,mfc='white',mew=1.8,lw=2.3,zorder=3),
            'geometric':dict(color='#d27827',marker='D',ms=4.5,mfc='#d27827',mew=1.0,lw=1.3,
                             ls='--',zorder=4)}
    for spacing in SPACINGS:
        group=sorted((r for r in rows if r['spacing']==spacing),key=lambda r:r['M'])
        ax.plot([r['M'] for r in group],[r['selected_h'] for r in group],
                label=f'{spacing.capitalize()} scan',**styles[spacing])
    x=list(ADDITIONAL_M+BENCHMARKED_M)
    ax.set_xscale('log',base=2)
    ax.set_xticks(x,[str(m) for m in x])
    ax.set_xlim(2.4,1300)
    ax.set_ylim(.075,.5)
    ax.set_yticks([.1,.2,.3,.4,.5])
    ax.set_xlabel('Refinement grid points $M$ (including endpoints; log scale)',labelpad=9)
    ax.set_ylabel('Selected bandwidth $h$',labelpad=8)
    ax.grid(axis='y',color='#e2e9ef',lw=.8)
    ax.set_axisbelow(True)
    handles,labels=ax.get_legend_handles_labels()
    ax.legend(handles[2:]+handles[:2],labels[2:]+labels[:2],frameon=False,
              loc='center right',fontsize=9.5,facecolor='white')
    fig.text(.11,.035,'The two curves overlap at the displayed scale. The first four $M$ values extend the earlier benchmark.',
             color='#627487',fontsize=9)
    fig.subplots_adjust(left=.11,right=.97,top=.82,bottom=.19)
    fig.savefig(OUTPUT/'refinement_sample54.png',dpi=180,facecolor='white')
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
