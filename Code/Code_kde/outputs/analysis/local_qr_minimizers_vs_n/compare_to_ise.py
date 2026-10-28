"""Compare empirical local-bandwidth distributions with the ISE reference.

Shape distances condition on the existence of a local minimum. Missing
outcomes are reported separately. These descriptive distances are not
independent-sample hypothesis tests or paired bandwidth-error measures.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
KDE_DIR = HERE.parents[2]
ROOT = KDE_DIR.parents[1]
sys.path.insert(0,str(KDE_DIR))
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/kde-hmax10-mpl-cache')
os.environ.setdefault('MPLBACKEND','Agg')

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
from scipy.stats import wasserstein_distance

from kde_tools.pdf_export import enable_automatic_pdf_saving

from kde_tools.energy_minimization import unified_energy_run_directory, energy_search_settings, energy_search_id

DATA_DIR = unified_energy_run_directory(KDE_DIR/'outputs/data/local_qr_minimizers_vs_n')
IMAGE_DIR = KDE_DIR/'outputs/images/local_qr_minimizers_vs_n'
DELIVERY_DIR = ROOT/'output/pdf/local_qr_minimizers_vs_n'
METHODS = ('Q0','Q13_14','Q1')
COLORS = {'Q0':'#0072B2','Q13_14':'#009E73','Q1':'#D55E00'}
STYLES = {'Q0':'-','Q13_14':'-.','Q1':'--'}
LABELS = {'Q0':r'$\widehat Q_0$','Q13_14':r'$\widehat Q_{13/14}$','Q1':r'$\widehat Q_1$'}


def save_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main():
    groups=defaultdict(list)
    histogram_path=DATA_DIR/'histogram.csv'
    for row in csv.DictReader(histogram_path.open()):
        groups[(int(row['n']),row['method'])].append(row)
    sample_sizes=sorted({key[0] for key in groups})
    distances=[]
    for n in sample_sizes:
        reference=sorted(groups[n,'ISE'],key=lambda row:int(row['bin_index']))
        h=np.array([float(row['bandwidth_center']) for row in reference])
        p=np.array([int(row['count']) for row in reference],dtype=float)
        assert len(h)==201 and p.sum()==1000 and np.all(np.diff(h)>0)
        p/=p.sum()
        for method in METHODS:
            part=sorted(groups[n,method],key=lambda row:int(row['bin_index']))
            count=np.array([int(row['count']) for row in part],dtype=float)
            available=int(count.sum())
            assert 0<available<=1000
            q=count/available
            distance=float(wasserstein_distance(h,h,p,q))
            cdf_gap=np.cumsum(p-q)
            integral=float(np.sum(np.abs(cdf_gap[:-1])*np.diff(h)))
            assert abs(distance-integral)<1e-12
            distances.append(dict(n=n,method=method,available_local_minima=available,
                                  missing_fraction=1-available/1000,
                                  wasserstein_h=distance,
                                  wasserstein_relative_to_mean_ise=distance/float(h@p),
                                  wasserstein_log_h=float(wasserstein_distance(np.log(h),np.log(h),p,q)),
                                  maximum_cdf_gap=float(np.max(np.abs(cdf_gap)))))
    save_csv(DATA_DIR/'distribution_distance_to_ise.csv',distances)
    winners={}
    for metric in ('wasserstein_h','wasserstein_log_h','maximum_cdf_gap'):
        winners[metric]=dict(Counter(min((row for row in distances if row['n']==n),key=lambda row:row[metric])['method'] for n in sample_sizes))

    # Verify the ranking without histogram binning wherever raw ISE selections survive.
    raw_ise=defaultdict(list)
    for row in csv.DictReader((KDE_DIR/'outputs/data/continuous_minimizers_hmax10/selections.csv').open()):
        if row['method']=='ISE':raw_ise[int(row['n'])].append(float(row['h']))
    raw_local=defaultdict(list)
    for row in csv.DictReader((DATA_DIR/'selections.csv').open()):
        n=int(row['n']);value=float(row['selected_h'])
        if n in raw_ise and np.isfinite(value):raw_local[n,row['method']].append(value)
    raw_checks=[]
    raw_winners={}
    for n in sorted(raw_ise):
        for method in METHODS:
            distance=float(wasserstein_distance(raw_ise[n],raw_local[n,method]))
            binned=next(row['wasserstein_h'] for row in distances if row['n']==n and row['method']==method)
            raw_checks.append(dict(n=n,method=method,raw_wasserstein_h=distance,
                                   histogram_wasserstein_h=binned,
                                   absolute_binning_difference=abs(distance-binned)))
        raw_winners[str(n)]=min((row for row in raw_checks if row['n']==n),key=lambda row:row['raw_wasserstein_h'])['method']
        assert raw_winners[str(n)]==min((row for row in distances if row['n']==n),key=lambda row:row['wasserstein_h'])['method']
    save_csv(DATA_DIR/'distribution_distance_raw_checks.csv',raw_checks)
    analysis=dict(representation='201-bin empirical probability distributions on actual bandwidth values',
                  local_distribution='conditional on a detected local minimum in (0.01,2)',
                  ise_distribution='all 1000 ISE-minimizing bandwidths on [0.01,2]',
                  missing_outcomes='reported separately, never mapped to a bandwidth',
                  descriptive_only_no_p_values=True,sample_sizes=sample_sizes,
                  winner_counts=winners,raw_data_winners=raw_winners,
                  largest_raw_vs_binned_distance_difference=max(row['absolute_binning_difference'] for row in raw_checks),
                  histogram_sha256=hashlib.sha256(histogram_path.read_bytes()).hexdigest(),
                  wasserstein_reference='https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html#notes',
                  maximum_cdf_gap_reference='https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html#notes')
    (DATA_DIR/'distribution_distance_validation.json').write_text(json.dumps(analysis,indent=2)+'\n')

    pdf_dir=enable_automatic_pdf_saving('local_qr_ise_distribution_distances')
    plt.style.use('default')
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.3))
    for method in METHODS:
        part=[row for row in distances if row['method']==method]
        n=[row['n'] for row in part]
        axes[0].plot(n,[row['wasserstein_h'] for row in part],color=COLORS[method],linestyle=STYLES[method],linewidth=2)
        axes[1].plot(n,[row['missing_fraction'] for row in part],color=COLORS[method],linestyle=STYLES[method],linewidth=2)
    axes[0].set(title='Distribution distance to ISE (lower is closer)',ylabel='Wasserstein-1 distance (bandwidth units)',ylim=(0,None))
    axes[1].set(title='Samples without a local minimum',ylabel='fraction of samples',ylim=(-.008,.38))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    for ax in axes:
        ax.set(xlabel='sample size n',xlim=(100,2000))
        ax.grid(color='#DDE5EC',linewidth=.7)
        ax.set_axisbelow(True)
        ax.tick_params(colors='#304967')
        for side in ('top','right'):ax.spines[side].set_visible(False)
        for side in ('left','bottom'):ax.spines[side].set_color('#B8C5D1')
    fig.suptitle('Closeness of local bandwidth distributions to ISE',fontsize=17,y=.985)
    handles=[Line2D([0],[0],color=COLORS[m],linestyle=STYLES[m],linewidth=2,label=LABELS[m]) for m in METHODS]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.91),ncol=3,frameon=False,fontsize=12)
    fig.text(.5,.025,'1,000 paired standard-normal samples per n; local minima inside (0.01,2); ISE minimum on [0.01,2].\nDistance compares the local distribution conditional on a minimum being found with the full ISE distribution.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.075,right=.985,bottom=.18,top=.73,wspace=.29)
    fig.savefig(IMAGE_DIR/'distribution_distance_to_ise.png',dpi=185)
    plt.show();plt.close(fig)
    source=pdf_dir/'001_closeness_of_local_bandwidth_distributions_to_ise.pdf'
    assert source.is_file()
    shutil.copyfile(source,DELIVERY_DIR/'distribution_distance_to_ise.pdf')
    print(json.dumps(analysis,indent=2),flush=True)
    print('Selected-size distances:',flush=True)
    for row in distances:
        if row['n'] in (100,1000,1500,2000):print(row,flush=True)


if __name__=='__main__':main()
