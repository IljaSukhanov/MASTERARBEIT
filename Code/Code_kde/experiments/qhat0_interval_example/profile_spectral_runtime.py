"""Time the existing spectral selector's assembly and Perron stages.

The original Perron implementation is instrumented with lightweight timing
wrappers; its algorithm, matrix construction and selected bandwidth are not
changed. Every timed call uses a fresh sample/grid calculation. The existing
paired experiment is used only to verify sample identity and selected bandwidths.
"""
from __future__ import annotations

import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter_ns

import numpy as np
import scipy
from threadpoolctl import threadpool_info, threadpool_limits

KDE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KDE_ROOT))
from kde_tools.bandwidths import bandwidth_grid_linear_in_psi_beta
from kde_tools.experiment_inputs import paired_gaussian_sample_bank
from kde_tools.fourier.similarities import (
    DEFAULT_FREQUENCY_CUTOFF, DEFAULT_FREQUENCY_INTERVALS,
    beta2_divergence_matrix_fourier,
)

perron_module = importlib.import_module('kde_tools.perron')
OUT = KDE_ROOT/'outputs/spectral_runtime_breakdown_alpha15'
PREVIOUS = KDE_ROOT/'outputs/brent_linear_refinements_spectral_alpha15/results.csv'
CHECKPOINTS = KDE_ROOT/'outputs/brent_refinements_spectral_alpha15/checkpoints'
SIZES = (100, 200, 400)
SAMPLE_COUNT = 100
SEED = 20261111


def benchmark_one(sample, size, timing):
    timing.clear()
    t0=perf_counter_ns()
    grid=bandwidth_grid_linear_in_psi_beta(.01,2.,2.,size)
    lam=float(size**1.5)
    t1=perf_counter_ns()
    divergence=beta2_divergence_matrix_fourier(sample,grid)
    t2=perf_counter_ns()
    result=perron_module.perron_eigenpair_from_divergence(divergence,lam)
    t3=perf_counter_ns()
    index=result.selected_index
    h=float(grid[index])
    t4=perf_counter_ns()
    if timing.get('solver_calls') != 1 or timing.get('eigh_calls') != 1:
        raise RuntimeError('Expected exactly one Perron solver and one eigh call.')
    solver=timing['solver_ns']
    eigensolve=timing['eigh_ns']
    prep=(t3-t2)-solver
    # Grid, divergence, shifted-similarity preprocessing, Perron solver, and
    # final selection partition the instrumented end-to-end interval.
    durations=dict(grid_ms=(t1-t0)/1e6, divergence_ms=(t2-t1)/1e6,
                   similarity_preparation_ms=prep/1e6, perron_solver_ms=solver/1e6,
                   eigensolver_ms=eigensolve/1e6, perron_other_ms=(solver-eigensolve)/1e6,
                   selection_ms=(t4-t3)/1e6, total_ms=(t4-t0)/1e6)
    durations['matrix_assembly_ms']=(t1-t0+t2-t1+prep)/1e6
    durations['perron_and_selection_ms']=(solver+t4-t3)/1e6
    if any(value < 0 for value in durations.values()):
        raise ArithmeticError('Inconsistent nested timing measurements.')
    return h,index,result.relative_residual,durations


def main():
    samples=paired_gaussian_sample_bank(repetitions=SAMPLE_COUNT,maximum_n=200)
    with PREVIOUS.open(newline='') as stream:
        previous={(int(r['sample_id']),int(r['N'])):float(r['selected_h'])
                  for r in csv.DictReader(stream) if r['family']=='spectral'}
    if len(previous)!=SAMPLE_COUNT*len(SIZES):
        raise ValueError('Incomplete prior spectral results.')
    timing={}
    solver_original=perron_module.perron_eigenpair
    eigh_original=perron_module.eigh

    def timed_solver(*args,**kwargs):
        timing['solver_calls']=timing.get('solver_calls',0)+1
        start=perf_counter_ns()
        try:
            return solver_original(*args,**kwargs)
        finally:
            timing['solver_ns']=timing.get('solver_ns',0)+(perf_counter_ns()-start)

    def timed_eigh(*args,**kwargs):
        timing['eigh_calls']=timing.get('eigh_calls',0)+1
        start=perf_counter_ns()
        try:
            return eigh_original(*args,**kwargs)
        finally:
            timing['eigh_ns']=timing.get('eigh_ns',0)+(perf_counter_ns()-start)

    perron_module.perron_eigenpair=timed_solver
    perron_module.eigh=timed_eigh
    rows=[]
    try:
        with threadpool_limits(limits=1):
            for size in SIZES:
                benchmark_one(samples[0],size,timing)
            pools=[{k:info.get(k) for k in ('internal_api','prefix','num_threads','version','architecture')}
                   for info in threadpool_info()]
            for index,sample in enumerate(samples):
                prior=json.loads((CHECKPOINTS/f'sample_{index:03d}.json').read_text())
                if hashlib.sha256(sample.tobytes()).hexdigest()!=prior['sample_sha256']:
                    raise ValueError(f'Sample hash mismatch for sample {index}.')
                order=np.random.default_rng(np.random.SeedSequence([SEED,index])).permutation(SIZES)
                for size in order:
                    size=int(size)
                    h,peak,residual,times=benchmark_one(sample,size,timing)
                    if not np.isclose(h,previous[(index,size)],rtol=0,atol=1e-14):
                        raise ArithmeticError(f'Peak disagrees with previous run: sample {index}, N={size}.')
                    rows.append(dict(sample_id=index,N=size,lambda_N=float(size**1.5),
                                     selected_h=h,selected_index=peak,relative_eigen_residual=residual,
                                     **times))
                if (index+1)%20==0:
                    print(f'Completed {index+1}/{SAMPLE_COUNT} samples',flush=True)
    finally:
        perron_module.perron_eigenpair=solver_original
        perron_module.eigh=eigh_original
    metrics=('grid_ms','divergence_ms','similarity_preparation_ms','matrix_assembly_ms',
             'perron_solver_ms','eigensolver_ms','perron_other_ms','selection_ms',
             'perron_and_selection_ms','total_ms')
    summary=[]
    for size in SIZES:
        group=[row for row in rows if row['N']==size]
        item=dict(N=size, samples=len(group))
        for metric in metrics:
            values=np.array([row[metric] for row in group])
            item[metric]=dict(mean=float(values.mean()),se=float(values.std(ddof=1)/np.sqrt(len(values))),
                              median=float(np.median(values)))
        item['matrix_fraction_of_mean_total']=item['matrix_assembly_ms']['mean']/item['total_ms']['mean']
        item['perron_fraction_of_mean_total']=item['perron_and_selection_ms']['mean']/item['total_ms']['mean']
        summary.append(item)
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'results.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    config=dict(samples=SAMPLE_COUNT,sample_size=200,distribution='N(0,1)',
                h_interval=[.01,2.],grid='geometric',N_values=SIZES,lambda_rule='N**1.5',
                frequency_cutoff=DEFAULT_FREQUENCY_CUTOFF,frequency_intervals=DEFAULT_FREQUENCY_INTERVALS,
                previous_results=str(PREVIOUS),sample_hashes_and_selected_bandwidths_match=True,
                timer='perf_counter_ns; one initialized fresh measurement per method and sample; '
                      'method order randomized separately for each sample',
                matrix_assembly='grid generation, Fourier divergence computation, and preparation '
                                'of the shifted similarity matrix, with associated checks/orchestration',
                perron='complete existing perron_eigenpair call (including preprocessing, eigh, '
                       'normalization and residual checks) plus final peak selection',
                eigensolver='nested scipy.linalg.eigh call (two largest eigenpairs)',
                validation_not_timed=True,thread_limit_requested=1,threadpools_reported=pools,
                thread_environment={k:os.environ.get(k) for k in ('VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS',
                                                                    'OMP_NUM_THREADS','MKL_NUM_THREADS')},
                python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                platform=platform.platform())
    (OUT/'summary.json').write_text(json.dumps(dict(configuration=config,results=summary),indent=2)+'\n')
    table='| N | Matrix assembly (ms) | Perron + selection (ms) | Total (ms) | Assembly share |\n|---:|---:|---:|---:|---:|\n'
    for item in summary:
        table+=f'| {item["N"]} | {item["matrix_assembly_ms"]["mean"]:.3f} ± {item["matrix_assembly_ms"]["se"]:.3f} | '
        table+=f'{item["perron_and_selection_ms"]["mean"]:.3f} ± {item["perron_and_selection_ms"]["se"]:.3f} | '
        table+=f'{item["total_ms"]["mean"]:.3f} ± {item["total_ms"]["se"]:.3f} | {100*item["matrix_fraction_of_mean_total"]:.1f}% |\n'
    (OUT/'table.md').write_text(table)
    print(table)
    print('Maximum eigen residual:',max(r['relative_eigen_residual'] for r in rows))


if __name__=='__main__':
    main()
