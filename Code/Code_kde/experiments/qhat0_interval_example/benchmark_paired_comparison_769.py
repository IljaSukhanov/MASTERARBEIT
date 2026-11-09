"""Jointly time all comparison methods and use paired Brent M=769 reference."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
from time import perf_counter_ns

for name in ('VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(name, '1')

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

KDE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KDE_ROOT))
from kde_tools.experiment_inputs import paired_gaussian_sample_bank
import benchmark_brent_reference_769 as brent
import profile_spectral_runtime as profile

OUT = brent.OUT
METHODS = [('brent', x) for x in brent.METHODS] + [('spectral', n) for n in brent.SIZES]
with (OUT/'brent_results.csv').open(newline='') as f:
    previous_brent = {(int(r['sample_id']),r['method']):float(r['selected_h']) for r in csv.DictReader(f)}
with brent.SPECTRAL.open(newline='') as f:
    previous_spectral = {(int(r['sample_id']),int(r['N'])):float(r['selected_h']) for r in csv.DictReader(f)}

timing = {}
perron = profile.perron_module
old_solver, old_eigh = perron.perron_eigenpair, perron.eigh


def timed_solver(*args, **kwargs):
    timing['solver_calls'] = timing.get('solver_calls',0)+1
    start = perf_counter_ns()
    try:
        return old_solver(*args, **kwargs)
    finally:
        timing['solver_ns'] = timing.get('solver_ns',0)+perf_counter_ns()-start


def timed_eigh(*args, **kwargs):
    timing['eigh_calls'] = timing.get('eigh_calls',0)+1
    start = perf_counter_ns()
    try:
        return old_eigh(*args, **kwargs)
    finally:
        timing['eigh_ns'] = timing.get('eigh_ns',0)+perf_counter_ns()-start


def run(sample, family, size):
    if family == 'spectral':
        h, index, residual, durations = profile.benchmark_one(sample, size, timing)
        return dict(selected_h=h, selected_index=index, relative_eigen_residual=residual,
                    total_ms=durations['total_ms'], assembly_ms=durations['matrix_assembly_ms'],
                    perron_ms=durations['perron_and_selection_ms'],
                    detected_valleys='', endpoint_selected=index in (0,size-1))
    t0 = perf_counter_ns()
    result = brent.search(sample, size)
    elapsed = (perf_counter_ns()-t0)/1e6
    return dict(selected_h=result['h'], selected_index='', relative_eigen_residual='',
                total_ms=elapsed, assembly_ms='', perron_ms='',
                detected_valleys=result.get('valleys_detected',''),
                endpoint_selected=result['boundary'] != 0)


def main():
    perron.perron_eigenpair, perron.eigh = timed_solver, timed_eigh
    records=[]
    samples=paired_gaussian_sample_bank(100,200)
    try:
        with threadpool_limits(limits=1):
            pools=threadpool_info()
            for family,size in METHODS:
                run(samples[0],family,size)
            for index,sample in enumerate(samples):
                check=json.loads((brent.CHECKPOINTS/f'sample_{index:03d}.json').read_text())
                if hashlib.sha256(sample.tobytes()).hexdigest()!=check['sample_sha256']:
                    raise ValueError(f'Sample {index} does not match earlier experiment.')
                order=np.random.default_rng(np.random.SeedSequence([20261112,index])).permutation(len(METHODS))
                for pos in order:
                    family,size=METHODS[int(pos)]
                    result=run(sample,family,size)
                    key = ('brent_unrefined' if size=='unrefined' else f'brent_linear_{size}') if family=='brent' else f'spectral_{size}'
                    expected=previous_brent[(index,key)] if family=='brent' else previous_spectral[(index,size)]
                    if abs(result['selected_h']-expected)>1e-12:
                        raise ArithmeticError(f'Selected bandwidth differs: {index}, {key}')
                    records.append(dict(sample_id=index,method=key,**result))
                if (index+1)%20==0:
                    print(f'Finished {index+1}/100 paired samples',flush=True)
    finally:
        perron.perron_eigenpair, perron.eigh = old_solver, old_eigh

    refs={r['sample_id']:r['selected_h'] for r in records if r['method']=='brent_linear_769'}
    rows=[]
    for family,size in METHODS:
        key=('brent_unrefined' if size=='unrefined' else f'brent_linear_{size}') if family=='brent' else f'spectral_{size}'
        group=[r for r in records if r['method']==key]
        if {r['sample_id'] for r in group}!=set(range(100)):
            raise ValueError(f'Incomplete records: {key}')
        times=[r['total_ms'] for r in group]
        errors=[100*abs(r['selected_h']-refs[r['sample_id']])/refs[r['sample_id']] for r in group]
        rows.append(dict(method=key,total_ms=statistics.mean(times),
                         total_se_ms=statistics.stdev(times)/10,
                         assembly_ms=statistics.mean(r['assembly_ms'] for r in group) if family=='spectral' else '',
                         perron_ms=statistics.mean(r['perron_ms'] for r in group) if family=='spectral' else '',
                         mean_error_percent=statistics.mean(errors),max_error_percent=max(errors),
                         mean_detected_minima=statistics.mean(r['detected_valleys'] for r in group)
                             if family=='brent' and size!='unrefined' else '',
                         endpoints=sum(r['endpoint_selected'] for r in group)))
    with (OUT/'paired_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    with (OUT/'comparison_paired.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (OUT/'paired_configuration.json').write_text(json.dumps(dict(
        samples=100,n=200,distribution='N(0,1)',r=0,interval=[.01,2],
        reference='linear refined Brent M=769, sample by sample',
        spectral_N=list(brent.SIZES),lambda_rule='N**1.5',spectral_grid='geometric',
        brent_grid='linear',brent_M=[33,129,513,769],
        timing='All ten methods timed on each paired sample, in randomized order; warmups included before timing. '
               'Spectral calls instrumented for phase split. Independent fresh calls, no reused objective cache.',
        all_1000_bandwidths_match_prior_runs=True,requested_thread_limit=1,
        detected_threadpools=pools),indent=2)+'\n')
    print((OUT/'comparison_paired.csv').read_text(),flush=True)


if __name__=='__main__':
    main()
