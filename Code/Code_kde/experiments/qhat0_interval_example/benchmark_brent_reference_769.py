"""Benchmark five paired Brent searches and rebase spectral errors to M=769."""
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
from kde_tools.qhat import bounded_qhat_minimum, grid_brent_qhat_minimum

OUT = KDE_ROOT / 'outputs/brent_spectral_comparison_m769_reference'
SPECTRAL = KDE_ROOT / 'outputs/spectral_runtime_breakdown_alpha15_n800/results.csv'
SUMMARY = KDE_ROOT / 'outputs/spectral_runtime_breakdown_alpha15_n800/summary.json'
PREVIOUS = KDE_ROOT / 'outputs/brent_linear_refinements_spectral_alpha15/results.csv'
CHECKPOINTS = KDE_ROOT / 'outputs/brent_refinements_spectral_alpha15/checkpoints'
METHODS = ('unrefined', 33, 129, 513, 769)
SIZES = (100, 200, 400, 600, 800)


def search(sample, method):
    if method == 'unrefined':
        return bounded_qhat_minimum(sample)
    return grid_brent_qhat_minimum(sample, grid_points=method, grid_spacing='linear')


def main():
    samples = paired_gaussian_sample_bank(100, 200)
    with PREVIOUS.open(newline='') as f:
        old = list(csv.DictReader(f))
    previous = {(int(row['sample_id']), row['method']): float(row['selected_h'])
                for row in old if row['family'].startswith('brent')}
    records = []
    with threadpool_limits(limits=1):
        pools = threadpool_info()
        for method in METHODS:
            search(samples[0], method)
        for index, sample in enumerate(samples):
            old_sample = json.loads((CHECKPOINTS / f'sample_{index:03d}.json').read_text())
            if hashlib.sha256(sample.tobytes()).hexdigest() != old_sample['sample_sha256']:
                raise ValueError(f'Sample mismatch: {index}')
            order = np.random.default_rng(np.random.SeedSequence([20261109, index])).permutation(len(METHODS))
            for mpos in order:
                method = METHODS[int(mpos)]
                t0 = perf_counter_ns()
                result = search(sample, method)
                elapsed = (perf_counter_ns()-t0)/1e6
                method_key = 'brent_unrefined' if method == 'unrefined' else f'brent_linear_{method}'
                if method != 769 and abs(result['h']-previous[(index, method_key)]) > 1e-12:
                    raise ArithmeticError(f'Earlier bandwidth mismatch: {index}, {method_key}')
                records.append(dict(sample_id=index, method=method_key, time_ms=elapsed,
                                    selected_h=result['h'], energy=result['energy'],
                                    detected_valleys=result.get('valleys_detected', ''),
                                    endpoint_selected=result['boundary'] != 0))
            if (index+1) % 20 == 0:
                print(f'Finished {index+1}/100 samples', flush=True)
    references = {r['sample_id']: r['selected_h'] for r in records if r['method']=='brent_linear_769'}
    if len(references) != 100:
        raise ValueError('Incomplete references')
    with SPECTRAL.open(newline='') as f:
        spectral = list(csv.DictReader(f))
    spectral_summary = {r['N']: r for r in json.loads(SUMMARY.read_text())['results']}

    def errors(group):
        return [100*abs(float(r['selected_h'])-references[int(r['sample_id'])]) /
                references[int(r['sample_id'])] for r in group]

    rows = []
    for method in METHODS:
        key = 'brent_unrefined' if method == 'unrefined' else f'brent_linear_{method}'
        group = [r for r in records if r['method']==key]
        times = [r['time_ms'] for r in group]
        e = errors(group)
        rows.append(dict(method=key, total_ms=statistics.mean(times),
                         total_se_ms=statistics.stdev(times)/10, assembly_ms='', perron_ms='',
                         mean_error_percent=statistics.mean(e), max_error_percent=max(e),
                         mean_detected_minima='' if method == 'unrefined' else
                         statistics.mean(r['detected_valleys'] for r in group),
                         endpoints=sum(r['endpoint_selected'] for r in group)))
    for n in SIZES:
        group = [r for r in spectral if int(r['N'])==n]
        if {int(r['sample_id']) for r in group} != set(references):
            raise ValueError(f'Incomplete or unpaired spectral measurements: {n}')
        item = spectral_summary[n]
        e = errors(group)
        rows.append(dict(method=f'spectral_{n}', total_ms=item['total_ms']['mean'],
                         total_se_ms=item['total_ms']['se'],
                         assembly_ms=item['matrix_assembly_ms']['mean'],
                         perron_ms=item['perron_and_selection_ms']['mean'],
                         mean_error_percent=statistics.mean(e), max_error_percent=max(e),
                         mean_detected_minima='',
                         endpoints=sum(int(r['selected_index']) in (0,n-1) for r in group)))
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'brent_results.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    with (OUT/'comparison.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (OUT/'configuration.json').write_text(json.dumps(dict(
        samples=100, sample_size=200, distribution='N(0,1)',
        interval=[.01,2.], r=0, spacing='linear', bandwidth_reference='brent_linear_769',
        timing='fresh warmed calls in randomized order per sample; no cross-method objective cache; '
               'spectral times are from a separate earlier instrumented run',
        requested_thread_limit=1, detected_threadpools=pools,
        previous_brent_selected_bandwidths_agreed=True), indent=2)+'\n')
    print((OUT/'comparison.csv').read_text(), flush=True)


if __name__ == '__main__':
    main()
