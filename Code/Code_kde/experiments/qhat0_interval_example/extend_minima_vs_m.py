"""Extend the paired linear scan using nested grids and the same objective."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
from scipy.optimize import minimize_scalar

KDE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KDE_ROOT))
from kde_tools.experiment_inputs import paired_gaussian_sample_bank
from kde_tools.qhat import GaussianQhatObjective

OUTPUT = KDE_ROOT / 'outputs/brent_minima_vs_m_extended'
OLD = KDE_ROOT / 'outputs/brent_linear_refinements_spectral_alpha15/results.csv'
CHECKS = KDE_ROOT / 'outputs/brent_refinements_spectral_alpha15/checkpoints'
INTERVALS = 49152
M_VALUES = sorted({2**k + 1 for k in range(5, 15)} |
                  {3*2**k + 1 for k in range(4, 15)})


def one_sample(item):
    index, sample = item
    path = OUTPUT / 'checkpoints' / f'sample_{index:03d}.json'
    digest = hashlib.sha256(sample.tobytes()).hexdigest()
    check = json.loads((CHECKS / f'sample_{index:03d}.json').read_text())
    if check['sample_sha256'] != digest:
        raise ValueError(f'Sample {index} differs from earlier experiment')
    if path.exists():
        old = json.loads(path.read_text())
        if old['sample_sha256'] == digest and old['M_values'] == M_VALUES:
            return old
        raise ValueError(f'Checkpoint mismatch for sample {index}')
    obj = GaussianQhatObjective(sample)
    fine_grid = np.linspace(.01, 2., INTERVALS + 1)
    energy = np.asarray([obj.energy(h) for h in fine_grid])
    results = []
    for m in M_VALUES:
        if INTERVALS % (m - 1):
            raise ValueError(f'Grid M={m} is not nested')
        step = INTERVALS // (m - 1)
        values, grid = energy[::step], fine_grid[::step]
        indices = np.flatnonzero((values[1:-1] < values[:-2]) &
                                 (values[1:-1] < values[2:])) + 1
        minima = []
        for i in indices:
            result = minimize_scalar(obj.energy, bounds=(grid[i-1], grid[i+1]),
                                     method='bounded',
                                     options={'xatol': 2e-12, 'maxiter': 1000})
            if not result.success:
                raise ArithmeticError(result.message)
            minima.append(float(result.x))
        results.append({'M': m, 'detected_minima': len(minima),
                        'candidate_bandwidths': minima})
    record = {'sample_id': index, 'sample_sha256': digest,
              'M_values': M_VALUES, 'results': results}
    path.write_text(json.dumps(record, indent=2) + '\n')
    return record


def main():
    (OUTPUT / 'checkpoints').mkdir(parents=True, exist_ok=True)
    bank = paired_gaussian_sample_bank(repetitions=100, maximum_n=200)
    records = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(one_sample, (i, sample)) for i, sample in enumerate(bank)]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records) % 10 == 0:
                print(f'Completed {len(records)}/100 samples', flush=True)
    records.sort(key=lambda r: r['sample_id'])
    with OLD.open(newline='') as f:
        original = {(int(r['sample_id']), int(r['M'])): int(r['detected_valleys'])
                    for r in csv.DictReader(f) if r['family'] == 'brent_linear'}
    differences = [(r['sample_id'], v['M'], v['detected_minima'],
                    original[(r['sample_id'], v['M'])])
                   for r in records for v in r['results']
                   if (r['sample_id'], v['M']) in original and
                   v['detected_minima'] != original[(r['sample_id'], v['M'])]]
    if differences:
        raise ValueError(f'Existing sample counts do not match: {differences}')
    rows = []
    for m in M_VALUES:
        counts = [next(v['detected_minima'] for v in r['results'] if v['M'] == m)
                  for r in records]
        rows.append({'M': m, 'samples': len(counts), 'detected_minima_total': sum(counts),
                     'mean_per_sample': sum(counts)/len(counts)})
    with (OUTPUT / 'minima_vs_m.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / 'configuration.json').write_text(json.dumps({
        'M_values': M_VALUES, 'samples': 100, 'n': 200, 'r': 0,
        'interval': [.01, 2.], 'spacing': 'linear', 'endpoints_included': True,
        'finest_grid_points': INTERVALS+1, 'previous_sample_counts_checked': len(original),
        'mismatches': differences, 'global_minimum_certificate': False,
    }, indent=2) + '\n')
    print('Completed all samples; all earlier counts agree.', flush=True)


if __name__ == '__main__':
    main()
