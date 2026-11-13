"""Evaluate the additional valid coarse linear Brent grids on paired samples."""
import csv
import hashlib
import json
from pathlib import Path
import sys

KDE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KDE_ROOT))
from kde_tools.experiment_inputs import paired_gaussian_sample_bank
from kde_tools.qhat import grid_brent_qhat_minimum

OUTPUT = KDE_ROOT / 'outputs/brent_minima_vs_subintervals'
SMALL = (2, 3, 4, 6, 8, 12, 16, 24)
records = []
for index, sample in enumerate(paired_gaussian_sample_bank(100, 200)):
    path = KDE_ROOT / 'outputs/brent_refinements_spectral_alpha15/checkpoints' / f'sample_{index:03d}.json'
    previous = json.loads(path.read_text())
    if hashlib.sha256(sample.tobytes()).hexdigest() != previous['sample_sha256']:
        raise ValueError(f'Sample {index} differs from earlier experiment')
    for intervals in SMALL:
        result = grid_brent_qhat_minimum(sample, grid_points=intervals+1, grid_spacing='linear')
        records.append({'sample_id': index, 'subintervals': intervals, 'M': intervals+1,
                        'detected_minima': result['valleys_detected'],
                        'endpoint_selected': bool(result['boundary'])})
OUTPUT.mkdir(parents=True, exist_ok=True)
with (OUTPUT / 'additional_coarse_samples.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
print('subintervals, detected minima across 100 samples')
for intervals in SMALL:
    print(intervals, sum(r['detected_minima'] for r in records if r['subintervals'] == intervals))
