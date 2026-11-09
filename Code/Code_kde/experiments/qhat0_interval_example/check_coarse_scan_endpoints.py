"""Check the M=33 initial grid against both endpoints for each paired sample."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

KDE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KDE_ROOT))
from kde_tools.experiment_inputs import paired_gaussian_sample_bank
from kde_tools.qhat import GaussianQhatObjective

grid = np.linspace(.01, 2., 33)
records = []
for index, sample in enumerate(paired_gaussian_sample_bank(100, 200)):
    checkpoint = json.loads((KDE_ROOT / 'outputs/brent_refinements_spectral_alpha15/checkpoints' /
                             f'sample_{index:03d}.json').read_text())
    if hashlib.sha256(sample.tobytes()).hexdigest() != checkpoint['sample_sha256']:
        raise ValueError(f'Sample mismatch: {index}')
    obj = GaussianQhatObjective(sample)
    energies = np.array([obj.energy(h) for h in grid])
    valleys = np.flatnonzero((energies[1:-1] < energies[:-2]) &
                             (energies[1:-1] < energies[2:])) + 1
    records.append(dict(sample_id=index, valleys=len(valleys),
                        min_endpoint_energy=min(energies[0], energies[-1]),
                        best_grid_valley_energy=min(energies[valleys]) if len(valleys) else '',
                        valley_below_both_endpoints=bool(len(valleys) and
                            np.any(energies[valleys] < min(energies[0], energies[-1])))))
path = KDE_ROOT / 'outputs/brent_minima_vs_subintervals/coarse_grid_endpoint_checks.csv'
path.parent.mkdir(parents=True, exist_ok=True)
with path.open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
print(f"{sum(r['valley_below_both_endpoints'] for r in records)}/{len(records)} samples")
