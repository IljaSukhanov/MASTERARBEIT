"""Check the first bounded-Brent evaluation against endpoints for paired samples."""
import csv
import hashlib
import json
from pathlib import Path
import sys

from scipy.optimize import minimize_scalar

KDE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KDE_ROOT))
from kde_tools.experiment_inputs import paired_gaussian_sample_bank
from kde_tools.qhat import GaussianQhatObjective

OUT = KDE_ROOT / 'outputs/brent_minima_vs_subintervals/unrefined_first_evaluation.csv'
records = []
for index, sample in enumerate(paired_gaussian_sample_bank(100, 200)):
    path = KDE_ROOT / 'outputs/brent_refinements_spectral_alpha15/checkpoints' / f'sample_{index:03d}.json'
    previous = json.loads(path.read_text())
    if hashlib.sha256(sample.tobytes()).hexdigest() != previous['sample_sha256']:
        raise ValueError(f'Sample {index} differs from the benchmark.')
    objective = GaussianQhatObjective(sample)
    evaluations = []

    def observed(h):
        value = objective.energy(h)
        evaluations.append((float(h), float(value)))
        return value

    result = minimize_scalar(observed, bounds=(.01, 2.), method='bounded',
                             options={'xatol': 2e-12, 'maxiter': 1000})
    if not result.success:
        raise ArithmeticError(result.message)
    h, energy = evaluations[0]
    low, high = objective.energy(.01), objective.energy(2.)
    records.append(dict(sample_id=index, first_h=h, first_energy=energy,
                        lower_endpoint_energy=low, upper_endpoint_energy=high,
                        first_below_lower=energy < low, first_below_upper=energy < high,
                        first_below_both=energy < min(low, high),
                        final_candidate_h=float(result.x),
                        final_candidate_below_both=float(result.fun) < min(low, high)))
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
for name in ('first_below_lower','first_below_upper','first_below_both','final_candidate_below_both'):
    print(name, sum(r[name] for r in records))
print('first_h', records[0]['first_h'])
print('not lower than both', [r['sample_id'] for r in records if not r['first_below_both']])
