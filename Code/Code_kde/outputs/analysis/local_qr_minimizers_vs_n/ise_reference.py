"""Compute the ISE histogram reference using the current shared minimizer.

References are regenerated from canonical sampling inputs, without reading
historical plots, selection tables, or checkpoints.
"""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
KDE_DIR = HERE.parents[2]
sys.path.insert(0, str(KDE_DIR))

from kde_tools.continuous_runs import CONTINUOUS_RUNNER_VERSION, run_continuous_point
from kde_tools.continuous_minimizers import CONTINUOUS_SELECTOR_VERSION
from kde_tools.energy_minimization import unified_energy_run_directory, energy_search_settings
from kde_tools.experiment_inputs import paired_gaussian_sample_bank, sampling_inputs

DATA_DIR = unified_energy_run_directory(KDE_DIR/'outputs/data/local_qr_minimizers_vs_n')
REFERENCE = DATA_DIR/'ise_histogram.csv'
VALIDATION = DATA_DIR/'ise_reference_validation.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_ise_reference(*, sample_sizes=tuple(range(100, 2001, 100)), repetitions=1000,
                         workers=4):
    """Return a current ISE reference, resuming current-version selections."""
    sample_sizes = tuple(sample_sizes)
    if (not sample_sizes or any(not isinstance(n, int) or n < 1 for n in sample_sizes)
            or len(set(sample_sizes)) != len(sample_sizes)):
        raise ValueError('Choose distinct positive integer sample sizes.')
    controls = energy_search_settings()
    settings = {key: controls[key] for key in ('scan_points', 'half_width', 'spatial_points')}
    settings.update(h_min=.01, h_max=2., sigma2=sampling_inputs()['paired_sample_bank']['sigma2'])
    samples = paired_gaussian_sample_bank(repetitions, max(sample_sizes))
    identity = dict(version='fresh_unified_ise_reference_v1', runner=CONTINUOUS_RUNNER_VERSION,
                    selector=CONTINUOUS_SELECTOR_VERSION, energy_search=controls,
                    settings=settings, sample_sizes=list(sample_sizes), repetitions=repetitions,
                    sample_bank_sha256=hashlib.sha256(samples.tobytes()).hexdigest(), histogram_points=201)
    if REFERENCE.exists() and VALIDATION.exists():
        saved = json.loads(VALIDATION.read_text())
        if saved.get('identity') == identity and saved.get('reference_csv_sha256') == digest(REFERENCE):
            return REFERENCE
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    centers = np.geomspace(.01, 2., 201)
    records = []
    for n in sample_sizes:
        result = run_continuous_point(n, samples, settings, ('ISE',), DATA_DIR/'ise_selections',
                                      configuration=identity, workers=workers)
        bandwidths = result['values'][:, 0, result['fields'].index('h')]
        positions = np.interp(np.log(bandwidths), np.log(centers), np.arange(201))
        counts, _ = np.histogram(positions, bins=np.arange(202)-.5)
        if counts.sum() != repetitions:
            raise ArithmeticError('ISE histogram lost sample outcomes.')
        records.extend(dict(n=n, method='ISE', bin_index=i, bandwidth_center=float(h),
                            count=int(count), relative_frequency_all_samples=float(count/repetitions))
                       for i, (h, count) in enumerate(zip(centers, counts)))
    temporary = REFERENCE.with_suffix('.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(REFERENCE)
    VALIDATION.write_text(json.dumps(dict(identity=identity,
        criterion='sample-specific ISE against the known normal density',
        all_histograms_sum_to_repetitions=True, reference_csv_sha256=digest(REFERENCE)), indent=2)+'\n')
    return REFERENCE


if __name__ == '__main__':
    print(ensure_ise_reference())
