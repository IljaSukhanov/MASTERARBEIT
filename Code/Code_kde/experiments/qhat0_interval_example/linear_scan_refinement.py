"""Compare three linear-scan resolutions on the same 100 paired samples.

The saved resolution-checked derivative-based reference is used only for
validation, outside the timed derivative-free scan and Brent refinements.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from scan_grid_spacing_comparison import (
    KDE_ROOT, LOG_TOLERANCE, M, SAMPLE_SIZE, analyze, load_reference,
)
from kde_tools.experiment_inputs import paired_gaussian_sample_bank
from kde_tools.qhat import grid_brent_qhat_minimum

GRID_COUNTS = (257, 513, 1025)
OUTPUT = KDE_ROOT / 'outputs/linear_scan_refinement_comparison'


def main():
    samples = paired_gaussian_sample_bank(repetitions=M, maximum_n=SAMPLE_SIZE)
    rows, details = [], []
    with threadpool_limits(limits=1):
        for points in GRID_COUNTS:
            grid_brent_qhat_minimum(samples[0], grid_points=points, grid_spacing='linear')
        for index, sample in enumerate(samples):
            reference = load_reference(index, sample)
            rng = np.random.default_rng(np.random.SeedSequence([20261110, index]))
            for points in rng.permutation(GRID_COUNTS):
                points = int(points)
                started = perf_counter()
                result = grid_brent_qhat_minimum(sample, grid_points=points, grid_spacing='linear')
                seconds = perf_counter() - started
                row = analyze(index, 'linear', result, reference, seconds)
                row['grid_points'] = points
                rows.append(row)
                details.append(dict(sample_id=index, grid_points=points, selected_h=result['h'],
                                    valleys=result['valley_candidates'],
                                    reference_valleys=reference['local_minima']))
            if (index+1) % 20 == 0:
                print(f'Completed {index+1}/{M} paired samples', flush=True)
    summaries = []
    for points in GRID_COUNTS:
        subset = [row for row in rows if row['grid_points'] == points]
        times = np.array([row['time_ms'] for row in subset])
        summaries.append(dict(
            grid_points=points, bandwidth_step=(2.0-0.01)/(points-1),
            samples=len(subset),
            selected_matches=sum(row['selected_bandwidth_matches'] for row in subset),
            selected_mismatch_samples=[row['sample_id'] for row in subset if not row['selected_bandwidth_matches']],
            all_valleys_match=sum(row['all_valleys_match'] for row in subset),
            valley_mismatch_samples=[row['sample_id'] for row in subset if not row['all_valleys_match']],
            mean_time_ms=float(times.mean()), median_time_ms=float(np.median(times)),
            standard_error_time_ms=float(times.std(ddof=1)/np.sqrt(len(times))),
            maximum_relative_bandwidth_error=max(row['relative_bandwidth_error'] for row in subset)))
    summary = dict(
        samples=M, sample_size=SAMPLE_SIZE, interval=[0.01,2.0], r=0.0,
        spacing='linear', match_log_bandwidth_tolerance=LOG_TOLERANCE,
        reference='Saved derivative-based numerical search; sample hashes verified; not a certificate.',
        timing='One initialized fresh call per grid and sample; randomized method order; '
               'includes pair-distance preparation, scan, Brent refinements and endpoint comparison; '
               'excludes reference loading and validation.',
        numerical_threads=1, summaries=summaries)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT/'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (OUTPUT/'candidate_details.json').write_text(json.dumps(details, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
