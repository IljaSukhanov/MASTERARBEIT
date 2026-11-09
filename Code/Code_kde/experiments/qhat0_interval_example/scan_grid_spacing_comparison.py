"""Paired comparison of linear and geometric derivative-free valley scans.

Reuse the saved, resolution-checked derivative-based references for the same
100 samples; reference loading and validation are outside timed calls. Both
scans minimize in h, compare endpoints, and use the same number of points.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import scipy
from threadpoolctl import threadpool_limits

KDE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KDE_ROOT))
from kde_tools.experiment_inputs import paired_gaussian_sample_bank
from kde_tools.qhat import grid_brent_qhat_minimum

M = 100
SAMPLE_SIZE = 200
GRID_POINTS = 257
LOG_TOLERANCE = 1e-6
SPACINGS = ("geometric", "linear")
REFERENCE_DIR = KDE_ROOT / "outputs/grid_brent_spectral_m100_comparison/checkpoints"
OUTPUT_DIR = KDE_ROOT / "outputs/scan_grid_spacing_comparison"


def load_reference(index, sample):
    record = json.loads((REFERENCE_DIR / f"sample_{index:03d}.json").read_text())
    if record["sample_sha256"] != hashlib.sha256(sample.tobytes()).hexdigest():
        raise ValueError(f"Reference sample identity mismatch for sample {index}.")
    if record["reference"]["diagnostics"]["resolution_stable"] is not True:
        raise ValueError(f"Reference is not resolution-stable for sample {index}.")
    return record["reference"]


def analyze(sample_id, spacing, result, reference, seconds):
    target = reference["global_minimum"]
    minima = reference["local_minima"]
    valleys = result["valley_candidates"]
    all_valleys_match = len(minima) == len(valleys) and all(
        abs(np.log(a["h"]/b["h"])) <= LOG_TOLERANCE for a, b in zip(minima, valleys))
    return dict(sample_id=sample_id, spacing=spacing, grid_points=GRID_POINTS,
                selected_h=result["h"], reference_h=target["h"],
                selected_energy=result["energy"], reference_energy=target["energy"],
                relative_bandwidth_error=abs(result["h"]-target["h"])/target["h"],
                relative_energy_excess=result["energy"]/target["energy"]-1,
                selected_bandwidth_matches=bool(abs(np.log(result["h"]/target["h"])) <= LOG_TOLERANCE),
                reference_local_minima=len(minima), detected_valleys=len(valleys),
                all_valleys_match=all_valleys_match, endpoint_selected=result["boundary"] != 0,
                grid_evaluations=result["grid_function_evaluations"],
                refinement_evaluations=result["minimizer_function_evaluations"],
                time_ms=1000*seconds)


def summarize(records, spacing):
    group = [rec for rec in records if rec["spacing"] == spacing]
    times = np.array([rec["time_ms"] for rec in group])
    errors = np.array([rec["relative_bandwidth_error"] for rec in group])
    return dict(spacing=spacing, samples=len(group),
                selected_matches=sum(rec["selected_bandwidth_matches"] for rec in group),
                selected_mismatch_samples=[rec["sample_id"] for rec in group if not rec["selected_bandwidth_matches"]],
                all_valleys_match=sum(rec["all_valleys_match"] for rec in group),
                valley_mismatch_samples=[rec["sample_id"] for rec in group if not rec["all_valleys_match"]],
                mean_relative_bandwidth_error=float(errors.mean()),
                maximum_relative_bandwidth_error=float(errors.max()),
                mean_time_ms=float(times.mean()), median_time_ms=float(np.median(times)),
                standard_error_time_ms=float(times.std(ddof=1)/np.sqrt(len(times))),
                mean_refinement_evaluations=float(np.mean([r["refinement_evaluations"] for r in group])))


def main():
    samples = paired_gaussian_sample_bank(repetitions=M, maximum_n=SAMPLE_SIZE)
    results, candidate_records = [], []
    with threadpool_limits(limits=1):
        for spacing in SPACINGS:
            grid_brent_qhat_minimum(samples[0], grid_points=GRID_POINTS, grid_spacing=spacing)
        for index, sample in enumerate(samples):
            reference = load_reference(index, sample)
            rng = np.random.default_rng(np.random.SeedSequence([20261109, index]))
            for spacing in rng.permutation(SPACINGS):
                start = perf_counter()
                result = grid_brent_qhat_minimum(sample, grid_points=GRID_POINTS, grid_spacing=spacing)
                seconds = perf_counter()-start
                results.append(analyze(index, spacing, result, reference, seconds))
                candidate_records.append(dict(sample_id=index, spacing=spacing,
                                              selected_h=result["h"],
                                              valleys=result["valley_candidates"],
                                              reference_valleys=reference["local_minima"]))
            if (index+1) % 20 == 0:
                print(f"Completed {index+1}/{M} paired samples", flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR/'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    configuration = dict(samples=M, sample_size=SAMPLE_SIZE, interval=[0.01,2.0], r=0.0,
                         grid_points=GRID_POINTS, spacings=SPACINGS, log_bandwidth_match_tolerance=LOG_TOLERANCE,
                         numerical_threads=1, seed_for_method_order=20261109,
                         timing='One initialized fresh call per sample and spacing, randomized order; '
                                'includes pair distances, entire scan, all valley refinements, and endpoint comparison. '
                                'Excludes sample loading, references, validation and saving.',
                         reference='Previously saved analytic derivative-based, resolution-checked search; '
                                   'sample hashes checked; not a global certificate.',
                         python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)
    summary = dict(configuration=configuration, results=[summarize(results, spacing) for spacing in SPACINGS])
    (OUTPUT_DIR/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (OUTPUT_DIR/'candidate_details.json').write_text(json.dumps(candidate_records, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
