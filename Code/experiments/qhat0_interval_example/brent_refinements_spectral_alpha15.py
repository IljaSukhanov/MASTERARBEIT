"""Paired benchmark of Brent refinements and spectral selection for alpha=3/2.

Use 100 samples, the numerical references from the earlier checked search,
N=100,200,400, and linear scans at six grid sizes. Linear M=1025 is the
reference for bandwidth errors. The separate derivative search is used for
checking detected valleys. Each
method starts with raw observations; reference work and validation are not
timed. Existing reference results are checked against sample hashes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import scipy
from threadpoolctl import threadpool_info, threadpool_limits

KDE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KDE_ROOT))
from kde_tools.bandwidths import bandwidth_grid_linear_in_psi_beta
from kde_tools.experiment_inputs import paired_gaussian_sample_bank, sampling_inputs
from kde_tools.energy_minimization import EnergySearchConfig
from kde_tools.fourier.similarities import (
    DEFAULT_FREQUENCY_CUTOFF, DEFAULT_FREQUENCY_INTERVALS,
    beta2_divergence_matrix_fourier, gaussian_kde_l2_gram_fourier,
)
from kde_tools.perron import PERRON_SOLVER_VERSION, perron_eigenpair_from_divergence
from kde_tools.qhat import GaussianQhatObjective, bounded_qhat_minimum, grid_brent_qhat_minimum, minimize_qhat
from kde_tools.similarities import beta2_divergence_matrix, squared_l2_distance_from_gram

N_VALUES = (100, 200, 400)
M_VALUES = (33, 65, 129, 257, 513, 1025)
SPACINGS = ('linear',)
METHODS = (('brent_unrefined', None),
           *((f'brent_{spacing}', points) for spacing in SPACINGS for points in M_VALUES),
           *((f'spectral', n) for n in N_VALUES))
REFERENCE_DIR = KDE_ROOT/'outputs/grid_brent_spectral_m100_comparison/checkpoints'
DEFAULT_OUTPUT = KDE_ROOT/'outputs/brent_linear_refinements_spectral_alpha15_new_run'
ORDER_SEED = 20261109
TOLERANCE = 1e-6


def method_key(method, size):
    return method if size is None else f'{method}_{size}'


def load_reference(index, sample):
    path = REFERENCE_DIR/f'sample_{index:03d}.json'
    if not path.is_file():
        # The original grid-comparison run was interrupted. Use the same
        # exact reference search as that experiment for missing checkpoints.
        reference = minimize_qhat(sample, .01, 2., r=0.,
                                  config=EnergySearchConfig(scan_backend='exact'))
        if reference['diagnostics']['resolution_stable'] is not True:
            raise ValueError(f'Recomputed reference not stable at index {index}.')
        return reference
    record = json.loads(path.read_text())
    sample_hash = hashlib.sha256(sample.tobytes()).hexdigest()
    if sample_hash != record['sample_sha256']:
        raise ValueError(f'Reference sample mismatch at index {index}.')
    reference = record['reference']
    if reference['diagnostics']['resolution_stable'] is not True:
        raise ValueError(f'Reference not stable at index {index}.')
    return reference


def execute(method, size, sample):
    if method == 'brent_unrefined':
        return bounded_qhat_minimum(sample)
    if method.startswith('brent_'):
        return grid_brent_qhat_minimum(sample, grid_points=size,
                                      grid_spacing=method.removeprefix('brent_'))
    grid = bandwidth_grid_linear_in_psi_beta(.01, 2., 2., size)
    divergence = beta2_divergence_matrix_fourier(sample, grid)
    perron = perron_eigenpair_from_divergence(divergence, float(size**1.5))
    return dict(h=float(grid[perron.selected_index]), grid=grid, divergence=divergence,
                perron=perron, selected_index=perron.selected_index)


def process_sample(index, sample):
    reference = load_reference(index, sample)
    sample_hash = hashlib.sha256(sample.tobytes()).hexdigest()
    rng = np.random.default_rng(np.random.SeedSequence([ORDER_SEED, index]))
    order = rng.permutation(len(METHODS))
    measured = {}
    for item in order:
        method, size = METHODS[int(item)]
        start = perf_counter()
        result = execute(method, size, sample)
        measured[method_key(method, size)] = (result, perf_counter()-start)
    # All energy/error calculations and numerical checks occur outside timing.
    target = measured['brent_linear_1025'][0]
    objective = GaussianQhatObjective(sample)
    rows, checks = [], []
    for method, size in METHODS:
        key = method_key(method, size)
        result, seconds = measured[key]
        h = result['h']
        energy = objective.energy(h)
        minima = reference['local_minima']
        all_valleys_match = None
        if method.startswith('brent_') and method != 'brent_unrefined':
            valleys = result['valley_candidates']
            all_valleys_match = bool(len(valleys) == len(minima) and all(
                abs(np.log(v['h']/m['h'])) <= TOLERANCE for v, m in zip(valleys, minima)))
        rows.append(dict(
            sample_id=index, method=key, family=method,
            N=size if method == 'spectral' else None,
            M=size if method.startswith('brent_') and method != 'brent_unrefined' else None,
            grid_spacing=method.removeprefix('brent_') if method in ('brent_linear','brent_geometric') else None,
            time_ms=1000*seconds, selected_h=h, energy=energy,
            reference_h=target['h'], reference_energy=target['energy'],
            relative_bandwidth_error_percent=100*abs(h-target['h'])/target['h'],
            relative_energy_excess_percent=100*(energy/target['energy']-1),
            reference_bandwidth_match=bool(abs(np.log(h/target['h'])) <= TOLERANCE),
            reference_valleys=len(minima), detected_valleys=result.get('valleys_detected'),
            all_reference_valleys_match=all_valleys_match,
            grid_evaluations=result.get('grid_function_evaluations'),
            brent_function_evaluations=result.get('minimizer_function_evaluations'),
            endpoint_selected=bool(result.get('boundary', 0)),
        ))
        if method == 'spectral':
            gram = gaussian_kde_l2_gram_fourier(
                sample, result['grid'], frequency_cutoff=DEFAULT_FREQUENCY_CUTOFF,
                frequency_points=2*DEFAULT_FREQUENCY_INTERVALS+1)
            fine_divergence = 0.5*squared_l2_distance_from_gram(gram)
            fine = perron_eigenpair_from_divergence(fine_divergence, float(size**1.5))
            check = dict(N=size, refinement_matches=bool(fine.selected_index == result['selected_index']),
                         coarse_index=int(result['selected_index']), fine_index=int(fine.selected_index),
                         relative_residual=float(result['perron'].relative_residual),
                         max_divergence_change=float(np.max(np.abs(result['divergence']-fine_divergence))))
            if size == 100 and index in (0,54,98):
                exact = beta2_divergence_matrix(sample, result['grid'])
                exact_perron = perron_eigenpair_from_divergence(exact, float(size**1.5))
                check['analytic_overlap_matches'] = bool(exact_perron.selected_index == result['selected_index'])
                check['max_analytic_divergence_error'] = float(np.max(np.abs(exact-result['divergence'])))
            checks.append(check)
    return dict(sample_id=index, sample_sha256=sample_hash, reference=reference,
                measured_order=[method_key(*METHODS[int(k)]) for k in order],
                rows=rows, spectral_checks=checks)


def label(method, size):
    if method == 'brent_unrefined':
        return 'Brent, unrefined'
    if method == 'spectral':
        return f'Spectral, N={size}'
    return f'Brent, {method.removeprefix("brent_")}, M={size}'


def aggregate(records):
    summaries = []
    for method, size in METHODS:
        key = method_key(method, size)
        group = [next(row for row in record['rows'] if row['method'] == key) for record in records]
        times = np.asarray([row['time_ms'] for row in group])
        bw = np.asarray([row['selected_h'] for row in group])
        err = np.asarray([row['relative_bandwidth_error_percent'] for row in group])
        energies = np.asarray([row['relative_energy_excess_percent'] for row in group])
        valleys = [row['all_reference_valleys_match'] for row in group if row['all_reference_valleys_match'] is not None]
        summaries.append(dict(method=key, label=label(method,size), family=method, size=size,
                              samples=len(group), mean_time_ms=float(times.mean()),
                              se_time_ms=float(times.std(ddof=1)/np.sqrt(len(times))) if len(times)>1 else None,
                              median_time_ms=float(np.median(times)), mean_selected_h=float(bw.mean()),
                              mean_relative_bandwidth_error_percent=float(err.mean()),
                              median_relative_bandwidth_error_percent=float(np.median(err)),
                              max_relative_bandwidth_error_percent=float(err.max()),
                              mean_relative_energy_excess_percent=float(energies.mean()),
                              detected_minimum_candidates_total=(
                                  len(group) if method == 'brent_unrefined' else
                                  sum(row['detected_valleys'] for row in group)
                                  if method.startswith('brent_') else None),
                              endpoint_selections_total=(
                                  sum(row['endpoint_selected'] for row in group)
                                  if method.startswith('brent_') else None),
                              reference_matches=sum(row['reference_bandwidth_match'] for row in group),
                              all_detected_valleys_match=sum(valleys) if valleys else None,
                              mismatch_samples=[row['sample_id'] for row in group if not row['reference_bandwidth_match']]))
    return summaries


def write_table(output, summaries, sample_count):
    header = '| Method | Mean time ± SE (ms) | Mean bandwidth | Mean absolute relative error (%) | Minimum candidates | Endpoint selections |\n|---|---:|---:|---:|---:|---:|\n'
    body = ''.join(f'| {r["label"]} | {r["mean_time_ms"]:.2f} ± {r["se_time_ms"]:.2f} | '
                   f'{r["mean_selected_h"]:.5f} | {r["mean_relative_bandwidth_error_percent"]:.6g} | '
                   f'{r["detected_minimum_candidates_total"] if r["detected_minimum_candidates_total"] is not None else "—"} | '
                   f'{r["endpoint_selections_total"] if r["endpoint_selections_total"] is not None else "—"} |\n' for r in summaries)
    (output/'table.md').write_text(header+body)
    lines = [r'\begin{table}[htbp]', r'    \centering', r'    \footnotesize',
             r'    \setlength{\tabcolsep}{3pt}', r'    \begin{tabular}{lccccc}', r'        \toprule',
             r'        Method & \shortstack{Mean time\\$\pm$ SE (ms)} & \shortstack{Mean\\bandwidth} & \shortstack{Mean absolute\\rel. error (\%)} & \shortstack{Minima\\found} & \shortstack{Endpoint\\selected} \\', r'        \midrule']
    last_family = None
    for r in summaries:
        if last_family is not None and r['family'] != last_family:
            lines.append(r'        \midrule')
        if r['family'] == 'brent_unrefined': name = 'Brent, unrefined'
        elif r['family'] == 'spectral': name = f'Spectral, $N={r["size"]}$'
        else: name = f'Brent, {r["family"].removeprefix("brent_")}, $M={r["size"]}$'
        e = r['mean_relative_bandwidth_error_percent']
        err = ('0' if r['method'] == 'brent_linear_1025' else
               r'$<10^{-5}$' if e < 1e-5 else f'{e:.2e}' if e < .001 else f'{e:.3f}')
        if 'e' in err:
            mantissa, exponent = err.split('e')
            err = f'${mantissa}\\times 10^{{{int(exponent)}}}$'
        candidates = str(r['detected_minimum_candidates_total']) if r['detected_minimum_candidates_total'] is not None else r'---'
        endpoints = str(r['endpoint_selections_total']) if r['endpoint_selections_total'] is not None else r'---'
        lines.append(f'        {name} & ${r["mean_time_ms"]:.2f} \\pm {r["se_time_ms"]:.2f}$ & '
                     f'${r["mean_selected_h"]:.5f}$ & {err} & {candidates} & {endpoints} \\\\')
        last_family = r['family']
    lines.extend([r'        \bottomrule', r'    \end{tabular}',
                  f'    \\caption{{Comparison over {sample_count} samples of size 200 from $N(0,1)$ on $[0.01,2]$. '
                  r'The spectral method uses $\lambda_N=N^{3/2}$ and geometric grids. '
                  r'For refined Brent searches, $M$ is the number of scan grid points. '
                  r'Times are means $\pm$ standard errors. Relative bandwidth errors are computed '
                  r'for each sample against the linear refined Brent result with $M=1025$; '
                  r'that reference row has zero error by construction. Minima found is the '
                  r'total number of numerical interior candidates over all samples; the unrefined '
                  r'method returns one per sample. Endpoint selections are counted across samples.}',
                  r'\end{table}'])
    (output/'table.tex').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not 2 <= args.samples <= 100:
        raise ValueError('samples must be between 2 and 100.')
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    samples = paired_gaussian_sample_bank(repetitions=args.samples, maximum_n=200)
    config = dict(version='brent_linear_refinements_spectral_alpha15_v2', samples=args.samples,
                  sample_size=200, distribution='N(0,1)', sample_seed=sampling_inputs()['paired_sample_bank']['seed'],
                  sample_sha256=hashlib.sha256(samples.tobytes()).hexdigest(), h_interval=[.01,2.], r=0.,
                  N_values=N_VALUES, lambda_rule='N**1.5', spectral_grid='geometric',
                  M_values=M_VALUES, brent_scan_spacings=SPACINGS,
                  brent_grid_point_meaning='M includes both endpoints',
                  brent_xatol=2e-12, brent_maxiter=1000,
                  reference='linear refined Brent with M=1025 for the same sample; not a global certificate',
                  derivative_validation_reference='previous resolution-checked analytic derivative-based '
                                                  'search; sample hashes checked',
                  match_log_bandwidth_tolerance=TOLERANCE,
                  timing='One initialized fresh call per sample and method; randomized separately per sample; '
                         'a one-thread limit is requested but system libraries may not be detectable. '
                         'Includes sample-specific pair distances, grid scan and Brent refinements '
                         'or similarity construction and eigensolve, plus endpoint comparisons. '
                         'Excludes the separate derivative-based validation, imports, sample generation and saving.',
                  thread_environment={k:os.environ.get(k) for k in (
                      'VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
                  quadrature_cutoff=DEFAULT_FREQUENCY_CUTOFF, quadrature_intervals=DEFAULT_FREQUENCY_INTERVALS,
                  spectral_solver=PERRON_SOLVER_VERSION,
                  python_version=platform.python_version(), numpy_version=np.__version__, scipy_version=scipy.__version__,
                  platform=platform.platform())
    config_path=output/'configuration.json'
    if config_path.exists() and json.loads(config_path.read_text()) != json.loads(json.dumps(config)):
        raise ValueError('Existing run configuration differs; choose a new output directory.')
    config_path.write_text(json.dumps(config,indent=2)+'\n')
    checkpoints=output/'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    records=[]
    start=perf_counter()
    with threadpool_limits(limits=1):
        for method,size in METHODS:
            execute(method,size,samples[0])
        pools=[{k: info.get(k) for k in ('internal_api','prefix','num_threads','version','architecture')}
               for info in threadpool_info()]
        for index,sample in enumerate(samples):
            path=checkpoints/f'sample_{index:03d}.json'
            if path.exists():
                record=json.loads(path.read_text())
                if record['sample_sha256'] != hashlib.sha256(sample.tobytes()).hexdigest():
                    raise ValueError('Checkpoint sample mismatch.')
            else:
                record=process_sample(index,sample)
                temporary=path.with_suffix('.tmp')
                temporary.write_text(json.dumps(record,indent=2)+'\n')
                temporary.replace(path)
            records.append(record)
            if (index+1)%5 == 0:
                print(f'Completed {index+1}/{args.samples} paired samples ({perf_counter()-start:.1f} s)',flush=True)
    summaries=aggregate(records)
    all_rows=[row for record in records for row in record['rows']]
    if len(all_rows) != args.samples*len(METHODS) or len({(r['sample_id'],r['method']) for r in all_rows}) != len(all_rows):
        raise ArithmeticError('Incomplete or duplicate measurements.')
    checks=[(record['sample_id'],check) for record in records for check in record['spectral_checks']]
    reference_differences = [abs(np.log(
        next(row for row in record['rows'] if row['method'] == 'brent_linear_1025')['selected_h'] /
        record['reference']['global_minimum']['h'])) for record in records]
    validation=dict(derivative_reference_resolution_stable=all(r['reference']['diagnostics']['resolution_stable'] is True for r in records),
                    linear_reference_matches_derivative=sum(bool(v <= TOLERANCE) for v in reference_differences),
                    max_log_difference_from_derivative=float(max(reference_differences)),
                    spectral_refinement_checks=len(checks),
                    spectral_refinement_mismatches=[dict(sample_id=i,**c) for i,c in checks if not c['refinement_matches']],
                    analytic_overlap_checks=sum('analytic_overlap_matches' in c for _,c in checks),
                    analytic_overlap_mismatches=[dict(sample_id=i,**c) for i,c in checks if c.get('analytic_overlap_matches') is False],
                    maximum_relative_eigen_residual=max(c['relative_residual'] for _,c in checks),
                    threads=pools)
    with (output/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    with (output/'aggregate_results.csv').open('w',newline='') as f:
        fields=[k for k in summaries[0] if k!='mismatch_samples']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(summaries)
    (output/'summary.json').write_text(json.dumps(dict(configuration=config,validation=validation,results=summaries),indent=2)+'\n')
    write_table(output,summaries,args.samples)
    print(json.dumps(dict(validation=validation,results=summaries),indent=2))


if __name__ == '__main__':
    main()
