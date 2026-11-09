"""Recompute the comparison using linear Brent M=1025 as reference.

Reads the already timed run; no method is rerun and timings are unchanged.
Retains the derivative-based reference in separate columns for validation.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

KDE_ROOT = Path(__file__).resolve().parents[2]
SOURCE = KDE_ROOT / 'outputs/brent_refinements_spectral_alpha15'
OUTPUT = KDE_ROOT / 'outputs/brent_linear_refinements_spectral_alpha15'
REFERENCE_METHOD = 'brent_linear_1025'
METHODS = [('brent_unrefined', None),
           *(('brent_linear', size) for size in (33, 65, 129, 257, 513, 1025)),
           *(('spectral', size) for size in (100, 200, 400))]


def key(family, size):
    return family if size is None else f'{family}_{size}'


def title(family, size):
    if family == 'brent_unrefined': return 'Brent, unrefined'
    if family == 'spectral': return f'Spectral, N={size}'
    return f'Brent, {family.removeprefix("brent_")}, M={size}'


def main():
    with (SOURCE/'results.csv').open(newline='') as stream:
        all_source_rows = list(csv.DictReader(stream))
    selected_methods = {key(family, size) for family, size in METHODS}
    source = [row for row in all_source_rows if row['method'] in selected_methods]
    source_summary = json.loads((SOURCE/'summary.json').read_text())
    refs = {int(row['sample_id']): row for row in source if row['method'] == REFERENCE_METHOD}
    n = source_summary['configuration']['samples']
    expected = {(i, key(family, size)) for i in range(n) for family, size in METHODS}
    if len(refs) != n or len(source) != len(expected) or {
            (int(row['sample_id']),row['method']) for row in source} != expected:
        raise ValueError('Missing or duplicate measurements or reference rows.')
    rows = []
    for old in source:
        ref = refs[int(old['sample_id'])]
        h, target = float(old['selected_h']), float(ref['selected_h'])
        energy, target_energy = float(old['energy']), float(ref['energy'])
        row = dict(old)
        row['derivative_reference_h'] = row.pop('reference_h')
        row['derivative_reference_energy'] = row.pop('reference_energy')
        row['derivative_reference_bandwidth_match'] = row.pop('reference_bandwidth_match')
        row['derivative_reference_valleys'] = row.pop('reference_valleys')
        row['all_derivative_reference_valleys_match'] = row.pop('all_reference_valleys_match')
        row['reference_h'] = target
        row['reference_energy'] = target_energy
        row['relative_bandwidth_error_percent'] = 100*abs(h-target)/target
        row['relative_energy_excess_percent'] = 100*(energy/target_energy-1)
        row['reference_bandwidth_match'] = bool(abs(np.log(h/target)) <= 1e-6)
        rows.append(row)
    aggregates = []
    for family,size in METHODS:
        k = key(family,size)
        group = [row for row in rows if row['method'] == k]
        times = np.array([float(row['time_ms']) for row in group])
        bw = np.array([float(row['selected_h']) for row in group])
        errors = np.array([row['relative_bandwidth_error_percent'] for row in group])
        is_brent = family.startswith('brent_')
        candidate_count = (len(group) if family == 'brent_unrefined' else
                           sum(int(row['detected_valleys']) for row in group) if is_brent else None)
        endpoint_count = (sum(row['endpoint_selected'] == 'True' for row in group)
                          if is_brent else None)
        aggregates.append(dict(method=k, family=family, size=size, label=title(family,size),
                               samples=len(group), mean_time_ms=float(times.mean()),
                               se_time_ms=float(times.std(ddof=1)/np.sqrt(len(times))),
                               median_time_ms=float(np.median(times)), mean_selected_h=float(bw.mean()),
                               mean_relative_bandwidth_error_percent=float(errors.mean()),
                               max_relative_bandwidth_error_percent=float(errors.max()),
                               detected_minimum_candidates_total=candidate_count,
                               endpoint_selections_total=endpoint_count,
                               reference_matches=sum(row['reference_bandwidth_match'] for row in group),
                               mismatch_samples=[int(row['sample_id']) for row in group if not row['reference_bandwidth_match']]))
    ref_rows = [row for row in rows if row['method'] == REFERENCE_METHOD]
    differences = [abs(np.log(float(row['reference_h'])/float(row['derivative_reference_h']))) for row in ref_rows]
    validation = dict(reference_method=REFERENCE_METHOD, same_sample_count=n,
                      derivative_reference_matches=sum(bool(v<=1e-6) for v in differences),
                      max_log_difference_from_derivative_reference=float(max(differences)),
                      references_certified_global=False,
                      note='The reference method agrees with itself by construction; '
                           'all error metrics use matched samples. Derivative reference was used only for validation.')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with (OUTPUT/'results.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (OUTPUT/'aggregate_results.csv').open('w',newline='') as stream:
        fields=[k for k in aggregates[0] if k!='mismatch_samples']
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(aggregates)
    (OUTPUT/'summary.json').write_text(json.dumps(dict(reference=validation,source_config=source_summary['configuration'],
                                                       results=aggregates),indent=2)+'\n')
    md='| Method | Mean time ± SE (ms) | Mean bandwidth | Mean relative error (%) | Minimum candidates | Endpoint selections |\n|---|---:|---:|---:|---:|---:|\n'
    md+=''.join(f'| {row["label"]}{" (reference)" if row["method"] == REFERENCE_METHOD else ""} | '
                f'{row["mean_time_ms"]:.2f} ± {row["se_time_ms"]:.2f} | {row["mean_selected_h"]:.5f} | '
                f'{row["mean_relative_bandwidth_error_percent"]:.6g} | '
                f'{row["detected_minimum_candidates_total"] if row["detected_minimum_candidates_total"] is not None else "—"} | '
                f'{row["endpoint_selections_total"] if row["endpoint_selections_total"] is not None else "—"} |\n' for row in aggregates)
    (OUTPUT/'table.md').write_text(md)
    tex=[r'\begin{table}[htbp]',r'    \centering',r'    \footnotesize',r'    \setlength{\tabcolsep}{3pt}',
         r'    \begin{tabular}{lccccc}',r'        \toprule',
         r'        Method & \shortstack{Mean time\\$\pm$ SE (ms)} & \shortstack{Mean\\bandwidth} & \shortstack{Mean absolute\\rel. error (\%)} & \shortstack{Minima\\found} & \shortstack{Endpoint\\selected} \\',r'        \midrule']
    family_before = None
    for row in aggregates:
        family,size=row['family'],row['size']
        if family_before is not None and family != family_before: tex.append(r'        \midrule')
        if family=='brent_unrefined': name='Brent, unrefined'
        elif family=='spectral': name=f'Spectral, $N={size}$'
        else: name=f'Brent, {family.removeprefix("brent_")}, $M={size}$'
        error=row['mean_relative_bandwidth_error_percent']
        if row['method']==REFERENCE_METHOD: err='0'
        elif error < 1e-5: err=r'$<10^{-5}$'
        elif error < .001:
            mantissa, exponent=f'{error:.2e}'.split('e')
            err=f'${mantissa}\\times 10^{{{int(exponent)}}}$'
        else: err=f'{error:.3f}'
        candidates = str(row['detected_minimum_candidates_total']) if row['detected_minimum_candidates_total'] is not None else r'---'
        endpoints = str(row['endpoint_selections_total']) if row['endpoint_selections_total'] is not None else r'---'
        tex.append(f'        {name} & ${row["mean_time_ms"]:.2f} \\pm {row["se_time_ms"]:.2f}$ & '
                   f'${row["mean_selected_h"]:.5f}$ & {err} & {candidates} & {endpoints} \\\\')
        family_before=family
    tex.extend([r'        \bottomrule',r'    \end{tabular}',
                r'    \caption{Comparison over 100 samples of size 200 from $N(0,1)$ on $[0.01,2]$. '
                r'The spectral method uses $\lambda_N=N^{3/2}$ and geometric grids. '
                r'For refined Brent searches, $M$ denotes the number of scan grid points. '
                r'Times are means $\pm$ standard errors. Relative bandwidth errors are computed '
                r'for each sample against the linear refined Brent result with $M=1025$; '
                r'that reference row has zero error by construction. Minima found is the '
                r'total number of numerical interior candidates over all samples; the unrefined '
                r'method returns one per sample. Endpoint selections are counted over the 100 samples.}',r'\end{table}'])
    (OUTPUT/'table.tex').write_text('\n'.join(tex)+'\n')
    print(json.dumps(dict(validation=validation,results=aggregates),indent=2))


if __name__=='__main__':
    main()
