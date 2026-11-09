"""Combine the paired Brent results with the instrumented spectral run."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/brent_spectral_combined_table'
BRENT = ROOT / 'outputs/brent_linear_refinements_spectral_alpha15/results.csv'
SPECTRAL = ROOT / 'outputs/spectral_runtime_breakdown_alpha15_n800/results.csv'
SUMMARY = ROOT / 'outputs/spectral_runtime_breakdown_alpha15_n800/summary.json'

with BRENT.open(newline='') as f:
    original = list(csv.DictReader(f))
with SPECTRAL.open(newline='') as f:
    spectral = list(csv.DictReader(f))
summary = json.loads(SUMMARY.read_text())
references = {int(r['sample_id']): float(r['selected_h']) for r in original
              if r['method'] == 'brent_linear_1025'}
if set(references) != set(range(100)):
    raise ValueError('Incomplete reference samples')

rows = []
for method in ['brent_unrefined'] + [f'brent_linear_{m}' for m in (33, 65, 129, 257, 513, 1025)]:
    group = [r for r in original if r['method'] == method]
    if {int(r['sample_id']) for r in group} != set(references):
        raise ValueError(f'Incomplete or unpaired records for {method}')
    times = [float(r['time_ms']) for r in group]
    errors = [100*abs(float(r['selected_h'])-references[int(r['sample_id'])]) /
              references[int(r['sample_id'])] for r in group]
    row = dict(method=method, total_ms=statistics.mean(times),
               total_se_ms=statistics.stdev(times)/10, assembly_ms='', perron_ms='',
               mean_error_percent=statistics.mean(errors), max_error_percent=max(errors),
               mean_detected_minima='' if method == 'brent_unrefined' else
               statistics.mean(int(r['detected_valleys']) for r in group),
               endpoints=sum(r['endpoint_selected'] == 'True' for r in group))
    rows.append(row)
for item in summary['results']:
    n = item['N']
    group = [r for r in spectral if int(r['N']) == n]
    if {int(r['sample_id']) for r in group} != set(references):
        raise ValueError(f'Incomplete or unpaired spectral records N={n}')
    errors = [100*abs(float(r['selected_h'])-references[int(r['sample_id'])]) /
              references[int(r['sample_id'])] for r in group]
    rows.append(dict(method=f'spectral_{n}', total_ms=item['total_ms']['mean'],
                     total_se_ms=item['total_ms']['se'],
                     assembly_ms=item['matrix_assembly_ms']['mean'],
                     perron_ms=item['perron_and_selection_ms']['mean'],
                     mean_error_percent=statistics.mean(errors),
                     max_error_percent=max(errors), mean_detected_minima='',
                     endpoints=sum(int(r['selected_index']) in (0, n-1) for r in group)))

OUT.mkdir(parents=True, exist_ok=True)
with (OUT/'comparison.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

def pct(x):
    return f'{x:.2e}' if 0 < x < 0.01 else f'{x:.2f}'

def latex_num(x):
    s = pct(x)
    if 'e' in s:
        mantissa, exponent = s.split('e')
        return rf'{mantissa}\!\times\!10^{{{int(exponent)}}}'
    return s

lines = [r'\begin{table}[htbp]', r'\centering', r'\scriptsize',
         r'\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{lrrrrrrr}', r'\toprule',
         r'& \shortstack{Total time\\(ms)} & \shortstack{Assembly\\(ms)} & \shortstack{Perron\\(ms)} & \shortstack{Mean error\\(\%)} & \shortstack{Max. error\\(\%)} & \shortstack{Mean\\minima} & \shortstack{Endpoint\\count} \\',
         r'\midrule']
for row in rows:
    method = row['method']
    if method == 'spectral_100':
        lines.append(r'\midrule')
    if method == 'brent_unrefined':
        label = 'Brent, unrefined'
    elif method.startswith('brent'):
        label = rf'Brent, $M={method.rsplit("_", 1)[-1]}$'
    else:
        label = rf'Spectral, $N={method.rsplit("_", 1)[-1]}$'
    tim = rf'${row["total_ms"]:.2f} \pm {row["total_se_ms"]:.2f}$'
    assembly = '--' if row['assembly_ms'] == '' else f'{row["assembly_ms"]:.2f}'
    perron = '--' if row['perron_ms'] == '' else f'{row["perron_ms"]:.2f}'
    err = '--' if method == 'brent_linear_1025' else f'${latex_num(row["mean_error_percent"])}$'
    maximum = '--' if method == 'brent_linear_1025' else f'${latex_num(row["max_error_percent"])}$'
    minima = '--' if row['mean_detected_minima'] == '' else f'{row["mean_detected_minima"]:.2f}'
    lines.append(f'{label} & {tim} & {assembly} & {perron} & {err} & {maximum} & {minima} & {row["endpoints"]} ' + r'\\')
lines += [r'\bottomrule', r'\end{tabular}',
          r'\caption{Comparison over 100 samples of size 200. Total times are means $\pm$ standard errors; assembly and Perron times are means. Relative bandwidth errors are calculated for each sample against linear refined Brent with $M=1025$, then averaged or maximized across samples. This numerical reference is not a certificate of the global minimum. The mean number of minima counts detected interior valleys; endpoint counts are totals over 100 samples. Dashes indicate quantities that do not apply or the reference itself. Spectral and Brent timings come from separate runs.}',
          r'\end{table}']
(OUT/'comparison.tex').write_text('\n'.join(lines)+'\n')
print((OUT/'comparison.csv').read_text())
