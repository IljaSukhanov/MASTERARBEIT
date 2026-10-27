"""Paired local Q0, Q13/14, Q1 histograms for n=100,200,...,2000.

One selected interior minimum per sample and method, with missing outcomes
retained. Reuse validated Q1 results and the earlier exact n=200 candidates.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time

HERE = Path(__file__).resolve().parent
KDE_DIR = HERE.parents[2]
ROOT = KDE_DIR.parents[1]
sys.path[:0] = [str(KDE_DIR), str(HERE)]
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/kde-hmax10-mpl-cache')
os.environ.setdefault('MPLBACKEND', 'Agg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
from threadpoolctl import threadpool_limits

from kde_tools.pdf_export import enable_automatic_pdf_saving
from local_search import GridScanner, H_MIN, H_MAX, sample_minima, same_minima, exact_dense_minima

NAME = 'local_qr_minimizers_vs_n'
DATA_DIR = KDE_DIR/'outputs/data'/NAME
IMAGE_DIR = KDE_DIR/'outputs/images'/NAME
DELIVERY_DIR = ROOT/'output/pdf'/NAME
SAMPLE_SIZES, SELECTED_N, M, SEED = tuple(range(100, 2001, 100)), (100, 1000, 1500, 2000), 1000, 20260830
METHODS = {'Q0': 0., 'Q13_14': 13/14, 'Q1': 1.}
COLORS = {'Q0': '#0072B2', 'Q13_14': '#009E73', 'Q1': '#D55E00'}
STYLES = {'Q0': '-', 'Q13_14': '-.', 'Q1': '--'}
LABELS = {'Q0': r'$\widehat Q_0$', 'Q13_14': r'$\widehat Q_{13/14}$', 'Q1': r'$\widehat Q_1$'}
OLD_WORKFLOW = KDE_DIR/'outputs/analysis/local_energy_minimizers_n200/run_experiment.py'


def save_csv(path, rows):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def selection(n, repetition, method, found, producer, diagnostics=None):
    found = sorted(found, key=lambda point: float(point['h']))
    best = min(range(len(found)), key=lambda i: (float(found[i]['energy']), float(found[i]['h']))) if found else None
    diagnostics = diagnostics or {}
    row = dict(n=n, repetition=repetition, method=method, r=METHODS[method],
               local_minimum_count=len(found), selected_h=float(found[best]['h']) if best is not None else np.nan,
               selected_energy=float(found[best]['energy']) if best is not None else np.nan,
               producer=producer, curvature_fallback=bool(diagnostics.get('curvature_fallback', False)),
               coarse_fine_signature_agrees=bool(diagnostics.get('coarse_fine_signature_agrees', True)))
    points = [dict(n=n, repetition=repetition, method=method, r=METHODS[method], candidate=i,
                   selected=i == best, **{key: float(point[key]) for key in
                   ('h', 'energy', 'log_derivative', 'log_curvature', 'newton_correction')},
                   producer=producer) for i, point in enumerate(found)]
    return row, points


def insert(connection, row, points):
    connection.execute('INSERT INTO results VALUES(?,?,?,?,?)',
                       (row['n'], row['repetition'], row['method'], json.dumps(row), json.dumps(points)))


def import_previous(connection, samples):
    q1_dir = KDE_DIR/'outputs/data/local_q1_minimizers_selected_n'
    q1_settings = json.loads((q1_dir/'settings.json').read_text())
    assert q1_settings['sample_bank_sha256'] == hashlib.sha256(samples.tobytes()).hexdigest()
    assert (q1_settings['h_min'], q1_settings['h_max'], q1_settings['endpoint_candidates']) == (H_MIN, H_MAX, False)
    q1_points = {}
    for point in csv.DictReader((q1_dir/'candidates.csv').open()):
        q1_points.setdefault((int(point['n']), int(point['repetition'])), []).append(point)
    imported = 0
    for old in csv.DictReader((q1_dir/'selections.csv').open()):
        n, repetition = int(old['n']), int(old['repetition'])
        found = q1_points.get((n, repetition), [])
        assert len(found) == int(old['local_minimum_count'])
        row, points = selection(n, repetition, 'Q1', found, 'validated_q1_interval_results')
        old_h = float(old['selected_h'] or 'nan')
        assert (np.isnan(old_h) and np.isnan(row['selected_h'])) or old_h == row['selected_h']
        insert(connection, row, points)
        imported += 1
    old_dir = KDE_DIR/'outputs/data/local_energy_minimizers_n200'
    old_settings = json.loads((old_dir/'settings.json').read_text())
    assert old_settings['sample_bank_sha256'] == hashlib.sha256(np.ascontiguousarray(samples[:, :200]).tobytes()).hexdigest()
    assert old_settings['qhat_source_sha256'] == hashlib.sha256((KDE_DIR/'kde_tools/qhat.py').read_bytes()).hexdigest()
    method_names = {'r0': 'Q0', 'r13over14': 'Q13_14', 'r1': 'Q1'}
    old_points = {}
    for point in csv.DictReader((old_dir/'candidates.csv').open()):
        if H_MIN < float(point['h']) < H_MAX:
            key = (int(point['sample_id']), method_names[point['method']])
            old_points.setdefault(key, []).append(point)
    for repetition in range(M):
        for method in METHODS:
            row, points = selection(200, repetition, method, old_points.get((repetition, method), []),
                                    'validated_n200_exact_candidates_filtered_to_interval')
            insert(connection, row, points)
            imported += 1
    connection.commit()
    assert imported == 7000
    return imported


def run_experiment():
    for directory in (DATA_DIR, IMAGE_DIR, DELIVERY_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    samples = np.concatenate((np.random.default_rng(SEED).normal(size=(M, 1000)),
        np.random.default_rng(np.random.SeedSequence([SEED, 1])).normal(size=(M, 1000))), axis=1)
    sample_hash = hashlib.sha256(samples.tobytes()).hexdigest()
    original = KDE_DIR/'outputs/data/continuous_ise_qhat_distributions_vs_n/thesis/5b5fa0b0a92f/settings.json'
    assert sample_hash == json.loads(original.read_text())['sample_bank_sha256']
    settings = dict(version='paired_local_qr_interval_vs_n_v1', sample_sizes=list(SAMPLE_SIZES),
                    selected_histogram_sizes=list(SELECTED_N), repetitions=M, seed=SEED, r_values=METHODS,
                    interval=[H_MIN, H_MAX], endpoints_excluded=True,
                    selection='lowest energy among detected interior local minima; no endpoint fallback',
                    normalization='bin count divided by all 1000 samples',
                    sample_bank_sha256=sample_hash, scan_points=1025, spatial_points=65536,
                    refined_scan_points=2049, refined_spatial_points=131072,
                    qhat_source_sha256=hashlib.sha256((KDE_DIR/'kde_tools/qhat.py').read_bytes()).hexdigest())
    path = DATA_DIR/'settings.json'
    if path.exists() and json.loads(path.read_text()) != settings:
        raise RuntimeError('Experiment settings differ from saved checkpoints.')
    path.write_text(json.dumps(settings, indent=2)+'\n')
    connection = sqlite3.connect(DATA_DIR/'checkpoint.sqlite')
    connection.execute('CREATE TABLE IF NOT EXISTS results(n INTEGER, repetition INTEGER, method TEXT, selection_json TEXT, points_json TEXT, PRIMARY KEY(n,repetition,method))')
    if connection.execute('SELECT count(*) FROM results').fetchone()[0] == 0:
        import_previous(connection, samples)
    done = set(connection.execute('SELECT n,repetition,method FROM results'))
    jobs = [(n, repetition, {m:r for m,r in METHODS.items() if (n,repetition,m) not in done})
            for n in SAMPLE_SIZES for repetition in range(M)]
    jobs = [job for job in jobs if job[2]]
    print(f'Reusing {len(done)} selections; computing {60000-len(done)} selections in {len(jobs)} paired sample jobs.', flush=True)
    started = time.monotonic()
    with threadpool_limits(limits=1):
        scanner, refined = GridScanner(), GridScanner(2049, 131072)

        def compute(job):
            n, repetition, pending = job
            sample = samples[repetition, :n]
            try:
                found, diagnostics = sample_minima(sample, pending, scanner, refined)
            except ArithmeticError:
                found = exact_dense_minima(sample, pending, OLD_WORKFLOW, points=2049)
                diagnostics = {method:dict(curvature_fallback=True, coarse_fine_signature_agrees=False) for method in pending}
            return [selection(n, repetition, method, found[method], 'exact_pair_refined_local_derivative_search', diagnostics[method]) for method in pending]

        if jobs:
            for row, points in compute(jobs[0]):
                insert(connection, row, points)
            connection.commit()
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(compute, job) for job in jobs[1:]]
            for completed, future in enumerate(as_completed(futures), start=2):
                for row, points in future.result():
                    insert(connection, row, points)
                if completed % 200 == 0:
                    connection.commit()
                    count = connection.execute('SELECT count(*) FROM results').fetchone()[0]
                    print(f'{count}/60000 selections; {completed}/{len(jobs)} new paired samples; elapsed {time.monotonic()-started:.0f}s', flush=True)
        connection.commit()
        stored = list(connection.execute('SELECT n,repetition,method,selection_json,points_json FROM results ORDER BY n,repetition,method'))
        assert len(stored) == 60000
        rows = [json.loads(record[3]) for record in stored]
        candidates = [point for record in stored for point in json.loads(record[4])]
        lookup = {(record[0], record[1], record[2]):json.loads(record[4]) for record in stored}
        save_csv(DATA_DIR/'selections.csv', rows)
        save_csv(DATA_DIR/'candidates.csv', candidates)
        connection.close()
        checks = []
        for n in SAMPLE_SIZES:
            for repetition in (0, M//2, M-1):
                actual, _ = sample_minima(samples[repetition, :n], METHODS, refined, refined)
                for method in METHODS:
                    expected = lookup[(n, repetition, method)]
                    assert same_minima(expected, actual[method]), (n, repetition, method)
                    checks.append(dict(n=n, repetition=repetition, method=method,
                                       kind='doubled scan and spatial resolution',
                                       maximum_h_difference=max([abs(a['h']-b['h']) for a,b in zip(expected, actual[method])] or [0.])))
        print('Doubled-resolution checks passed. Checking independent exact scans at every n.', flush=True)

        def independent(n):
            actual = exact_dense_minima(samples[0, :n], METHODS, OLD_WORKFLOW)
            result = []
            for method in METHODS:
                expected = lookup[(n, 0, method)]
                assert same_minima(expected, actual[method]), (n, method, 'exact dense scan')
                result.append(dict(n=n, repetition=0, method=method, kind='independent exact 1025-node derivative scan',
                                   maximum_h_difference=max([abs(a['h']-b['h']) for a,b in zip(expected, actual[method])] or [0.])))
            return result
        with ThreadPoolExecutor(max_workers=4) as pool:
            for result in pool.map(independent, SAMPLE_SIZES):
                checks.extend(result)
        empty, _ = sample_minima(np.zeros(200), METHODS, scanner, refined)
        assert all(not points for points in empty.values())
    assert all(H_MIN < point['h'] < H_MAX and point['log_curvature'] > 0 and point['newton_correction'] <= 1e-8 for point in candidates)
    summary = []
    for n in SAMPLE_SIZES:
        for method in METHODS:
            part = [row for row in rows if row['n'] == n and row['method'] == method]
            h = np.array([row['selected_h'] for row in part])
            h = h[np.isfinite(h)]
            assert len(part) == M
            summary.append(dict(n=n, method=method, r=METHODS[method], samples=M,
                                with_local_minimum=len(h), without_local_minimum=M-len(h),
                                multiple_local_minima=sum(row['local_minimum_count'] > 1 for row in part),
                                total_local_minima=sum(row['local_minimum_count'] for row in part),
                                median_h=float(np.median(h)), q05_h=float(np.quantile(h,.05)), q95_h=float(np.quantile(h,.95))))
    save_csv(DATA_DIR/'summary.csv', summary)
    validation = dict(total_selections=60000, paired_sample_jobs=20000, reused_selections=sum(row['producer'] != 'exact_pair_refined_local_derivative_search' for row in rows),
                      exact_candidate_count=len(candidates), sample_bank_matches_previous_experiments=True,
                      endpoint_substitutions=0, all_candidates_strictly_in_interval=True,
                      all_candidates_have_positive_curvature=True,
                      maximum_stationarity_correction=max(point['newton_correction'] for point in candidates),
                      maximum_validation_h_difference=max(check['maximum_h_difference'] for check in checks),
                      monotone_gaussian_has_no_local_minima=True,
                      numerical_search_is_not_an_exhaustiveness_certificate=True,
                      search_source_sha256=hashlib.sha256((HERE/'local_search.py').read_bytes()).hexdigest(), checks=checks)
    (DATA_DIR/'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    return rows, summary


def plot_results(rows, summary):
    from ise_reference import ensure_ise_reference
    ise_reference = list(csv.DictReader(ensure_ise_reference().open()))
    plot_methods = ('ISE', *METHODS)
    plot_colors = {'ISE':'#7B4AB2', **COLORS}
    plot_styles = {'ISE':'-', **STYLES}
    plot_labels = {'ISE':'ISE', **LABELS}
    pdf_dir = enable_automatic_pdf_saving(NAME)
    bandwidths, positions, edges = np.geomspace(H_MIN, H_MAX, 201), np.arange(201), np.arange(202)-.5
    frequencies, records = {}, []
    summary_lookup = {(row['n'],row['method']):row for row in summary}
    for n in SAMPLE_SIZES:
        frequencies[n] = {}
        reference_bins = sorted((row for row in ise_reference if int(row['n'])==n),key=lambda row:int(row['bin_index']))
        assert len(reference_bins)==201 and sum(int(row['count']) for row in reference_bins)==M
        frequencies[n]['ISE'] = np.array([int(row['count'])/M for row in reference_bins])
        records.extend(dict(n=n,method='ISE',bin_index=i,bandwidth_center=float(bandwidths[i]),
                            count=int(row['count']),relative_frequency_all_samples=int(row['count'])/M)
                       for i,row in enumerate(reference_bins))
        for method in METHODS:
            h = np.array([row['selected_h'] for row in rows if row['n'] == n and row['method'] == method])
            h = h[np.isfinite(h)]
            counts, _ = np.histogram(np.interp(np.log(h), np.log(bandwidths), positions), bins=edges)
            assert counts.sum() == len(h)
            frequencies[n][method] = counts/M
            records.extend(dict(n=n, method=method, bin_index=i, bandwidth_center=float(bandwidths[i]), count=int(counts[i]), relative_frequency_all_samples=float(counts[i]/M)) for i in range(201))
    save_csv(DATA_DIR/'histogram.csv', records)
    ticks = np.unique(np.linspace(0,200,7,dtype=int))
    ymax = 1.15*max(float(values.max()) for part in frequencies.values() for values in part.values())
    legend = [Line2D([0],[0],color=plot_colors[m],linestyle=plot_styles[m],linewidth=2,label=plot_labels[m]) for m in plot_methods]
    plt.style.use('default')

    def figure(sample_sizes, title, rows_count, figsize):
        fig, axes = plt.subplots(rows_count,2,figsize=figsize,sharex=True,sharey=True,squeeze=False)
        for ax,n in zip(axes.flat,sample_sizes):
            for method in plot_methods:
                ax.stairs(frequencies[n][method],edges,fill=True,color=plot_colors[method],alpha=.07,linewidth=0)
                ax.stairs(frequencies[n][method],edges,color=plot_colors[method],linestyle=plot_styles[method],linewidth=1.8 if method=='ISE' else 1.5)
            ax.set_title(rf'$n={n}$',fontsize=13,pad=9)
            ax.set_xticks(ticks,[f'{bandwidths[i]:.3g}' for i in ticks])
            ax.set(xlim=(-.7,200.7),ylim=(0,ymax),xlabel='selected bandwidth h',ylabel='relative frequency')
            found=' / '.join([str(M)]+[str(summary_lookup[(n,m)]['with_local_minimum']) for m in METHODS])
            ax.text(.025,.95,f'Histogram entries: {found}',transform=ax.transAxes,va='top',fontsize=8.5,color='#304967')
            ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
            ax.tick_params(colors='#304967',labelsize=9,labelbottom=True,labelleft=True)
            ax.grid(axis='y',color='#DDE5EC',linewidth=.7)
            ax.set_axisbelow(True)
            for side in ('top','right'):ax.spines[side].set_visible(False)
            for side in ('left','bottom'):ax.spines[side].set_color('#B8C5D1')
        fig.suptitle(title,fontsize=18,y=.989)
        if rows_count==2:
            subtitle_y,legend_y,top,bottom=.932,.915,.80,.13
        else:
            subtitle_y,legend_y,top,bottom=.96,.949,.90,.065
        fig.text(.5,subtitle_y,'Gaussian samples; M = 1000 per n; bandwidth interval [0.01, 2]',ha='center',fontsize=11)
        fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,legend_y),ncol=4,frameon=False,fontsize=12)
        fig.text(.5,.021,r'$\widehat Q_r$: lowest-energy interior local minimum. ISE: numerical minimum over [0.01, 2].'
                 +'\nFrequencies divide by all 1,000 samples; missing local minima contribute no entry. Counts follow legend order.',ha='center',fontsize=9)
        fig.subplots_adjust(left=.075,right=.985,bottom=bottom,top=top,hspace=.45,wspace=.20)
        return fig

    selected=figure(SELECTED_N,'Local energy minimizers and ISE-optimal bandwidths',2,(12.8,8.4))
    selected.savefig(IMAGE_DIR/'selected_histograms.png',dpi=180)
    plt.show()
    plt.close(selected)
    produced=pdf_dir/'001_local_energy_minimizers_and_ise_optimal_bandwidths.pdf'
    assert produced.is_file()
    shutil.copyfile(produced,DELIVERY_DIR/'selected_histograms.pdf')
    with PdfPages(DELIVERY_DIR/'all_sample_sizes.pdf') as overview:
        for page,ns in enumerate((SAMPLE_SIZES[:10],SAMPLE_SIZES[10:]),start=1):
            fig=figure(ns,f'Local energy and ISE minimizers: n = {ns[0]} to {ns[-1]}',5,(14,17.8))
            fig.savefig(IMAGE_DIR/f'all_sample_sizes_page{page}.png',dpi=155)
            overview.savefig(fig,bbox_inches='tight')
            plt.show()
            plt.close(fig)
    print('Saved selected histograms and the complete two-page overview.',flush=True)


if __name__=='__main__':
    if '--plot-only' in sys.argv:
        rows=[]
        for record in csv.DictReader((DATA_DIR/'selections.csv').open()):
            record.update(n=int(record['n']),selected_h=float(record['selected_h'] or 'nan'))
            rows.append(record)
        summary=[]
        for record in csv.DictReader((DATA_DIR/'summary.csv').open()):
            record.update(n=int(record['n']),with_local_minimum=int(record['with_local_minimum']))
            summary.append(record)
    else:
        rows,summary=run_experiment()
    plot_results(rows,summary)
