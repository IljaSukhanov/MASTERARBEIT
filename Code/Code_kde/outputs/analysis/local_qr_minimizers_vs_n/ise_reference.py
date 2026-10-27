"""Recover the original validated ISE histogram counts without resimulation.

The original full-interval vector plots retain every integer bin count. Their
201 geometric bins match the local-minimum plots. A separate check compares
all recovered bins with the saved per-sample ISE bandwidths at four n values.
"""
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import runpy
import subprocess
import sys

HERE = Path(__file__).resolve().parent
KDE_DIR = HERE.parents[2]
ROOT = KDE_DIR.parents[1]
DATA_DIR = KDE_DIR/'outputs/data/local_qr_minimizers_vs_n'
REFERENCE = DATA_DIR/'ise_histogram.csv'
VALIDATION = DATA_DIR/'ise_reference_validation.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_reference():
    exporter = runpy.run_path(str(ROOT/'output/analysis/continuous_minimizers_selected_n/export_figures.py'))
    source_dir = KDE_DIR/'outputs/pdfs/continuous_ise_qhat_distributions_vs_n'
    original_dir = KDE_DIR/'outputs/data/continuous_ise_qhat_distributions_vs_n/thesis/5b5fa0b0a92f'
    original = json.loads((original_dir/'settings.json').read_text())
    local = json.loads((DATA_DIR/'settings.json').read_text())
    assert original['sample_bank_sha256'] == local['sample_bank_sha256']
    assert (original['h_min'], original['h_max'], original['histogram_points'], original['repetitions']) == (.01, 2., 201, 1000)
    counts, sources, recovery = {}, [], []
    for page, prefix in enumerate(('001','002')):
        source, = source_dir.glob(f'{prefix}*.pdf')
        reader = exporter['PdfReader'](source)
        sources.append(dict(path=str(source.relative_to(ROOT)), sha256=digest(source)))
        for row in range(5):
            for column in range(2):
                n = 100*(page*10+row*2+column+1)
                histogram = exporter['filled_histograms'](reader, reader.pages[0], row, column)['ISE']
                counts[n] = histogram['counts']
                assert sum(counts[n]) == 1000
                recovery.append(dict(n=n, total_count=1000,
                                     maximum_vector_count_rounding_error=histogram['maximum_vector_count_rounding_error']))
    assert sorted(counts) == list(range(100,2001,100))
    saved_dir = KDE_DIR/'outputs/data/continuous_minimizers_hmax10'
    saved_settings = json.loads((saved_dir/'settings.json').read_text())
    assert saved_settings['sample_bank_sha256'] == original['sample_bank_sha256']
    independent = {}
    for point in csv.DictReader((saved_dir/'selections.csv').open()):
        if point['method'] != 'ISE':
            continue
        n, h = int(point['n']), float(point['h'])
        # A wider-interval minimizer inside [0.01,2] is also its restricted minimizer.
        assert .01 <= h <= 2.
        index = min(200,max(0,math.floor(200*math.log(h/.01)/math.log(200)+.5)))
        independent.setdefault(n,[0]*201)[index] += 1
    assert len(independent) == 4
    assert all(independent[n] == counts[n] for n in independent)
    records = [dict(n=n,method='ISE',bin_index=i,bandwidth_center=.01*200**(i/200),
                    count=count,relative_frequency_all_samples=count/1000)
               for n in sorted(counts) for i,count in enumerate(counts[n])]
    with REFERENCE.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    validation = dict(criterion='sample-specific ISE against the known standard-normal density',
                      optimization_interval=[.01,2.], repetitions_per_n=1000,
                      sample_bank_sha256=original['sample_bank_sha256'], sources=sources,
                      original_settings_sha256=digest(original_dir/'settings.json'),
                      original_validation_sha256=digest(original_dir/'validation.json'),
                      recovered_integer_counts_checked_to_tolerance=.0001,
                      all_20_histograms_sum_to_1000=True,
                      saved_per_sample_reference_sizes=sorted(independent),
                      saved_per_sample_reference_count=4000,
                      every_bin_matches_saved_per_sample_reference=True,
                      reference_csv_sha256=digest(REFERENCE), recovery=recovery)
    VALIDATION.write_text(json.dumps(validation,indent=2)+'\n')
    print('Recovered 20 ISE histograms; all 804 reference bins match the 4,000 saved ISE bandwidths.',flush=True)


def ensure_ise_reference():
    if REFERENCE.exists() and VALIDATION.exists():
        validation = json.loads(VALIDATION.read_text())
        if validation['reference_csv_sha256'] == digest(REFERENCE) and all(
                digest(ROOT/source['path']) == source['sha256'] for source in validation['sources']):
            return REFERENCE
    if all(importlib.util.find_spec(name) is not None for name in ('pypdf','reportlab')):
        extract_reference()
    else:
        bundled_python = Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3'
        if not bundled_python.is_file():
            raise RuntimeError('Recovering the ISE reference requires a Python environment with pypdf and reportlab.')
        subprocess.run([str(bundled_python),str(__file__),'--extract'],check=True)
    return REFERENCE


if __name__=='__main__':
    extract_reference() if '--extract' in sys.argv else ensure_ise_reference()
