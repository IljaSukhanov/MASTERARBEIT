"""Check retained settings, numerical exports, and PDFs after notebook consolidation."""

import ast
import contextlib
import csv
import io
import json
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from kde_tools.density import exact_gaussian_kde_l2_gram
from kde_tools.experiment_inputs import paired_gaussian_sample_bank
from kde_tools.fourier import gaussian_kde_l2_gram_fourier
from kde_tools.pdf_export import enable_automatic_pdf_saving, disable_automatic_pdf_saving
from kde_tools.qhat import minimize_qhat


ROOT = Path(__file__).resolve().parents[1]


def notebook(name):
    path, = ROOT.glob(f'xx_notebooks_*/{name}.ipynb')
    return json.loads(path.read_text())


def execute(cell, namespace, overrides=None):
    source = '\n'.join(line for line in ''.join(cell['source']).splitlines()
                       if not line.lstrip().startswith(('%', '!')))
    tree = ast.parse(source)
    if overrides:
        class Settings(ast.NodeTransformer):
            def visit_Assign(self, node):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in overrides:
                        node.value = ast.parse(overrides[target.id], mode='eval').body
                return self.generic_visit(node)
        tree = ast.fix_missing_locations(Settings().visit(tree))
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(tree, '<consolidated notebook>', 'exec'), namespace)


class NotebookCleanupTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')
        disable_automatic_pdf_saving()

    def test_single_sample_view_matches_direct_csv_and_reuses_repeated_cache(self):
        nb = notebook('g_rs_error_vs_k_repeated_samples')
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ns = {}
            exporter = lambda name: enable_automatic_pdf_saving(name, output_root=base/'pdfs')
            with patch('kde_tools.pdf_export.enable_automatic_pdf_saving', side_effect=exporter):
                execute(nb['cells'][1], ns)
            ns['code_directory'] = base
            execute(nb['cells'][2], ns, dict(n='20', realizations='2', N='5', workers='1',
                K_candidates='np.array([64, 128, 256])', highlight_K='128'))
            for cell in nb['cells'][3:]:
                if cell['cell_type'] == 'code':
                    execute(cell, ns)
            table = np.loadtxt(ns['single_sample_directory']/'g_rs_error_vs_k.csv',
                               delimiter=',', skiprows=1)
            sample = np.random.default_rng(20261024).normal(size=20)
            gram = exact_gaussian_kde_l2_gram(sample, ns['bandwidths'])
            for row, k in zip(table, ns['K_candidates']):
                delta = gaussian_kde_l2_gram_fourier(sample, ns['bandwidths'],
                    frequency_cutoff=450., frequency_points=int(k)+1)-gram
                r, s = np.unravel_index(np.abs(delta).argmax(), delta.shape)
                expected = [k, abs(delta[r, s]), r, s,
                            ns['bandwidths'][r], ns['bandwidths'][s], delta[r, s]]
                np.testing.assert_allclose(row, expected, rtol=0, atol=2e-14)
            np.testing.assert_array_equal(np.load(ns['single_sample_directory']/'sample.npy'), sample)
            self.assertEqual(len(list((base/'pdfs').rglob('*.pdf'))), 2)
            numerical_files = [ns['calculation_directory']/name for name in
                               ('samples.npy', 'bandwidths.npy', 'exact_grams.npy', 'per_sample_results.npz')]
            before = {path: path.read_bytes() for path in numerical_files}
            with patch.dict(ns, exact_gaussian_kde_l2_gram=lambda *a, **k: self.fail('Cache recomputed')):
                execute(nb['cells'][3], ns)
            self.assertEqual({path: path.read_bytes() for path in numerical_files}, before)

    def test_grid_setup_preserves_both_choices_and_separate_pdf_directories(self):
        nb = notebook('linear_grid_joint_N_lambda_limit')
        setup = next(c for c in nb['cells'] if c['cell_type'] == 'code')
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            exporter = lambda name: enable_automatic_pdf_saving(name, output_root=base/'pdfs')
            names = []
            for choice, expected_name in [('1', 'linear_grid_joint_N_lambda_limit'),
                                           ('13/14', 'r13over14_joint_N_lambda_limit')]:
                ns = {}
                with patch('kde_tools.pdf_export.enable_automatic_pdf_saving', side_effect=exporter):
                    execute(setup, ns, dict(GRID_R=repr(choice)))
                workflow = ns['workflow']
                self.assertEqual(ns['experiment_name'], expected_name)
                self.assertEqual(ns['pdf_directory'].name, expected_name.lower())
                grid = workflow['bandwidth_grid'](7)
                r = 1. if choice == '1' else 13/14
                expected = np.linspace(.01**r, 2.**r, 7)**(1/r)
                np.testing.assert_allclose(grid, expected, atol=1e-14)
                globals_ = workflow['run_experiment'].__globals__
                self.assertEqual(globals_['DATA_DIR'].name, expected_name.lower())
                # Switching back must reset both the grid and its output namespace.
                self.assertEqual(workflow['configure_grid']('1'), 'linear_grid_joint_N_lambda_limit')
                np.testing.assert_allclose(workflow['bandwidth_grid'](7), np.linspace(.01, 2., 7))
                names.append(ns['pdf_directory'])
            self.assertNotEqual(*names)

    def q1_workflow(self, base):
        functions = runpy.run_path(str(ROOT/'outputs/analysis/local_qr_minimizers_vs_n/run_experiment.py'))
        globals_ = functions['export_q1_view'].__globals__
        globals_.update(M=2, SELECTED_N=(100, 200), DATA_DIR=base/'data',
                        IMAGE_DIR=base/'images', DELIVERY_DIR=base/'delivery')
        globals_['DATA_DIR'].mkdir()
        globals_['enable_automatic_pdf_saving'] = lambda name: enable_automatic_pdf_saving(
            name, output_root=base/'pdfs')
        samples = paired_gaussian_sample_bank(2, 200)
        samples[1] = 0.  # No interior minimum: keep a missing outcome, not an endpoint.
        rows, summaries, reports = [], [], {}
        for n in globals_['SELECTED_N']:
            count = 0
            for repetition in range(2):
                analysis = minimize_qhat(samples[repetition, :n], .01, 2., r=1.)
                reports[n, repetition] = analysis
                row, _ = functions['selection'](n, repetition, 'Q1', analysis['local_minima'],
                    'test shared selection', dict(analysis=analysis))
                rows.append(row)
                count += np.isfinite(row['selected_h'])
            summaries.append(dict(n=n, method='Q1', with_local_minimum=int(count)))
        return functions, rows, summaries, reports

    def test_q1_view_keeps_full_reports_missing_outcomes_and_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            workflow, rows, summaries, reports = self.q1_workflow(base)
            # Recover the full report for a selection-only historical fallback.
            rows[0]['analysis_json'] = 'null'
            q1, summary = workflow['export_q1_view'](rows, summaries)
            for row in q1:
                reference = reports[row['n'], row['repetition']]
                self.assertEqual(row['local_minimum_count'], len(reference['local_minima']))
                self.assertAlmostEqual(row['interval_minimum_h'], reference['global_minimum']['h'])
                self.assertEqual(row['interval_minimum_boundary'], reference['global_minimum']['boundary'])
                self.assertTrue(json.loads(row['analysis_json'])['diagnostics'])
                if row['repetition'] == 1:
                    self.assertTrue(np.isnan(row['selected_h']))
                    self.assertEqual(row['interval_minimum_boundary'], 1)
            self.assertEqual(rows[0]['analysis_json'], 'null')
            original = json.loads((base/'data/q1_selected_n/validation.json').read_text())
            self.assertEqual(original['reference_checks']['fresh_exact_n200_samples_checked'], 1)
            with patch.dict(workflow['export_q1_view'].__globals__,
                            q1_reference_checks=lambda: self.fail('Reference checks repeated')):
                workflow['export_q1_view'](rows, summaries)
            with contextlib.redirect_stdout(io.StringIO()):
                workflow['plot_q1_results'](q1, summary)
            pdf, = (base/'pdfs').rglob('*.pdf')
            self.assertGreater(pdf.stat().st_size, 1000)
            delivery = base/'delivery/q1_selected_n/local_q1_minimizers_selected_n.pdf'
            self.assertEqual(delivery.read_bytes(), pdf.read_bytes())
            with (base/'data/q1_selected_n/histogram.csv').open() as stream:
                histogram = list(csv.DictReader(stream))
            for n in (100, 200):
                self.assertEqual(sum(int(r['count']) for r in histogram if int(r['n']) == n),
                                 sum(np.isfinite(r['selected_h']) for r in q1 if r['n'] == n))

    def test_q1_view_rejects_incomplete_shared_selections(self):
        with tempfile.TemporaryDirectory() as tmp:
            workflow, rows, summaries, _ = self.q1_workflow(Path(tmp))
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                workflow['export_q1_view'](rows[:-1], summaries)


if __name__ == '__main__':
    unittest.main()
