"""Run the thesis notebook sections with small, isolated inputs and exports."""

import ast
import contextlib
import io
import json
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from kde_tools.pdf_export import enable_automatic_pdf_saving, disable_automatic_pdf_saving


ROOT = Path(__file__).resolve().parents[1]


def cells(name):
    return {c['id']: c for c in json.loads((ROOT/'notebooks'/f'{name}.ipynb').read_text())['cells']}


def execute(cell, ns, overrides=None):
    source = '\n'.join(l for l in ''.join(cell['source']).splitlines()
                       if not l.lstrip().startswith(('%', '!')))
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
        exec(compile(tree, '<thesis notebook>', 'exec'), ns)


class ThesisWorkflowTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')
        disable_automatic_pdf_saving()

    def test_numerics_illustration_error_and_runtime_exports(self):
        c = cells('kde_numerics')
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            isolated = base/'Code/Code_kde'
            ns = {'display': lambda value: None}
            exporter = lambda name, **kw: enable_automatic_pdf_saving('kde_numerics', output_root=base/'pdfs')
            with patch('kde_tools.pdf_export.enable_automatic_pdf_saving', side_effect=exporter), \
                    patch('kde_tools.thesis_exports.KDE_DIRECTORY', isolated):
                ns['isolated'] = isolated
                execute(c['setup'], ns, dict(
                    data_directory='isolated/"outputs/data/classical_bandwidth_kde_illustration"',
                    image_directory='isolated/"outputs/images/classical_bandwidth_kde_illustration"'))
                ns['code_directory'] = isolated
                execute(c['settings'], ns, dict(cv_grid_size='31', integration_points='1001', plot_points='201'))
                for key in ('calculate-and-save', 'plot-and-export'):
                    execute(c[key], ns)
                self.assertEqual(len(list((isolated/'outputs/pdfs').rglob('*.pdf'))), 1)
                execute(c['grs-repeat-1'], ns)
                ns['code_directory'] = isolated
                execute(c['grs-repeat-2'], ns, dict(n='20', N='5', realizations='2', workers='1',
                    K_candidates='np.array([64, 128, 256])', highlight_K='128'))
                for key in ('grs-repeat-3','grs-repeat-4','grs-repeat-5'):
                    execute(c[key], ns)
                self.assertEqual(int(ns['per_sample_errors'].shape[0]), 2)
                execute(c['runtime-01'], ns)
                ns['code_directory'] = isolated
                execute(c['runtime-02'], ns, dict(n='20', N='5', realizations='2', K='1536', warmup_calls_per_method='1'))
                execute(c['runtime-03'], ns)
                execute(c['runtime-04'], ns)
                self.assertTrue((ns['data_directory']/'summary.md').is_file())
                self.assertTrue((ns['data_directory']/'report.json').is_file())
                self.assertEqual(len(list((base/'pdfs').rglob('*.pdf'))), 2)

    def test_both_presented_spectral_histogram_profiles(self):
        c = cells('spectral_scale_selection_kde')
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ns = {}
            with patch('kde_tools.pdf_export.enable_automatic_pdf_saving',
                       side_effect=lambda name: enable_automatic_pdf_saving(name, output_root=base/'pdfs')):
                execute(c['imports'], ns)
            ns.update(code_directory=base, kde_thesis_figure_dir=base/'thesis')
            ns['kde_thesis_figure_dir'].mkdir()
            for profile in ('figure_3_5','localized'):
                execute(c['beta2-configurable-parameters'], ns,
                        dict(beta2_profile=repr(profile), beta2_n='20', beta2_N='7', beta2_M='2',
                             beta2_binning='1', beta2_workers='1'))
                execute(c['beta2-configurable-selections'], ns)
                execute(c['beta2-configurable-histograms'], ns)
                self.assertTrue(all(int(values.sum()) == 2 for values in ns['beta2_run']['counts']))
            self.assertEqual(len(list((base/'pdfs').rglob('*.pdf'))), 2)

    def test_selected_energy_panels_and_local_minimum_control(self):
        c = cells('continuous_ise_qhat_distributions_vs_n')
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            isolated = base/'Code/Code_kde'
            ns = {'display': lambda value: None}
            with patch('kde_tools.pdf_export.enable_automatic_pdf_saving',
                       side_effect=lambda name, **kw: enable_automatic_pdf_saving(name, output_root=base/'pdfs')), \
                    patch('kde_tools.thesis_exports.KDE_DIRECTORY', isolated):
                execute(c['setup'], ns)
                ns.update(data_output_directory=base/'data', image_output_directory=base/'images')
                execute(c['settings'], ns, dict(RUN_PROFILE='"preview"', PROFILES='{"preview":{"repetitions":2}}',
                    SAMPLE_SIZES='[20,40]', SAMPLE_BANK_REPETITIONS='2', SAMPLE_BANK_MAXIMUM_N='40',
                    SAMPLE_BANK_BLOCK_SIZE='20', SCAN_POINTS='65', SPATIAL_POINTS='4096', WORKERS='1',
                    continuous_cache_root='data_output_directory/"shared"'))
                for key in ('simulation','saved-data','validation','figures'):
                    execute(c[key], ns)
                self.assertEqual(len(list((base/'output/pdf/continuous_minimizers_selected_n').glob('*.pdf'))), 2)
                execute(c['qhat-control-1'], ns)
                ns['KDE_DIR'] = isolated
                execute(c['qhat-control-2'], ns, dict(SEARCH='EnergySearchConfig(scan_points=65, spatial_points=4096)'))
                execute(c['qhat-control-3'], ns)
                execute(c['qhat-control-4'], ns)
                self.assertTrue((ns['output_directory']/'analysis.json').is_file())

    def test_relocated_grid_and_planned_diagnostic_helpers_import(self):
        grid = runpy.run_path(str(ROOT/'experiments/joint_limits/run_experiment.py'))
        for r, name in [('1','linear_grid_joint_N_lambda_limit'), ('13/14','r13over14_joint_N_lambda_limit')]:
            self.assertEqual(grid['configure_grid'](r), name)
            self.assertEqual(grid['bandwidth_grid'](9).shape, (9,))
        for p in sorted((ROOT/'experiments').rglob('*.py')):
            namespace = runpy.run_path(str(p))
            directory = namespace.get('CODE_DIRECTORY', namespace.get('KDE_DIR'))
            self.assertEqual(directory, ROOT)


if __name__ == '__main__':
    unittest.main()
