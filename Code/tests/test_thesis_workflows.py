"""Run the thesis notebook sections with small, isolated inputs and exports."""

# Permit running individual tests and unittest discovery from the project root.
import sys as _sys
from pathlib import Path as _Path
_code_root = str(_Path(__file__).resolve().parents[1])
if _code_root not in _sys.path:
    _sys.path.insert(0, _code_root)

import ast
import contextlib
import io
import json
from pathlib import Path
import runpy
import sys
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
    with contextlib.redirect_stdout(io.StringIO()), \
            patch.object(sys, 'path', [str(ROOT/'notebooks'), *sys.path]):
        exec(compile(tree, '<thesis notebook>', 'exec'), ns)


class ThesisWorkflowTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')
        disable_automatic_pdf_saving()

    def test_numerics_exports(self):
        c = cells('02_kde_numerics')
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            isolated = base/'Code'
            ns = {'display': lambda value: None}
            exporter = lambda name, **kw: enable_automatic_pdf_saving(name, output_root=base/'pdfs')
            with patch('kde_tools.pdf_export.enable_automatic_pdf_saving', side_effect=exporter), \
                    patch('kde_tools.thesis_exports.KDE_DIRECTORY', isolated):
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
                pdfs = list((base/'pdfs/02_kde_numerics').glob('*.pdf'))
                self.assertEqual(len(pdfs), 2)  # Explicit thesis filename and automatic export.
                self.assertTrue(all(pdf.read_bytes().startswith(b'%PDF-') for pdf in pdfs))


    def test_local_minimum_control(self):
        control = cells('06_additional_experiments')
        with tempfile.TemporaryDirectory() as tmp:
            isolated = Path(tmp) / 'Code'
            ns = {'display': lambda value: None}
            with patch('kde_tools.pdf_export.enable_automatic_pdf_saving',
                       side_effect=lambda name, **kw: enable_automatic_pdf_saving(name, output_root=Path(tmp)/'pdfs')):
                execute(control['setup'], ns)
                execute(control['qhat-control-1'], ns)
                ns['KDE_DIR'] = isolated
                execute(control['qhat-control-2'], ns, dict(SEARCH='EnergySearchConfig(scan_points=65, spatial_points=4096)'))
                execute(control['qhat-control-3'], ns)
                execute(control['qhat-control-4'], ns)
                self.assertTrue((ns['output_directory']/'analysis.json').is_file())


    def test_relocated_grid_and_planned_diagnostic_helpers_import(self):
        grid = runpy.run_path(str(ROOT/'experiments/joint_limits/run_experiment.py'))
        for r, name in [('1','linear_grid_joint_N_lambda_limit'), ('13/14','r13over14_joint_N_lambda_limit')]:
            self.assertEqual(grid['configure_grid'](r), name)
            self.assertEqual(grid['bandwidth_grid'](9).shape, (9,))
        for p in sorted((ROOT/'experiments').rglob('*.py')):
            # Several scripts legitimately write results at module scope. Check
            # their syntax without executing or overwriting existing results.
            with self.subTest(script=p.relative_to(ROOT)):
                ast.parse(p.read_text(), filename=str(p))


if __name__ == '__main__':
    unittest.main()
