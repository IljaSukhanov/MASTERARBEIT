"""Check the notebook/module boundary and automatic figure delivery."""

import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from IPython.core.interactiveshell import InteractiveShell
import matplotlib.pyplot as plt
import numpy as np

from kde_tools.joint_experiments import local_custom_beta_metric, psi_beta, inverse_psi_beta
from kde_tools.numerical_experiments import run_fourier_error_experiment
from kde_tools.pdf_export import enable_automatic_pdf_saving, disable_automatic_pdf_saving
from kde_tools.qhat import GaussianQhatObjective


ROOT = Path(__file__).resolve().parents[1]


class NotebookStructureTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')
        disable_automatic_pdf_saving()

    def test_notebooks_import_functions_instead_of_defining_them(self):
        for path in (ROOT/'notebooks').glob('*.ipynb'):
            for cell in json.loads(path.read_text())['cells']:
                if cell['cell_type'] != 'code':
                    continue
                tree = ast.parse(''.join(cell['source']))
                with self.subTest(notebook=path.name, cell=cell['id']):
                    self.assertFalse(any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                                         for n in ast.walk(tree)))
                    self.assertFalse(any(isinstance(n, ast.ImportFrom) and
                                         any(a.name == '*' for a in n.names) for n in ast.walk(tree)))

    def test_imported_metric_agrees_with_shared_energy_objective(self):
        samples = np.random.default_rng(31).normal(size=31)
        differences = (samples[:, None] - samples[None, :])**2
        objective = GaussianQhatObjective(samples)
        for beta in (1/7, 1., 2.):
            for h in (.1, .4, 1.5):
                self.assertAlmostEqual(
                    local_custom_beta_metric(h, differences, beta=beta),
                    objective.log_energy(np.log(h), 1-beta/2)[0], places=12,
                )
                self.assertAlmostEqual(inverse_psi_beta(psi_beta(h, beta), beta), h, places=12)

    def test_implicit_notebook_display_saves_one_pdf(self):
        shell = InteractiveShell()
        with tempfile.TemporaryDirectory() as tmp, patch('IPython.get_ipython', return_value=shell):
            figure, ax = plt.subplots()
            ax.plot([0, 1], [1, 2])
            shell.events.register('post_execute', lambda: plt.close('all'))
            original_callbacks = list(shell.events.callbacks['post_execute'])
            directory = enable_automatic_pdf_saving('inline_figure', output_root=tmp)
            shell.events.trigger('post_execute')
            shell.events.trigger('post_execute')
            files = list(directory.glob('*.pdf'))
            self.assertEqual(len(files), 1)
            self.assertTrue(files[0].read_bytes().startswith(b'%PDF'))
            disable_automatic_pdf_saving()
            self.assertEqual(shell.events.callbacks['post_execute'], original_callbacks)

    def test_fourier_error_runner_reuses_its_saved_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/'run'
            directory.mkdir()
            settings = {'summary_curves': ['maximum']}
            (directory/'settings.json').write_text(json.dumps(settings))
            arguments = dict(
                K_candidates=np.array([32, 64]), N=3, bandwidths=np.geomspace(.1, 1., 3),
                data_directory=directory, frequency_cutoff=30., realizations=2,
                samples=np.random.default_rng(17).normal(size=(2, 12)), settings=settings, workers=1,
            )
            first = run_fourier_error_experiment(**arguments)
            with patch('kde_tools.numerical_experiments.calculate_one_sample',
                       side_effect=AssertionError('A cache hit must not recompute the sample')):
                resumed = run_fourier_error_experiment(**arguments)
            for original, cached in zip(first[:3], resumed[:3]):
                np.testing.assert_array_equal(original, cached)
            self.assertEqual(resumed[3], 0.)
            self.assertEqual(resumed[4], directory)


if __name__ == '__main__':
    unittest.main()
