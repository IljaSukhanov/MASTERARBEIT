"""Run each thesis experiment group independently with isolated small inputs."""

from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from experiments.joint_limits import run_experiment as grid_experiment
from kde_tools.pdf_export import enable_automatic_pdf_saving, disable_automatic_pdf_saving
from tests.test_thesis_workflows import ROOT, cells, execute


GROUPS = (
    '03_fixed_sample_experiments', '04_repeated_sampling_experiments',
    '05_increasing_sample_size_experiments', '06_additional_experiments',
)


class NotebookReorganizationTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')
        disable_automatic_pdf_saving()

    def test_thesis_groups_have_distinct_entry_points(self):
        expected = {'01_classical_bandwidth_kde_illustration', '02_kde_numerics', *GROUPS}
        self.assertEqual({p.stem for p in (ROOT/'notebooks').glob('*.ipynb')}, expected)
        required = {
            GROUPS[0]: ('beta2-stationary-grid-comparison', 'joint-limit-computation'),
            GROUPS[1]: ('beta2-configurable-selections', 'repeated-sample-computation', 'e1081780'),
            GROUPS[2]: ('beta2-sample-sweep', 'simulation', 'simultaneous-asymptotic-bridge'),
            GROUPS[3]: ('mesh-perturbation-computation', 'local-row-sum-mechanism', 'qhat-control-2'),
        }
        for name, cell_ids in required.items():
            self.assertTrue(all(cell_id in cells(name) for cell_id in cell_ids), name)

    def test_all_experiment_groups_run_independently_and_export_pdfs(self):
        overrides = dict(
            N_DATA='20', N_VALUES='np.array([15, 25])',
            stationary_example_n='20', stationary_example_N='9',
            stationary_example_lambda_grid_N='9',
            stationary_example_grid_fineness_N_values='np.array([9, 15])',
            stationary_example_grid_fineness_tick_values='np.array([9, 15])',
            beta2_n='20', beta2_N='7', beta2_M='2', beta2_binning='1', beta2_workers='1',
            REPEATED_SAMPLE_M='2', REPEATED_SAMPLE_N='9',
            MC_N='9', MC_SAMPLE_SIZE='20', MC_REPETITIONS='2',
            beta2_n_sweep_N='7', beta2_n_sweep_M='2', beta2_n_sweep_workers='1',
            beta2_n_sweep_binning='1', SAMPLE_SIZE_SWEEP_VALUES='np.array([20, 40])',
            SAMPLE_SIZE_SWEEP_M='2', SAMPLE_SIZE_SWEEP_N='9', RUN_PROFILE='"preview"',
            PROFILES='{"preview": {"repetitions": 2}}', SAMPLE_SIZES='[20, 40]',
            SAMPLE_BANK_REPETITIONS='2', SAMPLE_BANK_MAXIMUM_N='40', SAMPLE_BANK_BLOCK_SIZE='20',
            SCAN_POINTS='65', SPATIAL_POINTS='4096', WORKERS='1',
            ASYMPTOTIC_M='2', ASYMPTOTIC_N_VALUES='np.array([20, 40])', ASYMPTOTIC_GRID_BASE='25',
            ZOOM_N_MIN='0', MESH_STRESS_N='15', MESH_STRESS_N_VALUES='np.array([15, 25])',
            MESH_ALPHA_VALUES='np.array([0., .5, 1., 1.5, 2., 3.])',
            STRESS_GAP='.5', DISPLAY_N='25',
            SEARCH='EnergySearchConfig(scan_points=65, spatial_points=4096)',
        )
        for name in GROUPS:
            with self.subTest(notebook=name), tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp)
                isolated = base/'Code/Code_kde'
                shutil.copytree(ROOT/'kde_tools', isolated/'kde_tools',
                                ignore=shutil.ignore_patterns('__pycache__'))
                namespace = {}
                exporter = lambda name, **kw: enable_automatic_pdf_saving(name, output_root=base/'pdfs')
                with patch('kde_tools.pdf_export.enable_automatic_pdf_saving', side_effect=exporter), \
                     patch('kde_tools.thesis_exports.KDE_DIRECTORY', isolated), \
                     patch.dict(grid_experiment.__dict__,
                                KDE_DIR=isolated, N_VALUES=np.array([15, 25]), N_DATA=20):
                    for cell_id, cell in cells(name).items():
                        if cell['cell_type'] != 'code':
                            continue
                        execute(cell, namespace, overrides)
                        if cell_id == 'setup':
                            namespace.update(code_directory=isolated, project_root=base,
                                             kde_thesis_figure_dir=base/'thesis')
                            namespace['kde_thesis_figure_dir'].mkdir()
                    pdfs = list((base/'pdfs'/name).glob('*.pdf'))
                    self.assertTrue(pdfs, name)
                    self.assertTrue(all(p.read_bytes().startswith(b'%PDF') for p in pdfs))
                plt.close('all')
                disable_automatic_pdf_saving()


if __name__ == '__main__':
    unittest.main()
