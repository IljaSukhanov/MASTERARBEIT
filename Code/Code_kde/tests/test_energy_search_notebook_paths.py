"""Execute migrated notebook search cells with reduced, isolated inputs."""

import ast
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from kde_tools.energy_minimization import EnergySearchConfig, minimize_bandwidth_energy
from kde_tools.qhat import minimize_qhat, expected_gaussian_qhat_log_energy
from kde_tools.pdf_export import disable_automatic_pdf_saving
from kde_tools.joint_experiments import (
    local_custom_beta_metric, expected_q_standard_gaussian,
    exact_expected_q_minimizer,
)


ROOT = Path(__file__).resolve().parents[1]


def notebook(name):
    path, = ROOT.glob("notebooks/"+name+".ipynb")
    return json.loads(path.read_text())


def execute(source, namespace, overrides=None):
    source = "\n".join(line for line in source.splitlines()
                       if not line.lstrip().startswith(("%", "!")))
    tree = ast.parse(source)
    if overrides:
        class Settings(ast.NodeTransformer):
            def visit_Assign(self, node):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in overrides:
                        node.value = ast.parse(overrides[target.id], mode="eval").body
                return self.generic_visit(node)
        tree = Settings().visit(tree)
        ast.fix_missing_locations(tree)
    with contextlib.redirect_stdout(io.StringIO()), \
            patch.object(sys, 'path', [str(ROOT/'notebooks'), *sys.path]):
        exec(compile(tree, "<unified notebook integration>", "exec"), namespace)


class UnifiedNotebookSearchTests(unittest.TestCase):
    def tearDown(self):
        disable_automatic_pdf_saving()


    def test_continuous_comparison_cells_run_validate_and_export(self):
        nb = notebook("05_increasing_sample_size_experiments")
        by_id = {cell['id']: cell for cell in nb['cells']}
        with tempfile.TemporaryDirectory() as tmp:
            ns = {"display": lambda value: None}
            execute("".join(by_id['setup']["source"]), ns)
            ns.update(data_output_directory=Path(tmp)/"data", image_output_directory=Path(tmp)/"images")
            execute("".join(by_id['settings']["source"]), ns, dict(
                RUN_PROFILE='"preview"', PROFILES='{"preview": {"repetitions": 2}}',
                SAMPLE_SIZES="[20, 40]", SAMPLE_BANK_REPETITIONS="2", SAMPLE_BANK_MAXIMUM_N="40",
                SAMPLE_BANK_BLOCK_SIZE="20", SCAN_POINTS="65", SPATIAL_POINTS="4096", WORKERS="1",
                continuous_cache_root="data_output_directory/'shared'", continuous_import_directories="[]"))
            for cell_id in ('simulation', 'saved-data', 'validation'):
                execute("".join(by_id[cell_id]["source"]), ns)
            self.assertLess(ns["validation_report"]["maximum_refinement_discrepancy"], 1e-7)
            self.assertEqual(len(list(Path(tmp).rglob("n*_energy_analysis.json"))), 2)

    def test_joint_notebook_parameter_mapping_and_expected_objective(self):
        nb = notebook("03_fixed_sample_experiments")
        samples = np.random.default_rng(21).normal(size=50)
        ns = dict(np=np, samples=samples, H_MIN=.01, H_MAX=2., CUSTOM_BETA=1/7,
                  ENERGY_SEARCH=EnergySearchConfig(scan_points=65), minimize_qhat=minimize_qhat,
                  minimize_bandwidth_energy=minimize_bandwidth_energy,
                  expected_gaussian_qhat_log_energy=expected_gaussian_qhat_log_energy)
        ns["squared_differences"] = (samples[:, None]-samples[None, :])**2
        by_id = {cell['id']: cell for cell in nb['cells']}
        for cell_id in ('continuous-reference',):
            tree = ast.parse("".join(by_id[cell_id]["source"]))
            allowed = {"continuous_minimum", "custom_beta_continuous_minimum"}
            tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) or
                         isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in allowed
                                                             for t in node.targets)]
            exec(compile(tree, "<joint search>", "exec"), ns)
        point = minimize_qhat(samples, .01, 2., r=13/14,
                              config=ns['ENERGY_SEARCH'])["global_minimum"]
        self.assertAlmostEqual(point["energy"], local_custom_beta_metric(
            point["h"], ns["squared_differences"], beta=1/7), places=12)
        value = expected_q_standard_gaussian(.4, 200, 1/7)
        self.assertAlmostEqual(value, expected_gaussian_qhat_log_energy(np.log(.4), 200, 13/14)[0])
        h = exact_expected_q_minimizer(200, 1/7, .5, 5., search=ns['ENERGY_SEARCH'])
        self.assertTrue(.5*200**(-.2) <= h <= 5*200**(-.2))



if __name__ == "__main__":
    unittest.main()
