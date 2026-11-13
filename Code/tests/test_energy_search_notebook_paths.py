"""Execute migrated notebook search cells with reduced, isolated inputs."""

# Permit running individual tests and unittest discovery from the project root.
import sys as _sys
from pathlib import Path as _Path
_code_root = str(_Path(__file__).resolve().parents[1])
if _code_root not in _sys.path:
    _sys.path.insert(0, _code_root)

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
