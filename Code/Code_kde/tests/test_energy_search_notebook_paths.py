"""Execute migrated notebook search cells with reduced, isolated inputs."""

import ast
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import runpy
import tempfile
import unittest

import numpy as np
import pandas as pd

from kde_tools.energy_minimization import EnergySearchConfig, minimize_bandwidth_energy
from kde_tools.qhat import minimize_qhat, expected_gaussian_qhat_log_energy
from kde_tools.pdf_export import disable_automatic_pdf_saving


ROOT = Path(__file__).resolve().parents[1]


def notebook(name):
    path, = ROOT.glob("xx_notebooks_*/"+name+".ipynb")
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
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(tree, "<unified notebook integration>", "exec"), namespace)


class UnifiedNotebookSearchTests(unittest.TestCase):
    def tearDown(self):
        disable_automatic_pdf_saving()

    def test_fast_overlay_cell_runs_and_exports_full_reports(self):
        nb = notebook("fast_l2_bandwidth_distributions_vs_n_lambda")
        with tempfile.TemporaryDirectory() as tmp:
            ns = dict(np=np, pd=pd, Path=Path, hashlib=hashlib, json=json, csv=csv,
                      code_directory=ROOT, data_output_directory=Path(tmp),
                      overlay_settings=dict(seed=61, sigma2=1., maximum_sample_size=40),
                      overlay_common_configuration={"settings": {"seed": 61, "sigma2": 1.,
                                                                 "maximum_sample_size": 40}},
                      overlay_repetitions=2, overlay_sample_sizes=[20, 40],
                      overlay_h_grid=np.geomspace(.01, 2., 17), OVERLAY_PROFILE="preview")
            execute("".join(nb["cells"][23]["source"]), ns, dict(
                QHAT_SCAN_POINTS="65", QHAT_SPATIAL_POINTS="4096", QHAT_WORKERS="1",
                continuous_cache_root="data_output_directory/'shared'",
                continuous_import_directories="[]"))
            reports = list(Path(tmp).rglob("n*_energy_analysis.json"))
            self.assertEqual(len(reports), 2)
            for path in reports:
                self.assertEqual(len(json.loads(path.read_text())), 2)

    def test_continuous_comparison_cells_run_validate_and_export(self):
        nb = notebook("continuous_ise_qhat_distributions_vs_n")
        with tempfile.TemporaryDirectory() as tmp:
            ns = {"display": lambda value: None}
            execute("".join(nb["cells"][1]["source"]), ns)
            ns.update(data_output_directory=Path(tmp)/"data", image_output_directory=Path(tmp)/"images")
            execute("".join(nb["cells"][3]["source"]), ns, dict(
                RUN_PROFILE='"preview"', PROFILES='{"preview": {"repetitions": 2}}',
                SAMPLE_SIZES="[20, 40]", SAMPLE_BANK_REPETITIONS="2", SAMPLE_BANK_MAXIMUM_N="40",
                SAMPLE_BANK_BLOCK_SIZE="20", SCAN_POINTS="65", SPATIAL_POINTS="4096", WORKERS="1",
                continuous_cache_root="data_output_directory/'shared'", continuous_import_directories="[]"))
            for index in (5, 7, 9):
                execute("".join(nb["cells"][index]["source"]), ns)
            self.assertLess(ns["validation_report"]["maximum_refinement_discrepancy"], 1e-7)
            self.assertEqual(len(list(Path(tmp).rglob("n*_energy_analysis.json"))), 2)

    def test_joint_notebook_parameter_mapping_and_expected_objective(self):
        nb = notebook("experiment_1_joint_N_lambda_limit")
        samples = np.random.default_rng(21).normal(size=50)
        ns = dict(np=np, samples=samples, H_MIN=.01, H_MAX=2., CUSTOM_BETA=1/7,
                  ENERGY_SEARCH=EnergySearchConfig(scan_points=65), minimize_qhat=minimize_qhat,
                  minimize_bandwidth_energy=minimize_bandwidth_energy,
                  expected_gaussian_qhat_log_energy=expected_gaussian_qhat_log_energy)
        ns["squared_differences"] = (samples[:, None]-samples[None, :])**2
        for index in (4, 13):
            tree = ast.parse("".join(nb["cells"][index]["source"]))
            allowed = {"continuous_minimum", "custom_beta_continuous_minimum"}
            tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) or
                         isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in allowed
                                                             for t in node.targets)]
            exec(compile(tree, "<joint search>", "exec"), ns)
        point = ns["custom_beta_continuous_minimum"]["global_minimum"]
        self.assertAlmostEqual(point["energy"], ns["local_custom_beta_metric"](
            point["h"], ns["squared_differences"]), places=12)
        tree = ast.parse("".join(nb["cells"][37]["source"]))
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and
                     node.name in ("expected_q_standard_gaussian", "exact_expected_q_minimizer")]
        exec(compile(tree, "<expected search>", "exec"), ns)
        value = ns["expected_q_standard_gaussian"](.4, 200, 1/7)
        self.assertAlmostEqual(value, expected_gaussian_qhat_log_energy(np.log(.4), 200, 13/14)[0])
        h = ns["exact_expected_q_minimizer"](200, 1/7, .5, 5.)
        self.assertTrue(.5*200**(-.2) <= h <= 5*200**(-.2))

    def test_local_experiment_adapters_recover_fixed_sample_reference(self):
        workflow = runpy.run_path(str(ROOT/"outputs/analysis/local_energy_minimizers_n200/run_experiment.py"))
        report = workflow["validate_reference_cases"]()
        self.assertTrue(report["monotone_single_gaussian_has_no_local_minimum"])
        self.assertLess(report["maximum_previous_sample_h_error"], 5e-7)


if __name__ == "__main__":
    unittest.main()
