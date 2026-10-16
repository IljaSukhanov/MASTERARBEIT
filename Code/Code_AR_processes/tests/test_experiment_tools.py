"""Regression checks for notebook entry points and shared experiment workers."""

import ast
import contextlib
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
import pandas as pd

from Code.Code_AR_processes.ar_tools.experiments import (
    run_ar3_coefficient_mse_experiment,
    run_ar3_mse_process,
)


AR_DIR = Path(__file__).resolve().parents[1]


class ExperimentToolsTests(unittest.TestCase):
    def test_experiment_folder_has_no_top_level_python_files(self):
        self.assertEqual(list(AR_DIR.glob("*.py")), [])

    def test_every_former_script_experiment_has_a_notebook_entry_point(self):
        required = {
            "run_ar3_coefficient_mse_experiment", "run_ar3_warm_start_experiment",
            "run_ar3_population_selector_experiment", "run_ar3_lp_descent_experiment",
            "fit_ar3_gradient_examples", "plot_ar3_gradient_examples",
        }
        called = set()
        for path in AR_DIR.glob("*.ipynb"):
            notebook = json.loads(path.read_text())
            trees = [ast.parse("".join(cell["source"]))
                     for cell in notebook["cells"] if cell["cell_type"] == "code"]
            imports = {}
            for tree in trees:
                for node in ast.walk(tree):
                    if (isinstance(node, ast.ImportFrom) and node.module
                            and node.module.startswith("Code.Code_AR_processes.ar_tools.")):
                        imports.update({alias.asname or alias.name: alias.name for alias in node.names})
            for tree in trees:
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                        called.add(imports.get(node.func.id))
        self.assertTrue(required <= called, required - called)

    def test_shared_tools_do_not_import_the_script_launchers(self):
        for path in (AR_DIR / "ar_tools").glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and node.module:
                    self.assertFalse(node.module.startswith("Code.Code_AR_processes.ar3_"), path.name)

    def test_notebook_setup_finds_the_project_without_launchers(self):
        notebook_names = [
            "ar3_gradient_figure_2_4_examples", "ar3_descent_chain_lp",
            "ar3_order_selection_warm_starts", "ar3_population_breakpoint_selectors",
        ]
        for directory in (AR_DIR, AR_DIR.parents[1]):
            for name in notebook_names:
                with self.subTest(notebook=name, directory=directory), contextlib.chdir(directory):
                    notebook = json.loads((AR_DIR / f"{name}.ipynb").read_text())
                    setup = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
                    namespace = {"__name__": "__notebook_setup__"}
                    exec(compile("".join(setup["source"]), name, "exec"), namespace)
                    self.assertEqual(namespace["PROJECT_ROOT"], AR_DIR.parents[1])

    def test_mse_worker_is_picklable_and_uses_explicit_settings(self):
        self.assertIs(pickle.loads(pickle.dumps(run_ar3_mse_process)), run_ar3_mse_process)
        configuration = {
            "order": 3, "embedding_dimension": 4, "replications": 1,
            "sample_sizes": [40, 56], "burnin": 20, "experiment_seed": 51,
            "optimizer_seed": 52, "starts": 2, "iterations": 12,
            "initial_step_scale": .2, "geometric_decay": .97,
        }
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            run_ar3_coefficient_mse_experiment(
                configuration=configuration, output_directory=directory, n_jobs=1,
            )
            results = pd.read_csv(Path(directory) / "replication_results.csv")
            resumed = run_ar3_mse_process(
                0, configuration=configuration, cache_directory=Path(directory) / "processes",
            )
        self.assertEqual(results["T"].tolist(), [40, 56])
        self.assertTrue(results["status"].eq("ok").all(), results["error_message"].tolist())
        self.assertTrue(np.isfinite(results["coefficient_mse"]).all())
        pd.testing.assert_frame_equal(results, resumed)


if __name__ == "__main__":
    unittest.main()
