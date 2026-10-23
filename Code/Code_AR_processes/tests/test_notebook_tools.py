"""Checks that AR notebooks use importable helpers with explicit settings."""

import ast
import importlib
import inspect
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from Code.Code_AR_processes.ar_tools.breakpoint_selectors import (
    select_tail_ratio,
    tail_excess_ratios,
)
from Code.Code_AR_processes.ar_tools.experiments import run_mae_replication


AR_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_PATHS = sorted([
    *AR_DIR.glob("*.ipynb"),
    *(AR_DIR / "xx_notebooks_estimation").glob("*.ipynb"),
    *(AR_DIR / "xx_notebooks_tuning").glob("*.ipynb"),
    *(AR_DIR / "xx_notebooks_order_selection").glob("*.ipynb"),
    *(AR_DIR / "xx_notebooks_figures").glob("*.ipynb"),
])


class NotebookToolsTests(unittest.TestCase):
    def test_notebooks_have_no_local_function_definitions(self):
        for path in NOTEBOOK_PATHS:
            notebook = json.loads(path.read_text())
            for index, cell in enumerate(notebook["cells"], start=1):
                if cell["cell_type"] != "code":
                    continue
                with self.subTest(notebook=path.name, cell=index):
                    tree = ast.parse("".join(cell["source"]))
                    self.assertFalse(any(
                        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                        for node in ast.walk(tree)
                    ))

    def test_notebook_calls_match_shared_function_signatures(self):
        for path in NOTEBOOK_PATHS:
            notebook = json.loads(path.read_text())
            trees = [ast.parse("".join(cell["source"]))
                     for cell in notebook["cells"] if cell["cell_type"] == "code"]
            imports = {}
            for tree in trees:
                for node in ast.walk(tree):
                    if (isinstance(node, ast.ImportFrom) and node.module
                            and node.module.startswith("Code.Code_AR_processes.ar_tools.")):
                        module = importlib.import_module(node.module)
                        for alias in node.names:
                            imports[alias.asname or alias.name] = getattr(module, alias.name)
            for tree in trees:
                for node in ast.walk(tree):
                    if (not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name)
                            or node.func.id not in imports
                            or any(isinstance(arg, ast.Starred) for arg in node.args)
                            or any(keyword.arg is None for keyword in node.keywords)):
                        continue
                    with self.subTest(notebook=path.name, function=node.func.id):
                        inspect.signature(imports[node.func.id]).bind(
                            *[None for _ in node.args],
                            **{keyword.arg: None for keyword in node.keywords},
                        )

    def test_tail_selectors_keep_their_distinct_zero_denominator_rules(self):
        widths = np.array([10., 7., 5., 2., 1., 1.])
        orders = np.arange(1, 5)
        signed, denominators = tail_excess_ratios(
            widths, max_dimension=6, candidate_orders=orders,
        )
        selected, scores = select_tail_ratio(
            widths, max_dimension=6, candidate_orders=orders,
        )
        self.assertEqual(denominators[-1], 0.)
        self.assertTrue(np.isnan(signed[-1]))
        self.assertTrue(np.isposinf(scores[-1]))
        self.assertEqual(selected, 4)

    def test_mae_worker_runs_all_estimators_without_notebook_globals(self):
        configuration = {
            "phi": [.4, -.1], "T_values": [48], "k": 3, "K": 4,
            "burnin": 20, "half_width": 1., "process_seed": 51,
            "optimizer_seed": 52, "starts": 1, "iterations": 8,
            "initial_step": .2, "geometric_decay": .97,
            "invalid_u_0_threshold": 1e-10, "use_convex_hull": False,
            "methods": ["Gradient descent", "Yule-Walker", "Conditional QML",
                        "Horvath L-infinity LP"],
        }
        with tempfile.TemporaryDirectory() as directory:
            result = run_mae_replication(1, configuration, directory, resume=False)
        self.assertEqual(len(result), 4)
        self.assertTrue(result["status"].eq("ok").all(), result["error_message"].tolist())
        self.assertTrue(np.isfinite(result["coefficient_mae"]).all())
        estimates = result[["estimated_phi_1", "estimated_phi_2"]].to_numpy()
        absolute_errors = np.abs(estimates - np.array(configuration["phi"]))
        np.testing.assert_allclose(result["coefficient_mae"], absolute_errors.mean(axis=1))
        np.testing.assert_allclose(
            result[["absolute_error_phi_1", "absolute_error_phi_2"]], absolute_errors,
        )


if __name__ == "__main__":
    unittest.main()
