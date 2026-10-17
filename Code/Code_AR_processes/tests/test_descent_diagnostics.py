import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import Mock

import numpy as np

from Code.Code_AR_processes.ar_tools.descent import projected_subgradient_empirical_width


class DescentDiagnosticsTests(unittest.TestCase):
    def test_notebook_descent_cell_refreshes_a_stale_function_binding(self):
        notebook_path = (
            Path(__file__).resolve().parents[1]
            / "ar_single_process_coefficient_estimation.ipynb"
        )
        notebook = json.loads(notebook_path.read_text())
        sources = {
            cell["id"]: "".join(cell["source"])
            for cell in notebook["cells"] if cell["cell_type"] == "code"
        }
        namespace = {}
        stale_optimizer = Mock(side_effect=AssertionError("Stale optimizer was called"))
        with contextlib.redirect_stdout(io.StringIO()):
            exec(sources["imports"], namespace)
            exec(sources["settings"], namespace)
            namespace.update(
                sample_size=40, burnin=20,
                number_of_starts=2, optimization_iterations=3,
            )
            exec(sources["simulate-one-process"], namespace)
            namespace["projected_subgradient_empirical_width"] = stale_optimizer
            # Rerun only the descent cell, without rerunning setup.
            exec(sources["find-and-print-direction"], namespace)
        stale_optimizer.assert_not_called()
        self.assertEqual(len(namespace["start_summary"]), 2)
        for result in namespace["descent_results"]:
            self.assertIn("termination_reason", result)
            self.assertIn("width_history", result)

    def test_constant_direction_keeps_full_history(self):
        result = projected_subgradient_empirical_width(
            np.array([[-2.0, 0.0], [1.0, 0.0]]),
            iterations=10,
            initial_direction=np.array([1.0, 0.0]),
        )
        self.assertEqual(result["termination_reason"], "iteration_limit")
        self.assertEqual(len(result["width_history"]), 10)
        np.testing.assert_allclose(result["width_history"], np.full(10, 3.0))
        np.testing.assert_allclose(result["best_width_history"], np.full(10, 3.0))

    def test_constant_width_uses_patience_counter(self):
        result = projected_subgradient_empirical_width(
            np.array([[-2.0, 0.0], [1.0, 0.0]]),
            iterations=10,
            patience=3,
            initial_direction=np.array([1.0, 0.0]),
        )
        self.assertEqual(result["termination_reason"], "patience")
        self.assertEqual(len(result["width_history"]), 4)
        np.testing.assert_allclose(result["width_history"], np.full(4, 3.0))

    def test_patience_records_aligned_width_and_step_histories(self):
        result = projected_subgradient_empirical_width(
            np.array([[0.0, 0.0], [3.0, 4.0]]),
            iterations=10,
            patience=1,
            improvement_tolerance=100.0,
            initial_direction=np.array([1.0, 0.0]),
        )
        self.assertEqual(result["termination_reason"], "patience")
        self.assertEqual(len(result["width_history"]), 2)
        for key in ("best_width_history", "step_history"):
            self.assertEqual(len(result[key]), len(result["width_history"]))
            self.assertTrue(np.isfinite(result[key]).all())


if __name__ == "__main__":
    unittest.main()
