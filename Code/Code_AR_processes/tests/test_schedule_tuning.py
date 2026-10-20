"""Checks for paired geometric-schedule tuning on one fixed cloud."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from Code.Code_AR_processes.ar_tools.descent import projected_subgradient_empirical_width
from Code.Code_AR_processes.ar_tools.tuning import tune_geometric_width_schedule


class ScheduleTuningTests(unittest.TestCase):
    def setUp(self):
        self.cloud = np.array([[-2.0, -0.1], [2.0, -0.1], [2.0, 0.1], [-2.0, 0.1]])
        self.directions = np.array([[1.0, 1.0], [-0.4, 1.0]])
        self.settings = dict(iterations=30, patience=8, improvement_tolerance=1e-10,
                             optimizer_seed=17)

    def test_singleton_grid_reproduces_direct_descents(self):
        result = tune_geometric_width_schedule(
            self.cloud, self.directions, [0.03], [0.97], **self.settings,
        )
        direct = [
            projected_subgradient_empirical_width(
                self.cloud, initial_direction=direction,
                initial_step=0.03, geometric_decay=0.97, step_schedule="geometric",
                rng=np.random.SeedSequence([17, 0, 2, 21, index]),
                iterations=30, patience=8, improvement_tolerance=1e-10,
            )
            for index, direction in enumerate(self.directions)
        ]
        for actual, expected in zip(result["best_results"], direct):
            np.testing.assert_array_equal(actual["width_history"], expected["width_history"])
            np.testing.assert_array_equal(actual["direction"], expected["direction"])
        self.assertEqual(result["selected"]["minimum_width"], min(r["minimum_width"] for r in direct))

    def test_grid_selection_is_order_independent_and_preserves_inputs(self):
        original = self.directions.copy()
        forward = tune_geometric_width_schedule(
            self.cloud, self.directions, [0.01, 0.05], [0.9, 0.99], **self.settings,
        )
        reverse = tune_geometric_width_schedule(
            self.cloud, self.directions, [0.05, 0.01], [0.99, 0.9], **self.settings,
        )
        self.assertEqual(len(forward["candidates"]), 4)
        self.assertEqual(len(forward["per_start"]), 8)
        self.assertEqual(forward["selected"]["gamma0"], reverse["selected"]["gamma0"])
        self.assertEqual(forward["selected"]["beta"], reverse["selected"]["beta"])
        np.testing.assert_array_equal(self.directions, original)
        np.testing.assert_array_equal(forward["direction"], reverse["direction"])
        selected = forward["selected"]
        self.assertEqual(selected["minimum_width"], min(r["minimum_width"] for r in forward["candidates"]))
        self.assertAlmostEqual(np.ptp(self.cloud @ forward["direction"]), selected["minimum_width"])
        history = forward["histories"][selected["candidate_id"]]
        self.assertTrue(np.all(np.diff(history) <= 0))
        self.assertAlmostEqual(history[-1], selected["minimum_width"])

    def test_invalid_grids_and_directions_are_rejected(self):
        for gamma, beta in (([], [0.9]), ([0.1], []), ([0.0], [0.9]),
                            ([np.nan], [0.9]), ([0.1], [1.0]), ([0.1], [0.0])):
            with self.subTest(gamma=gamma, beta=beta), self.assertRaises(ValueError):
                tune_geometric_width_schedule(self.cloud, self.directions, gamma, beta, **self.settings)
        for directions in (np.zeros((1, 2)), np.ones((1, 3)), np.array([[np.inf, 1.0]])):
            with self.subTest(directions=directions), self.assertRaises(ValueError):
                tune_geometric_width_schedule(self.cloud, directions, [0.1], [0.9], **self.settings)

    def test_notebook_bottom_runs_and_exports_a_small_search(self):
        notebook = json.loads((
            Path(__file__).resolve().parents[1] / "xx_notebooks_tuning"
            / "ar_single_process_coefficient_estimation.ipynb"
        ).read_text())
        sources = {cell["id"]: "".join(cell["source"])
                   for cell in notebook["cells"] if cell["cell_type"] == "code"}
        namespace = {}
        with contextlib.redirect_stdout(io.StringIO()), tempfile.TemporaryDirectory() as directory:
            exec(sources["imports"], namespace)
            exec(sources["settings"], namespace)
            namespace.update(p=2, embedding_dimension=3, sample_size=60, burnin=20,
                             number_of_starts=2, optimization_iterations=20, patience=8)
            exec(sources["simulate-one-process"], namespace)
            exec(sources["find-and-print-direction"], namespace)
            exec(sources["gamma0-beta-tuning-settings"], namespace)
            namespace.update(
                gamma0_grid=np.array([namespace["initial_step"]]),
                beta_grid=np.array([0.9, namespace["geometric_decay"]]),
                TUNING_DATA_DIR=Path(directory) / "data",
                TUNING_PDF_DIR=Path(directory) / "pdfs",
                TUNING_IMAGE_DIR=Path(directory) / "images",
            )
            exec(sources["run-gamma0-beta-tuning"], namespace)
            exec(sources["report-gamma0-beta-tuning"], namespace)
            exec(sources["plot-gamma0-beta-tuning"], namespace)
            selected = json.loads((Path(directory) / "data/selected_schedule.json").read_text())
            self.assertEqual(len(namespace["tuning_ranking"]), 2)
            self.assertLessEqual(selected["minimum_width"], selected["baseline_minimum_width"])
            self.assertEqual(namespace["tuned_estimated_phi"].shape, (2,))
            for name in ("data/grid_results.csv", "data/per_start.csv", "data/configuration.json",
                         "pdfs/gamma0_beta_tuning.pdf", "images/gamma0_beta_tuning.png"):
                self.assertTrue((Path(directory) / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
