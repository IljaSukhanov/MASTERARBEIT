"""Regression checks for the thesis's fixed-support Beta innovations."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from Code.Code_AR_processes.ar_tools import functions as ar_functions
from Code.Code_AR_processes.ar_tools.functions import (
    simulate_beta_ar,
    simulate_uniform_ar,
)


class BetaInnovationTests(unittest.TestCase):
    def test_plotting_cell_refreshes_a_cached_simulator(self):
        notebook = json.loads((
            Path(__file__).resolve().parents[1] / "ar_beta_sliding_window_2d.ipynb"
        ).read_text())
        sources = {
            cell["id"]: "".join(cell["source"])
            for cell in notebook["cells"] if cell["cell_type"] == "code"
        }
        namespace = {}
        exec(sources["setup"], namespace)
        exec(sources["settings"], namespace)
        exec(sources["theoretical-zonoid"], namespace)
        stale_simulator = Mock(side_effect=TypeError(
            "simulate_beta_ar() got an unexpected keyword argument 'innovation_bounds'"
        ))
        with tempfile.TemporaryDirectory() as directory:
            namespace.update(
                sample_size=100, burnin=20, beta_pairs=[(2.0, 5.0), (5.0, 2.0)],
                PDF_DIR=Path(directory), IMAGE_DIR=Path(directory),
                simulate_beta_ar=stale_simulator,
            )
            with (
                patch.object(ar_functions, "simulate_beta_ar", stale_simulator),
                patch.object(namespace["plt"], "show"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                exec(sources["plot-clouds"], namespace)
            self.assertTrue((Path(directory) / "beta_sliding_window_clouds_2d.pdf").is_file())
            self.assertTrue((Path(directory) / "beta_sliding_window_clouds_2d.png").is_file())
        stale_simulator.assert_not_called()
        for cloud in namespace["clouds"]:
            self.assertEqual(cloud.shape, (99, 2))
            residuals = cloud[:, 1] - 0.7 * cloud[:, 0]
            self.assertTrue(np.all(residuals >= -1.0))
            self.assertTrue(np.all(residuals <= 1.0))
        np.testing.assert_array_equal(*namespace["zonoid_boundaries"])

    def test_asymmetric_shapes_keep_fixed_support_and_nonzero_mean(self):
        for alpha, beta, expected_mean in ((2, 5, -3 / 7), (5, 2, 3 / 7)):
            with self.subTest(alpha=alpha, beta=beta):
                innovations = simulate_beta_ar(
                    [0.0], 50_000, alpha=alpha, beta=beta, rng=42,
                )
                self.assertTrue(np.all(innovations >= -1.0))
                self.assertTrue(np.all(innovations <= 1.0))
                self.assertAlmostEqual(innovations.mean(), expected_mean, delta=0.01)
                self.assertAlmostEqual(innovations.var(), 5 / 49, delta=0.005)

    def test_ar_residuals_follow_the_chosen_interval(self):
        series = simulate_beta_ar(
            [0.4, -0.2], 20_000, innovation_bounds=(-2.0, 0.5),
            alpha=2, beta=5, intercept=0.25, rng=123,
        )
        residuals = series[2:] - 0.25 - 0.4 * series[1:-1] + 0.2 * series[:-2]
        self.assertTrue(np.all(residuals >= -2.0))
        self.assertTrue(np.all(residuals <= 0.5))
        self.assertAlmostEqual(residuals.mean(), -9 / 7, delta=0.015)

    def test_general_interval_has_the_expected_beta_moments(self):
        innovations = simulate_beta_ar(
            [0.0], 50_000, innovation_bounds=(-2.0, 4.0),
            alpha=5, beta=2, rng=17,
        )
        self.assertTrue(np.all(innovations >= -2.0))
        self.assertTrue(np.all(innovations <= 4.0))
        self.assertAlmostEqual(innovations.mean(), 16 / 7, delta=0.02)
        self.assertAlmostEqual(innovations.var(), 45 / 49, delta=0.02)

    def test_initial_history_accounts_for_the_innovation_mean(self):
        # Nearly constant innovations at 3 and intercept 0.2 give mean
        # (3 + 0.2) / (1 - 0.4 - 0.2) = 8, including before burn-in.
        series = simulate_beta_ar(
            [0.4, 0.2], 10, burnin=0, innovation_bounds=(3.0, 3.0 + 1e-10),
            alpha=2, beta=5, intercept=0.2, rng=42,
        )
        np.testing.assert_allclose(series, 8.0, rtol=0, atol=1e-9)

    def test_uniform_wrapper_preserves_its_support(self):
        innovations = simulate_uniform_ar([0.0], 50_000, half_width=1.5, rng=42)
        self.assertTrue(np.all(innovations >= -1.5))
        self.assertTrue(np.all(innovations <= 1.5))
        self.assertAlmostEqual(innovations.mean(), 0.0, delta=0.015)
        self.assertAlmostEqual(innovations.var(), 0.75, delta=0.015)

    def test_rejects_invalid_shapes_and_support(self):
        for kwargs in (
            {"alpha": 0}, {"beta": -1}, {"alpha": np.nan}, {"beta": np.inf},
            {"innovation_bounds": (1.0, -1.0)},
            {"innovation_bounds": (1.0, 1.0)},
            {"innovation_bounds": (0.0, np.inf)},
            {"innovation_bounds": ((-1.0, 1.0),)},
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                simulate_beta_ar([0.5], 10, **kwargs)


if __name__ == "__main__":
    unittest.main()
