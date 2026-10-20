"""Distribution and integration checks for centered and fixed-support innovations."""

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
    beta_innovation_bounds,
    sample_beta_innovations,
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
                innovation_bounds=(-1.0, 1.0), innovation_width=None,
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
                    innovation_bounds=(-1.0, 1.0),
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

    def test_sampler_default_preserves_the_uniform_random_stream(self):
        expected_generator = np.random.default_rng(42)
        actual_generator = np.random.default_rng(42)
        expected = expected_generator.uniform(-1.0, 1.0, size=(8, 11))
        actual = sample_beta_innovations((8, 11), rng=actual_generator)
        np.testing.assert_array_equal(actual, expected)
        # The sampler must advance an existing generator exactly once.
        self.assertEqual(actual_generator.random(), expected_generator.random())

    def test_asymmetric_sampler_defaults_to_centered_width_two(self):
        from scipy.stats import beta as beta_distribution
        from scipy.stats import kstest

        for alpha, beta in ((2.0, 5.0), (5.0, 2.0), (0.5, 0.75)):
            with self.subTest(alpha=alpha, beta=beta):
                lower, upper = beta_innovation_bounds(alpha, beta)
                self.assertAlmostEqual(upper - lower, 2.0)
                self.assertAlmostEqual(lower + 2 * alpha / (alpha + beta), 0.0)
                values = sample_beta_innovations(50_000, alpha, beta, rng=42)
                self.assertTrue(np.all((values >= lower) & (values <= upper)))
                self.assertAlmostEqual(values.mean(), 0.0, delta=0.012)
                variance = 4 * alpha * beta / ((alpha + beta)**2 * (alpha + beta + 1))
                self.assertAlmostEqual(values.var(), variance, delta=0.012)
                # Check the whole distribution, including asymmetric shapes.
                self.assertLess(kstest(
                    values, beta_distribution(alpha, beta, loc=lower, scale=2).cdf,
                ).statistic, 0.01)
                # Population centering must not subtract the observed sample mean.
                self.assertNotEqual(float(values.mean()), 0.0)

    def test_custom_width_and_fixed_endpoints_have_distinct_means(self):
        centered = sample_beta_innovations(50_000, 2, 5, L=3, rng=17)
        self.assertAlmostEqual(centered.mean(), 0.0, delta=0.015)
        np.testing.assert_allclose(beta_innovation_bounds(2, 5, L=3), (-6 / 7, 15 / 7))
        fixed = sample_beta_innovations(50_000, 2, 5, a=-1, b=2, L=3, rng=17)
        self.assertAlmostEqual(fixed.mean(), -1 / 7, delta=0.015)
        self.assertTrue(np.all((fixed >= -1) & (fixed <= 2)))

    def test_new_and_legacy_fixed_interval_arguments_agree(self):
        legacy = simulate_beta_ar(
            [0.4, -0.2], 200, innovation_bounds=(-2, 4), alpha=2, beta=5, rng=42,
        )
        explicit = simulate_beta_ar(
            [0.4, -0.2], 200, alpha=2, beta=5, a=-2, b=4, L=6, rng=42,
        )
        np.testing.assert_array_equal(explicit, legacy)

    def test_centered_ar_residuals_use_requested_width(self):
        series = simulate_beta_ar(
            [0.4, -0.2], 50_000, alpha=2, beta=5, L=3, intercept=0.25, rng=42,
        )
        residuals = series[2:] - 0.25 - 0.4 * series[1:-1] + 0.2 * series[:-2]
        lower, upper = beta_innovation_bounds(2, 5, L=3)
        self.assertTrue(np.all((residuals >= lower) & (residuals <= upper)))
        self.assertAlmostEqual(residuals.mean(), 0.0, delta=0.015)
        self.assertAlmostEqual(series.mean(), 0.25 / (1 - 0.4 + 0.2), delta=0.02)

    def test_uniform_wrapper_and_general_simulator_are_identical(self):
        uniform = simulate_uniform_ar([0.4, -0.2], 200, half_width=1.5, rng=42)
        general = simulate_beta_ar([0.4, -0.2], 200, L=3, rng=42)
        np.testing.assert_array_equal(uniform, general)

    def test_scalar_empty_and_shaped_samples(self):
        self.assertIsInstance(sample_beta_innovations(rng=42), float)
        self.assertEqual(sample_beta_innovations(0, rng=42).shape, (0,))
        self.assertEqual(sample_beta_innovations((2, 3), 2, 5, rng=42).shape, (2, 3))

    def test_centered_plotting_uses_a_different_support_for_each_shape(self):
        notebook = json.loads((
            Path(__file__).resolve().parents[1] / "ar_beta_sliding_window_2d.ipynb"
        ).read_text())
        sources = {cell["id"]: "".join(cell["source"])
                   for cell in notebook["cells"] if cell["cell_type"] == "code"}
        namespace = {}
        with contextlib.redirect_stdout(io.StringIO()):
            for cell_id in ("setup", "settings", "theoretical-zonoid"):
                exec(sources[cell_id], namespace)
            with tempfile.TemporaryDirectory() as directory:
                namespace.update(
                    sample_size=500, burnin=100, beta_pairs=[(2.0, 5.0), (5.0, 2.0)],
                    PDF_DIR=Path(directory), IMAGE_DIR=Path(directory),
                )
                with patch.object(namespace["plt"], "show"):
                    exec(sources["plot-clouds"], namespace)
        expected_bounds = [beta_innovation_bounds(2, 5), beta_innovation_bounds(5, 2)]
        np.testing.assert_allclose(namespace["panel_innovation_bounds"], expected_bounds)
        self.assertFalse(np.allclose(*namespace["zonoid_boundaries"]))
        for cloud, (lower, upper) in zip(namespace["clouds"], expected_bounds):
            residuals = cloud[:, 1] - 0.7 * cloud[:, 0]
            self.assertTrue(np.all((residuals >= lower) & (residuals <= upper)))

    def test_ar1_notebook_preserves_its_original_uniform_draws(self):
        notebook = json.loads((
            Path(__file__).resolve().parents[1] / "ar1_sliding_window_embedding.ipynb"
        ).read_text())
        source = next("".join(c["source"]) for c in notebook["cells"]
                      if c["id"] == "simulate-ar1")
        namespace = {"np": np, "sample_beta_innovations": sample_beta_innovations}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(source, namespace)
        expected = np.random.default_rng(namespace["seed"]).uniform(
            -namespace["innovation_radius"], namespace["innovation_radius"],
            size=namespace["total_steps"],
        )
        np.testing.assert_array_equal(namespace["innovations"], expected)

    def test_rejects_inconsistent_widths_and_ambiguous_intervals(self):
        for kwargs in (
            {"a": -1}, {"b": 1}, {"L": 0}, {"L": -2}, {"L": np.inf},
            {"a": -1, "b": 1, "L": 3}, {"a": -1, "b": 1, "L": np.nan},
            {"a": np.nan, "b": 1}, {"a": 2, "b": 1},
            {"a": -1e308, "b": 1e308}, {"alpha": [1, 2]},
            {"alpha": True}, {"beta": "2"},
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                sample_beta_innovations(10, **kwargs)
        with self.assertRaises(ValueError):
            simulate_beta_ar([0.4], 10, innovation_bounds=(-1, 1), a=-1, b=1)

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
