"""Checks for the thesis's common mean-zero, variance-1/3 Beta innovations."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.stats import beta as beta_distribution, kstest

from Code.Code_AR_processes.ar_tools import functions as ar_functions
from Code.Code_AR_processes.ar_tools.functions import (
    beta_innovation_bounds,
    sample_beta_innovations,
    simulate_beta_ar,
    simulate_uniform_ar,
)

AR_DIRECTORY = Path(__file__).resolve().parents[1]


class BetaInnovationTests(unittest.TestCase):
    def test_endpoints_satisfy_both_population_conditions(self):
        for alpha, beta in ((1, 1), (2, 5), (5, 2), (0.5, 0.75), (5, 5), (0.1, 10)):
            with self.subTest(alpha=alpha, beta=beta):
                lower, upper = beta_innovation_bounds(alpha, beta)
                width = upper - lower
                expected_width = (alpha + beta) * np.sqrt(
                    (alpha + beta + 1) / (3 * alpha * beta)
                )
                self.assertLess(lower, 0)
                self.assertGreater(upper, 0)
                self.assertAlmostEqual(upper * alpha + lower * beta, 0, places=12)
                self.assertAlmostEqual(width, expected_width, places=12)
                variance = width**2 * alpha * beta / (
                    (alpha + beta)**2 * (alpha + beta + 1)
                )
                self.assertAlmostEqual(variance, 1 / 3, places=13)
        self.assertEqual(beta_innovation_bounds(), (-1.0, 1.0))

    def test_normalized_samples_match_the_full_beta_distribution(self):
        for alpha, beta in ((2.0, 5.0), (5.0, 2.0), (0.5, 0.75), (5.0, 5.0)):
            with self.subTest(alpha=alpha, beta=beta):
                # Compute expected support directly from the mathematical result.
                lower = -np.sqrt(alpha * (alpha + beta + 1) / (3 * beta))
                upper = np.sqrt(beta * (alpha + beta + 1) / (3 * alpha))
                values = sample_beta_innovations(100_000, alpha, beta, rng=42)
                self.assertTrue(np.all((values >= lower - 1e-14) & (values <= upper + 1e-14)))
                self.assertAlmostEqual(values.mean(), 0.0, delta=0.008)
                self.assertAlmostEqual(values.var(), 1 / 3, delta=0.008)
                self.assertLess(kstest(
                    values, beta_distribution(alpha, beta, loc=lower, scale=upper-lower).cdf,
                ).statistic, 0.008)
                # Theoretical normalization must not force empirical moments.
                self.assertNotEqual(float(values.mean()), 0.0)
                self.assertNotEqual(float(values.var()), 1 / 3)

    def test_default_preserves_the_uniform_random_stream(self):
        expected_generator = np.random.default_rng(42)
        actual_generator = np.random.default_rng(42)
        expected = expected_generator.uniform(-1.0, 1.0, size=(8, 11))
        actual = sample_beta_innovations((8, 11), rng=actual_generator)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual_generator.random(), expected_generator.random())

    def test_uniform_wrapper_and_general_simulator_are_identical(self):
        uniform = simulate_uniform_ar([0.4, -0.2], 200, rng=42)
        general = simulate_beta_ar([0.4, -0.2], 200, rng=42)
        np.testing.assert_array_equal(uniform, general)

    def test_both_simulators_call_the_single_sampler(self):
        for name, kwargs in (
            ("simulate_beta_ar", {"alpha": 2, "beta": 5}),
            ("simulate_uniform_ar", {}),
        ):
            with self.subTest(entry_point=name), patch.object(
                ar_functions, "sample_beta_innovations", wraps=sample_beta_innovations,
            ) as sampler:
                values = getattr(ar_functions, name)([0.4], 50, burnin=10, rng=42, **kwargs)
                self.assertEqual(values.shape, (50,))
                sampler.assert_called_once()

    def test_legacy_arguments_are_consistency_checks_only(self):
        alpha, beta = 2, 5
        lower, upper = beta_innovation_bounds(alpha, beta)
        width = upper - lower
        expected = sample_beta_innovations(200, alpha, beta, rng=42)
        actual = sample_beta_innovations(
            200, alpha, beta, a=lower, b=upper, L=width, rng=42,
        )
        np.testing.assert_array_equal(actual, expected)
        expected = simulate_beta_ar([0.4, -0.2], 200, alpha=alpha, beta=beta, rng=42)
        for kwargs in (
            {"innovation_bounds": (lower, upper), "L": width},
            {"a": lower, "b": upper, "L": width},
        ):
            with self.subTest(**kwargs):
                actual = simulate_beta_ar(
                    [0.4, -0.2], 200, alpha=alpha, beta=beta, rng=42, **kwargs,
                )
                np.testing.assert_array_equal(actual, expected)

    def test_no_endpoint_or_width_override_can_break_normalization(self):
        for kwargs in (
            {"a": -1, "b": 1},   # Former fixed-support asymmetric mode.
            {"L": 2},             # Former fixed-width asymmetric mode.
            {"a": -2 / 7, "b": 5 / 7, "L": 1},
            {"a": -1, "b": 2, "L": 3},
        ):
            with self.subTest(**kwargs), self.assertRaisesRegex(ValueError, "determined"):
                sample_beta_innovations(10, alpha=2, beta=5, **kwargs)
            with self.subTest(simulator=kwargs), self.assertRaisesRegex(ValueError, "determined"):
                simulate_beta_ar([0.4], 10, alpha=2, beta=5, **kwargs)
        with self.assertRaisesRegex(ValueError, "determined"):
            simulate_beta_ar([0.4], 10, alpha=2, beta=5, innovation_bounds=(-1, 1))
        with self.assertRaisesRegex(ValueError, "determined"):
            simulate_uniform_ar([0.4], 10, half_width=1.5)

    def test_ar_residuals_and_stationary_mean_use_the_normalization(self):
        for alpha, beta in ((1, 1), (2, 5), (5, 2), (5, 5)):
            with self.subTest(alpha=alpha, beta=beta):
                series = simulate_beta_ar(
                    [0.4, -0.2], 50_000, alpha=alpha, beta=beta, intercept=0.25, rng=42,
                )
                residuals = series[2:] - 0.25 - 0.4 * series[1:-1] + 0.2 * series[:-2]
                lower, upper = beta_innovation_bounds(alpha, beta)
                self.assertTrue(np.all((residuals >= lower - 1e-14) &
                                       (residuals <= upper + 1e-14)))
                self.assertAlmostEqual(residuals.mean(), 0.0, delta=0.012)
                self.assertAlmostEqual(residuals.var(), 1 / 3, delta=0.01)
                self.assertAlmostEqual(series.mean(), 0.25 / 0.8, delta=0.02)

    def test_initial_history_uses_zero_innovation_mean(self):
        with patch.object(ar_functions, "sample_beta_innovations", return_value=np.zeros(12)):
            series = simulate_beta_ar(
                [0.4, 0.2], 10, burnin=0, alpha=2, beta=5, intercept=0.2, rng=42,
            )
        np.testing.assert_allclose(series, 0.2 / (1 - 0.4 - 0.2), rtol=0, atol=1e-15)

    def test_scalar_empty_and_shaped_samples(self):
        self.assertIsInstance(sample_beta_innovations(rng=42), float)
        self.assertEqual(sample_beta_innovations(0, rng=42).shape, (0,))
        self.assertEqual(sample_beta_innovations((2, 3), 2, 5, rng=42).shape, (2, 3))

    def test_rejects_invalid_shapes_and_derived_parameter_checks(self):
        for kwargs in (
            {"alpha": 0}, {"beta": -1}, {"alpha": np.nan}, {"beta": np.inf},
            {"alpha": [1, 2]}, {"alpha": True}, {"beta": "2"}, {"beta": np.complex128(1j)},
            {"a": -1}, {"b": 1}, {"L": 0}, {"L": -2}, {"L": np.inf},
            {"L": np.nan}, {"a": np.nan, "b": 1}, {"a": 2, "b": 1},
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                sample_beta_innovations(10, **kwargs)
            with self.subTest(simulator=kwargs), self.assertRaises(ValueError):
                simulate_beta_ar([0.4], 10, **kwargs)
        for bounds in ((1, -1), (1, 1), (0, np.inf), ((-1, 1),)):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                simulate_beta_ar([0.4], 10, innovation_bounds=bounds)
        with self.assertRaises(ValueError):
            simulate_beta_ar([0.4], 10, innovation_bounds=(-1, 1), a=-1, b=1)

    def test_plotting_refreshes_the_simulator_and_uses_normalized_supports(self):
        notebook = json.loads((AR_DIRECTORY / "ar_beta_sliding_window_2d.ipynb").read_text())
        sources = {cell["id"]: "".join(cell["source"])
                   for cell in notebook["cells"] if cell["cell_type"] == "code"}
        namespace = {}
        stale = Mock(side_effect=TypeError("stale simulator"))
        with contextlib.redirect_stdout(io.StringIO()):
            for cell_id in ("setup", "settings", "theoretical-zonoid"):
                exec(sources[cell_id], namespace)
            with tempfile.TemporaryDirectory() as directory:
                namespace.update(
                    sample_size=5_000, burnin=100, beta_pairs=[(2.0, 5.0), (5.0, 2.0)],
                    PDF_DIR=Path(directory), IMAGE_DIR=Path(directory), simulate_beta_ar=stale,
                )
                with patch.object(ar_functions, "simulate_beta_ar", stale), patch.object(
                    namespace["plt"], "show",
                ):
                    exec(sources["plot-clouds"], namespace)
                self.assertTrue((Path(directory) / "beta_sliding_window_clouds_2d.pdf").is_file())
                self.assertTrue((Path(directory) / "beta_sliding_window_clouds_2d.png").is_file())
        stale.assert_not_called()
        expected_bounds = [beta_innovation_bounds(2, 5), beta_innovation_bounds(5, 2)]
        np.testing.assert_allclose(namespace["panel_innovation_bounds"], expected_bounds)
        self.assertFalse(np.allclose(*namespace["zonoid_boundaries"]))
        for cloud, (lower, upper) in zip(namespace["clouds"], expected_bounds):
            self.assertEqual(cloud.shape, (4_999, 2))
            residuals = cloud[:, 1] - 0.7 * cloud[:, 0]
            self.assertTrue(np.all((residuals >= lower) & (residuals <= upper)))
            self.assertAlmostEqual(residuals.mean(), 0, delta=0.025)
            self.assertAlmostEqual(residuals.var(), 1 / 3, delta=0.02)
        self.assertIn("variance 1/3", namespace["fig"]._suptitle.get_text())

    def test_ar1_notebook_preserves_its_original_uniform_draws(self):
        notebook = json.loads((AR_DIRECTORY / "ar1_sliding_window_embedding.ipynb").read_text())
        source = next("".join(c["source"]) for c in notebook["cells"]
                      if c["id"] == "simulate-ar1")
        namespace = {"np": np, "sample_beta_innovations": sample_beta_innovations,
                     "beta_innovation_bounds": beta_innovation_bounds}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(source, namespace)
        expected = np.random.default_rng(namespace["seed"]).uniform(
            -1.0, 1.0, size=namespace["total_steps"],
        )
        np.testing.assert_array_equal(namespace["innovations"], expected)


if __name__ == "__main__":
    unittest.main()
