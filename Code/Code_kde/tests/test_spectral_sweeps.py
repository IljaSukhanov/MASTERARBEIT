"""Numerical equivalence and cross-notebook cache checks for spectral sweeps."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from kde_tools.fourier import (
    DEFAULT_FOURIER_HALF_WIDTH, DEFAULT_FOURIER_SPATIAL_POINTS,
    FOURIER_L2_METHOD_VERSION, beta2_divergence_matrix_fourier,
)
from kde_tools.perron import PERRON_SOLVER_VERSION, perron_eigenpair_from_divergence
from kde_tools.spectral_sweeps import run_l2_point


class SpectralSweepTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.grid = np.geomspace(0.08, 1.2, 9)
        self.settings = {"repetitions": 3, "maximum_sample_size": 22,
                         "fft_half_width": 12.0, "fft_spatial_points": 4096,
                         "seed": 20260830, "sigma2": 1.0}
        self.samples = np.random.default_rng(20260830).normal(size=(3, 22))

    def run_point(self, output="main", lambdas=(1.0, 1000.0), **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return run_l2_point(14, self.samples, self.grid, self.settings, lambdas,
                                self.root / output, cache_root=self.root / "shared", **kwargs)

    def reference(self, lambdas=(1.0, 1000.0)):
        return np.array([
            [perron_eigenpair_from_divergence(
                beta2_divergence_matrix_fourier(sample[:14], self.grid, 12.0, 4096), lam
            ).selected_index for lam in lambdas]
            for sample in self.samples
        ])

    def test_numerical_choices_and_histograms_match_independent_solves(self):
        actual = self.run_point()
        np.testing.assert_array_equal(actual["selected_indices"], self.reference())
        np.testing.assert_array_equal(actual["counts"].sum(axis=1), [3, 3])
        np.testing.assert_allclose(actual["frequencies"].sum(axis=1), [1, 1])
        self.assertEqual(actual["divergence_evaluations"], 3)

    def test_other_notebook_reuses_choices_without_recomputing_divergences(self):
        first = self.run_point()
        self.settings = self.settings | {"grid_size": len(self.grid), "report_title": "Another notebook"}
        with patch("kde_tools.spectral_sweeps.beta2_divergence_matrix_fourier", side_effect=AssertionError("Duplicate divergence")):
            second = self.run_point(output="fast")
        np.testing.assert_array_equal(second["selected_indices"], first["selected_indices"])
        self.assertEqual(first["cache_directory"], second["cache_directory"])
        self.assertEqual(second["reused_selections"], 6)

    def test_adding_lambda_reuses_completed_columns_and_preserves_order(self):
        first = self.run_point(lambdas=(1.0,))
        extended = self.run_point(output="more", lambdas=(1000.0, 1.0))
        np.testing.assert_array_equal(extended["selected_indices"][:, 1], first["selected_indices"][:, 0])
        self.assertEqual(extended["reused_selections"], 3)
        self.assertEqual(extended["divergence_evaluations"], 3)
        with patch("kde_tools.spectral_sweeps.beta2_divergence_matrix_fourier", side_effect=AssertionError("Duplicate subset")):
            subset = self.run_point(output="subset", lambdas=(1000.0,))
        np.testing.assert_array_equal(subset["selected_indices"][:, 0], extended["selected_indices"][:, 0])

    def test_actual_sample_changes_do_not_reuse_the_previous_cache(self):
        first = self.run_point()
        self.samples[1, 0] += 0.1
        second = self.run_point(output="changed")
        self.assertNotEqual(first["cache_directory"], second["cache_directory"])
        self.assertEqual(second["reused_selections"], 0)

    def test_changed_frequency_grid_does_not_import_or_overwrite_old_results(self):
        previous = self.run_point(output="old_grid")
        checkpoint = previous["point_directory"] / "checkpoint.npz"
        original_bytes = checkpoint.read_bytes()
        self.settings = self.settings | {
            "fft_half_width": DEFAULT_FOURIER_HALF_WIDTH,
            "fft_spatial_points": DEFAULT_FOURIER_SPATIAL_POINTS,
        }
        current = self.run_point(
            output="current_grid", import_directories=[self.root / "old_grid"]
        )
        self.assertNotEqual(previous["cache_directory"], current["cache_directory"])
        self.assertEqual(current["reused_selections"], 0)
        self.assertEqual(current["divergence_evaluations"], len(self.samples))
        self.assertEqual(len(current["skipped_imports"]), 1)
        self.assertEqual(checkpoint.read_bytes(), original_bytes)

    def test_compatible_legacy_checkpoint_import_is_read_only(self):
        directory = self.root / "legacy/n14"
        directory.mkdir(parents=True)
        config = {"settings": self.settings, "sample_size": 14, "lambda_values": [1.0, 1000.0],
                  "implementation_version": 2, "l2_method": FOURIER_L2_METHOD_VERSION,
                  "perron_solver": PERRON_SOLVER_VERSION}
        path = directory / "checkpoint.npz"
        expected = self.reference()
        np.savez_compressed(path, configuration_json=json.dumps(config, sort_keys=True),
                            h_grid=self.grid, selected_indices=expected, completed=3)
        before = path.read_bytes()
        with patch("kde_tools.spectral_sweeps.beta2_divergence_matrix_fourier", side_effect=AssertionError("Legacy results recomputed")):
            imported = self.run_point(import_directories=[self.root / "legacy"])
        np.testing.assert_array_equal(imported["selected_indices"], expected)
        self.assertEqual(imported["reused_selections"], 6)
        self.assertEqual(path.read_bytes(), before)

    def test_wrong_method_versions_in_local_checkpoint_are_rejected(self):
        result = self.run_point()
        path = result["point_directory"] / "checkpoint.npz"
        with np.load(path, allow_pickle=False) as saved:
            original = {key: saved[key].copy() for key in saved.files}
        for field in ("perron_solver", "l2_method"):
            payload = dict(original)
            config = json.loads(str(payload["configuration_json"].item()))
            config[field] = "legacy"
            payload["configuration_json"] = json.dumps(config, sort_keys=True)
            np.savez_compressed(path, **payload)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.run_point()

    def test_failure_keeps_completed_lambda_choices_for_resume(self):
        calls = 0
        def fail_once(divergence, lam):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ArithmeticError("Deliberate interruption")
            return perron_eigenpair_from_divergence(divergence, lam)
        with patch("kde_tools.spectral_sweeps.perron_eigenpair_from_divergence", side_effect=fail_once):
            with self.assertRaisesRegex(ArithmeticError, "Deliberate interruption"):
                self.run_point()
        resumed = self.run_point()
        self.assertEqual(resumed["reused_selections"], 1)
        np.testing.assert_array_equal(resumed["selected_indices"], self.reference())

    def test_invalid_shared_cached_index_is_rejected(self):
        result = self.run_point()
        path = result["cache_directory"] / "checkpoint.npz"
        with np.load(path, allow_pickle=False) as saved:
            values = {key: saved[key].copy() for key in saved.files}
        values["selected_indices"][0, 0] = len(self.grid)
        np.savez_compressed(path, **values)
        with self.assertRaisesRegex(ValueError, "invalid selections"):
            self.run_point(output="another")

    def test_diagnostics_upgrade_selection_only_cache_then_resume(self):
        first = self.run_point()
        path = first["cache_directory"] / "checkpoint.npz"
        # Earlier shared caches contain selections and completion flags only.
        with np.load(path, allow_pickle=False) as saved:
            old = {key: saved[key].copy() for key in saved.files if key != "solver_diagnostics"}
        np.savez_compressed(path, **old)
        upgraded = self.run_point(output="joint", diagnostics=True)
        np.testing.assert_array_equal(first["selected_indices"], upgraded["selected_indices"])
        self.assertEqual(upgraded["divergence_evaluations"], 3)
        self.assertTrue(np.all(upgraded["solver_diagnostics"][:, :, 0] < 1e-10))
        self.assertTrue(np.all(upgraded["solver_diagnostics"][:, :, 1] > 0))
        self.assertTrue(np.all(upgraded["solver_diagnostics"][:, :, 2] == 0))
        with patch("kde_tools.spectral_sweeps.beta2_divergence_matrix_fourier", side_effect=AssertionError("Repeated diagnostics")):
            resumed = self.run_point(output="joint", diagnostics=True)
        np.testing.assert_array_equal(resumed["solver_diagnostics"], upgraded["solver_diagnostics"])

    def test_new_selection_runs_save_diagnostics_for_other_notebooks(self):
        first = self.run_point()
        with patch("kde_tools.spectral_sweeps.beta2_divergence_matrix_fourier", side_effect=AssertionError("Repeated fresh diagnostics")):
            other = self.run_point(output="joint", diagnostics=True)
        np.testing.assert_array_equal(first["solver_diagnostics"], other["solver_diagnostics"])

    def test_diagnostics_match_independent_eigenpair_results(self):
        actual = self.run_point(diagnostics=True)
        for r, sample in enumerate(self.samples):
            divergence = beta2_divergence_matrix_fourier(sample[:14], self.grid, 12.0, 4096)
            for c, lam in enumerate((1.0, 1000.0)):
                expected = perron_eigenpair_from_divergence(divergence, lam)
                np.testing.assert_array_equal(actual["solver_diagnostics"][r, c],
                                              [expected.relative_residual, expected.relative_gap, expected.matvecs])


if __name__ == "__main__":
    unittest.main()
