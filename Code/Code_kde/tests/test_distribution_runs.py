"""Numerical and checkpoint checks for the consolidated histogram profiles."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from kde_tools.density import exact_oracle_ise_values
from kde_tools.distribution_runs import run_beta2_distribution
from kde_tools.fourier import beta2_divergence_matrix_fourier
from kde_tools.perron import perron_eigenpair_from_divergence
from kde_tools.qhat import gaussian_qhat
from kde_tools.similarities import beta2_divergence_matrix


class DistributionRunTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.directory = Path(self.scratch.name)
        self.grid = np.geomspace(0.08, 1.2, 11)
        self.options = dict(n=18, repetitions=3, lambda_value=1.0, seed=20260828,
                            data_root=self.directory, workers=1, spatial_points=4096,
                            validate_fourier=True)

    def run_profile(self, **changes):
        return run_beta2_distribution(self.grid, **(self.options | changes))

    def test_both_profiles_match_independent_selector_calculations(self):
        samples = np.random.default_rng(self.options["seed"]).normal(size=(3, 18))
        for profile, lam in (("figure_3_5", 1.0), ("localized", len(self.grid) ** 1.5)):
            with self.subTest(profile=profile):
                actual = self.run_profile(profile=profile, lambda_value=lam, reuse=False)
                expected = []
                for sample in samples:
                    divergence = beta2_divergence_matrix_fourier(sample, self.grid, 12.0, 4096)
                    spectral = perron_eigenpair_from_divergence(divergence, lam).selected_index
                    ise = int(np.argmin(exact_oracle_ise_values(sample, self.grid, 1.0)))
                    qhat = int(np.argmin(gaussian_qhat(sample, self.grid)))
                    expected.append((spectral, ise, qhat))
                np.testing.assert_array_equal(actual["selected_indices"], expected)
                np.testing.assert_array_equal(actual["counts"].sum(axis=1), [3, 3, 3])
                self.assertTrue(np.all(actual["solver_diagnostics"][:, 0] < 1e-10))
                self.assertTrue(np.all(actual["solver_diagnostics"][:, 1] > 0))
                for filename in ("checkpoint.npz", "settings.json", "validation.json", "selections.csv",
                                 "frequencies.csv", "solver_diagnostics.csv", "summary.csv", "frequencies_bins5.csv"):
                    self.assertTrue((actual["run_directory"] / filename).is_file())

    def test_completed_checkpoint_resumes_without_computing_eigenpairs(self):
        original = self.run_profile()
        with patch("kde_tools.distribution_runs.perron_eigenpair_from_divergence", side_effect=AssertionError("Unexpected solve")):
            resumed = self.run_profile()
        np.testing.assert_array_equal(resumed["selected_indices"], original["selected_indices"])
        np.testing.assert_array_equal(resumed["solver_diagnostics"], original["solver_diagnostics"])
        self.assertEqual(resumed["validation"]["reused_repetitions"], 3)

    def test_localized_profile_reuses_exact_controls_from_matching_baseline(self):
        baseline = self.run_profile()
        with patch("kde_tools.distribution_runs.exact_oracle_ise_values", side_effect=AssertionError("Repeated ISE")), \
             patch("kde_tools.distribution_runs.exact_beta2_qhat_values", side_effect=AssertionError("Repeated Q-hat")):
            localized = self.run_profile(profile="localized", lambda_value=len(self.grid) ** 1.5)
        np.testing.assert_array_equal(localized["selected_indices"][:, 1:], baseline["selected_indices"][:, 1:])
        self.assertEqual(localized["validation"]["reused_control_source"], str(baseline["run_directory"]))
        self.assertNotEqual(localized["run_directory"], baseline["run_directory"])

    def test_compatible_prefix_import_preserves_the_source_checkpoint(self):
        prefix = self.run_profile(profile="localized", lambda_value=len(self.grid) ** 1.5, repetitions=2)
        path = prefix["run_directory"] / "checkpoint.npz"
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        with patch("kde_tools.distribution_runs.perron_eigenpair_from_divergence", wraps=perron_eigenpair_from_divergence) as solve:
            extended = self.run_profile(profile="localized", lambda_value=len(self.grid) ** 1.5,
                                        import_directories=[prefix["run_directory"]])
        self.assertEqual(solve.call_count, 1)
        self.assertEqual(extended["validation"]["reused_repetitions"], 2)
        np.testing.assert_array_equal(extended["selected_indices"][:2], prefix["selected_indices"])
        np.testing.assert_array_equal(extended["solver_diagnostics"][:2], prefix["solver_diagnostics"])
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_legacy_solver_checkpoint_is_skipped_without_relabeling_it(self):
        prefix = self.run_profile(profile="localized", lambda_value=len(self.grid) ** 1.5, repetitions=2)
        path = prefix["run_directory"] / "checkpoint.npz"
        with np.load(path, allow_pickle=False) as saved:
            values = {name: saved[name].copy() for name in saved.files}
        settings = json.loads(str(values["settings_json"].item()))
        settings["perron_solver"] = "stabilized_lanczos_v1"
        values["settings_json"] = json.dumps(settings, sort_keys=True)
        np.savez_compressed(path, **values)
        before = path.read_bytes()
        actual = self.run_profile(profile="localized", lambda_value=len(self.grid) ** 1.5,
                                  import_directories=[prefix["run_directory"]])
        self.assertEqual(actual["validation"]["reused_repetitions"], 0)
        self.assertEqual(actual["validation"]["checkpoint_imports"][0]["status"], "skipped")
        self.assertEqual(path.read_bytes(), before)

    def test_invalid_completed_indices_in_own_checkpoint_are_rejected(self):
        original = self.run_profile()
        path = original["run_directory"] / "checkpoint.npz"
        with np.load(path, allow_pickle=False) as saved:
            values = {name: saved[name].copy() for name in saved.files}
        values["selected_indices"][0, 0] = len(self.grid)
        np.savez_compressed(path, **values)
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            self.run_profile()

    def test_failed_run_saves_a_resumable_checkpoint(self):
        with patch("kde_tools.distribution_runs.perron_eigenpair_from_divergence", side_effect=ArithmeticError("Deliberate failure")):
            with self.assertRaisesRegex(ArithmeticError, "Deliberate failure"):
                self.run_profile()
        path, = self.directory.glob("figure_3_5/*/checkpoint.npz")
        with np.load(path, allow_pickle=False) as saved:
            self.assertFalse(np.any(saved["completed"]))
        resumed = self.run_profile()
        self.assertEqual(resumed["validation"]["completed_repetitions"], 3)

    def test_exact_gaussian_profile_preserves_independent_seeds_and_ise_values(self):
        actual = self.run_profile(profile="exact_gaussian", seed=910000, sigma2=1.7)
        for r in range(3):
            sample = np.random.default_rng(910000 + r).normal(0, np.sqrt(1.7), size=18)
            errors = exact_oracle_ise_values(sample, self.grid, 1.7)
            spectral = perron_eigenpair_from_divergence(beta2_divergence_matrix(sample, self.grid), 1).selected_index
            expected = [spectral, int(np.argmin(errors))]
            np.testing.assert_array_equal(actual["selected_indices"][r], expected)
            np.testing.assert_array_equal(actual["selected_ise_values"][r], errors[expected])
        self.assertEqual(actual["methods"], ("L2", "ISE"))
        with patch("kde_tools.distribution_runs.beta2_divergence_matrix", side_effect=AssertionError("Repeated exact divergence")):
            resumed = self.run_profile(profile="exact_gaussian", seed=910000, sigma2=1.7)
        np.testing.assert_array_equal(actual["selected_ise_values"], resumed["selected_ise_values"])


if __name__ == "__main__":
    unittest.main()
