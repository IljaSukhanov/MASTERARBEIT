"""Input preservation, paired estimation, and failure accounting for the final study."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from Code.Code_AR_processes.ar_tools import final_coefficient_estimation as study
from Code.Code_AR_processes.ar_tools.functions import jones_ar_coefficients, simulate_beta_ar


class FinalCoefficientEstimationTests(unittest.TestCase):
    def configuration(self):
        config = study.default_dataset_configuration()
        config.update(models_per_order=2, sample_size=48, burnin=30)
        return config

    def test_saved_inputs_reproduce_seeds_and_pair_coefficients_across_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            data = study.prepare_coefficient_dataset(directory, self.configuration(), progress=False)
            self.assertEqual(data["series"].shape, (50, 48))
            self.assertEqual(data["coefficients"].shape, (10, 5))
            self.assertFalse(data["series"].flags.writeable)
            for model in data["models"].itertuples(index=False):
                np.testing.assert_array_equal(data["coefficients"][model.model_index, :model.order],
                                              jones_ar_coefficients(model.order, rng=int(model.coefficient_seed)))
            for row in data["metadata"].itertuples(index=False):
                phi = data["coefficients"][row.model_index, :row.order]
                expected = simulate_beta_ar(phi, 48, burnin=30, alpha=row.alpha, beta=row.beta, rng=int(row.innovation_seed))
                np.testing.assert_array_equal(data["series"][row.sample_id - 1], expected)
            self.assertTrue(data["metadata"].groupby(["order", "model_id"])["shape_id"].nunique().eq(5).all())
            hashes = data["manifest"]["file_sha256"].copy()
            with patch.object(study, "jones_ar_coefficients", side_effect=AssertionError("Unexpected redraw")):
                loaded = study.prepare_coefficient_dataset(directory, self.configuration(), progress=False)
            np.testing.assert_array_equal(loaded["series"], data["series"])
            self.assertEqual(loaded["manifest"]["file_sha256"], hashes)

    def test_corrupted_inputs_are_rejected_without_regeneration(self):
        with tempfile.TemporaryDirectory() as directory:
            data = study.prepare_coefficient_dataset(directory, self.configuration(), progress=False)
            path = data["directory"] / "dataset.npz"
            path.write_bytes(path.read_bytes() + b"corrupted")
            with patch.object(study, "jones_ar_coefficients", side_effect=AssertionError("Unexpected redraw")):
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    study.prepare_coefficient_dataset(directory, self.configuration(), progress=False)

    def test_existing_samples_are_reused_across_numpy_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            original = study.prepare_coefficient_dataset(directory, self.configuration(), progress=False)
            with patch.object(study.np, "__version__", "999.0.0"), patch.object(
                study, "jones_ar_coefficients", side_effect=AssertionError("Unexpected redraw")
            ):
                loaded = study.prepare_coefficient_dataset(directory, self.configuration(), progress=False)
            self.assertEqual(loaded["dataset_id"], original["dataset_id"])
            np.testing.assert_array_equal(loaded["series"], original["series"])

    def test_design_changes_get_a_separate_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.configuration()
            first = study.prepare_coefficient_dataset(directory, config, progress=False)
            config["sample_size"] += 1
            second = study.prepare_coefficient_dataset(directory, config, progress=False)
            self.assertNotEqual(first["directory"], second["directory"])
            self.assertTrue((first["directory"] / "dataset.npz").is_file())

    def test_estimators_use_saved_series_and_the_tuned_single_start(self):
        config = self.configuration()
        config.update(orders=[1, 3], models_per_order=2)
        with tempfile.TemporaryDirectory() as directory:
            data = study.prepare_coefficient_dataset(Path(directory) / "inputs", config, progress=False)
            settings = study.default_fit_configuration()
            settings.update(iterations=12, patience=6)
            with patch.object(study, "projected_subgradient_empirical_width", wraps=study.projected_subgradient_empirical_width) as optimizer:
                run = study.run_coefficient_estimation(data, Path(directory) / "outputs", settings, n_jobs=1, progress=False)
            results = run["results"]
            self.assertEqual(len(results), 80)
            self.assertTrue(results["status"].eq("ok").all(), results["error_message"].tolist())
            self.assertEqual(optimizer.call_count, 20)
            for call in optimizer.call_args_list:
                points = call.args[0]
                kwargs = call.kwargs
                expected = np.zeros(points.shape[1])
                expected[0] = 1
                np.testing.assert_array_equal(kwargs["initial_direction"], expected)
                self.assertAlmostEqual(kwargs["initial_step"] * study.pdist(points).max(), .1)
                self.assertAlmostEqual(kwargs["geometric_decay"], np.exp2(-1 / 100))
                self.assertFalse(kwargs["stationarity_check"])
                self.assertEqual(kwargs["max_restarts"], 0)
            for row in results.itertuples(index=False):
                phi = data["coefficients"][row.model_index, :row.order]
                estimate = np.array([getattr(row, f"estimated_phi_{j+1}") for j in range(row.order)])
                self.assertAlmostEqual(row.coefficient_mae, np.abs(phi - estimate).mean())
            with patch.object(study, "_fit_case", side_effect=AssertionError("Unexpected refit")):
                resumed = study.run_coefficient_estimation(data, Path(directory) / "outputs", settings, n_jobs=1, progress=False)
            pd.testing.assert_frame_equal(results, resumed["results"])
            summary, paired = study.summarize_coefficient_estimation(run, bootstrap_count=50)
            self.assertEqual(len(summary), 40)
            self.assertEqual(len(paired), 30)
            for row in paired.itertuples(index=False):
                sample = results.loc[results["order"].eq(row.order) & results["shape_id"].eq(row.shape_id)]
                matrix = sample.pivot(index="model_id", columns="method", values="coefficient_mae")
                self.assertAlmostEqual(row.mean_mae_difference, (matrix["Minimum width"] - matrix[row.comparator]).mean())

    def test_failures_are_retained_and_make_unconditional_scores_unavailable(self):
        config = self.configuration()
        config.update(orders=[1], models_per_order=2, beta_shapes=[config["beta_shapes"][0]])
        with tempfile.TemporaryDirectory() as directory:
            data = study.prepare_coefficient_dataset(Path(directory) / "inputs", config, progress=False)
            settings = study.default_fit_configuration()
            settings.update(iterations=6, patience=3)
            with patch.object(study, "horvath_linf_coefficients", side_effect=RuntimeError("deliberate failure")):
                run = study.run_coefficient_estimation(data, Path(directory) / "outputs", settings, n_jobs=1, progress=False)
            self.assertEqual(len(run["results"]), 8)
            summary, paired = study.summarize_coefficient_estimation(run, bootstrap_count=20)
            failed = summary.loc[summary["method"].eq("Horvath L-infinity")].iloc[0]
            self.assertEqual(failed.failed_fits, 2)
            self.assertTrue(np.isnan(failed.mean_mae))
            self.assertTrue(np.isnan(failed.mean_ci_lower))
            self.assertTrue(np.isnan(paired.loc[paired["comparator"].eq("Horvath L-infinity"), "mean_mae_difference"].iloc[0]))
            self.assertEqual(len(pd.read_csv(run["directory"] / "failed_fits.csv")), 2)


if __name__ == "__main__":
    unittest.main()
