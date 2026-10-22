"""Archival checks: numeric fidelity, paired inputs and preservation of old runs."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from Code.Code_AR_processes.ar_tools.functions import simulate_uniform_ar, sliding_window_point_cloud
from Code.Code_AR_processes.ar_tools.tuning_exports import TuningTextArchive


class TuningExportTests(unittest.TestCase):
    def test_text_tables_round_trip_full_precision_and_uint64_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            seed = 2**64 - 1
            archive = TuningTextArchive(directory, {"seed": seed, "delta": 1e-8})
            values = np.array([np.nextafter(1., 2.), 1e-300, np.nan, np.inf])
            table = pd.DataFrame({"process_seed": [seed] * 4, "minimum_width": values})
            path = archive.save_table("individual_results", table)
            archive.checkpoint("Round-trip check complete.", complete=True)
            restored = pd.read_csv(path, sep="\t", float_precision="round_trip",
                                   dtype={"process_seed": "uint64"})
            np.testing.assert_array_equal(restored["minimum_width"], values)
            self.assertEqual(restored["process_seed"].tolist(), [seed] * 4)
            self.assertEqual(json.loads((archive.directory / "settings.txt").read_text())["seed"], seed)
            self.assertTrue(json.loads((archive.directory / "status.txt").read_text())["complete"])
            manifest = pd.read_csv(archive.directory / "manifest.txt", sep="\t")
            for row in manifest.itertuples(index=False):
                exported = archive.directory / row.path
                self.assertEqual(row.sha256, hashlib.sha256(exported.read_bytes()).hexdigest())
            self.assertEqual({p.suffix for p in archive.directory.rglob("*") if p.is_file()}, {".txt"})

    def test_new_archives_preserve_previous_runs_and_failed_writes_are_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            first = TuningTextArchive(directory, {"run": 1})
            second = TuningTextArchive(directory, {"run": 1})
            self.assertNotEqual(first.directory, second.directory)
            path = first.save_table("results", [{"width": 1.25}])
            previous = path.read_bytes()
            with patch.object(pd.DataFrame, "to_csv", side_effect=OSError("Disk failure")):
                with self.assertRaisesRegex(OSError, "Disk failure"):
                    first.save_table("results", [{"width": 2.5}])
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(list(first.directory.rglob("*.tmp")), [])
            self.assertFalse((second.directory / "results.txt").exists())
            with self.assertRaises(ValueError):
                first.save_text("../outside", "must not be written")

    def test_exported_inputs_reproduce_the_paired_series_and_cloud(self):
        phi = np.array([.2, -.1, .05])
        seed, burnin, sample_size = 2**63 + 123, 12, 24
        series = simulate_uniform_ar(phi, sample_size=sample_size, burnin=burnin,
                                     half_width=1., rng=seed)
        cloud = sliding_window_point_cloud(series, 4)
        true_direction = np.r_[1., -phi]
        true_direction /= np.linalg.norm(true_direction)
        case = {
            "replication": 1, "T": sample_size, "process_seed": seed,
            **{f"phi_{j+1}": value for j, value in enumerate(phi)},
            "empirical_true_direction_width": np.ptp(cloud @ true_direction),
        }
        with tempfile.TemporaryDirectory() as directory:
            archive = TuningTextArchive(directory, {})
            archive.save_inputs("process_001", [case], burnin=burnin)
            series_path = archive.directory / "inputs/process_001/time_series.txt"
            cloud_path = archive.directory / "inputs/process_001/point_clouds.txt"
            restored_series = pd.read_csv(series_path, sep="\t", float_precision="round_trip")
            restored_cloud = pd.read_csv(cloud_path, sep="\t", float_precision="round_trip")
            np.testing.assert_array_equal(restored_series["value"], series)
            np.testing.assert_array_equal(restored_cloud[[f"x_{j}" for j in range(4)]], cloud)
            self.assertEqual(restored_series["process_seed"].tolist(), [seed] * sample_size)
            with patch("Code.Code_AR_processes.ar_tools.tuning_exports.simulate_uniform_ar",
                       side_effect=AssertionError("Paired inputs must only be exported once.")):
                archive.save_inputs("process_001", [{**case, "initial_u_0": 1.}], burnin=burnin)
            with self.assertRaisesRegex(ValueError, "different paths"):
                archive.save_inputs("process_001", [{**case, "process_seed": seed + 1}], burnin=burnin)


if __name__ == "__main__":
    unittest.main()
