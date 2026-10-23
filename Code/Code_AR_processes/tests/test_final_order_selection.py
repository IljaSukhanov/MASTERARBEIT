"""Frozen-input pairing, padded starts, checkpoint reuse, and outcome accounting."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from Code.Code_AR_processes.ar_tools import final_order_selection as study
from Code.Code_AR_processes.ar_tools.functions import sliding_window_point_cloud


def make_dataset(directory):
    directory = Path(directory)
    values = np.random.default_rng(62).normal(size=(4, 64))
    orders = np.array([1, 1, 2, 2])
    source_ids = np.array([13, 27, 56, 61])
    np.savez_compressed(directory / "dataset.npz", series=values,
                        source_sample_ids=source_ids, orders=orders)
    pd.DataFrame({
        "subset_index": np.arange(4), "sample_id": np.arange(1, 5),
        "source_sample_id": source_ids, "order": orders, "shape_id": "uniform",
        "innovation_seed": np.arange(4, dtype=np.uint64),
        "series_sha256": [hashlib.sha256(row.astype("<f8").tobytes()).hexdigest() for row in values],
    }).to_csv(directory / "series_metadata.csv", index=False)
    spec = {"orders": [1, 2], "series_per_order": 2}
    manifest = {
        "specification": spec,
        "dataset_id": hashlib.sha256(study._json(spec).encode()).hexdigest()[:16],
        "series_count": 4, "sample_size": 64,
        "file_sha256": {name: study._digest(directory / name)
                        for name in ["dataset.npz", "series_metadata.csv"]},
    }
    (directory / "manifest.json").write_text(study._json(manifest))
    return values


class FinalOrderSelectionTests(unittest.TestCase):
    def configuration(self):
        config = study.default_configuration()
        config.update(K=5, maximum_classical_order=4, iterations=12, patience=6)
        return config

    def test_common_rows_and_single_padded_starts_use_the_shared_optimizer(self):
        values = np.random.default_rng(19).uniform(-1, 1, 64)
        config = self.configuration()
        with patch.object(study, "projected_subgradient_empirical_width",
                          wraps=study.projected_subgradient_empirical_width) as optimizer:
            fit = study.fit_width_curve(values, config, sample_id=1)
        self.assertEqual(optimizer.call_count, 4)
        cloud = sliding_window_point_cloud(values, 5)
        for k, call in enumerate(optimizer.call_args_list, 2):
            np.testing.assert_array_equal(call.args[0], cloud[:, :k])
            np.testing.assert_array_equal(call.kwargs["initial_direction"],
                                          np.r_[fit["directions"][k-2, :k-1], 0])
            self.assertAlmostEqual(call.kwargs["initial_step"] * fit["diameters"][k-1], .1)
            self.assertAlmostEqual(call.kwargs["geometric_decay"], np.exp2(-1 / 100))
            self.assertFalse(call.kwargs["stationarity_check"])
            self.assertEqual(call.kwargs["max_restarts"], 0)
        self.assertTrue(np.all(np.diff(fit["widths"]) <= 1e-12))

    def test_paired_results_are_complete_and_reuse_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "inputs"
            source.mkdir()
            make_dataset(source)
            first = study.run_order_selection(source, root / "outputs", self.configuration(), n_jobs=1, progress=False)
            self.assertEqual(len(first["results"]), 12)
            self.assertEqual(first["results"].groupby("sample_id").method.nunique().tolist(), [3] * 4)
            self.assertTrue(first["results"].status.eq("ok").all())
            self.assertEqual(first["verification"]["width_values"], 20)
            with patch.object(study, "_fit_case", side_effect=AssertionError("Unexpected refit")):
                resumed = study.run_order_selection(source, root / "outputs", self.configuration(), n_jobs=1, progress=False)
            pd.testing.assert_frame_equal(first["results"], resumed["results"])
            self.assertEqual(first["run_id"], resumed["run_id"])
            self.assertTrue((first["directory"] / "table_2_2_style.tex").is_file())

    def test_corrupt_inputs_fail_before_any_fit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_dataset(root)
            path = root / "dataset.npz"
            path.write_bytes(path.read_bytes() + b"changed")
            with patch.object(study, "_fit_case", side_effect=AssertionError("Unexpected fit")):
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    study.run_order_selection(root, root / "outputs", self.configuration(), n_jobs=1, progress=False)

    def test_failed_fits_are_not_counted_as_underselections(self):
        rows = []
        for method in study.METHODS:
            rows.extend([
                {"method": method, "order": 2, "selected_order": 2, "status": "ok"},
                {"method": method, "order": 2, "selected_order": 1, "status": "ok"},
                {"method": method, "order": 2, "selected_order": None, "status": "failed"},
            ])
        with tempfile.TemporaryDirectory() as temporary:
            summary, matrix = study.save_selection_tables(pd.DataFrame(rows), temporary)
        self.assertTrue(summary.correct.eq(1).all())
        self.assertTrue(summary.under.eq(1).all())
        self.assertTrue(summary.failed.eq(1).all())
        self.assertEqual(matrix.iloc[0, 0], "1/3 (1/0); failed=1")


if __name__ == "__main__":
    unittest.main()
