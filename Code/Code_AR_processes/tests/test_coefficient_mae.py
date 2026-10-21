"""MAE normalization, failed fits, and protection against legacy score caches."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from Code.Code_AR_processes.ar_tools.coefficient_errors import coefficient_mae
from Code.Code_AR_processes.ar_tools.experiments import (
    run_ar3_coefficient_mae_experiment, run_mae_replication,
)


class CoefficientMAETests(unittest.TestCase):
    def test_normalizes_by_order_instead_of_summing_or_squaring(self):
        self.assertEqual(coefficient_mae([2., -1., 9.], [1., 1., 3.]), 3.)
        for order in (1, 2, 3, 5):
            with self.subTest(order=order):
                self.assertEqual(coefficient_mae(np.full(order, .5), np.zeros(order)), .5)

    def test_broadcasts_repeated_fits_and_averages_absolute_errors(self):
        actual = coefficient_mae([[2., 0., 5.], [0., -2., 1.]], [1., 0., 2.])
        np.testing.assert_allclose(actual, [4 / 3, 4 / 3])

    def test_nonfinite_fit_is_unavailable_without_affecting_other_rows(self):
        actual = coefficient_mae([[1., 2.], [np.nan, 2.], [np.inf, 3.]], [0., 0.])
        self.assertEqual(actual[0], 1.5)
        self.assertTrue(np.isnan(actual[1:]).all())
        self.assertTrue(np.isnan(np.mean(actual)))

    def test_rejects_empty_scalar_and_different_order_inputs(self):
        for estimate, truth in (([], []), (1., 0.), ([1., 2.], [0.])):
            with self.subTest(estimate=estimate), self.assertRaises(ValueError):
                coefficient_mae(estimate, truth)

    def test_worker_refuses_legacy_cache_instead_of_relabelling_it(self):
        with tempfile.TemporaryDirectory() as directory:
            pd.DataFrame([{"replication": 1, "T": 40, "method": "Yule-Walker",
                           "squared_coefficient_error": .25}]).to_csv(
                Path(directory) / "replication_0001.csv", index=False,
            )
            with self.assertRaisesRegex(RuntimeError, "cache"):
                run_mae_replication(1, {"T_values": [40], "methods": ["Yule-Walker"]}, directory)

    def test_unconditional_experiment_mean_retains_failed_fit(self):
        config = {"sample_sizes": [20], "replications": 2, "experiment_seed": 51,
                  "burnin": 4}
        with tempfile.TemporaryDirectory() as directory, \
                patch("Code.Code_AR_processes.ar_tools.experiments.jones_ar_coefficients",
                      return_value=np.array([.4, -.1, .1])), \
                patch("Code.Code_AR_processes.ar_tools.experiments.fit_ar3_mae_prefix",
                      side_effect=[{"estimate": np.array([.6, -.3, .5])}, ValueError("failed fit")]), \
                contextlib.redirect_stdout(io.StringIO()):
            run_ar3_coefficient_mae_experiment(configuration=config, output_directory=directory, n_jobs=1)
            results = pd.read_csv(Path(directory) / "replication_results.csv")
            summary = pd.read_csv(Path(directory) / "mae_summary.csv")
        self.assertAlmostEqual(results.coefficient_mae.iloc[0], .8 / 3)
        self.assertTrue(np.isnan(results.coefficient_mae.iloc[1]))
        self.assertTrue(np.isnan(summary.coefficient_mae.iloc[0]))
        self.assertEqual(summary.failed_fits.iloc[0], 1)


if __name__ == "__main__":
    unittest.main()
