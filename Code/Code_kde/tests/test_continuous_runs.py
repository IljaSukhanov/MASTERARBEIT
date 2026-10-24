"""Numerical equivalence, cross-notebook reuse, and legacy provenance checks."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from kde_tools.continuous_minimizers import METHODS, continuous_ise_qr_minima
from kde_tools.continuous_runs import FIELDS, _QHAT_SOURCE, run_continuous_point
from kde_tools.qhat import QHAT_VERSION, continuous_qhat_minimum


class ContinuousRunTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.samples = np.random.default_rng(61).normal(size=(3, 24))
        self.settings = dict(h_min=.03, h_max=1.4, sigma2=1., scan_points=65,
                             half_width=16., spatial_points=4096)

    def run_point(self, methods=METHODS, output="continuous", **kwargs):
        return run_continuous_point(18, self.samples, self.settings, methods, self.root/output,
                                    cache_root=self.root/"shared", workers=1, **kwargs)

    def test_shared_results_agree_with_original_selectors(self):
        actual = self.run_point()
        for r, sample in enumerate(self.samples):
            original = continuous_ise_qr_minima(sample[:18], **self.settings)
            for c, method in enumerate(METHODS):
                np.testing.assert_allclose(actual["values"][r, c],
                                           [original[method][f] for f in FIELDS], rtol=2e-7, atol=1e-8)

    def test_qhat_view_and_larger_bank_reuse_q0(self):
        first = self.run_point(("Q0",), "fast")
        # An extended n-bank with exactly the same 18-observation prefixes.
        self.samples = np.column_stack([self.samples, np.ones((3, 5))])
        with patch("kde_tools.continuous_runs.continuous_qhat_minimum", side_effect=AssertionError("Repeated Q0")):
            other = self.run_point()
        np.testing.assert_array_equal(first["values"][:, 0], other["values"][:, 1])
        self.assertEqual(other["reused_selections"], 3)
        with patch("kde_tools.continuous_runs.continuous_ise_qr_minima", side_effect=AssertionError("Repeated ISE/Qr")):
            resumed = self.run_point()
        self.assertEqual(resumed["reused_selections"], 12)

    def test_actual_sample_change_uses_a_different_cache(self):
        first = self.run_point(("Q0",))
        self.samples[1, 0] += .1
        other = self.run_point(("Q0",))
        self.assertNotEqual(first["cache_directory"], other["cache_directory"])
        self.assertEqual(other["reused_selections"], 0)

    def legacy_qhat(self):
        path = self.root/"legacy"; path.mkdir()
        config = {"version": QHAT_VERSION, "source_sha256": _QHAT_SOURCE,
                  "sampling_configuration": {"settings": {"seed": 61, "sigma2": 1., "repetitions": 3,
                                                              "maximum_sample_size": 24}},
                  "sample_bank_sha256": hashlib.sha256(self.samples.tobytes()).hexdigest(),
                  **{k: v for k, v in self.settings.items() if k != "sigma2"}}
        fields = ("h", "qhat", *FIELDS[2:])
        values = np.array([[continuous_qhat_minimum(s[:18], **{k: v for k, v in self.settings.items() if k != "sigma2"})[f]
                            for f in fields] for s in self.samples])
        np.savez_compressed(path/"n18.npz", configuration_json=json.dumps(config, sort_keys=True),
                            sample_size=18, fields=np.array(fields), values=values, completed=3)
        return path

    def test_verified_legacy_qhat_import_is_read_only(self):
        path = self.legacy_qhat()
        before = (path/"n18.npz").read_bytes()
        with patch("kde_tools.continuous_runs.continuous_qhat_minimum", side_effect=AssertionError("Repeated legacy Q0")):
            actual = self.run_point(("Q0",), import_directories=[path])
        self.assertEqual(actual["reused_selections"], 3)
        self.assertEqual(actual["imports"][0]["status"], "imported")
        self.assertEqual(before, (path/"n18.npz").read_bytes())

    def test_incompatible_legacy_samples_are_skipped(self):
        path = self.legacy_qhat()
        self.samples[0, 0] += .01
        actual = self.run_point(("Q0",), import_directories=[path])
        self.assertEqual(actual["reused_selections"], 0)
        self.assertEqual(actual["imports"][0]["status"], "skipped")

    def test_interruption_preserves_completed_selections(self):
        real = continuous_qhat_minimum
        calls = 0
        def fail(sample, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ArithmeticError("Deliberate interruption")
            return real(sample, **kwargs)
        with patch("kde_tools.continuous_runs.continuous_qhat_minimum", side_effect=fail):
            with self.assertRaises(ArithmeticError):
                self.run_point(("Q0",))
        self.assertEqual(self.run_point(("Q0",))["reused_selections"], 1)

    def test_corrupt_completed_values_are_rejected(self):
        result = self.run_point(("Q0",))
        path = result["cache_directory"]/"checkpoint.npz"
        with np.load(path, allow_pickle=False) as saved:
            payload = {k: saved[k].copy() for k in saved.files}
        payload["values"][0, 1, 0] = np.nan
        np.savez_compressed(path, **payload)
        with self.assertRaisesRegex(ValueError, "invalid completed"):
            self.run_point(("Q0",))
