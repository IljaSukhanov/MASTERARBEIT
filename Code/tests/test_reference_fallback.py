"""Checks for incomplete saved reference runs."""

# Permit running individual tests and unittest discovery from the project root.
import sys as _sys
from pathlib import Path as _Path
_code_root = str(_Path(__file__).resolve().parents[1])
if _code_root not in _sys.path:
    _sys.path.insert(0, _code_root)

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.qhat0_interval_example import brent_refinements_spectral_alpha15 as experiment


class ReferenceFallbackTests(unittest.TestCase):
    def setUp(self):
        self.sample = np.array([-1.0, 0.0, 0.4, 1.0])
        self.reference = {'diagnostics': {'resolution_stable': True}, 'global_minimum': {'h': 0.4}}

    def test_missing_checkpoint_runs_exact_reference_search(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(experiment, 'REFERENCE_DIR', Path(directory)), \
             patch.object(experiment, 'minimize_qhat', return_value=self.reference) as search:
            self.assertIs(experiment.load_reference(12, self.sample), self.reference)
            args, kwargs = search.call_args
            np.testing.assert_array_equal(args[0], self.sample)
            self.assertEqual(args[1:], (.01, 2.))
            self.assertEqual(kwargs['config'].scan_backend, 'exact')

    def test_recomputed_reference_must_be_resolution_stable(self):
        unstable = {'diagnostics': {'resolution_stable': False}}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(experiment, 'REFERENCE_DIR', Path(directory)), \
             patch.object(experiment, 'minimize_qhat', return_value=unstable):
            with self.assertRaisesRegex(ValueError, 'not stable'):
                experiment.load_reference(12, self.sample)

    def test_existing_checkpoint_rejects_a_mismatched_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'sample_000.json'
            path.write_text(json.dumps({'sample_sha256': hashlib.sha256(b'other').hexdigest(),
                                        'reference': self.reference}))
            with patch.object(experiment, 'REFERENCE_DIR', Path(directory)):
                with self.assertRaisesRegex(ValueError, 'sample mismatch'):
                    experiment.load_reference(0, self.sample)


if __name__ == '__main__':
    unittest.main()
