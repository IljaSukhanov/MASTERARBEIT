"""Check reruns remain independent of removed historical result artifacts."""
import csv
import hashlib
import json
from pathlib import Path
import runpy
import tempfile
import unittest

import numpy as np

from kde_tools.experiment_inputs import load_reference_sample, paired_gaussian_sample_bank, sampling_inputs


ROOT = Path(__file__).resolve().parents[1]


class ExperimentInputTests(unittest.TestCase):
    def test_fixed_input_identity(self):
        samples, metadata = load_reference_sample()
        self.assertEqual(len(samples), 200)
        self.assertEqual(hashlib.sha256(samples.tobytes()).hexdigest(), metadata['sample_sha256'])

    def test_reduced_bank_preserves_original_row_prefixes(self):
        full = paired_gaussian_sample_bank()
        self.assertEqual(hashlib.sha256(full.tobytes()).hexdigest(),
                         sampling_inputs()['paired_sample_bank']['sample_bank_sha256'])
        np.testing.assert_array_equal(paired_gaussian_sample_bank(3, 25), full[:3, :25])
        np.testing.assert_array_equal(paired_gaussian_sample_bank(3, 1100), full[:3, :1100])



if __name__ == '__main__':
    unittest.main()
