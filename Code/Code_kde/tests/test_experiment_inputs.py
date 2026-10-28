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

    def test_ise_reference_computes_and_reuses_fresh_results(self):
        namespace = runpy.run_path(str(ROOT/'outputs/analysis/local_qr_minimizers_vs_n/ise_reference.py'))
        function = namespace['ensure_ise_reference']
        # Isolate outputs so the test cannot create experiment results.
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            function.__globals__.update(DATA_DIR=directory, REFERENCE=directory/'ise_histogram.csv',
                                        VALIDATION=directory/'ise_reference_validation.json')
            result = function(sample_sizes=(20, 40), repetitions=2, workers=1)
            with result.open() as stream:
                records = list(csv.DictReader(stream))
            for n in (20, 40):
                self.assertEqual(sum(int(r['count']) for r in records if int(r['n']) == n), 2)
            reference_time = result.stat().st_mtime_ns
            function(sample_sizes=(20, 40), repetitions=2, workers=1)
            self.assertEqual(result.stat().st_mtime_ns, reference_time)
            validation = json.loads((directory/'ise_reference_validation.json').read_text())
            self.assertNotIn('sources', validation)
            self.assertEqual(len(list(directory.rglob('n*_energy_analysis.json'))), 2)


if __name__ == '__main__':
    unittest.main()
