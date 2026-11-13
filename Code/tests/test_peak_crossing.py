"""Reproduce the fixed-sample switch discussed in the thesis."""

from pathlib import Path
import tempfile
import unittest

from experiments.spectral_lambda_jump_study.repeated_peak_switch_analysis import (
    figure_3_3_crossing, run_analysis,
)


class PeakCrossingTests(unittest.TestCase):
    def test_thesis_crossing_can_be_computed_without_a_checkpoint(self):
        event = figure_3_3_crossing()
        self.assertEqual(event['status'], 'confirmed')
        self.assertEqual(event['peak_count_at_crossing'], 2)
        self.assertAlmostEqual(event['lambda_crossing'], 17.8737506447, places=7)

    def test_missing_sample_checkpoint_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, '--reference-only'):
                run_analysis(source_checkpoint=Path(directory)/'missing.npz')


if __name__ == '__main__':
    unittest.main()
