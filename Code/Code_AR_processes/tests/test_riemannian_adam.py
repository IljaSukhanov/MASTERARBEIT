"""Tests for the sphere-aware Adam minimum-width optimizer."""

import unittest

import numpy as np

from Code.Code_AR_processes.functions import (
    batched_riemannian_adam_width_curves,
    sliding_window_point_cloud,
)


class RiemannianAdamTests(unittest.TestCase):
    def test_returns_reproducible_unit_directions(self) -> None:
        series = np.sin(np.linspace(0.0, 8.0, 80))
        arguments = {
            "series": series,
            "k_values": [2, 3],
            "seed": 17,
            "replication": 4,
            "starts": 3,
            "iterations": 40,
            "initial_step": 0.1,
            "geometric_decay": 0.95,
            "return_directions": True,
        }

        first = batched_riemannian_adam_width_curves(**arguments)
        second = batched_riemannian_adam_width_curves(**arguments)

        np.testing.assert_allclose(first[0], second[0])
        np.testing.assert_allclose(first[1], second[1])
        for direction in first[3]:
            self.assertAlmostEqual(np.linalg.norm(direction), 1.0)

    def test_best_width_does_not_exceed_first_start_initial_width(self) -> None:
        series = np.cos(np.linspace(0.0, 6.0, 60))
        cloud = sliding_window_point_cloud(series, 3)
        _, eigenvectors = np.linalg.eigh(np.cov(cloud, rowvar=False))
        initial_direction = eigenvectors[:, 0]
        initial_width = np.ptp(cloud @ initial_direction)

        single_widths, best_widths, _ = (
            batched_riemannian_adam_width_curves(
                series,
                [3],
                seed=5,
                replication=2,
                starts=2,
                iterations=20,
                initial_step=0.1,
            )
        )

        self.assertLessEqual(single_widths[0], initial_width + 1e-12)
        self.assertLessEqual(best_widths[0], single_widths[0] + 1e-12)

    def test_rejects_invalid_adam_parameters(self) -> None:
        series = np.arange(10.0)
        with self.assertRaisesRegex(ValueError, "beta_1"):
            batched_riemannian_adam_width_curves(
                series,
                [2],
                seed=1,
                replication=1,
                beta_1=1.0,
            )


if __name__ == "__main__":
    unittest.main()
