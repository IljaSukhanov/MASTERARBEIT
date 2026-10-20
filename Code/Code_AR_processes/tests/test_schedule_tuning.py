"""Checks for paired geometric-schedule tuning on one fixed cloud."""

import contextlib
import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from Code.Code_AR_processes.ar_tools.descent import projected_subgradient_empirical_width
from Code.Code_AR_processes.ar_tools.tuning import (
    compare_ar_schedules, tune_ar_sample_sizes,
    summarize_configuration_wins, summarize_geometric_replications, tune_fixed_ar_realizations,
    tune_geometric_half_life_schedules, tune_geometric_width_schedule,
    tune_jones_ar_realizations,
)
from Code.Code_AR_processes.ar_tools.functions import (
    is_stationary, jones_ar_coefficients, simulate_uniform_ar, sliding_window_point_cloud,
)


class ScheduleTuningTests(unittest.TestCase):
    def setUp(self):
        self.cloud = np.array([[-2.0, -0.1], [2.0, -0.1], [2.0, 0.1], [-2.0, 0.1]])
        self.directions = np.array([[1.0, 1.0], [-0.4, 1.0]])
        self.settings = dict(iterations=30, patience=8, improvement_tolerance=1e-10,
                             optimizer_seed=17)

    def test_singleton_grid_reproduces_direct_descents(self):
        result = tune_geometric_width_schedule(
            self.cloud, self.directions, [0.03], [0.97], **self.settings,
        )
        direct = [
            projected_subgradient_empirical_width(
                self.cloud, initial_direction=direction,
                initial_step=0.03, geometric_decay=0.97, step_schedule="geometric",
                rng=np.random.SeedSequence([17, 0, 2, 21, index]),
                iterations=30, patience=8, improvement_tolerance=1e-10,
            )
            for index, direction in enumerate(self.directions)
        ]
        for actual, expected in zip(result["best_results"], direct):
            np.testing.assert_array_equal(actual["width_history"], expected["width_history"])
            np.testing.assert_array_equal(actual["direction"], expected["direction"])
        self.assertEqual(result["selected"]["minimum_width"], min(r["minimum_width"] for r in direct))

    def test_grid_selection_is_order_independent_and_preserves_inputs(self):
        original = self.directions.copy()
        forward = tune_geometric_width_schedule(
            self.cloud, self.directions, [0.01, 0.05], [0.9, 0.99], **self.settings,
        )
        reverse = tune_geometric_width_schedule(
            self.cloud, self.directions, [0.05, 0.01], [0.99, 0.9], **self.settings,
        )
        self.assertEqual(len(forward["candidates"]), 4)
        self.assertEqual(len(forward["per_start"]), 8)
        self.assertEqual(forward["selected"]["gamma0"], reverse["selected"]["gamma0"])
        self.assertEqual(forward["selected"]["beta"], reverse["selected"]["beta"])
        np.testing.assert_array_equal(self.directions, original)
        np.testing.assert_array_equal(forward["direction"], reverse["direction"])
        selected = forward["selected"]
        self.assertEqual(selected["minimum_width"], min(r["minimum_width"] for r in forward["candidates"]))
        self.assertAlmostEqual(np.ptp(self.cloud @ forward["direction"]), selected["minimum_width"])
        history = forward["histories"][selected["candidate_id"]]
        self.assertTrue(np.all(np.diff(history) <= 0))
        self.assertAlmostEqual(history[-1], selected["minimum_width"])

    def test_invalid_grids_and_directions_are_rejected(self):
        for gamma, beta in (([], [0.9]), ([0.1], []), ([0.0], [0.9]),
                            ([np.nan], [0.9]), ([0.1], [1.0]), ([0.1], [0.0])):
            with self.subTest(gamma=gamma, beta=beta), self.assertRaises(ValueError):
                tune_geometric_width_schedule(self.cloud, self.directions, gamma, beta, **self.settings)
        for directions in (np.zeros((1, 2)), np.ones((1, 3)), np.array([[np.inf, 1.0]])):
            with self.subTest(directions=directions), self.assertRaises(ValueError):
                tune_geometric_width_schedule(self.cloud, directions, [0.1], [0.9], **self.settings)

    def test_half_life_rules_use_the_actual_diameter_and_reproduce_direct_steps(self):
        cloud = np.array([[0., 0.], [3., 0.], [1., 1.]])
        direction = np.array([0.4, 0.9])
        result = tune_geometric_half_life_schedules(
            cloud, direction, [0.9], [4], **self.settings,
        )
        self.assertEqual(result["diameter"], 3.)  # Box diagonal would be sqrt(10).
        self.assertEqual(len(result["candidates"]), 2)
        for row in result["candidates"]:
            expected_step = 0.9 if row["step_rule"] == "absolute" else 0.3
            self.assertAlmostEqual(row["gamma0"], expected_step)
            self.assertAlmostEqual(row["beta"] ** row["half_life"], 0.5)
            direct = projected_subgradient_empirical_width(
                cloud, initial_direction=direction, initial_step=expected_step,
                geometric_decay=2**(-1/4), step_schedule="geometric",
                rng=np.random.SeedSequence([17, 0, 2, 21, 0]),
                iterations=30, patience=8, improvement_tolerance=1e-10,
            )
            self.assertAlmostEqual(row["minimum_width"], direct["minimum_width"])
            np.testing.assert_allclose(result["histories"][row["candidate_id"]],
                                       direct["best_width_history"], atol=1e-13, rtol=1e-13)

    def test_normalized_rule_is_invariant_to_cloud_scaling(self):
        cloud = np.array([[0., 0.], [3., 0.], [1., 1.], [-0.2, 0.5]])
        direction = np.array([0.4, 0.9])
        settings = dict(self.settings, improvement_tolerance=0.)
        original = tune_geometric_half_life_schedules(
            cloud, direction, [0.07, 0.3], [2, 20], **settings,
        )
        scaled = tune_geometric_half_life_schedules(
            7*cloud, direction, [0.07, 0.3], [2, 20], **settings,
        )
        for row, other in zip(original["candidates"], scaled["candidates"]):
            if row["step_rule"] != "diameter_normalized":
                continue
            self.assertAlmostEqual(7*row["minimum_width"], other["minimum_width"], places=10)
            np.testing.assert_allclose([row["u_0"], row["u_1"]],
                                       [other["u_0"], other["u_1"]], atol=1e-11, rtol=1e-11)

    def test_coincident_effective_schedules_are_evaluated_once(self):
        result = tune_geometric_half_life_schedules(
            np.array([[0., 0.], [1., 0.]]), [0.4, 0.9], [0.03], [4], **self.settings,
        )
        self.assertEqual(result["effective_schedule_count"], 1)
        self.assertEqual(len(result["candidates"]), 2)
        self.assertEqual(result["candidates"][0]["minimum_width"],
                         result["candidates"][1]["minimum_width"])

    def test_unified_notebook_runs_and_exports_only_figures(self):
        notebook = json.loads((
            Path(__file__).resolve().parents[1] / "xx_notebooks_tuning"
            / "ar_unified_tuning_experiments.ipynb"
        ).read_text())
        sources = {cell["id"]: "".join(cell["source"])
                   for cell in notebook["cells"] if cell["cell_type"] == "code"}
        self.assertNotIn("numerical-audit", sources)
        for removed in ["exact_empirical_width_low_dimension", "audit_width", "target_relative_gap"]:
            self.assertNotIn(removed, "\n".join(sources.values()))
        namespace = {}
        with contextlib.redirect_stdout(io.StringIO()), tempfile.TemporaryDirectory() as directory:
            exec(sources["imports"], namespace)
            exec(sources["settings"], namespace)
            self.assertEqual(namespace["max_iterations"], 2000)
            self.assertEqual(namespace["patience"], 200)
            namespace.update(T=48, burnin=20, max_iterations=30, patience=12,
                             scale_grid=np.array([0.05, 0.2]),
                             half_life_grid=np.array([2, 20]),
                             number_of_realizations=2, OUTPUT_ROOT=Path(directory))
            for cell_id in ["fixed-inputs", "process-plot", "run-grid", "matrix-plots",
                            "coefficient-recovery", "export-selection",
                            "run-realizations", "realization-matrices", "realization-curves",
                            "realization-selection",
                            "epsilon-win-counts", "epsilon-win-matrices"]:
                exec(sources[cell_id], namespace)
            selected = namespace["selected"]
            self.assertEqual(list(Path(directory).rglob("*.json")), [])
            self.assertEqual(list(Path(directory).rglob("*.csv")), [])
            self.assertEqual(len(namespace["ranking"]), 8)
            self.assertEqual(namespace["initial_direction"].shape, (4,))
            self.assertEqual(selected["minimum_width"],
                             namespace["ranking"]["minimum_width"].min())
            for rule in ["absolute", "diameter_normalized"]:
                self.assertEqual(namespace["width_matrices"][rule].shape, (2, 2))
            self.assertEqual(namespace["coefficient_comparison"].shape[0], 6)
            self.assertEqual(len(namespace["realization_results"]), 16)
            self.assertTrue(namespace["realization_results"]["total_evaluations"].between(1, 30).all())
            self.assertEqual(len(namespace["realization_ranking"]), 8)
            self.assertEqual(len(namespace["realization_cases"]), 2)
            means = namespace["realization_results"].groupby("candidate_id")["minimum_width"].mean()
            self.assertAlmostEqual(namespace["realization_selected"]["mean_minimum_width"], means.min())
            for rule in ["absolute", "diameter_normalized"]:
                self.assertEqual(namespace["realization_matrices"][rule].shape, (2, 2))
            for name in ["01_fixed_process", "02_minimum_width_matrices",
                         "04_realization_mean_minimum_widths", "05_realization_selected_widths"]:
                self.assertTrue((namespace["PDF_DIR"] / f"{name}.pdf").is_file())
                self.assertTrue((namespace["IMAGE_DIR"] / f"{name}.png").is_file())
            self.assertEqual(namespace["max_iterations"], 30)
            self.assertEqual(namespace["patience"], 12)
            self.assertEqual(namespace["epsilon_win_tolerance"], 1e-8)
            self.assertEqual(namespace["epsilon_win_experiment"]["epsilon"], 1e-8)
            self.assertGreaterEqual(namespace["epsilon_win_ranking"]["wins"].sum(), 2)
            self.assertTrue(namespace["epsilon_win_ranking"]["wins"].between(0, 2).all())
            self.assertEqual(len(namespace["epsilon_win_ranking"]), 8)
            self.assertEqual(len(namespace["epsilon_win_cases"]), 2)
            self.assertEqual(len(namespace["epsilon_win_winners"]),
                             namespace["epsilon_win_experiment"]["total_points"])
            epsilon_expected = set()
            for replication, rows in namespace["realization_results"].groupby("replication"):
                threshold = rows["minimum_width"].min() + 1e-8
                epsilon_expected.update((replication, candidate) for candidate in
                                        rows.loc[rows["minimum_width"] <= threshold, "candidate_id"])
            epsilon_actual = {(row["replication"], row["candidate_id"])
                              for row in namespace["epsilon_win_winners"].to_dict("records")}
            self.assertEqual(epsilon_actual, epsilon_expected)
            for matrix in namespace["epsilon_win_matrices"].values():
                self.assertEqual(matrix.shape, (2, 2))
            for output, suffix in [("PDF_DIR", ".pdf"), ("IMAGE_DIR", ".png")]:
                self.assertTrue((namespace[output] / (namespace["epsilon_win_figure_stem"] + suffix)).is_file())
                self.assertEqual(len(list(namespace[output].glob("*" + suffix))), 5)
            ids = [cell["id"] for cell in notebook["cells"]]
            self.assertLess(ids.index("realization-selection"), ids.index("epsilon-win-frequency-description"))
            exec(sources["jones-settings"], namespace)
            self.assertEqual(namespace["number_of_coefficient_realizations"], 100)
            namespace["number_of_coefficient_realizations"] = 2
            for cell_id in ["run-jones-realizations", "jones-models", "jones-mean-matrices",
                            "jones-win-counts", "jones-win-matrices", "jones-selection",
                            "jones-coefficient-diagnostics", "jones-width-benchmarks"]:
                exec(sources[cell_id], namespace)
            self.assertEqual(len(namespace["jones_results"]), 16)
            self.assertEqual(len(namespace["jones_cases"]), 2)
            self.assertEqual(len(namespace["jones_win_ranking"]), 8)
            self.assertTrue(namespace["jones_results"]["selected_start"].eq(1).all())
            self.assertTrue(namespace["jones_results"]["total_evaluations"].between(1, 30).all())
            self.assertTrue(namespace["jones_cases"]["spectral_radius"].lt(1).all())
            self.assertEqual(namespace["jones_cases"][["phi_1", "phi_2", "phi_3"]].drop_duplicates().shape[0], 2)
            diagnostics = namespace["jones_coefficient_diagnostics"]
            self.assertEqual(len(diagnostics), 16)
            self.assertEqual(namespace["jones_failure_counts"].iloc[0]["runs"], 16)
            self.assertEqual(len(namespace["jones_grid_winner_coefficients"]), 2)
            for row in diagnostics.to_dict("records"):
                estimate = np.array([row[f"estimated_phi_{j}"] for j in range(1, 4)])
                self.assertEqual(row["nonstationary_estimate"], not is_stationary(estimate)[0])
                self.assertEqual(row["undefined_coefficients"], False)
            self.assertEqual(namespace["jones_failure_by_configuration"]["nonstationary_estimates"].sum(),
                             diagnostics["nonstationary_estimate"].sum())
            width_diagnostics = namespace["jones_width_diagnostics"]
            self.assertEqual(len(width_diagnostics), 2)
            np.testing.assert_allclose(
                width_diagnostics["grid_population_width_gap"],
                width_diagnostics["grid_best_width"] - width_diagnostics["population_minimum_width"],
            )
            self.assertTrue((width_diagnostics["empirical_suboptimality_lower_bound"] >= 0).all())
            expected = set()
            for replication, rows in namespace["jones_results"].groupby("replication"):
                threshold = rows["minimum_width"].min() + 1e-8
                expected.update((replication, candidate) for candidate in
                                rows.loc[rows["minimum_width"] <= threshold, "candidate_id"])
            self.assertEqual({(row["replication"], row["candidate_id"])
                              for row in namespace["jones_win_winners"].to_dict("records")}, expected)
            for output, suffix in [("JONES_PDF_DIR", ".pdf"), ("JONES_IMAGE_DIR", ".png")]:
                self.assertEqual(len(list(namespace[output].glob("*" + suffix))), 2)
            self.assertEqual(list(Path(directory).rglob("*.json")), [])
            self.assertEqual(list(Path(directory).rglob("*.csv")), [])
            self.assertLess(ids.index("epsilon-win-matrices"), ids.index("jones-description"))

    def test_notebook_setup_refreshes_a_cached_module_missing_new_helper(self):
        notebook = json.loads((
            Path(__file__).resolve().parents[1] / "xx_notebooks_tuning"
            / "ar_unified_tuning_experiments.ipynb"
        ).read_text())
        setup = next("".join(cell["source"]) for cell in notebook["cells"]
                     if cell.get("id") == "imports")
        module = importlib.import_module("Code.Code_AR_processes.ar_tools.tuning")
        descent_module = importlib.import_module("Code.Code_AR_processes.ar_tools.descent")
        # Emulate a kernel that imported tuning.py before experiment 2 existed.
        del module.tune_fixed_ar_realizations
        namespace = {}
        try:
            self.assertFalse(hasattr(module, "tune_fixed_ar_realizations"))
            with patch.object(descent_module, "projected_subgradient_empirical_width",
                              side_effect=RuntimeError("Stale optimizer in the notebook kernel.")):
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(setup, namespace)
                self.assertIs(namespace["descent_tools"].projected_subgradient_empirical_width,
                              descent_module.projected_subgradient_empirical_width)
                result = namespace["descent_tools"].projected_subgradient_empirical_width(
                    np.array([[-2., 0.], [1., 0.]]), iterations=5, patience=3,
                    initial_direction=np.array([1., 0.]),
                )
                self.assertEqual(result["termination_reason"], "patience")
                self.assertEqual(len(result["width_history"]), 4)
            self.assertTrue(callable(namespace["tune_fixed_ar_realizations"]))
            self.assertIs(namespace["tune_fixed_ar_realizations"], module.tune_fixed_ar_realizations)
        finally:
            importlib.reload(descent_module)
            importlib.reload(module)

    def test_replication_ranking_uses_mean_achieved_width_on_all_paths(self):
        rows = []
        for candidate, widths, evaluations in [
            ("lowest_mean", [1.5, 1.5], 50),
            ("lowest_single_run", [1.0, 2.1], 10),
            ("same_mean_slower", [1.5, 1.5], 60),
        ]:
            for replication, width in enumerate(widths, start=1):
                rows.append(dict(candidate_id=candidate, replication=replication,
                                 step_rule="absolute", step_value=0.1, half_life=20., beta=0.97,
                                 gamma0=0.1, minimum_width=width, total_evaluations=evaluations))
        ranking = summarize_geometric_replications(rows)
        self.assertEqual([row["candidate_id"] for row in ranking],
                         ["lowest_mean", "same_mean_slower", "lowest_single_run"])
        self.assertEqual(ranking[0]["mean_minimum_width"], 1.5)
        self.assertEqual(ranking[0]["replications"], 2)
        self.assertEqual(ranking[2]["mean_minimum_width"], 1.55)
        self.assertEqual(ranking[2]["worst_minimum_width"], 2.1)
        self.assertEqual(ranking[2]["median_minimum_width"], 1.55)
        with self.assertRaises(ValueError):
            summarize_geometric_replications(rows[:-1])
        with self.assertRaises(ValueError):
            summarize_geometric_replications(rows + [rows[0]])
        for invalid in [np.nan, np.inf, -0.1]:
            with self.subTest(width=invalid), self.assertRaises(ValueError):
                summarize_geometric_replications([dict(rows[0], minimum_width=invalid)])

    def test_win_frequency_can_disagree_with_mean_and_includes_zero_win_candidates(self):
        rows = []
        for candidate, scores in [("frequent", [1., 1., 100.]),
                                  ("small_mean", [2., 2., 2.]),
                                  ("never", [3., 3., 101.])]:
            for replication, score in enumerate(scores, start=1):
                rows.append(dict(candidate_id=candidate, replication=replication, T=40,
                                 step_rule="absolute", step_value=.1, half_life=5., beta=.9,
                                 minimum_width=score, total_evaluations=30))
        result = summarize_configuration_wins(rows, score_key="minimum_width")
        self.assertEqual(result["selected"]["candidate_id"], "frequent")
        self.assertEqual(result["selected"]["wins"], 2)
        self.assertAlmostEqual(result["selected"]["win_rate"], 2/3)
        self.assertEqual(result["selected"]["mean_score"], 34.)
        self.assertEqual([r["wins"] for r in result["ranking"]], [2, 1, 0])
        self.assertEqual(sum(r["wins"] for r in result["ranking"]), 3)
        self.assertEqual(min(result["ranking"], key=lambda r: r["mean_score"])["candidate_id"], "small_mean")
        for incomplete in [rows[:-1], rows + [rows[0]]]:
            with self.assertRaises(ValueError):
                summarize_configuration_wins(incomplete, score_key="minimum_width")
        with self.assertRaises(ValueError):
            summarize_configuration_wins([dict(rows[0], minimum_width=np.nan)],
                                         score_key="minimum_width")

    def test_absolute_gap_wins_use_raw_scores_and_include_the_threshold(self):
        epsilon = 1e-8
        rows = []
        for replication, scores in [
            (1, [1.000000004, 1.000000006, 1.00000002]),
            (2, [1.5, 1.5 + epsilon, 1.5 + 2 * epsilon]),
            (3, [100., 100. + 1e-7, 100. + 1e-9]),
        ]:
            for candidate, score, evaluations in zip(["A", "B", "C"], scores, [30, 10, 1]):
                rows.append(dict(candidate_id=candidate, replication=replication, T=40,
                                 step_rule="absolute", step_value=.1, half_life=5., beta=.9,
                                 minimum_width=score, total_evaluations=evaluations))
        result = summarize_configuration_wins(rows, epsilon=epsilon)
        self.assertEqual([(row["replication"], row["candidate_id"]) for row in result["winners"]],
                         [(1, "A"), (1, "B"), (2, "A"), (2, "B"), (3, "A"), (3, "C")])
        self.assertEqual(result["total_points"], 6)
        self.assertEqual(result["epsilon"], epsilon)
        self.assertEqual(result["ranking"][0]["wins"], 3)
        self.assertTrue(all(row["winning_score"] <= row["winning_threshold"]
                            for row in result["winners"]))
        self.assertEqual(summarize_configuration_wins(rows), result)
        reversed_result = summarize_configuration_wins(list(reversed(rows)), epsilon=epsilon)
        self.assertEqual(reversed_result["winners"], result["winners"])
        self.assertEqual(reversed_result["case_summaries"], result["case_summaries"])
        self.assertEqual([(row["candidate_id"], row["wins"]) for row in reversed_result["ranking"]],
                         [(row["candidate_id"], row["wins"]) for row in result["ranking"]])
        for actual, expected in zip(reversed_result["ranking"], result["ranking"]):
            self.assertAlmostEqual(actual["mean_score"], expected["mean_score"])
        exact = summarize_configuration_wins(rows, epsilon=0.)
        self.assertEqual([row["candidate_id"] for row in exact["winners"]], ["A", "A", "A"])
        for invalid in [-1., np.nan, np.inf, True, "1e-8", None]:
            with self.subTest(epsilon=invalid), self.assertRaises(ValueError):
                summarize_configuration_wins(rows, epsilon=invalid)

    def test_fixed_phi_paths_reproduce_paired_grid_without_external_optimum(self):
        phi = np.array([.6, -.3, .2])
        direction = np.array([.1, -.8, .3, .5])
        original = direction.copy()
        with patch("Code.Code_AR_processes.ar_tools.lp_widths.exact_empirical_width_low_dimension",
                   side_effect=AssertionError("External optimum must not be computed.")):
            result = tune_fixed_ar_realizations(
                phi, 40, direction, [.1], [5], replications=2, process_seed=20261020,
                burnin=20, **self.settings,
            )
        seeds = [row["process_seed"] for row in result["cases"]]
        self.assertEqual(len(set(seeds)), 2)
        np.testing.assert_array_equal(direction, original)
        for case in result["cases"]:
            series = simulate_uniform_ar(phi, sample_size=40, burnin=20, half_width=1.,
                                         rng=case["process_seed"])
            cloud = sliding_window_point_cloud(series, 4)
            true_direction = np.r_[1., -phi]
            normalizer = np.linalg.norm(true_direction)
            self.assertAlmostEqual(case["population_minimum_width"], 2./normalizer)
            self.assertAlmostEqual(case["empirical_true_direction_width"],
                                   np.ptp(cloud @ (true_direction / normalizer)))
            self.assertLessEqual(case["empirical_true_direction_width"], case["population_minimum_width"] + 1e-12)
            direct = tune_geometric_half_life_schedules(
                cloud, direction / np.linalg.norm(direction), [.1], [5], **self.settings,
            )
            self.assertAlmostEqual(case["diameter"], direct["diameter"])
            self.assertEqual(case["grid_best_width"], direct["selected"]["minimum_width"])
            self.assertNotIn("reference_width", case)
            for row in (row for row in result["results"] if row["replication"] == case["replication"]):
                history = direct["histories"][row["candidate_id"]]
                self.assertAlmostEqual(row["minimum_width"], history[-1])
                self.assertNotIn("relative_width_gap", row)
                self.assertNotIn("success", row)
                if row["step_rule"] == "diameter_normalized":
                    self.assertAlmostEqual(row["gamma0"], .1 / case["diameter"])
        for summary in result["ranking"]:
            widths = [row["minimum_width"] for row in result["results"]
                      if row["candidate_id"] == summary["candidate_id"]]
            self.assertEqual(summary["mean_minimum_width"], np.mean(widths))
        self.assertEqual(result["selected"]["mean_minimum_width"],
                         min(row["mean_minimum_width"] for row in result["ranking"]))

    def test_jones_models_reproduce_seeded_draws_and_descents_from_one_fixed_start(self):
        direction = np.array([.1, -.8, .3, .5])
        original = direction.copy()
        arguments = dict(replications=2, coefficient_seed=20261023, process_seed=20261024,
                         burnin=20, **self.settings)
        result = tune_jones_ar_realizations(40, direction, [.1], [5], **arguments)
        np.testing.assert_array_equal(direction, original)
        np.testing.assert_allclose(result["initial_direction"], direction / np.linalg.norm(direction))
        self.assertEqual(len(result["cases"]), 2)
        self.assertEqual(len(result["results"]), 4)
        self.assertEqual(len({case["coefficient_seed"] for case in result["cases"]}), 2)
        self.assertEqual(len({case["process_seed"] for case in result["cases"]}), 2)
        for case in result["cases"]:
            phi, pacf = jones_ar_coefficients(3, rng=case["coefficient_seed"], return_pacf=True)
            np.testing.assert_array_equal([case[f"phi_{j}"] for j in range(1, 4)], phi)
            np.testing.assert_array_equal([case[f"pacf_{j}"] for j in range(1, 4)], pacf)
            self.assertTrue(is_stationary(phi)[0])
            self.assertLess(case["spectral_radius"], 1.)
            series = simulate_uniform_ar(phi, sample_size=40, burnin=20, rng=case["process_seed"])
            cloud = sliding_window_point_cloud(series, 4)
            direct = tune_geometric_half_life_schedules(
                cloud, direction / np.linalg.norm(direction), [.1], [5], **self.settings,
            )
            for row, expected in zip(
                [row for row in result["results"] if row["replication"] == case["replication"]],
                direct["candidates"],
            ):
                self.assertEqual(row["minimum_width"], expected["minimum_width"])
                self.assertEqual(row["total_evaluations"], expected["total_evaluations"])
                self.assertEqual(row["selected_start"], 1)
                np.testing.assert_array_equal([row[f"phi_{j}"] for j in range(1, 4)], phi)
        changed_innovations = tune_jones_ar_realizations(
            40, direction, [.1], [5], **dict(arguments, process_seed=20261025),
        )
        for original_case, changed_case in zip(result["cases"], changed_innovations["cases"]):
            np.testing.assert_array_equal([original_case[f"phi_{j}"] for j in range(1, 4)],
                                          [changed_case[f"phi_{j}"] for j in range(1, 4)])
            self.assertNotEqual(original_case["process_seed"], changed_case["process_seed"])
        for invalid_seed in [None, -1, True, 1.5]:
            with self.subTest(coefficient_seed=invalid_seed), self.assertRaises(ValueError):
                tune_jones_ar_realizations(40, direction, [.1], [5],
                                           **dict(arguments, coefficient_seed=invalid_seed))

    def test_population_width_gap_is_signed_and_does_not_change_the_search(self):
        phi = np.array([.6, -.3, .2])
        direction = np.r_[1., -phi]
        direction /= np.linalg.norm(direction)
        result = tune_fixed_ar_realizations(
            phi, 40, direction, [.1], [5], replications=1, process_seed=81,
            burnin=20, **dict(self.settings, iterations=1),
        )
        for row in result["results"]:
            self.assertLess(row["population_width_gap"], 0.)
            self.assertAlmostEqual(row["population_width_gap"],
                                   row["minimum_width"] - 2./np.sqrt(1.+phi @ phi))
            self.assertAlmostEqual(row["empirical_true_direction_gap"], 0., places=12)
            self.assertEqual(row["total_evaluations"], 1)

    def test_sample_size_study_preserves_single_size_results_and_case_weights(self):
        phi = np.array([.6, -.3, .2])
        direction = np.array([.1, -.8, .3, .5])
        settings = dict(self.settings, replications=2, process_seed=81, burnin=20)
        joint = tune_ar_sample_sizes(phi, [32, 48], direction, [.1], [5], **settings)
        self.assertEqual(len(joint["results"]), 8)
        for size in [32, 48]:
            single = tune_fixed_ar_realizations(phi, size, direction, [.1], [5], **settings)
            actual = [row for row in joint["results"] if row["T"] == size]
            for row, expected in zip(actual, single["results"]):
                self.assertEqual(row["process_seed"], expected["process_seed"])
                self.assertEqual(row["minimum_width"], expected["minimum_width"])
                self.assertEqual(row["total_evaluations"], expected["total_evaluations"])
        for row in joint["ranking"]:
            widths = [case["minimum_width"] for case in joint["results"]
                      if case["candidate_id"] == row["candidate_id"]]
            self.assertEqual(row["replications"], 2)
            self.assertEqual(row["cases"], 4)
            self.assertEqual(row["mean_minimum_width"], np.mean(widths))
        with self.assertRaises(ValueError):
            summarize_geometric_replications(joint["results"][:-1])

    def test_validation_evaluates_frozen_schedules_without_search(self):
        phi = np.array([.6, -.3, .2])
        direction = np.array([.1, -.8, .3, .5])
        schedules = [dict(candidate_id=name, step_rule="diameter_normalized", step_value=.2, half_life=5.)
                     for name in ["selected", "baseline"]]
        original = [dict(row) for row in schedules]
        with patch("Code.Code_AR_processes.ar_tools.tuning.tune_geometric_half_life_schedules",
                   side_effect=AssertionError("Validation must not search the grid.")):
            result = compare_ar_schedules(
                phi, [32, 48], direction, schedules, replications=2,
                process_seed=82, burnin=20, record_histories=True, **self.settings,
            )
        self.assertEqual(schedules, original)
        self.assertEqual(len(result["results"]), 8)
        self.assertEqual(len(result["histories"]), 8)
        for row in result["results"]:
            trace = result["histories"][(row["replication"], row["T"], row["candidate_id"])]
            self.assertEqual(len(trace["width_history"]), row["total_evaluations"])
            self.assertEqual(trace["minimum_width"], row["minimum_width"])
            self.assertEqual(trace["best_width_history"][-1], row["minimum_width"])
            self.assertTrue(np.all(np.diff(trace["best_width_history"]) <= 0))
        for first, second in zip(result["results"][::2], result["results"][1::2]):
            self.assertEqual(first["minimum_width"], second["minimum_width"])
            self.assertEqual(first["total_evaluations"], second["total_evaluations"])
            self.assertEqual(first["process_seed"], second["process_seed"])
            self.assertAlmostEqual(first["gamma0"], .2/first["lipschitz_constant"])
            np.testing.assert_array_equal([first[f"u_{j}"] for j in range(4)],
                                          [second[f"u_{j}"] for j in range(4)])

    def test_sample_size_validation_notebook_runs_without_data_exports(self):
        directory = Path(__file__).resolve().parents[1] / "xx_notebooks_tuning"
        notebook = json.loads((directory / "ar_tuning_sample_size_and_validation.ipynb").read_text())
        sources = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]
                   if cell["cell_type"] == "code"}
        namespace = {}
        with contextlib.redirect_stdout(io.StringIO()), tempfile.TemporaryDirectory() as output:
            exec(sources["imports"], namespace)
            # Redirect exports before settings creates its output directories.
            namespace["OUTPUT_ROOT"] = Path(output)
            small_settings = sources["settings"].replace("np.array([500, 2000], dtype=int)",
                                                       "np.array([32, 48], dtype=int)")
            small_settings = small_settings.replace("burnin = 2000", "burnin = 20")
            small_settings = small_settings.replace("replications = 6", "replications = 2")
            small_settings = small_settings.replace("max_iterations = 2000", "max_iterations = 10")
            small_settings = small_settings.replace("patience = 200", "patience = 4")
            small_settings = small_settings.replace("np.arange(1, 10) / 10.0", "np.array([.1, .2])")
            small_settings = small_settings.replace("np.array([2, 5, 10, 20, 50, 100, 200, 1000])",
                                                   "np.array([2, 5])")
            exec(small_settings, namespace)
            for cell in ["tune-sample-sizes", "sample-size-matrices", "validate-frozen-schedules",
                         "validation-plot", "summary"]:
                exec(sources[cell], namespace)
            self.assertEqual(len(namespace["tuning_results"]), 32)
            self.assertEqual(len(namespace["validation_results"]), 8)
            self.assertEqual(len(namespace["paired_comparison"]), 4)
            self.assertEqual(len(namespace["joint_ranking"]), 8)
            self.assertEqual(list(Path(output).rglob("*.json")), [])
            self.assertEqual(list(Path(output).rglob("*.csv")), [])
            self.assertEqual(len(list(Path(output).rglob("*.pdf"))), 2)
            self.assertEqual(len(list(Path(output).rglob("*.png"))), 2)

    def test_tuning_folder_keeps_only_unified_and_distinct_extensions(self):
        directory = Path(__file__).resolve().parents[1] / "xx_notebooks_tuning"
        self.assertEqual({path.name for path in directory.glob("*.ipynb")},
                         {"ar_unified_tuning_experiments.ipynb", "ar_tuning_sample_size_and_validation.ipynb"})


if __name__ == "__main__":
    unittest.main()
