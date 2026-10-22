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

    def test_stationarity_tuning_reuses_optimizer_and_counts_discarded_attempts(self):
        cloud = np.array([[-1., 0.], [1., 0.]])
        settings = dict(self.settings, stationarity_check=True, max_restarts=20)
        result = tune_geometric_width_schedule(
            cloud, [[0., 1.]], [0.03], [0.97], **settings,
        )
        direct = projected_subgradient_empirical_width(
            cloud, initial_direction=[0., 1.], initial_step=0.03,
            geometric_decay=0.97, step_schedule="geometric",
            rng=np.random.SeedSequence([17, 0, 2, 21, 0]),
            **{key: value for key, value in settings.items() if key != "optimizer_seed"},
        )
        row = result["selected"]
        self.assertGreater(row["restart_count"], 0)
        self.assertTrue(row["stationarity_check_passed"])
        self.assertFalse(row["restart_limit_reached"])
        self.assertEqual(row["restart_count"], direct["restart_count"])
        self.assertEqual(row["total_evaluations"], direct["total_evaluations"])
        self.assertEqual(result["per_start"][0]["evaluations"], direct["total_evaluations"])
        self.assertEqual(row["total_evaluations"],
                         8 * row["restart_count"] + row["final_evaluations"])
        self.assertGreater(row["minimum_width"], 0.)  # Failed attempt's width was zero.
        np.testing.assert_array_equal(result["direction"], direct["direction"])
        np.testing.assert_array_equal(result["histories"][row["candidate_id"]],
                                      direct["best_width_history"])

    def test_stationarity_tuning_does_not_rank_an_exhausted_attempt(self):
        with patch("Code.Code_AR_processes.ar_tools.descent.is_stationary",
                   return_value=(False, np.array([2.]))):
            result = tune_geometric_half_life_schedules(
                self.cloud, self.directions[0], [0.03], [4],
                stationarity_check=True, max_restarts=0, **self.settings,
            )
        row = result["candidates"][0]
        self.assertFalse(row["stationarity_check_passed"])
        self.assertTrue(row["restart_limit_reached"])
        self.assertTrue(np.isnan(row["minimum_width"]))
        self.assertTrue(np.isnan(result["direction"]).all())
        self.assertTrue(np.isnan(result["histories"][row["candidate_id"]]).all())
        self.assertFalse(result["per_start"][0]["selected"])
        summary = summarize_configuration_wins(
            [dict(row, replication=1, T=10)], allow_nonfinite=True,
        )
        self.assertEqual(summary["total_points"], 0)
        self.assertEqual(summary["undefined_cases"], 1)
        self.assertEqual(summary["ranking"][0]["wins"], 0)

    def test_stationarity_tuning_selects_a_passed_start_over_a_failed_smaller_width(self):
        result = tune_geometric_width_schedule(
            np.array([[-1., 0.], [1., 0.]]), [[0., 1.], [1., 0.]], [0.03], [0.97],
            stationarity_check=True, max_restarts=0, **self.settings,
        )
        self.assertEqual(result["selected"]["selected_start"], 2)
        self.assertEqual(result["selected"]["minimum_width"], 2.)
        self.assertTrue(result["selected"]["stationarity_check_passed"])
        self.assertTrue(np.isnan(result["per_start"][0]["minimum_width"]))
        self.assertFalse(result["per_start"][0]["selected"])
        np.testing.assert_array_equal(result["histories"]["grid_000"],
                                      result["best_results"][1]["best_width_history"])

    def test_parallel_fixed_paths_preserve_seeds_restarts_and_results(self):
        settings = dict(
            self.settings, replications=2, process_seed=20261025, burnin=20,
            summarize=False, stationarity_check=True, max_restarts=3,
        )
        args = ([0.8981283425999981, -0.2059579137943896, -0.08127461108863643],
                40, [0.1, -0.8, 0.3, 0.5], [0.05, 0.2], [2, 20])
        serial = tune_fixed_ar_realizations(*args, n_jobs=1, **settings)
        parallel = tune_fixed_ar_realizations(*args, n_jobs=2, **settings)
        self.assertEqual(len(parallel["results"]), 8)
        self.assertEqual(len(parallel["cases"]), 2)
        for actual, expected in zip(parallel["results"], serial["results"]):
            for key in ["candidate_id", "replication", "process_seed", "restart_count",
                        "attempt_count", "total_evaluations", "stationarity_check_passed",
                        "restart_limit_reached"]:
                self.assertEqual(actual[key], expected[key])
            for key in ["minimum_width", "checkpoint_spectral_radius", "u_0", "u_1", "u_2", "u_3"]:
                np.testing.assert_allclose(actual[key], expected[key], atol=1e-12, rtol=1e-12, equal_nan=True)
        for jobs in [0, True, 1.5]:
            with self.subTest(n_jobs=jobs), self.assertRaisesRegex(ValueError, "n_jobs"):
                tune_fixed_ar_realizations(*args, n_jobs=jobs, **settings)

    def test_invalid_grids_and_directions_are_rejected(self):
        for gamma, beta in (([], [0.9]), ([0.1], []), ([0.0], [0.9]),
                            ([np.nan], [0.9]), ([0.1], [1.0]), ([0.1], [0.0])):
            with self.subTest(gamma=gamma, beta=beta), self.assertRaises(ValueError):
                tune_geometric_width_schedule(self.cloud, self.directions, gamma, beta, **self.settings)
        for directions in (np.zeros((1, 2)), np.ones((1, 3)), np.array([[np.inf, 1.0]])):
            with self.subTest(directions=directions), self.assertRaises(ValueError):
                tune_geometric_width_schedule(self.cloud, directions, [0.1], [0.9], **self.settings)

    def test_absolute_half_life_grid_reproduces_direct_steps(self):
        cloud = np.array([[0., 0.], [3., 0.], [1., 1.]])
        direction = np.array([0.4, 0.9])
        result = tune_geometric_half_life_schedules(
            cloud, direction, [0.9], [4], **self.settings,
        )
        self.assertEqual(len(result["candidates"]), 1)
        for row in result["candidates"]:
            expected_step = 0.9
            self.assertEqual(row["step_rule"], "absolute")
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

    def test_initial_steps_are_independent_of_cloud_scaling(self):
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
            self.assertEqual(row["gamma0"], row["step_value"])
            self.assertEqual(other["gamma0"], row["gamma0"])
            self.assertEqual(other["beta"], row["beta"])

    def test_diameter_normalized_grid_reuses_direct_optimizer_and_rescales(self):
        cloud = np.array([[0., 0.], [3., 0.], [0., 4.]])  # Diameter is exactly 5.
        direction = np.array([0.4, 0.9])
        settings = dict(self.settings, step_rule="diameter_normalized", improvement_tolerance=0.)
        result = tune_geometric_half_life_schedules(cloud, direction, [.1, .9], [4], **settings)
        scaled = tune_geometric_half_life_schedules(2 * cloud, direction, [.1, .9], [4], **settings)
        for row, other in zip(result["candidates"], scaled["candidates"]):
            self.assertEqual(row["cloud_diameter"], 5.)
            self.assertEqual(other["cloud_diameter"], 10.)
            self.assertEqual(row["gamma0"], row["step_value"] / 5.)
            self.assertEqual(other["gamma0"], row["gamma0"] / 2.)
            direct = projected_subgradient_empirical_width(
                cloud, initial_direction=direction, initial_step=row["gamma0"],
                geometric_decay=row["beta"], step_schedule="geometric",
                rng=np.random.SeedSequence([17, 0, 2, 21, 0]),
                iterations=30, patience=8, improvement_tolerance=0.,
            )
            np.testing.assert_array_equal(result["histories"][row["candidate_id"]],
                                          direct["best_width_history"])
            np.testing.assert_array_equal([row["u_0"], row["u_1"]], [other["u_0"], other["u_1"]])
            self.assertEqual(other["minimum_width"], 2 * row["minimum_width"])
        with self.assertRaisesRegex(ValueError, "positive finite cloud diameter"):
            tune_geometric_half_life_schedules(np.ones((3, 2)), direction, [.1], [4], **settings)
        with self.assertRaisesRegex(ValueError, "step_rule"):
            tune_geometric_half_life_schedules(cloud, direction, [.1], [4],
                                               **dict(settings, step_rule="unknown"))

    def test_parallel_normalized_jones_paths_preserve_inputs_and_actual_steps(self):
        args = (40, [.1, -.8, .3, .5], [.1, .9], [4])
        settings = dict(self.settings, replications=2, process_seed=20261024,
                        coefficient_seed=20261023, burnin=20, summarize=False,
                        stationarity_check=True, max_restarts=3, step_rule="diameter_normalized")
        serial = tune_jones_ar_realizations(*args, n_jobs=1, **settings)
        parallel = tune_jones_ar_realizations(*args, n_jobs=2, **settings)
        for row, other in zip(serial["results"], parallel["results"]):
            for key in set(row) - {"runtime_seconds"}:
                np.testing.assert_equal(row[key], other[key])
            phi = [row[f"phi_{j}"] for j in range(1, 4)]
            cloud = sliding_window_point_cloud(simulate_uniform_ar(
                phi, sample_size=40, burnin=20, rng=row["process_seed"],
            ), 4)
            expected_diameter = np.linalg.norm(cloud[:, None, :] - cloud[None, :, :], axis=2).max()
            self.assertAlmostEqual(row["cloud_diameter"], expected_diameter)
            self.assertAlmostEqual(row["gamma0"] * expected_diameter, row["step_value"])

    def test_single_grid_pair_produces_one_candidate(self):
        result = tune_geometric_half_life_schedules(
            np.array([[0., 0.], [1., 0.]]), [0.4, 0.9], [0.03], [4], **self.settings,
        )
        self.assertEqual(result["effective_schedule_count"], 1)
        self.assertEqual(len(result["candidates"]), 1)
        self.assertEqual(len(result["per_start"]), 1)
        self.assertEqual(len(result["histories"]), 1)

    def test_tuning_notebook_runs_and_exports_only_figures(self):
        notebook = json.loads((
            Path(__file__).resolve().parents[1] / "xx_notebooks_tuning"
            / "ar_tuning_experiments.ipynb"
        ).read_text())
        sources = {cell["id"]: "".join(cell["source"])
                   for cell in notebook["cells"] if cell["cell_type"] == "code"}
        all_source = "\n".join(sources.values())
        for removed in ("coefficient_mae", "coefficient_wins", "coefficient_errors",
                        "estimated_phi", "nonstationary_estimate", "coefficient_epsilon",
                        "mean_minimum_width", "exact_empirical_width_low_dimension"):
            self.assertNotIn(removed, all_source)
        namespace = {}
        with contextlib.redirect_stdout(io.StringIO()), tempfile.TemporaryDirectory() as directory:
            exec(sources["imports"], namespace)
            exec(sources["settings"], namespace)
            self.assertEqual(namespace["max_iterations"], 2000)
            self.assertEqual(namespace["patience"], 200)
            namespace.update(T=48, burnin=20, max_iterations=30, patience=12,
                             scale_grid=np.array([0.1, 0.2]),
                             half_life_grid=np.array([2, 20]), OUTPUT_ROOT=Path(directory))
            with patch.object(namespace["plt"], "show"):
                for cell_id, source in sources.items():
                    if cell_id in {"imports", "settings"}:
                        continue
                    if cell_id == "e1-combined-results":
                        namespace["tune_jones_ar_realizations"] = unittest.mock.Mock(
                            side_effect=AssertionError("Comparison must reuse completed descents."))
                    with patch("Code.Code_AR_processes.ar_tools.tuning.summarize_geometric_replications",
                               side_effect=AssertionError("Selection must use width co-winners.")):
                        exec(source, namespace)
                    if cell_id == "jones-settings":
                        self.assertEqual(namespace["number_of_coefficient_realizations"], 100)
                        namespace["number_of_coefficient_realizations"] = 2
                    elif cell_id == "jones-normalized-settings":
                        namespace["jones_normalized_n_jobs"] = 1
                    elif cell_id == "e1-settings":
                        namespace["e1_n_jobs"] = 1
            expected_direction = np.random.default_rng(20261019).normal(size=4)
            expected_direction /= np.linalg.norm(expected_direction)
            np.testing.assert_array_equal(namespace["initial_direction"], expected_direction)
            for results_name, ranking_name in (
                ("jones_results", "jones_win_ranking"),
                ("jones_normalized_results", "jones_normalized_ranking"),
                ("e1_absolute_results", "e1_absolute_ranking"),
                ("e1_normalized_results", "e1_normalized_ranking"),
                ("factorial_results", "factorial_ranking"),
            ):
                results, ranking = namespace[results_name], namespace[ranking_name]
                expected_wins = dict.fromkeys(results["candidate_id"].unique(), 0)
                for _, rows in results.groupby(["replication", "T"]):
                    for candidate in rows.loc[
                        rows["minimum_width"] <= rows["minimum_width"].min() + 1e-8, "candidate_id"
                    ]:
                        expected_wins[candidate] += 1
                self.assertEqual(ranking.set_index("candidate_id")["wins"].to_dict(), expected_wins)
                expected_max = {
                    candidate: sum(rows["total_evaluations"] == 30)
                    for candidate, rows in results.groupby("candidate_id")
                }
                self.assertEqual(ranking.set_index("candidate_id")["iteration_limit_count"].to_dict(),
                                 expected_max)
                self.assertTrue(results["selected_start"].eq(1).all())
                self.assertNotIn("coefficient_mae", results.columns)
            for cases_name in ("jones_cases", "jones_normalized_cases"):
                self.assertEqual(len(namespace[cases_name]), 2)
                self.assertNotIn("coefficient_mae", namespace[cases_name].columns)
                self.assertNotIn("estimated_phi_1", namespace[cases_name].columns)
            for experiment in namespace["e1_experiments"].values():
                self.assertNotIn("coefficient_mae", experiment["cases"][0])
            self.assertEqual(len(namespace["factorial_results"]), 32)
            self.assertEqual(len(namespace["factorial_ranking"]), 16)
            max_annotations = [text for ax in namespace["factorial_axes"].flat
                               for text in ax.texts if text.get_text().startswith("MAX ")]
            self.assertEqual(len(max_annotations), 16)
            self.assertTrue(all(text.get_color() == "#c75f00" for text in max_annotations))
            self.assertEqual(len(list(Path(directory).rglob("*.pdf"))), 5)
            self.assertEqual(len(list(Path(directory).rglob("*.png"))), 5)
            self.assertEqual(list(Path(directory).rglob("*.json")), [])
            self.assertEqual(list(Path(directory).rglob("*.csv")), [])

    def test_notebook_setup_refreshes_a_cached_module_missing_new_helper(self):
        notebook = json.loads((
            Path(__file__).resolve().parents[1] / "xx_notebooks_tuning"
            / "ar_tuning_experiments.ipynb"
        ).read_text())
        setup = next("".join(cell["source"]) for cell in notebook["cells"]
                     if cell.get("id") == "imports")
        module = importlib.import_module("Code.Code_AR_processes.ar_tools.tuning")
        descent_module = importlib.import_module("Code.Code_AR_processes.ar_tools.descent")
        # Emulate a kernel that imported tuning.py before the Jones runner existed.
        del module.tune_jones_ar_realizations
        namespace = {}
        try:
            self.assertFalse(hasattr(module, "tune_jones_ar_realizations"))
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
            self.assertTrue(callable(namespace["tune_jones_ar_realizations"]))
            self.assertIs(namespace["tune_jones_ar_realizations"], module.tune_jones_ar_realizations)
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

    def test_win_frequency_ignores_score_magnitude_and_includes_zero_win_candidates(self):
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
        self.assertNotIn("mean_score", result["selected"])
        self.assertEqual([r["wins"] for r in result["ranking"]], [2, 1, 0])
        self.assertEqual(sum(r["wins"] for r in result["ranking"]), 3)
        # Equal win counts retain label order even if one candidate's miss is much larger.
        tied_rows = [dict(rows[0], candidate_id=candidate, replication=replication,
                         minimum_width=score)
                     for candidate, scores in [("A", [0., 100.]), ("B", [1., 0.])]
                     for replication, score in enumerate(scores, start=1)]
        tied = summarize_configuration_wins(tied_rows)
        self.assertEqual([(row["candidate_id"], row["wins"]) for row in tied["ranking"]],
                         [("A", 1), ("B", 1)])
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
        self.assertEqual(reversed_result["ranking"], result["ranking"])
        exact = summarize_configuration_wins(rows, epsilon=0.)
        self.assertEqual([row["candidate_id"] for row in exact["winners"]], ["A", "A", "A"])
        for invalid in [-1., np.nan, np.inf, True, "1e-8", None]:
            with self.subTest(epsilon=invalid), self.assertRaises(ValueError):
                summarize_configuration_wins(rows, epsilon=invalid)

    def test_coefficient_wins_include_epsilon_boundary_and_exclude_undefined_scores(self):
        rows = []
        for replication, scores in [(1, [0., .01, .0100000001, np.nan]),
                                    (2, [np.nan, np.inf, -np.inf, np.nan]),
                                    (3, [.03, .01, .02, np.nan])]:
            for candidate, score in zip("ABCD", scores):
                rows.append(dict(candidate_id=candidate, replication=replication, T=40,
                                 step_rule="absolute", step_value=.1, half_life=5., beta=.9,
                                 coefficient_mae=score, total_evaluations=30))
        result = summarize_configuration_wins(rows, score_key="coefficient_mae",
                                              epsilon=.01, allow_nonfinite=True)
        self.assertEqual([(r["replication"], r["candidate_id"]) for r in result["winners"]],
                         [(1, "A"), (1, "B"), (3, "B"), (3, "C")])
        self.assertEqual(result["cases"], 3)
        self.assertEqual(result["undefined_cases"], 1)
        self.assertEqual(result["case_summaries"][1]["co_winners"], 0)
        ranking = {r["candidate_id"]: r for r in result["ranking"]}
        self.assertEqual(ranking["D"]["wins"], 0)
        self.assertEqual(ranking["D"]["undefined_scores"], 3)
        self.assertEqual(ranking["B"]["win_rate"], 2 / 3)
        with self.assertRaises(ValueError):
            summarize_configuration_wins(rows, score_key="coefficient_mae", epsilon=.01)
        with self.assertRaises(ValueError):
            summarize_configuration_wins(rows[:-1], score_key="coefficient_mae",
                                          epsilon=.01, allow_nonfinite=True)

    def test_fixed_phi_paths_reproduce_paired_grid_without_external_optimum(self):
        phi = np.array([.6, -.3, .2])
        direction = np.array([.1, -.8, .3, .5])
        original = direction.copy()
        progress_output = io.StringIO()
        with patch("Code.Code_AR_processes.ar_tools.lp_widths.exact_empirical_width_low_dimension",
                   side_effect=AssertionError("External optimum must not be computed.")), \
                contextlib.redirect_stdout(progress_output):
            result = tune_fixed_ar_realizations(
                phi, 40, direction, [.1], [5], replications=2, process_seed=20261020,
                burnin=20, progress=True, **self.settings,
            )
        self.assertIn("estimated phi=", progress_output.getvalue())
        self.assertIn("coefficient MAE=", progress_output.getvalue())
        self.assertNotIn("population min=", progress_output.getvalue())
        self.assertNotIn("empirical-population gap=", progress_output.getvalue())
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
            self.assertEqual(case["grid_best_width"], direct["selected"]["minimum_width"])
            winner = direct["selected"]
            expected_phi = -np.array([winner[f"u_{j}"] for j in range(1, 4)]) / winner["u_0"]
            np.testing.assert_allclose([case[f"estimated_phi_{j}"] for j in range(1, 4)], expected_phi)
            self.assertAlmostEqual(case["coefficient_mae"], np.abs(expected_phi - phi).mean())
            self.assertNotIn("reference_width", case)
            for row in (row for row in result["results"] if row["replication"] == case["replication"]):
                history = direct["histories"][row["candidate_id"]]
                self.assertAlmostEqual(row["minimum_width"], history[-1])
                self.assertNotIn("relative_width_gap", row)
                self.assertNotIn("success", row)
                self.assertAlmostEqual(row["gamma0"], .1)
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
        self.assertEqual(len(result["results"]), 2)
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
        joint = tune_ar_sample_sizes(phi, [32, 48], direction, [.1, .2], [5], **settings)
        self.assertEqual(len(joint["results"]), 8)
        for size in [32, 48]:
            single = tune_fixed_ar_realizations(phi, size, direction, [.1, .2], [5], **settings)
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
        schedules = [dict(candidate_id=name, step_rule="absolute", step_value=.2, half_life=5.)
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
            self.assertAlmostEqual(first["gamma0"], .2)
            np.testing.assert_array_equal([first[f"u_{j}"] for j in range(4)],
                                          [second[f"u_{j}"] for j in range(4)])
        with self.assertRaisesRegex(ValueError, "absolute steps"):
            compare_ar_schedules(
                phi, [32], direction, [dict(schedules[0], step_rule="diameter_normalized")],
                replications=1, process_seed=82, burnin=20, **self.settings,
            )

    def test_tuning_folder_keeps_only_tuning_notebook(self):
        directory = Path(__file__).resolve().parents[1] / "xx_notebooks_tuning"
        self.assertEqual({path.name for path in directory.glob("*.ipynb")},
                         {"ar_tuning_experiments.ipynb"})


if __name__ == "__main__":
    unittest.main()
