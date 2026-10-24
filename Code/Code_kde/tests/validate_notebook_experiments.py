"""Run reduced notebook experiments without regenerating historical figures.

Run from Code/Code_kde with: python3 tests/validate_notebook_experiments.py
Preview overrides are applied only in memory; notebook sources are never edited.
"""

import ast
import argparse
import contextlib
import importlib.util
import io
import json
import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.linalg import eigh
from kde_tools.perron import PERRON_SOLVER_VERSION, perron_eigenpair_from_divergence
from kde_tools.fourier import FOURIER_L2_METHOD_VERSION


def load(name):
    paths = list(root.glob(f"xx_notebooks_*/{name}.ipynb"))
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected one categorized notebook for {name}.")
    return json.loads(paths[0].read_text())


def textcell(n, i):
    return "".join(n["cells"][i]["source"])


def execute(s, ns, overrides=None):
    tree = ast.parse(s)
    if overrides:

        class PreviewSettings(ast.NodeTransformer):
            def visit_Assign(self, node):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in overrides:
                        node.value = ast.parse(overrides[target.id], mode="eval").body
                return self.generic_visit(node)

        tree = PreviewSettings().visit(tree)
        ast.fix_missing_locations(tree)
    exec(compile(tree, "<notebook validation>", "exec"), ns)


def definitions(n, ns):
    for c in n["cells"]:
        if c["cell_type"] != "code":
            continue
        tree = ast.parse("".join(c["source"]))
        tree.body = [
            x for x in tree.body if isinstance(x, (ast.FunctionDef, ast.ClassDef))
        ]
        if tree.body:
            exec(compile(tree, "<notebook definitions>", "exec"), ns)


def setup(name, pdf_root=None):
    import matplotlib.pyplot as plt
    from kde_tools.pdf_export import enable_automatic_pdf_saving

    plt.close("all")
    n = load(name)
    ns = {"__name__": "__main__", "display": lambda value: None}
    if pdf_root is None:
        execute(textcell(n, 1), ns)
    else:
        with patch(
            "kde_tools.pdf_export.enable_automatic_pdf_saving",
            side_effect=lambda notebook_name, **kwargs: enable_automatic_pdf_saving(
                notebook_name, output_root=pdf_root
            ),
        ):
            execute(textcell(n, 1), ns)
    return n, ns


def strip_plot(s, marker="fig, ax = plt.subplots"):
    if marker in s:
        return s.split(marker, 1)[0]
    for node in ast.parse(s).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            f = node.value.func
            if (
                isinstance(f, ast.Attribute)
                and isinstance(f.value, ast.Name)
                and f.value.id == "plt"
                and f.attr in ["subplots", "figure"]
            ):
                return "".join(s.splitlines(keepends=True)[: node.lineno - 1])
    return s


def main(report_path=None):
    report = {"solver_version": PERRON_SOLVER_VERSION,
              "l2_method": FOURIER_L2_METHOD_VERSION, "notebooks": {}, "checks": []}
    unit_log = io.StringIO()
    unit_result = unittest.TextTestRunner(stream=unit_log).run(
        unittest.TestLoader().discover(str(root / "tests"))
    )
    if not unit_result.wasSuccessful():
        raise RuntimeError(unit_log.getvalue())
    report["unit_tests"] = {"count": unit_result.testsRun, "status": "passed"}

    with tempfile.TemporaryDirectory(prefix="kde_perron_validation_") as td:
        scratch = Path(td)
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            name = "spectral_scale_selection_kde"
            n, ns = setup(name, scratch / "pdfs")
            execute(strip_plot(textcell(n, 3)), ns)
            execute(strip_plot(textcell(n, 5)), ns)
            d = ns["stationary_example_lambda_divergence"]
            references = []
            for lam in ns["stationary_example_lambda_values"]:
                _, v = eigh(np.exp(-lam * d), subset_by_index=[len(d) - 1, len(d) - 1])
                references.append(int(np.argmax(v[:, 0] / v[:, 0].sum())))
            assert np.array_equal(
                references, ns["stationary_example_lambda_selected_indices"]
            )
            report["checks"].append(
                "All 101 Figure 3.3 choices match dense reference solves."
            )
            execute(textcell(n, 7), ns)
            execute(
                strip_plot(textcell(n, 10)),
                ns,
                {"stationary_example_grid_fineness_N_values": "np.array([11,31,201])"},
            )
            ns.update(validation_data_root=scratch / "histogram_data",
                      validation_image_root=scratch / "histogram_images",
                      kde_thesis_figure_dir=scratch / "thesis_figures")
            ns["kde_thesis_figure_dir"].mkdir()
            histogram_overrides = {
                "beta2_n": "30", "beta2_N": "21", "beta2_M": "3",
                "beta2_data_root": "validation_data_root",
                "beta2_image_root": "validation_image_root",
            }
            execute(textcell(n, 12), ns, histogram_overrides)
            execute(textcell(n, 13), ns)
            baseline_choices = ns["beta2_run"]["selected_indices"].copy()
            execute(textcell(n, 14), ns)
            canonical_figure = ns["kde_thesis_figure_dir"] / "distribution_of_chosen_bandwidths.pdf"
            canonical_hash = hashlib.sha256(canonical_figure.read_bytes()).hexdigest()
            with patch("kde_tools.distribution_runs.perron_eigenpair_from_divergence",
                       side_effect=AssertionError("A completed histogram checkpoint was recomputed.")):
                execute(textcell(n, 13), ns)
            assert ns["beta2_run"]["validation"]["reused_repetitions"] == 3
            execute(textcell(n, 12), ns, histogram_overrides | {"beta2_profile": "'localized'"})
            execute(textcell(n, 13), ns)
            np.testing.assert_array_equal(ns["beta2_run"]["selected_indices"][:, 1:], baseline_choices[:, 1:])
            assert ns["beta2_run"]["validation"]["reused_control_source"] is not None
            execute(textcell(n, 14), ns)
            assert hashlib.sha256(canonical_figure.read_bytes()).hexdigest() == canonical_hash
            report["checks"].append(
                "Both main-notebook histogram profiles run and export PDFs; checkpoints resume, "
                "localized controls reuse matching baseline results, and the localized plot preserves Figure 3.5."
            )
            pdf_count = len(list(ns["pdf_output_directory"].glob("*.pdf")))
            execute(textcell(n, 12), ns, histogram_overrides | {
                "beta2_profile": "'exact_gaussian'", "beta2_gaussian_run_profile": "'preview'",
            })
            execute(textcell(n, 13), ns)
            exact_result = ns["beta2_run"]
            from kde_tools.similarities import beta2_divergence_matrix
            from kde_tools.density import exact_oracle_ise_values
            for repetition in range(3):
                sample = np.random.default_rng(910000+repetition).normal(size=30)
                spectral = perron_eigenpair_from_divergence(beta2_divergence_matrix(sample, ns["beta2_h_grid"]), 1).selected_index
                errors = exact_oracle_ise_values(sample, ns["beta2_h_grid"], 1)
                choices = [spectral, int(np.argmin(errors))]
                np.testing.assert_array_equal(exact_result["selected_indices"][repetition], choices)
                np.testing.assert_array_equal(exact_result["selected_ise_values"][repetition], errors[choices])
            execute(textcell(n, 14), ns)
            assert len(list(ns["pdf_output_directory"].glob("*.pdf"))) == pdf_count + 3
            assert len(list(ns["beta2_image_directory"].glob("*.png"))) == 3
            assert hashlib.sha256(canonical_figure.read_bytes()).hexdigest() == canonical_hash
            with patch("kde_tools.distribution_runs.perron_eigenpair_from_divergence",
                       side_effect=AssertionError("Repeated Gaussian benchmark")):
                execute(textcell(n, 13), ns)
            assert ns["beta2_run"]["validation"]["reused_repetitions"] == 3
            report["checks"].append("Main exact-Gaussian profile preserves independent seeds, both selections, selected ISE values, three plots, and checkpoint reuse.")
            execute(
                textcell(n, 16),
                ns,
                {
                    "beta2_n_sweep_values": "np.array([20,40])",
                    "beta2_n_sweep_N": "21",
                    "beta2_n_sweep_M": "3",
                    "beta2_n_sweep_output_root": "validation_data_root / 'sample_size_sweep'",
                    "beta2_n_sweep_cache_root": "validation_data_root / 'shared_spectral_sweeps'",
                },
            )
            main_sweep_samples = ns["beta2_n_sweep_full_samples"].copy()
            main_sweep_grid = ns["beta2_n_sweep_h_grid"].copy()
            main_sweep_settings = ns["beta2_n_sweep_settings"].copy()
            main_sweep_result = ns["beta2_n_sweep_spectral_results"][20]
            main_sweep_cache = ns["beta2_n_sweep_cache_root"]
            shared_sweep_runner = ns["run_l2_point"]
            execute(
                textcell(n, 21),
                ns,
                {"perron_comparison_n": "40", "perron_comparison_N": "31"},
            )
            execute(
                textcell(n, 21),
                ns,
                {
                    "perron_comparison_n": "40",
                    "perron_comparison_N": "31",
                    "perron_comparison_lambda": ".7",
                },
            )
            expected_first_order = ns["first_order_perron_vector"](
                ns["perron_comparison_population_divergence"], 0.7
            )
            np.testing.assert_array_equal(
                ns["perron_comparison_first_order_vector"], expected_first_order
            )
            report["notebooks"][name] = {
                "cells": [3, 5, 7, 10, 12, 13, 14, 16, 21],
                "histogram_profiles": ["figure_3_5", "localized", "exact_gaussian"],
                "figure_3_3_points": 101,
                "status": "passed",
            }

            name = "experiment_1_joint_N_lambda_limit"
            n, ns = setup(name, scratch / "pdfs")
            ns.update(
                N_DATA=40,
                N_VALUES=np.array([31, 101]),
                pdf_output_directory=scratch / "pdfs" / name,
            )
            execute(textcell(n, 2), ns)
            execute(strip_plot(textcell(n, 4)), ns)
            execute(textcell(n, 6), ns)
            execute(textcell(n, 7), ns)
            execute(textcell(n, 13), ns)
            execute(
                strip_plot(textcell(n, 15)),
                ns,
                {"MC_REPETITIONS": "3", "MC_N": "31", "MC_SAMPLE_SIZE": "30"},
            )
            execute(
                textcell(n, 17),
                ns,
                {
                    "MESH_STRESS_N": "31",
                    "MESH_GAP_FRACTIONS": "np.array([1.,.6,.4,.3,.2,.1])",
                },
            )
            execute(
                textcell(n, 25),
                ns,
                {
                    "MESH_STRESS_N": "31",
                    "MESH_GAP_FRACTIONS": "np.array([1.,.6,.4,.3,.2,.1])",
                },
            )
            execute(
                textcell(n, 31),
                ns,
                {"REPEATED_SAMPLE_M": "3", "REPEATED_SAMPLE_N": "31",
                 "joint_spectral_cache_root": "pdf_output_directory.parents[1] / 'data/shared_spectral_sweeps'"},
            )
            execute(strip_plot(textcell(n, 32), "repeated_histogram = "), ns)
            execute(
                textcell(n, 35),
                ns,
                {
                    "SAMPLE_SIZE_SWEEP_VALUES": "np.array([20,40])",
                    "SAMPLE_SIZE_SWEEP_M": "3",
                    "SAMPLE_SIZE_SWEEP_N": "31",
                },
            )
            joint_choices = ns["repeated_selected_index"].copy()
            joint_diagnostics = ns["repeated_diagnostics"].copy()
            with patch("kde_tools.spectral_sweeps.beta2_divergence_matrix_fourier",
                       side_effect=AssertionError("Repeated joint-limit sweep")):
                execute(textcell(n, 31), ns, {
                    "REPEATED_SAMPLE_M": "3", "REPEATED_SAMPLE_N": "31",
                    "joint_spectral_cache_root": "pdf_output_directory.parents[1] / 'data/shared_spectral_sweeps'",
                })
                execute(textcell(n, 35), ns, {
                    "SAMPLE_SIZE_SWEEP_VALUES": "np.array([20,40])",
                    "SAMPLE_SIZE_SWEEP_M": "3", "SAMPLE_SIZE_SWEEP_N": "31",
                })
            np.testing.assert_array_equal(joint_choices, ns["repeated_selected_index"])
            np.testing.assert_array_equal(joint_diagnostics, ns["repeated_diagnostics"])
            assert "solver_matvecs" in ns["repeated_sample_results"].columns
            assert "power_iterations" not in ns["repeated_sample_results"].columns
            assert set(ns["repeated_sample_results"]["perron_solver"]) == {
                PERRON_SOLVER_VERSION
            }
            report["notebooks"][name] = {
                "cells": [2, 4, 6, 7, 13, 15, 17, 25, 31, 32, 35],
                "status": "passed",
            }

            h = np.geomspace(0.05, 1.0, 11)
            name = "fast_l2_bandwidth_distributions_vs_n_lambda"
            n, ns = setup(name, scratch / "pdfs")
            definitions(n, ns)
            assert ns["run_l2_point"] is shared_sweep_runner
            with patch("kde_tools.spectral_sweeps.beta2_divergence_matrix_fourier",
                       side_effect=AssertionError("The other notebook recomputed a shared sweep.")):
                shared_result = ns["run_l2_point"](
                    20, main_sweep_samples, main_sweep_grid, main_sweep_settings,
                    [1.0], scratch / "fast_shared_sweep", cache_root=main_sweep_cache,
                )
            np.testing.assert_array_equal(shared_result["selected_indices"], main_sweep_result["selected_indices"])
            assert shared_result["divergence_evaluations"] == 0
            report["checks"].append(
                "Main and fast notebooks use the same spectral runner and reuse a matching sweep without recomputing divergences."
            )
            ns.update(data_output_directory=scratch / "fast_public_data",
                      image_output_directory=scratch / "fast_public_images",
                      validation_sweep_cache=main_sweep_cache)
            execute(textcell(n, 3), ns, {
                "RUN_PROFILE": "'preview'",
                "PROFILES": "{'preview': {'repetitions': 3, 'grid_size': 21}}",
                "SAMPLE_SIZES": "[20, 40]",
                "spectral_sweep_cache_root": "validation_sweep_cache",
            })
            execute(textcell(n, 7), ns)
            with patch("kde_tools.spectral_sweeps.beta2_divergence_matrix_fourier",
                       side_effect=AssertionError("The fast notebook repeated the main sweep.")):
                execute(textcell(n, 9), ns)
            execute(textcell(n, 11), ns)
            assert all(result["divergence_evaluations"] == 0 for result in ns["results_by_n"].values())
            execute(textcell(n, 15), ns, {"ADDITIONAL_LAMBDA_VALUES": "[1.0, 1000.0]"})
            execute(textcell(n, 16), ns)
            execute(textcell(n, 18), ns)
            assert all(result["reused_selections"] == 3 and result["divergence_evaluations"] == 3
                       for result in ns["additional_results_by_n"].values())
            settings = {
                "repetitions": 2,
                "maximum_sample_size": 40,
                "fft_half_width": 12.0,
                "fft_spatial_points": 8192,
            }
            full = np.random.default_rng(20260830).normal(size=(2, 40))
            first = ns["run_l2_point"](
                20, full, h, settings, [1.0, 1e9], scratch / name, reuse=False
            )
            resumed = ns["run_l2_point"](
                20, full, h, settings, [1.0, 1e9], scratch / name, reuse=True
            )
            assert np.array_equal(
                first["selected_indices"], resumed["selected_indices"]
            )
            cp = scratch / name / "n20" / "checkpoint.npz"
            with np.load(cp, allow_pickle=False) as z:
                payload = {k: z[k].copy() for k in z.files}
            current = json.loads(str(payload["configuration_json"].item()))
            assert current["l2_method"] == FOURIER_L2_METHOD_VERSION
            for obsolete_field in ("perron_solver", "l2_method"):
                old = dict(current)
                old[obsolete_field] = "legacy"
                payload["configuration_json"] = np.array(json.dumps(old, sort_keys=True))
                np.savez_compressed(cp, **payload)
                try:
                    ns["run_l2_point"](
                        20, full, h, settings, [1.0, 1e9], scratch / name, reuse=True
                    )
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"Old {obsolete_field} checkpoint was accepted.")
            ns.update(
                overlay_settings=dict(ns["settings"]), overlay_repetitions=3,
                overlay_h_grid=ns["h_grid"].copy(), overlay_sample_sizes=[20,40],
                overlay_common_configuration={"settings": dict(ns["settings"])}, OVERLAY_PROFILE="preview",
                validation_continuous_cache=scratch / "shared_continuous",
            )
            execute(textcell(n, 23), ns, {
                "QHAT_SCAN_POINTS": "65", "QHAT_SPATIAL_POINTS": "4096",
                "continuous_cache_root": "validation_continuous_cache",
                "continuous_import_directories": "[]",
            })
            fast_qhat_results = ns["overlay_qhat_results"]
            report["notebooks"][name] = {
                "cross_notebook_sweep_reuse": "passed",
                "public_sweep_cells_and_csv_exports": "passed",
                "additional_lambdas_reuse_completed_columns": "passed",
                "checkpoint_resume": "passed",
                "legacy_checkpoint_rejection": "passed",
                "status": "passed",
            }

            name = "continuous_ise_qhat_distributions_vs_n"
            n, ns = setup(name, scratch / "pdfs")
            ns.update(data_output_directory=scratch / "continuous_data",
                      image_output_directory=scratch / "continuous_images",
                      validation_continuous_cache=scratch / "shared_continuous")
            execute(textcell(n, 3), ns, {
                "RUN_PROFILE": "'preview'", "PROFILES": "{'preview': {'repetitions': 3}}",
                "SAMPLE_SIZES": "[20,40]", "SAMPLE_BANK_REPETITIONS": "3",
                "SAMPLE_BANK_BLOCK_SIZE": "40", "SAMPLE_BANK_MAXIMUM_N": "80",
                "SCAN_POINTS": "65", "SPATIAL_POINTS": "4096",
                "continuous_cache_root": "validation_continuous_cache",
                "continuous_import_directories": "[]",
            })
            with patch("kde_tools.continuous_runs.continuous_qhat_minimum",
                       side_effect=AssertionError("Continuous notebook repeated fast Q0")):
                execute(textcell(n, 5), ns)
            for current_n in [20,40]:
                np.testing.assert_array_equal(ns["results_by_n"][current_n][:, 1, 0], fast_qhat_results[current_n]["h"])
            with patch("kde_tools.continuous_runs.continuous_ise_qr_minima",
                       side_effect=AssertionError("Repeated continuous checkpoint")):
                execute(textcell(n, 5), ns)
            for i in (7,9,11):
                execute(textcell(n, i), ns)
            report["notebooks"][name] = {
                "cross_notebook_q0_reuse": "passed", "checkpoint_resume": "passed",
                "csv_refinement_and_pdf_exports": "passed", "status": "passed",
            }

        analysis_path = (
            root / "outputs/analysis/fast_l2_lambda_comparison/analyse_saved_results.py"
        )
        analysis_spec = importlib.util.spec_from_file_location(
            "saved_results_analysis", analysis_path
        )
        analysis = importlib.util.module_from_spec(analysis_spec)
        analysis_spec.loader.exec_module(analysis)
        d = np.array([[0.0, 1.0, 4.0], [1.0, 0.0, 3.0], [4.0, 3.0, 0.0]])
        for lam in (1.0, 1000.0, 1e9):
            assert (
                analysis.population_choice(d, lam)
                == perron_eigenpair_from_divergence(d, lam).selected_index
            )
        report["checks"].append(
            "Saved-results population references use the shared selector, including extreme lambda."
        )
        report["checks"].append(
            "Four notebook workflows passed reduced numerical execution, including all three main histogram profiles and both continuous views."
        )
        report["checks"].append(
            "Current checkpoints resume; legacy solver and L2-method checkpoints are rejected."
        )
        report["captured_execution_log"] = captured.getvalue().splitlines()
    output = (Path(report_path) if report_path is not None else
              root / "outputs/analysis/perron_refactor_validation/validation_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "captured_execution_log"}, indent=2
        )
    )
    print("Validation report:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Save this validation report separately.")
    main(parser.parse_args().output)
