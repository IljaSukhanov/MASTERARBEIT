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
    return json.loads((root / (name + ".ipynb")).read_text())


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
            execute(
                textcell(n, 12), ns, {"beta2_n": "30", "beta2_N": "21", "beta2_M": "3"}
            )
            execute(textcell(n, 13), ns)
            execute(
                textcell(n, 16),
                ns,
                {
                    "beta2_n_sweep_values": "np.array([20,40])",
                    "beta2_n_sweep_N": "21",
                    "beta2_n_sweep_M": "3",
                },
            )
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
                "cells": [3, 5, 7, 10, 12, 13, 16, 21],
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
                {"REPEATED_SAMPLE_M": "3", "REPEATED_SAMPLE_N": "31"},
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
            name = "gaussian_symmetric_divergence_bandwidth_distributions"
            n, ns = setup(name, scratch / "pdfs")
            from kde_tools.pdf_export import enable_automatic_pdf_saving

            ns.update(
                data_output_directory=scratch / "data" / name,
                image_output_directory=scratch / "images" / name,
                pdf_output_directory=enable_automatic_pdf_saving(
                    name, output_root=scratch / "pdfs"
                ),
            )
            ns["image_output_directory"].mkdir(parents=True)
            execute(
                textcell(n, 3), ns,
                {
                    "RUN_PROFILE": '"preview"',
                    "PROFILES": '{"preview": {"sample_size": 20, "repetitions": 2, "grid_size": 11}}',
                },
            )
            execute(textcell(n, 5), ns)
            sample = np.random.default_rng(7).normal(size=20)
            choices, errors = ns["select_one_sample"](sample, ns["h_grid"], ns["settings"])
            assert set(choices) == {"L2", "ISE"}
            reference_divergence = ns["beta2_divergence_matrix"](sample, ns["h_grid"])
            assert choices["L2"] == perron_eigenpair_from_divergence(
                reference_divergence, ns["settings"]["lambda_value"]
            ).selected_index
            assert choices["ISE"] == int(np.argmin(ns["exact_oracle_ise_values"](
                sample, ns["h_grid"], ns["settings"]["sigma2"]
            )))
            execute(textcell(n, 7), ns)
            first = ns["selected_indices"].copy()
            execute(textcell(n, 7), ns)
            np.testing.assert_array_equal(first, ns["selected_indices"])
            assert ns["completed"] == ns["settings"]["repetitions"]
            # Reject stale configurations before their saved choices are loaded.
            current_configuration = ns["configuration_json"]
            ns["configuration_json"] = '{"obsolete_configuration": true}'
            try:
                execute(textcell(n, 7), ns)
            except ValueError:
                pass
            else:
                raise AssertionError("Obsolete distribution checkpoint was accepted.")
            ns["configuration_json"] = current_configuration
            execute(textcell(n, 7), ns)
            for i in (9, 11, 13):
                execute(textcell(n, i), ns)
            saved_settings = json.loads((ns["run_directory"] / "settings.json").read_text())
            assert saved_settings["methods"] == ["L2", "ISE"]
            assert len(list(ns["pdf_output_directory"].glob("*.pdf"))) == 3
            report["notebooks"][name] = {
                "methods": list(choices),
                "checkpoint_resume": "passed",
                "obsolete_checkpoint_rejection": "passed",
                "csv_and_pdf_exports": "passed",
                "status": "passed",
            }

            name = "fast_l2_bandwidth_distributions_vs_n_lambda"
            n, ns = setup(name, scratch / "pdfs")
            definitions(n, ns)
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
            report["notebooks"][name] = {
                "checkpoint_resume": "passed",
                "legacy_checkpoint_rejection": "passed",
                "status": "passed",
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
            "Four notebook experiment paths passed reduced-size numerical execution."
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
