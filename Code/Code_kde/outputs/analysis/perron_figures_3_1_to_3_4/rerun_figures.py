"""Regenerate thesis Figures 3.1--3.4 and compare with their saved PDFs.

Execute the current notebook cells without changing their settings or
notebook files. Previous plotted data are recovered by the companion PDF
extraction script. All displayed figures use the shared PDF exporter.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

DIRECTORY = Path(__file__).resolve().parent
CODE = DIRECTORY.parents[2]
sys.path.insert(0, str(CODE))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import kde_tools.pdf_export as pdf_export
from kde_tools.perron import PERRON_SOLVER_VERSION, perron_eigenpair_from_divergence


def execute(notebook, cell, namespace, *, calculation_only=False):
    source = "".join(notebook["cells"][cell]["source"])
    if calculation_only:
        source = source.split("fig, ax = plt.subplots", 1)[0]
    exec(
        compile(source, f"<{namespace['notebook_name']} cell {cell}>", "exec"),
        namespace,
    )


def setup(name):
    paths = list(CODE.glob(f"xx_notebooks_*/{name}.ipynb"))
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected one categorized notebook for {name}.")
    notebook = json.loads(paths[0].read_text())
    namespace = {
        "__name__": "__main__",
        "display": lambda value: None,
        "notebook_name": name,
    }
    execute(notebook, 1, namespace)
    return notebook, namespace


def compare_choices(name, old, indices, bandwidths, labels, positions):
    previous_indices = np.asarray(old[name]["selected_indices"])
    previous_bandwidths = np.asarray(old[name]["selected_bandwidths"])
    assert previous_indices.shape == indices.shape
    records, summaries = [], []
    for row, label in enumerate(labels):
        difference = indices[row] - previous_indices[row]
        summaries.append(
            {
                "series": str(label),
                "points": len(positions),
                "changed_points": int(np.count_nonzero(difference)),
                "maximum_grid_index_change": int(np.max(np.abs(difference))),
                "maximum_absolute_bandwidth_change": float(
                    np.max(np.abs(bandwidths[row] - previous_bandwidths[row]))
                ),
                "maximum_relative_bandwidth_change": float(
                    np.max(np.abs(bandwidths[row] / previous_bandwidths[row] - 1))
                ),
                "old_final_bandwidth": float(previous_bandwidths[row, -1]),
                "new_final_bandwidth": float(bandwidths[row, -1]),
            }
        )
        for column, position in enumerate(positions):
            records.append(
                {
                    "series": str(label),
                    "position": float(position),
                    "old_index": int(previous_indices[row, column]),
                    "new_index": int(indices[row, column]),
                    "grid_index_change": int(difference[column]),
                    "old_bandwidth": float(previous_bandwidths[row, column]),
                    "new_bandwidth": float(bandwidths[row, column]),
                }
            )
    pd.DataFrame(records).to_csv(DIRECTORY / f"{name}_comparison.csv", index=False)
    return {
        "total_points": int(indices.size),
        "changed_points": int(np.count_nonzero(indices - previous_indices)),
        "series": summaries,
        "previous_pdf_grid_recovery_error": old[name]["grid_snap_max_error"],
    }


def main():
    previous = json.loads((DIRECTORY / "previous_figure_data.json").read_text())
    report = {
        "perron_solver": PERRON_SOLVER_VERSION,
        "comparison_baseline": "Marker values recovered from the four thesis PDFs saved before this rerun.",
        "figures": {},
    }
    started = time.perf_counter()
    notebook, namespace = setup("spectral_scale_selection_kde")

    print("Regenerating Figure 3.1: four Perron profiles, N=191.", flush=True)
    execute(notebook, 3, namespace)
    namespace["fig"].savefig(
        namespace["kde_thesis_figure_dir"] / "stationary_distribution_plot_001.png",
        dpi=100,
    )
    vectors = np.array(
        [
            namespace["stationary_example_perron_vectors"][value]
            for value in namespace["stationary_example_lambdas"]
        ]
    )
    old_vectors = np.asarray(previous["figure_3_1"]["recovered_values"])
    profile_rows = []
    for row, value in enumerate(namespace["stationary_example_lambdas"]):
        profile_rows.append(
            {
                "lambda": float(value),
                "maximum_absolute_component_difference": float(
                    np.max(abs(vectors[row] - old_vectors[row]))
                ),
                "old_peak_index": int(np.argmax(old_vectors[row])),
                "new_peak_index": int(
                    namespace["stationary_example_peak_indices"][value]
                ),
                "selected_bandwidth": float(
                    namespace["stationary_example_peak_bandwidths"][value]
                ),
            }
        )
    report["figures"]["figure_3_1"] = {
        "profiles": profile_rows,
        "maximum_absolute_component_difference": float(
            np.max(abs(vectors - old_vectors))
        ),
    }
    pd.DataFrame(profile_rows).to_csv(
        DIRECTORY / "figure_3_1_comparison.csv", index=False
    )
    np.savez_compressed(
        DIRECTORY / "figure_3_1_vectors.npz",
        previous=old_vectors,
        current=vectors,
        h_grid=namespace["stationary_example_grid"],
    )
    plt.close("all")

    print("Regenerating Figure 3.3: all 101 lambda values.", flush=True)
    execute(notebook, 5, namespace)
    namespace["fig"].savefig(
        namespace["kde_thesis_figure_dir"]
        / "chosen_bandwidth_vs_lambda_M1_fixed_sample.png",
        dpi=100,
    )
    report["figures"]["figure_3_3"] = compare_choices(
        "figure_3_3",
        previous,
        namespace["stationary_example_lambda_selected_indices"][None, :],
        namespace["stationary_example_lambda_selected_bandwidths"][None, :],
        ["fixed N=201"],
        namespace["stationary_example_lambda_values"],
    )
    report["figures"]["figure_3_3"]["closest_pair_bandwidths"] = namespace[
        "stationary_example_lambda_closest_pair_bandwidths"
    ].tolist()
    plt.close("all")

    print("Regenerating Figure 3.2: all 199 grids and four lambda values.", flush=True)
    # The omitted Monte Carlo plot occupies slot 003 in the complete notebook.
    pdf_export._ACTIVE_EXPORTER._counter = itertools.count(4)
    execute(notebook, 10, namespace)
    namespace["fig"].savefig(
        namespace["kde_thesis_figure_dir"] / "grid_fineness_chosen_bandwidths.png",
        dpi=100,
    )
    report["figures"]["figure_3_2"] = compare_choices(
        "figure_3_2",
        previous,
        namespace["stationary_example_grid_fineness_selected_indices"],
        namespace["stationary_example_grid_fineness_selected_bandwidths"],
        namespace["stationary_example_grid_fineness_lambdas"],
        namespace["stationary_example_grid_fineness_N_values"],
    )
    plt.close("all")

    print("Regenerating Figure 3.4: N=100,...,3000 and all six regimes.", flush=True)
    notebook, namespace = setup("experiment_1_joint_N_lambda_limit")
    execute(notebook, 2, namespace)
    execute(notebook, 4, namespace, calculation_only=True)
    execute(notebook, 6, namespace)
    original_divergence = namespace["beta2_divergence_matrix_fourier"]
    diagnostics = []

    def divergence_with_progress(samples, h_grid, **kwargs):
        print(f"Figure 3.4: computing grid N={len(h_grid)}", flush=True)
        return original_divergence(samples, h_grid, **kwargs)

    def vector_with_diagnostics(divergence, lambda_value):
        result = perron_eigenpair_from_divergence(divergence, lambda_value)
        diagnostics.append(
            {
                "N": len(divergence),
                "lambda": float(lambda_value),
                "relative_residual": result.relative_residual,
                "relative_gap": result.relative_gap,
                "matvecs": result.matvecs,
            }
        )
        return result.vector

    namespace["beta2_divergence_matrix_fourier"] = divergence_with_progress
    namespace["perron_vector_from_divergence"] = vector_with_diagnostics
    execute(notebook, 7, namespace)
    execute(notebook, 8, namespace)
    # Figure 3.4 is slot 003 in the full notebook's automatic PDF exports.
    pdf_export._ACTIVE_EXPORTER._counter = itertools.count(3)
    execute(notebook, 10, namespace)
    plt.close("all")
    data = namespace["results"]
    labels = namespace["REGIME_ORDER"]
    sizes = namespace["N_VALUES"]
    indices = np.array(
        [
            data[data["regime"] == label].sort_values("N")["selected_index"].to_numpy()
            for label in labels
        ]
    )
    bandwidths = np.array(
        [
            data[data["regime"] == label].sort_values("N")["selected_h"].to_numpy()
            for label in labels
        ]
    )
    report["figures"]["figure_3_4"] = compare_choices(
        "figure_3_4", previous, indices, bandwidths, labels, sizes
    )
    report["figures"]["figure_3_4"]["continuous_reference"] = namespace["h_infinity"]
    report["figures"]["figure_3_4"]["maximum_relative_residual"] = max(
        row["relative_residual"] for row in diagnostics
    )
    data.to_csv(DIRECTORY / "figure_3_4_results.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(
        DIRECTORY / "figure_3_4_solver_diagnostics.csv", index=False
    )

    outputs = {
        "figure_3_1": CODE
        / "outputs/images/thesis/stationary_distribution_plot_001.pdf",
        "figure_3_2": CODE
        / "outputs/images/thesis/grid_fineness_chosen_bandwidths.pdf",
        "figure_3_3": CODE
        / "outputs/images/thesis/chosen_bandwidth_vs_lambda_M1_fixed_sample.pdf",
        "figure_3_4": CODE
        / "outputs/pdfs/experiment_1_joint_n_lambda_limit/003_finite_selector_versus_continuous_reference.pdf",
    }
    for name, path in outputs.items():
        report["figures"][name].update(
            output_pdf=str(path),
            before_sha256=hashlib.sha256(
                (DIRECTORY / "before" / f"{name}.pdf").read_bytes()
            ).hexdigest(),
            after_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    report["elapsed_seconds"] = time.perf_counter() - started
    (DIRECTORY / "comparison_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
