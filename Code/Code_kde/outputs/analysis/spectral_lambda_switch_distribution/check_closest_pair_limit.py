"""Check the fixed-h-grid closest-pair limit in the 100 archived samples.

References: thesis Section 3.5.1 and Yu, Wang, Samworth, Theorem 2:
https://arxiv.org/pdf/1405.0680

Our matrix bound: if d* is a unique off-diagonal minimum, and delta is
the gap to the second-smallest upper-triangle entry, then the remainder
from the limiting single-edge adjacency matrix has operator norm at most
(N-1)*exp(-lambda*delta). This uses its maximum absolute row sum.
The bound here is evaluated on the computed divergence matrix; it is not
an interval certificate for exact Gaussian integrals.
"""
from pathlib import Path
import json
import os
import sys

CODE_DIRECTORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE_DIRECTORY))
os.environ.setdefault("MPLCONFIGDIR", str(CODE_DIRECTORY / ".matplotlib-cache"))

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from kde_tools.fourier import beta2_divergence_matrix_fft
from kde_tools.perron import perron_eigenpair_from_divergence

DEFAULT_RUN_DIRECTORY = CODE_DIRECTORY / "outputs/data/spectral_lambda_switch_distribution/59e74dc1d05b"


def run_checks(run_directory=DEFAULT_RUN_DIRECTORY):
    run_directory = Path(run_directory)
    output_directory = run_directory / "closest_pair_limit"
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads((run_directory / "checkpoint.json").read_text())
    records = {r["repetition"]: r for r in checkpoint["records"]}
    settings = checkpoint["settings"]
    with np.load(run_directory / "samples_and_grids.npz", allow_pickle=False) as source:
        samples, h_grid = source["samples"].copy(), source["bandwidth_grid"].copy()
    example_ids = [r["repetition"] for r in json.loads((run_directory / "ten_example_paths.json").read_text())["samples"]]
    rows, endpoint_vectors, terminal_vectors, targets = [], [], [], []
    error_target = 1e-10
    upper_indices = np.triu_indices(len(h_grid), k=1)
    with threadpool_limits(limits=1):
        for repetition, sample in enumerate(samples):
            divergence = beta2_divergence_matrix_fft(sample, h_grid,
                half_width=settings["fft_half_width"], spatial_points=settings["fft_spatial_points"])
            values = divergence[upper_indices]
            order = np.argsort(values)
            j, k = int(upper_indices[0][order[0]]), int(upper_indices[1][order[0]])
            minimum, next_minimum = float(values[order[0]]), float(values[order[1]])
            delta = next_minimum - minimum
            unique = delta > 0
            if not unique:
                raise ValueError(f"Sample {repetition} has a tied minimum; single-edge limit does not apply.")
            endpoint = perron_eigenpair_from_divergence(divergence, 1e4)
            assert endpoint.selected_index == records[repetition]["coarse_selected_indices"][-1]
            # Conservative terminal lambda: all other edge weights <=
            # error_target/(N-1), hence ||B(lambda)-B(infinity)||_2 <= target.
            terminal_lambda = max(1e4, float(np.log((len(h_grid) - 1) / error_target) / delta))
            terminal = perron_eigenpair_from_divergence(divergence, terminal_lambda)
            target = np.zeros(len(h_grid))
            target[[j, k]] = 0.5
            terminal_error = float(np.abs(terminal.vector - target).sum())
            assert terminal_error < 1e-8
            assert terminal.selected_index in (j, k)
            rows.append({"repetition": repetition, "is_displayed_example": repetition in example_ids,
                         "unique_minimum_pair": unique, "adjacent_pair": k == j + 1,
                         "lower_index": j, "upper_index": k,
                         "lower_pair_h": float(h_grid[j]), "upper_pair_h": float(h_grid[k]),
                         "minimum_divergence": minimum, "gap_to_next_pair": delta,
                         "h_selected_at_10000": float(h_grid[endpoint.selected_index]),
                         "selected_at_10000_in_pair": endpoint.selected_index in (j, k),
                         "perron_mass_on_pair_at_10000": float(endpoint.vector[[j, k]].sum()),
                         "l1_vector_error_at_10000": float(np.abs(endpoint.vector - target).sum()),
                         "terminal_lambda": terminal_lambda,
                         "terminal_matrix_remainder_bound": float((len(h_grid) - 1) * np.exp(-terminal_lambda * delta)),
                         "terminal_selected_h": float(h_grid[terminal.selected_index]),
                         "terminal_perron_mass_on_pair": float(terminal.vector[[j, k]].sum()),
                         "terminal_l1_vector_error": terminal_error,
                         "terminal_relative_residual": terminal.relative_residual})
            endpoint_vectors.append(endpoint.vector)
            terminal_vectors.append(terminal.vector)
            targets.append(target)
    table = pd.DataFrame(rows)
    table.to_csv(output_directory / "sample_results.csv", index=False)
    example_table = table.set_index("repetition").loc[example_ids].reset_index()
    example_table.to_csv(output_directory / "ten_example_results.csv", index=False)
    np.savez_compressed(output_directory / "vectors.npz", bandwidth_grid=h_grid,
                        endpoint_vectors=endpoint_vectors, terminal_vectors=terminal_vectors,
                        limiting_vectors=targets)
    summary = {"samples": len(rows), "unique_closest_pairs": int(table["unique_minimum_pair"].sum()),
               "adjacent_closest_pairs": int(table["adjacent_pair"].sum()),
               "selected_at_10000_in_closest_pair": int(table["selected_at_10000_in_pair"].sum()),
               "ten_examples_selected_at_10000_in_pair": int(example_table["selected_at_10000_in_pair"].sum()),
               "pair_mass_at_10000": {"minimum": float(table["perron_mass_on_pair_at_10000"].min()),
                                      "median": float(table["perron_mass_on_pair_at_10000"].median()),
                                      "maximum": float(table["perron_mass_on_pair_at_10000"].max())},
               "terminal_lambda_range": [float(table["terminal_lambda"].min()), float(table["terminal_lambda"].max())],
               "minimum_gap_to_next_pair": float(table["gap_to_next_pair"].min()),
               "maximum_terminal_l1_vector_error": float(table["terminal_l1_vector_error"].max()),
               "maximum_terminal_relative_residual": float(table["terminal_relative_residual"].max()),
               "matrix_remainder_target": error_target,
               "fixed_grid": "201 logarithmic nodes from 0.01 to 2, same as Figure 3.3",
               "scope": "Numerical confirmation on each saved sample and its computed divergence matrix; terminal lambda chosen separately from each sample's minimum-edge gap. This is the fixed-N limit, not a joint N/lambda limit or a population/sample-size limit."}
    (output_directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(example_table[["repetition", "h_selected_at_10000", "lower_pair_h", "upper_pair_h",
                         "selected_at_10000_in_pair", "perron_mass_on_pair_at_10000",
                         "terminal_perron_mass_on_pair"]].to_string(index=False))
    return summary


if __name__ == "__main__":
    run_checks()
