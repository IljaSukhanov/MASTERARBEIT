"""Paired coefficient-MSE experiment for the AR(3) minimum-width estimator.

Run from the project root with
    python Code/Code_AR_processes/ar3_minimum_width_mse_vs_T.py

Each of 100 independently drawn causal AR(3) processes supplies one length-2000
series. The five fits use nested prefixes of that series. Completed process
results are cached, so an interrupted run can be resumed safely.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import pdist

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Code.Code_AR_processes.ar_tools.descent import projected_subgradient_empirical_width
from Code.Code_AR_processes.ar_tools.functions import (
    jones_ar_coefficients,
    simulate_uniform_ar,
    sliding_window_point_cloud,
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "Code/Code_AR_processes/outputs/csv/ar3_minimum_width_k4/mse_vs_T_100_processes"
)
CACHE_DIR = OUTPUT_DIR / "processes"
CONFIG = {
    "order": 3,
    "embedding_dimension": 4,
    "replications": 100,
    "sample_sizes": [100, 500, 1000, 1500, 2000],
    "burnin": 1000,
    "innovation_distribution": "uniform[-1,1]",
    "experiment_seed": 20260928,
    "optimizer_seed": 20260929,
    "starts": 12,
    "iterations": 5000,
    "initial_step_scale": 0.2,
    "geometric_decay": 0.997,
    "hull_vertex_reduction": True,
    "mse_definition": "mean of three squared coefficient errors per fit; then mean over 100 processes",
}


def fit_prefix(series: np.ndarray, replication_index: int, sample_size: int) -> dict:
    """Fit only k=4, with the same random starts and schedule as the 50-run study."""
    cloud = sliding_window_point_cloud(series[:sample_size], embedding_dimension=4)
    optimization_cloud = cloud
    try:
        hull = ConvexHull(cloud)
        optimization_cloud = cloud[np.sort(hull.vertices)]
    except QhullError:
        pass

    # A point-cloud diameter is attained by hull vertices. Thus the step-size
    # normalization and every directional width use the original cloud values.
    diameter = float(pdist(optimization_cloud).max())
    if not np.isfinite(diameter) or diameter <= 0:
        raise ValueError("The point-cloud diameter is invalid.")
    initial_step = CONFIG["initial_step_scale"] / diameter

    generator = np.random.default_rng(
        np.random.SeedSequence([CONFIG["optimizer_seed"], replication_index, 4, 20])
    )
    initial_directions = generator.normal(size=(CONFIG["starts"], 4))
    initial_directions /= np.linalg.norm(initial_directions, axis=1, keepdims=True)

    results = [
        projected_subgradient_empirical_width(
            optimization_cloud,
            iterations=CONFIG["iterations"],
            initial_step=initial_step,
            patience=CONFIG["iterations"] + 1,
            rng=np.random.SeedSequence(
                [CONFIG["optimizer_seed"], replication_index, 4, 21, start_index]
            ),
            initial_direction=direction,
            step_schedule="geometric",
            geometric_decay=CONFIG["geometric_decay"],
        )
        for start_index, direction in enumerate(initial_directions)
    ]
    best_start = int(np.argmin([result["minimum_width"] for result in results]))
    result = results[best_start]
    direction = result["direction"]
    if abs(direction[0]) <= np.finfo(float).eps:
        raise ValueError("The first direction component is numerically zero.")
    estimate = -direction[1:] / direction[0]
    if not np.all(np.isfinite(estimate)):
        raise ValueError("The coefficient estimate is not finite.")
    if not np.isclose(np.ptp(cloud @ direction), result["minimum_width"], atol=1e-9):
        raise ValueError("The selected direction does not reproduce its width.")
    return {
        "estimate": estimate,
        "minimum_width": float(result["minimum_width"]),
        "abs_u_0": float(abs(direction[0])),
        "best_start": best_start + 1,
        "diameter": diameter,
        "initial_step": initial_step,
        "hull_vertices": len(optimization_cloud),
    }


def run_process(replication_index: int) -> pd.DataFrame:
    cache_path = CACHE_DIR / f"process_{replication_index + 1:03d}.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        if (
            len(cached) == len(CONFIG["sample_sizes"])
            and cached["T"].tolist() == CONFIG["sample_sizes"]
            and cached["replication"].eq(replication_index + 1).all()
        ):
            return cached

    generator = np.random.default_rng(
        np.random.SeedSequence([CONFIG["experiment_seed"], replication_index, 0])
    )
    true_phi = jones_ar_coefficients(3, rng=generator)
    series = simulate_uniform_ar(
        true_phi,
        sample_size=max(CONFIG["sample_sizes"]),
        burnin=CONFIG["burnin"],
        half_width=1.0,
        rng=generator,
    )
    rows = []
    for sample_size in CONFIG["sample_sizes"]:
        start_time = perf_counter()
        row = {
            "replication": replication_index + 1,
            "T": sample_size,
            "status": "ok",
            "error_message": "",
        }
        for index, value in enumerate(true_phi, start=1):
            row[f"true_phi_{index}"] = float(value)
        try:
            result = fit_prefix(series, replication_index, sample_size)
            estimate = result.pop("estimate")
            row.update(result)
            squared_errors = (estimate - true_phi) ** 2
            for index, (value, error) in enumerate(zip(estimate, squared_errors), start=1):
                row[f"estimated_phi_{index}"] = float(value)
                row[f"squared_error_phi_{index}"] = float(error)
            row["coefficient_mse"] = float(np.mean(squared_errors))
        except Exception as exc:
            row["status"] = "failed"
            row["error_message"] = f"{type(exc).__name__}: {exc}"
            row["coefficient_mse"] = np.nan
        row["runtime_seconds"] = perf_counter() - start_time
        rows.append(row)

    frame = pd.DataFrame(rows)
    temporary_path = cache_path.with_suffix(f".tmp.{os.getpid()}")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(cache_path)
    return frame


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    config_path = OUTPUT_DIR / "configuration.json"
    config_text = json.dumps(CONFIG, indent=2, sort_keys=True) + "\n"
    if config_path.exists() and config_path.read_text() != config_text:
        raise RuntimeError("The existing output folder has a different configuration.")
    config_path.write_text(config_text)

    run_start = perf_counter()
    frames = []
    with parallel_config(backend="loky", inner_max_num_threads=1):
        completed = Parallel(n_jobs=4, return_as="generator_unordered")(
            delayed(run_process)(index) for index in range(CONFIG["replications"])
        )
        for count, frame in enumerate(completed, start=1):
            frames.append(frame)
            if count == 1 or count % 10 == 0 or count == CONFIG["replications"]:
                partial = pd.concat(frames, ignore_index=True)
                partial.to_csv(OUTPUT_DIR / "replication_results_partial.csv", index=False)
                print(
                    f"Completed {count}/{CONFIG['replications']} processes "
                    f"in {(perf_counter() - run_start) / 60:.1f} minutes",
                    flush=True,
                )

    results = pd.concat(frames, ignore_index=True).sort_values(
        ["replication", "T"]
    ).reset_index(drop=True)
    expected = CONFIG["replications"] * len(CONFIG["sample_sizes"])
    if len(results) != expected or results.duplicated(["replication", "T"]).any():
        raise RuntimeError("The experiment results are incomplete or duplicated.")
    results.to_csv(OUTPUT_DIR / "replication_results.csv", index=False)
    failures = results.loc[results["status"] != "ok"]
    failures.to_csv(OUTPUT_DIR / "failed_fits.csv", index=False)

    summary_rows = []
    for sample_size in CONFIG["sample_sizes"]:
        group = results.loc[results["T"] == sample_size]
        losses = group["coefficient_mse"].to_numpy(dtype=float)
        failed_count = int((group["status"] != "ok").sum())
        summary_rows.append(
            {
                "T": sample_size,
                "processes": len(group),
                "failed_fits": failed_count,
                # A failed fit makes the unconditional MSE unavailable.
                "coefficient_mse": float(np.mean(losses)),
                "median_process_mse": float(np.median(losses)),
                "max_process_mse": float(np.max(losses)),
                "mean_runtime_seconds": float(group["runtime_seconds"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "mse_summary.csv", index=False)
    print(f"Finished {expected} fits; failed fits: {len(failures)}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.8g}"))
    print(f"Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
