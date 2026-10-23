"""Paired order selection on frozen series, with auditable width directions."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np
import pandas as pd
import scipy
from scipy.spatial.distance import pdist
from threadpoolctl import threadpool_limits

from .breakpoint_selectors import descent_chain
from .classical_selectors import aic_bic_ar_order, pacf_cutoff_order
from .descent import projected_subgradient_empirical_width
from .functions import sliding_window_point_cloud


METHODS = ("Minimum width + descent chain", "BIC", "PACF cutoff")


def default_configuration():
    return {
        "K": 10, "maximum_classical_order": 9, "scale": 0.1,
        "half_life": 100.0, "iterations": 2000, "patience": 200,
        "delta": 1e-8, "improvement_tolerance": 0.0,
        "optimizer_seed": 2026102306, "pacf_confidence_multiplier": 1.96,
    }


def _json(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_order_dataset(directory):
    """Load the saved subset, verifying file and per-series checksums."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    expected_id = hashlib.sha256(_json(manifest["specification"]).encode()).hexdigest()[:16]
    if manifest["dataset_id"] != expected_id:
        raise ValueError("Subset specification checksum mismatch.")
    for name, checksum in manifest["file_sha256"].items():
        if _digest(directory / name) != checksum:
            raise ValueError(f"Subset checksum mismatch: {name}.")
    metadata = pd.read_csv(directory / "series_metadata.csv", float_precision="round_trip",
                           dtype={"innovation_seed": "uint64"})
    with np.load(directory / "dataset.npz", allow_pickle=False) as saved:
        series = saved["series"].copy()
        saved_ids, saved_orders = saved["source_sample_ids"], saved["orders"]
    if (series.dtype != np.float64 or series.shape != (manifest["series_count"], manifest["sample_size"])
            or len(metadata) != len(series) or not np.isfinite(series).all()
            or not np.array_equal(metadata.subset_index, np.arange(len(series)))
            or not np.array_equal(metadata.sample_id, np.arange(1, len(series) + 1))
            or not np.array_equal(metadata.source_sample_id, saved_ids)
            or not np.array_equal(metadata.order, saved_orders)
            or not metadata.source_sample_id.is_unique):
        raise ValueError("Inconsistent saved series identities or dimensions.")
    counts = metadata.groupby("order").size().to_dict()
    expected = {int(p): manifest["specification"]["series_per_order"]
                for p in manifest["specification"]["orders"]}
    if counts != expected:
        raise ValueError("Subset does not have the declared number of series per order.")
    for row in metadata.itertuples(index=False):
        checksum = hashlib.sha256(np.ascontiguousarray(series[row.subset_index], dtype="<f8").tobytes()).hexdigest()
        if checksum != row.series_sha256:
            raise ValueError(f"Series checksum mismatch for sample {row.sample_id}.")
    series.flags.writeable = False
    return {"directory": directory, "dataset_id": manifest["dataset_id"],
            "manifest": manifest, "metadata": metadata, "series": series}


def fit_width_curve(series, configuration, *, sample_id):
    """Fit k=1,...,K on common rows, with one padded warm start per k."""
    config = configuration
    K = config["K"]
    cloud = sliding_window_point_cloud(series, K)
    widths = np.empty(K)
    directions = np.zeros((K, K))
    diameters = np.empty(K)
    evaluations = np.ones(K, dtype=np.int64)
    reasons = ["one_dimensional_range"]
    widths[0] = np.ptp(cloud[:, 0])
    directions[0, 0] = 1.0
    diameters[0] = widths[0]
    for k in range(2, K + 1):
        points = np.ascontiguousarray(cloud[:, :k])
        diameter = float(pdist(points).max())
        if not np.isfinite(diameter) or diameter <= 0:
            raise ValueError(f"Invalid cloud diameter at k={k}.")
        initial = np.r_[directions[k - 2, :k - 1], 0.0]
        fit = projected_subgradient_empirical_width(
            points, initial_direction=initial,
            initial_step=config["scale"] / diameter,
            step_schedule="geometric", geometric_decay=float(np.exp2(-1.0 / config["half_life"])),
            iterations=config["iterations"], patience=config["patience"],
            delta=config["delta"], improvement_tolerance=config["improvement_tolerance"],
            stationarity_check=False, max_restarts=0,
            rng=np.random.SeedSequence([config["optimizer_seed"], int(sample_id), k]),
        )
        directions[k - 1, :k] = fit["direction"]
        widths[k - 1] = fit["minimum_width"]
        diameters[k - 1] = diameter
        evaluations[k - 1] = len(fit["width_history"])
        reasons.append(fit["termination_reason"])
    reconstructed = np.array([np.ptp(cloud[:, :k] @ directions[k - 1, :k])
                              for k in range(1, K + 1)])
    norms = np.linalg.norm(directions, axis=1)
    if (not np.isfinite(widths).all() or np.any(widths <= 0)
            or not np.allclose(reconstructed, widths, rtol=0, atol=1e-10)
            or not np.allclose(norms, 1, rtol=0, atol=1e-12)):
        raise ValueError("Saved directions fail the width or unit-norm check.")
    chain = descent_chain(widths)  # rejects substantive upward steps; no smoothing
    return {
        "widths": widths, "directions": directions, "diameters": diameters,
        "evaluations": evaluations, "termination_reasons": np.array(reasons),
        "selected_order": chain[0], "drops": chain[1], "remaining_reduction": chain[2],
        "alpha": chain[3], "stationary_probability": chain[4], "odds": chain[5],
        "stationarity_residual": chain[6],
        "maximum_reconstruction_error": float(np.max(abs(reconstructed - widths))),
        "maximum_unit_norm_error": float(np.max(abs(norms - 1))),
        "maximum_upward_step": max(0.0, float(np.diff(widths).max())),
    }


def _fit_case(series, metadata, configuration):
    rows, arrays = [], {}
    for method in METHODS:
        started = perf_counter()
        row = {"method": method, "selected_order": None, "status": "failed", "error_message": ""}
        try:
            if method == METHODS[0]:
                fit = fit_width_curve(series, configuration, sample_id=metadata["sample_id"])
                selected = fit["selected_order"]
                arrays.update({key: value for key, value in fit.items() if isinstance(value, np.ndarray)})
                row.update({key: value for key, value in fit.items() if not isinstance(value, np.ndarray)})
            elif method == "BIC":
                _, selected, criteria = aic_bic_ar_order(series, configuration["maximum_classical_order"])
                arrays["bic_orders"] = criteria.order.to_numpy()
                arrays["bic_values"] = criteria.BIC.to_numpy()
            else:
                selected, values, cutoff = pacf_cutoff_order(
                    series, configuration["maximum_classical_order"],
                    confidence_multiplier=configuration["pacf_confidence_multiplier"],
                )
                arrays["pacf_values"] = values
                row["pacf_cutoff"] = cutoff
            row.update(selected_order=int(selected), status="ok")
        except Exception as error:
            row["error_message"] = f"{type(error).__name__}: {error}"
        row["fit_seconds"] = perf_counter() - started
        rows.append(row)
    identity = {key: metadata[key] for key in
                ["sample_id", "source_sample_id", "order", "shape_id", "series_sha256"]}
    return {"identity": identity, "results": rows}, arrays


def _cached_case(series, metadata, configuration, directory, run_id):
    path = Path(directory) / f"sample_{int(metadata['sample_id']):04d}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            record = json.loads(str(saved["record"]))
            arrays = {key: saved[key].copy() for key in saved.files if key != "record"}
        if (record["run_id"] != run_id
                or any(record["identity"][key] != metadata[key] for key in record["identity"])
                or [row["method"] for row in record["results"]] != list(METHODS)):
            raise ValueError(f"Inconsistent checkpoint: {path}")
        return record, arrays
    with threadpool_limits(limits=1):
        record, arrays = _fit_case(series, metadata, configuration)
    record["run_id"] = run_id
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".writing-", delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, record=np.array(_json(record)), **arrays)
    temporary.replace(path)
    return record, arrays


def _configuration(configuration):
    config = default_configuration() if configuration is None else deepcopy(configuration)
    if set(config) != set(default_configuration()):
        raise ValueError("Use exactly the fields in default_configuration().")
    for name in ["K", "maximum_classical_order", "iterations", "patience", "optimizer_seed"]:
        value = config[name]
        minimum = 0 if name == "optimizer_seed" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"Invalid integer setting: {name}.")
    if config["K"] < 4 or config["maximum_classical_order"] != config["K"] - 1:
        raise ValueError("Use K >= 4 and maximum_classical_order = K - 1.")
    for name in ["scale", "half_life", "delta", "improvement_tolerance", "pacf_confidence_multiplier"]:
        config[name] = float(config[name])
        if not np.isfinite(config[name]) or config[name] < 0 or (
                name in ["scale", "half_life", "pacf_confidence_multiplier"] and config[name] == 0):
            raise ValueError(f"Invalid setting: {name}.")
    return config


def save_selection_tables(results, directory):
    """Export the paired exact/under/over counts in the reference-table style."""
    directory = Path(directory)
    frame = results.copy()
    frame["correct"] = frame.status.eq("ok") & frame.selected_order.eq(frame.order)
    frame["under"] = frame.status.eq("ok") & frame.selected_order.lt(frame.order)
    frame["over"] = frame.status.eq("ok") & frame.selected_order.gt(frame.order)
    frame["failed"] = frame.status.ne("ok")
    summary = frame.groupby(["method", "order"], sort=False)[["correct", "under", "over", "failed"]].sum().astype(int)
    summary["total"] = frame.groupby(["method", "order"], sort=False).size()
    if not summary[["correct", "under", "over", "failed"]].sum(axis=1).eq(summary.total).all():
        raise AssertionError("Selection outcomes do not sum to the sample count.")
    orders = sorted(frame.order.unique())
    matrix = pd.DataFrame(index=pd.Index(METHODS, name="Selector"), columns=[f"AR({p})" for p in orders])
    tex_rows = []
    for method in METHODS:
        tex_cells = []
        for p in orders:
            row = summary.loc[(method, p)]
            text = f"{row.correct}/{row.total} ({row.under}/{row.over})"
            if row.failed:
                text += f"; failed={row.failed}"
            matrix.loc[method, f"AR({p})"] = text
            fraction = f"{row.correct}/{row.total}"
            if row.correct == summary.xs(p, level="order").correct.max():
                fraction = r"\textbf{" + fraction + "}"
            tail = f"({row.under}/{row.over})" + (f"; failed={row.failed}" if row.failed else "")
            tex_cells.append(r"\shortstack{" + fraction + r" \\ " + tail + "}")
        label = r"\shortstack[l]{Empirical minimum width\\(descent chain)}" if method == METHODS[0] else method
        tex_rows.append(label + " & " + " & ".join(tex_cells) + r" \\")
    summary.reset_index().to_csv(directory / "summary.csv", index=False)
    matrix.to_csv(directory / "table_2_2_style.csv")
    lines = ["| Selector | " + " | ".join(matrix.columns) + " |",
             "|---|" + "---:|" * len(orders)]
    lines += ["| " + method + " | " + " | ".join(matrix.loc[method]) + " |" for method in METHODS]
    (directory / "table_2_2_style.md").write_text(
        "Each cell: correct/50 (under/over). All methods use the same saved series.\n\n" + "\n".join(lines) + "\n")
    tex = [r"\begin{table}[htbp]", r"\centering", r"\small",
           r"\caption{AR-order selections on 50 saved series per true order, pooled across five Beta innovation shapes. "
           r"Each entry reports correct selections, followed by (underselections/overselections). $T=1000$, $K=10$.}",
           r"\begin{tabular}{l" + "c" * len(orders) + "}", r"\toprule",
           "Selector & " + " & ".join(matrix.columns) + r" \\", r"\midrule",
           *tex_rows, r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (directory / "table_2_2_style.tex").write_text("\n".join(tex) + "\n")
    return summary.reset_index(), matrix


def run_order_selection(dataset_directory, output_root, configuration=None, *, n_jobs=4, progress=True):
    config = _configuration(configuration)
    dataset = load_order_dataset(dataset_directory)
    if config["K"] >= dataset["series"].shape[1]:
        raise ValueError("K must be smaller than the saved series length.")
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer.")
    source_root = Path(__file__).resolve().parent
    spec = {
        "schema_version": 1, "dataset_id": dataset["dataset_id"],
        "input_manifest_sha256": _digest(dataset["directory"] / "manifest.json"),
        "configuration": config, "methods": list(METHODS),
        "width_rows": "common newest-first windows, endpoints t=K,...,T, for every k",
        "width_initialization": "k=1 range; one previous best direction padded on the right for each k>=2",
        "step_rule": "gamma_m=(scale/Euclidean_cloud_diameter)*2**(-m/half_life)",
        "width_postprocessing": "none", "stationarity_check": False, "restarts": 0,
        "classical_candidate_orders": list(range(config["K"])),
        "descent_chain_candidate_orders": list(range(1, config["K"] - 1)),
        "bic_intercept": True, "pacf_series_length": dataset["series"].shape[1],
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "source_sha256": {name: _digest(source_root / name) for name in
                          [Path(__file__).name, "descent.py", "functions.py", "classical_selectors.py", "breakpoint_selectors.py"]},
    }
    run_id = hashlib.sha256(_json(spec).encode()).hexdigest()[:16]
    directory = Path(output_root).resolve() / dataset["dataset_id"] / run_id
    checkpoints = directory / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    path = directory / "configuration.json"
    if path.exists() and json.loads(path.read_text()) != spec:
        raise ValueError("Saved experiment configuration mismatch.")
    path.write_text(_json(spec))
    from joblib import Parallel, delayed, parallel_config
    cases = []
    started = perf_counter()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        completed = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
            delayed(_cached_case)(dataset["series"][row["subset_index"]], row, config, checkpoints, run_id)
            for row in dataset["metadata"].to_dict("records")
        )
        for count, case in enumerate(completed, 1):
            cases.append(case)
            if progress and (count % 10 == 0 or count == len(dataset["series"])):
                print(f"Completed {count}/{len(dataset['series'])} series ({perf_counter()-started:.1f}s)", flush=True)
    cases.sort(key=lambda case: case[0]["identity"]["sample_id"])
    result_rows, curve_rows, chain_rows = [], [], []
    for record, arrays in cases:
        identity = record["identity"]
        result_rows.extend(dict(identity, **row) for row in record["results"])
        if "widths" not in arrays:
            continue
        for index in range(config["K"]):
            curve_rows.append(dict(identity, k=index+1, minimum_width=arrays["widths"][index],
                                   cloud_diameter=arrays["diameters"][index],
                                   evaluated_states=arrays["evaluations"][index],
                                   termination_reason=arrays["termination_reasons"][index],
                                   **{f"u_{j}": arrays["directions"][index, j] for j in range(config["K"])}))
            chain_rows.append(dict(identity, k=index+1, candidate=0<index<config["K"]-1,
                                   **{key: arrays[key][index] for key in
                                      ["drops", "remaining_reduction", "alpha", "stationary_probability", "odds"]}))
    results = pd.DataFrame(result_rows).reindex(columns=[
        "sample_id", "source_sample_id", "order", "shape_id", "series_sha256",
        "method", "selected_order", "status", "error_message", "fit_seconds",
        "stationarity_residual", "maximum_reconstruction_error",
        "maximum_unit_norm_error", "maximum_upward_step", "pacf_cutoff",
    ])
    results.to_csv(directory / "selections.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(directory / "width_curves.csv", index=False)
    pd.DataFrame(chain_rows).to_csv(directory / "descent_chain_scores.csv", index=False)
    results[results.status.ne("ok")].to_csv(directory / "failed_fits.csv", index=False)
    summary, matrix = save_selection_tables(results, directory)
    diagnostic = results[results.method.eq(METHODS[0])]
    verification = {
        "series": len(cases), "methods": len(METHODS), "selections": len(results),
        "failed_fits": int(results.status.ne("ok").sum()),
        "successful_width_curves": int(diagnostic.status.eq("ok").sum()),
        "width_values": len(curve_rows),
        "maximum_width_reconstruction_error": float(diagnostic.maximum_reconstruction_error.max()) if curve_rows else None,
        "maximum_unit_norm_error": float(diagnostic.maximum_unit_norm_error.max()) if curve_rows else None,
        "maximum_upward_step": float(diagnostic.maximum_upward_step.max()) if curve_rows else None,
        "maximum_chain_stationarity_residual": float(diagnostic.stationarity_residual.max()) if curve_rows else None,
        "checkpoint_sha256": {p.name: _digest(p) for p in sorted(checkpoints.glob("*.npz"))},
        "input_files_still_match_manifest": all(
            _digest(dataset["directory"] / name) == checksum
            for name, checksum in dataset["manifest"]["file_sha256"].items()),
    }
    if len(results) != len(dataset["series"]) * len(METHODS) or not verification["input_files_still_match_manifest"]:
        raise AssertionError("Incomplete results or modified input archive.")
    (directory / "verification.json").write_text(_json(verification))
    if progress:
        print(matrix.to_string(), flush=True)
        print(f"Saved results: {directory}", flush=True)
    return {"run_id": run_id, "directory": directory, "configuration": spec,
            "results": results, "summary": summary, "matrix": matrix, "verification": verification}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    run_order_selection(args.dataset, args.output_root, n_jobs=args.jobs)


if __name__ == "__main__":
    main()
