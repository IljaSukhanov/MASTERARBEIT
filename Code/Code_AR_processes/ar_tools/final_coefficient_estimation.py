"""Saved, paired coefficient benchmarks across AR orders and Beta shapes.

The input archive is independent of the fitting settings. It is never silently
regenerated or overwritten. All estimators consume its original float64 series.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
from time import perf_counter

import numpy as np
import pandas as pd
import scipy
from scipy.spatial.distance import pdist
from threadpoolctl import threadpool_limits

from .classical_selectors import (
    fit_ols_known_order, horvath_linf_coefficients, yule_walker_coefficients,
)
from .coefficient_errors import COEFFICIENT_ERROR_METRIC, coefficient_mae
from .descent import projected_subgradient_empirical_width
from .functions import (
    is_stationary, jones_ar_coefficients, simulate_beta_ar, sliding_window_point_cloud,
)
from .innovations import beta_innovation_bounds


METHODS = ("Minimum width", "OLS", "Yule-Walker", "Horvath L-infinity")
BETA_SHAPES = (
    {"id": "uniform", "alpha": 1.0, "beta": 1.0},
    {"id": "symmetric_u", "alpha": 0.5, "beta": 0.5},
    {"id": "symmetric_peak", "alpha": 2.0, "beta": 2.0},
    {"id": "right_skew", "alpha": 1.0, "beta": 2.0},
    {"id": "left_skew", "alpha": 2.0, "beta": 1.0},
)


def default_dataset_configuration():
    return {
        "orders": [1, 2, 3, 4, 5], "models_per_order": 100,
        "sample_size": 1000, "burnin": 2000,
        "coefficient_seed": 2026102301, "innovation_seed": 2026102302,
        "beta_shapes": deepcopy(list(BETA_SHAPES)),
    }


def default_fit_configuration():
    return {
        "scale": 0.1, "half_life": 100.0, "iterations": 2000,
        "patience": 200, "delta": 1e-8, "improvement_tolerance": 0.0,
        "optimizer_seed": 2026102303,
    }


def _json(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype="<f8").tobytes()).hexdigest()


def _source_digest(names):
    root = Path(__file__).resolve().parent
    return hashlib.sha256(b"".join((root / name).read_bytes() for name in names)).hexdigest()


def _seed(root, *coordinates):
    return int(np.random.SeedSequence([root, *coordinates]).generate_state(1, dtype=np.uint64)[0])


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _dataset_spec(configuration):
    config = deepcopy(configuration)
    if set(config) != set(default_dataset_configuration()):
        raise ValueError("Use exactly the fields in default_dataset_configuration().")
    for name in ("models_per_order", "sample_size", "burnin", "coefficient_seed", "innovation_seed"):
        config[name] = _integer(config[name], name, 0 if name in ("burnin", "coefficient_seed", "innovation_seed") else 1)
    orders = [_integer(order, "order") for order in config["orders"]]
    if not orders or orders != sorted(set(orders)) or config["sample_size"] <= max(orders) + 1:
        raise ValueError("Use distinct increasing orders and sample_size > max(order)+1.")
    config["orders"] = orders
    shapes = config["beta_shapes"]
    if not shapes or len({shape["id"] for shape in shapes}) != len(shapes):
        raise ValueError("Beta shape IDs must be distinct and nonempty.")
    for shape in shapes:
        if set(shape) != {"id", "alpha", "beta"} or not isinstance(shape["id"], str) or not shape["id"]:
            raise ValueError("Each Beta shape needs an id, alpha, and beta.")
        shape["alpha"], shape["beta"] = float(shape["alpha"]), float(shape["beta"])
        beta_innovation_bounds(shape["alpha"], shape["beta"])
    spec = {
        "schema_version": 1, "configuration": config,
        "normalization": "population_mean_0_variance_1_over_3",
        "initial_history": "zero (population mean); fixed discarded burn-in",
        "coefficient_pairing": "same coefficient vector across all Beta shapes",
        "generator_source_sha256": _source_digest(["functions.py", "innovations.py"]),
        "numpy_version": np.__version__,
    }
    return spec


def load_coefficient_dataset(directory):
    """Verify checksums and design completeness before exposing saved inputs."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    spec = manifest["specification"]
    dataset_id = hashlib.sha256(_json(spec).encode()).hexdigest()[:16]
    if manifest["dataset_id"] != dataset_id:
        raise ValueError("Dataset specification checksum mismatch.")
    for name in ("dataset.npz", "models.csv", "series_metadata.csv"):
        if _digest(directory / name) != manifest["file_sha256"][name]:
            raise ValueError(f"Dataset checksum mismatch: {name}; saved inputs were not regenerated.")
    with np.load(directory / "dataset.npz", allow_pickle=False) as archive:
        series = archive["series"].copy()
        coefficients = archive["coefficients"].copy()
    models = pd.read_csv(directory / "models.csv", dtype={"coefficient_seed": "uint64"}, float_precision="round_trip")
    metadata = pd.read_csv(directory / "series_metadata.csv", dtype={"innovation_seed": "uint64"}, float_precision="round_trip")
    config = spec["configuration"]
    expected_models = len(config["orders"]) * config["models_per_order"]
    expected_series = expected_models * len(config["beta_shapes"])
    if (series.dtype != np.float64 or series.shape != (expected_series, config["sample_size"])
            or coefficients.shape != (expected_models, max(config["orders"]))
            or len(models) != expected_models or len(metadata) != expected_series
            or not np.isfinite(series).all() or not np.isfinite(coefficients).all()):
        raise ValueError("Saved dataset has incorrect shapes or nonfinite values.")
    if (models.duplicated(["order", "model_id"]).any()
            or metadata.duplicated(["order", "model_id", "shape_id"]).any()
            or not np.array_equal(models["model_index"], np.arange(expected_models))
            or not np.array_equal(metadata["sample_id"], np.arange(1, expected_series + 1))
            or not models["coefficient_seed"].is_unique or not metadata["innovation_seed"].is_unique):
        raise ValueError("Saved model identities, pairing, or seeds are inconsistent.")
    expected_groups = {(p, shape["id"]) for p in config["orders"] for shape in config["beta_shapes"]}
    counts = metadata.groupby(["order", "shape_id"]).size()
    if set(counts.index) != expected_groups or not counts.eq(config["models_per_order"]).all():
        raise ValueError("Dataset does not contain every order/shape condition.")
    for row in models.itertuples(index=False):
        phi = coefficients[row.model_index, :row.order]
        stationary, _ = is_stationary(phi)
        if not stationary or np.any(coefficients[row.model_index, row.order:] != 0):
            raise ValueError("Invalid generating coefficient vector.")
    for row in metadata.itertuples(index=False):
        model = models.iloc[row.model_index]
        shape = config["beta_shapes"][row.shape_index]
        if (row.order != model["order"] or row.model_id != model["model_id"]
                or row.shape_id != shape["id"] or row.alpha != shape["alpha"] or row.beta != shape["beta"]
                or row.series_sha256 != _array_digest(series[row.sample_id - 1])):
            raise ValueError("Saved series/metadata pairing checksum mismatch.")
    series.flags.writeable = False
    coefficients.flags.writeable = False
    return {
        "dataset_id": dataset_id, "directory": directory, "manifest": manifest,
        "configuration": config, "series": series, "coefficients": coefficients,
        "models": models, "metadata": metadata,
    }


def prepare_coefficient_dataset(input_root, configuration=None, *, progress=True):
    """Generate once, atomically save, then reload the verified original arrays.

    A changed design gets a different directory. Corrupt or incomplete existing
    inputs raise an error, rather than triggering fresh random draws.
    """
    spec = _dataset_spec(default_dataset_configuration() if configuration is None else configuration)
    config = spec["configuration"]
    dataset_id = hashlib.sha256(_json(spec).encode()).hexdigest()[:16]
    input_root = Path(input_root).resolve()
    target = input_root / dataset_id
    # The saved arrays, not the current NumPy RNG implementation, define the
    # evaluation samples. Reuse an existing matching design across runtimes.
    requested = {key: value for key, value in spec.items() if key != "numpy_version"}
    if not target.exists() and input_root.exists():
        matches = []
        for path in sorted(input_root.glob("*/manifest.json")):
            if path.parent.name.startswith("."):
                continue
            saved = json.loads(path.read_text())["specification"]
            comparable = {key: value for key, value in saved.items() if key != "numpy_version"}
            if comparable == requested:
                matches.append(path.parent)
        if len(matches) > 1:
            raise ValueError("Multiple saved archives match this design; select one explicitly with load_coefficient_dataset().")
        if matches:
            target = matches[0]
    if target.exists():
        dataset = load_coefficient_dataset(target)
        saved = dataset["manifest"]["specification"]
        if {key: value for key, value in saved.items() if key != "numpy_version"} != requested:
            raise ValueError("Saved dataset does not match the requested design.")
        if progress:
            print(f"Loaded {len(dataset['series']):,} verified series from {target}")
        return dataset
    input_root.mkdir(parents=True, exist_ok=True)
    model_rows, metadata_rows, coefficient_rows, series_rows = [], [], [], []
    for order in config["orders"]:
        for model_id in range(1, config["models_per_order"] + 1):
            model_index = len(model_rows)
            coefficient_seed = _seed(config["coefficient_seed"], order, model_id)
            phi = jones_ar_coefficients(order, rng=coefficient_seed)
            stationary, eigenvalues = is_stationary(phi)
            if not stationary or not np.isfinite(phi).all():
                raise ValueError(f"Jones draw p={order}, model={model_id} failed stationarity; no redraw was made.")
            padded_phi = np.zeros(max(config["orders"]), dtype=np.float64)
            padded_phi[:order] = phi
            coefficient_rows.append(padded_phi)
            model_row = {
                "model_index": model_index, "order": order, "model_id": model_id,
                "coefficient_seed": coefficient_seed,
                "true_spectral_radius": float(np.max(np.abs(eigenvalues))),
            }
            model_row.update({f"true_phi_{j+1}": float(value) for j, value in enumerate(padded_phi)})
            model_rows.append(model_row)
            for shape_index, shape in enumerate(config["beta_shapes"]):
                innovation_seed = _seed(config["innovation_seed"], order, model_id, shape_index)
                values = simulate_beta_ar(
                    phi, config["sample_size"], burnin=config["burnin"],
                    alpha=shape["alpha"], beta=shape["beta"], rng=innovation_seed,
                )
                if values.shape != (config["sample_size"],) or not np.isfinite(values).all():
                    raise ValueError("Simulation returned invalid values; no replacement series was drawn.")
                lower, upper = beta_innovation_bounds(shape["alpha"], shape["beta"])
                series_rows.append(values)
                metadata_rows.append({
                    "sample_id": len(series_rows), "model_index": model_index,
                    "order": order, "model_id": model_id, "shape_index": shape_index,
                    "shape_id": shape["id"], "alpha": shape["alpha"], "beta": shape["beta"],
                    "T": config["sample_size"], "burnin": config["burnin"],
                    "innovation_seed": innovation_seed, "innovation_lower": lower,
                    "innovation_upper": upper, "innovation_variance": 1.0 / 3.0,
                    "series_sha256": _array_digest(values),
                })
        if progress:
            print(f"Saved-input preparation: AR({order}), {config['models_per_order']} models x {len(config['beta_shapes'])} shapes")
    with tempfile.TemporaryDirectory(prefix=".preparing-", dir=input_root) as temporary:
        staging = Path(temporary) / "complete"
        staging.mkdir()
        np.savez_compressed(staging / "dataset.npz", series=np.stack(series_rows), coefficients=np.stack(coefficient_rows))
        pd.DataFrame(model_rows).to_csv(staging / "models.csv", index=False, float_format="%.17g")
        pd.DataFrame(metadata_rows).to_csv(staging / "series_metadata.csv", index=False, float_format="%.17g")
        manifest = {
            "dataset_id": dataset_id, "specification": spec,
            "model_count": len(model_rows), "series_count": len(series_rows),
            "python_version": platform.python_version(), "scipy_version": scipy.__version__,
            "coefficient_padding": "unused coordinates are zero; slice by recorded order",
            "file_sha256": {name: _digest(staging / name) for name in ("dataset.npz", "models.csv", "series_metadata.csv")},
        }
        (staging / "manifest.json").write_text(_json(manifest))
        load_coefficient_dataset(staging)
        staging.rename(target)
    return load_coefficient_dataset(target)


def _atomic_csv(frame, path):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".writing-", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            frame.to_csv(handle, index=False, float_format="%.17g")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def _fit_case(values, phi, metadata, configuration):
    if _array_digest(values) != metadata["series_sha256"]:
        raise ValueError("The series differs from the saved input checksum.")
    order = int(metadata["order"])
    cloud = sliding_window_point_cloud(values, order + 1)
    rows = []
    for method in METHODS:
        row = dict(metadata, method=method, status="failed", error_message="",
                   coefficient_mae=np.nan, stationary_estimate=False,
                   spectral_radius=np.nan, fit_seconds=np.nan, evaluated_states=np.nan,
                   minimum_width=np.nan, cloud_diameter=np.nan, initial_step=np.nan,
                   termination_reason="", linf_radius=np.nan, linf_duality_gap=np.nan)
        for j in range(len(phi)):
            row[f"true_phi_{j+1}"] = float(phi[j])
            row[f"estimated_phi_{j+1}"] = np.nan
            row[f"absolute_error_phi_{j+1}"] = np.nan
        started = perf_counter()
        try:
            if method == "Minimum width":
                diameter = float(pdist(cloud).max())
                if not np.isfinite(diameter) or diameter <= 0:
                    raise ValueError("A positive finite point-cloud diameter is required.")
                start = np.zeros(order + 1)
                start[0] = 1.0
                fit = projected_subgradient_empirical_width(
                    cloud, initial_direction=start,
                    initial_step=configuration["scale"] / diameter,
                    step_schedule="geometric", geometric_decay=float(np.exp2(-1.0 / configuration["half_life"])),
                    iterations=configuration["iterations"], patience=configuration["patience"],
                    improvement_tolerance=configuration["improvement_tolerance"], delta=configuration["delta"],
                    stationarity_check=False, max_restarts=0,
                    rng=_seed(configuration["optimizer_seed"], int(metadata["sample_id"])),
                )
                direction = fit["direction"]
                row.update(minimum_width=fit["minimum_width"], cloud_diameter=diameter,
                           initial_step=configuration["scale"] / diameter,
                           evaluated_states=len(fit["width_history"]), termination_reason=fit["termination_reason"])
                if direction[0] == 0:
                    raise ValueError("The returned direction has a zero first coordinate.")
                with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                    estimate = -direction[1:] / direction[0]
            elif method == "OLS":
                estimate = fit_ols_known_order(cloud)
            elif method == "Yule-Walker":
                estimate = yule_walker_coefficients(values, order)
            else:
                estimate, radius, gap = horvath_linf_coefficients(values, order, use_convex_hull=False)
                row.update(linf_radius=radius, linf_duality_gap=gap)
            estimate = np.asarray(estimate, dtype=float)
            if estimate.shape != phi.shape or not np.isfinite(estimate).all():
                raise ValueError("The returned coefficients are undefined or nonfinite.")
            for j, value in enumerate(estimate):
                row[f"estimated_phi_{j+1}"] = float(value)
                row[f"absolute_error_phi_{j+1}"] = float(abs(value - phi[j]))
            error = coefficient_mae(estimate, phi)
            if not np.isfinite(error):
                raise ValueError("The coefficient MAE is nonfinite.")
            row.update(coefficient_mae=error, status="ok")
            try:
                stationary, eigenvalues = is_stationary(estimate)
                row.update(stationary_estimate=stationary, spectral_radius=float(np.max(np.abs(eigenvalues))))
            except np.linalg.LinAlgError:
                row["error_message"] = "Coefficient fit is finite; spectral-radius diagnostic failed."
        except Exception as error:
            row["error_message"] = f"{type(error).__name__}: {error}"
        row["fit_seconds"] = perf_counter() - started
        rows.append(row)
    # Mixed signed/unsigned seeds otherwise become float64 when case frames
    # are concatenated, losing low bits of some 64-bit seed values.
    return pd.DataFrame(rows).astype({"innovation_seed": "uint64"})


def _cached_case(values, phi, metadata, configuration, checkpoint_directory):
    path = Path(checkpoint_directory) / f"sample_{int(metadata['sample_id']):04d}.csv"
    if path.exists():
        frame = pd.read_csv(path, dtype={"innovation_seed": "uint64"}, float_precision="round_trip", keep_default_na=False,
                            na_values=[""])
        for column in ("error_message", "termination_reason"):
            frame[column] = frame[column].fillna("").astype(str)
        if (len(frame) != len(METHODS) or set(frame["method"]) != set(METHODS)
                or not frame["sample_id"].eq(metadata["sample_id"]).all()
                or not frame["series_sha256"].eq(metadata["series_sha256"]).all()
                or not frame["status"].isin(["ok", "failed"]).all()):
            raise ValueError(f"Invalid fit checkpoint: {path}")
        return frame
    with threadpool_limits(limits=1):
        frame = _fit_case(values, phi, metadata, configuration)
    _atomic_csv(frame, path)
    return frame


def run_coefficient_estimation(dataset, output_root, configuration=None, *, n_jobs=4, progress=True):
    """Fit four estimators with per-series checkpoints and no truth-based selection."""
    config = default_fit_configuration() if configuration is None else deepcopy(configuration)
    if set(config) != set(default_fit_configuration()):
        raise ValueError("Use exactly the fields in default_fit_configuration().")
    for name in ("iterations", "patience", "optimizer_seed"):
        config[name] = _integer(config[name], name, 0 if name == "optimizer_seed" else 1)
    for name in ("scale", "half_life", "delta", "improvement_tolerance"):
        config[name] = float(config[name])
        if not np.isfinite(config[name]) or config[name] < 0 or (name in ("scale", "half_life") and config[name] == 0):
            raise ValueError(f"Invalid fitting setting: {name}")
    n_jobs = _integer(n_jobs, "n_jobs")
    spec = {
        "schema_version": 1, "dataset_id": dataset["dataset_id"], "configuration": config,
        "methods": list(METHODS), "coefficient_error_metric": COEFFICIENT_ERROR_METRIC,
        "initial_direction": "e1", "step_rule": "gamma_m = (scale / cloud_diameter) * 2**(-m/half_life)",
        "stationarity_check": False, "restarts": 0, "horvath_intercept": False,
        "source_sha256": _source_digest([Path(__file__).name, "descent.py", "functions.py", "innovations.py",
                                          "classical_selectors.py", "coefficient_errors.py"]),
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
    }
    fit_id = hashlib.sha256(_json(spec).encode()).hexdigest()[:16]
    directory = Path(output_root).resolve() / dataset["dataset_id"] / fit_id
    checkpoints = directory / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    config_path = directory / "configuration.json"
    if config_path.exists() and json.loads(config_path.read_text()) != spec:
        raise ValueError("Existing fit configuration mismatch.")
    if not config_path.exists():
        config_path.write_text(_json(spec))
    metadata_records = dataset["metadata"].to_dict("records")
    from joblib import Parallel, delayed, parallel_config
    with parallel_config(backend="loky", inner_max_num_threads=1):
        completed = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
            delayed(_cached_case)(
                dataset["series"][int(meta["sample_id"]) - 1],
                dataset["coefficients"][int(meta["model_index"]), :int(meta["order"])],
                meta, config, checkpoints,
            ) for meta in metadata_records
        )
        frames = []
        for count, frame in enumerate(completed, 1):
            frames.append(frame)
            if progress and (count == 1 or count % 100 == 0 or count == len(metadata_records)):
                print(f"Coefficient estimates: {count:,}/{len(metadata_records):,} saved series")
    results = pd.concat(frames, ignore_index=True).sort_values(["order", "shape_index", "model_id", "method"]).reset_index(drop=True)
    expected = len(metadata_records) * len(METHODS)
    if len(results) != expected or results.duplicated(["sample_id", "method"]).any():
        raise ValueError("Incomplete or duplicated estimator results.")
    _atomic_csv(results, directory / "estimates.csv")
    _atomic_csv(results.loc[results["status"] != "ok"], directory / "failed_fits.csv")
    manifest = {"fit_id": fit_id, "dataset_id": dataset["dataset_id"], "fit_count": len(results),
                "failed_fits": int(results["status"].ne("ok").sum()),
                "estimates_sha256": _digest(directory / "estimates.csv")}
    (directory / "completion.json").write_text(_json(manifest))
    return {"directory": directory, "fit_id": fit_id, "specification": spec, "results": results}


def summarize_coefficient_estimation(run, *, bootstrap_count=2000, bootstrap_seed=2026102304):
    """Bootstrap coefficient-vector IDs within order, paired across methods/shapes.

    A failed fit makes an unconditional mean/interval unavailable. Successful
    nonstationary estimates and all finite extreme errors remain in the scores.
    """
    bootstrap_count = _integer(bootstrap_count, "bootstrap_count", 2)
    bootstrap_seed = _integer(bootstrap_seed, "bootstrap_seed", 0)
    results = run["results"]
    summaries, comparisons = [], []
    for order, order_rows in results.groupby("order", sort=True):
        model_ids = sorted(order_rows["model_id"].unique())
        count = len(model_ids)
        indices = np.random.default_rng(_seed(bootstrap_seed, int(order))).integers(0, count, size=(bootstrap_count, count))
        for (_, shape_id), group in order_rows.groupby(["shape_index", "shape_id"], sort=True):
            matrix = group.pivot(index="model_id", columns="method", values="coefficient_mae").reindex(index=model_ids, columns=METHODS)
            alpha, beta = float(group["alpha"].iloc[0]), float(group["beta"].iloc[0])
            for method in METHODS:
                rows = group.loc[group["method"].eq(method)]
                errors = matrix[method].to_numpy()
                valid = np.isfinite(errors) & rows.set_index("model_id").reindex(model_ids)["status"].eq("ok").to_numpy()
                complete = bool(valid.all())
                boot = np.mean(errors[indices], axis=1) if complete else np.full(bootstrap_count, np.nan)
                summaries.append({
                    "order": order, "shape_id": shape_id, "alpha": alpha, "beta": beta, "method": method,
                    "process_count": count, "failed_fits": int((~valid).sum()),
                    "nonstationary_fits": int((rows["status"].eq("ok") & ~rows["stationary_estimate"].astype(bool)).sum()),
                    "mean_mae": float(np.mean(errors)) if complete else np.nan,
                    "median_mae": float(np.median(errors)) if complete else np.nan,
                    "q90_mae": float(np.quantile(errors, .9)) if complete else np.nan,
                    "maximum_mae": float(np.max(errors)) if complete else np.nan,
                    "mae_mcse": float(np.std(errors, ddof=1) / np.sqrt(count)) if complete and count > 1 else np.nan,
                    "mean_ci_lower": float(np.quantile(boot, .025)) if complete else np.nan,
                    "mean_ci_upper": float(np.quantile(boot, .975)) if complete else np.nan,
                    "mean_mae_successful_fits": float(np.mean(errors[valid])) if valid.any() else np.nan,
                })
            for comparator in METHODS[1:]:
                differences = matrix["Minimum width"].to_numpy() - matrix[comparator].to_numpy()
                complete = bool(np.isfinite(differences).all())
                boot = np.mean(differences[indices], axis=1) if complete else np.full(bootstrap_count, np.nan)
                comparisons.append({
                    "order": order, "shape_id": shape_id, "alpha": alpha, "beta": beta,
                    "comparator": comparator, "process_count": count,
                    "available_pairs": int(np.isfinite(differences).sum()),
                    "mean_mae_difference": float(np.mean(differences)) if complete else np.nan,
                    "difference_mcse": float(np.std(differences, ddof=1) / np.sqrt(count)) if complete and count > 1 else np.nan,
                    "ci_lower": float(np.quantile(boot, .025)) if complete else np.nan,
                    "ci_upper": float(np.quantile(boot, .975)) if complete else np.nan,
                    "minimum_width_lower_error_count": int((differences < 0).sum()) if complete else np.nan,
                })
    summary, paired = pd.DataFrame(summaries), pd.DataFrame(comparisons)
    _atomic_csv(summary, run["directory"] / "accuracy_summary.csv")
    _atomic_csv(paired, run["directory"] / "paired_comparisons.csv")
    (run["directory"] / "summary_configuration.json").write_text(_json({
        "bootstrap_count": bootstrap_count, "bootstrap_seed": bootstrap_seed,
        "resampling_unit": "coefficient-vector model_id within order; shared indices across shapes and methods",
        "paired_difference": "minimum-width MAE minus comparator MAE; negative favors minimum width",
    }))
    return summary, paired


def plot_coefficient_estimation(summary, paired, pdf_directory, image_directory):
    """Display and export the final accuracy and paired-comparison figures."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    pdf_directory, image_directory = Path(pdf_directory), Path(image_directory)
    pdf_directory.mkdir(parents=True, exist_ok=True)
    image_directory.mkdir(parents=True, exist_ok=True)
    shapes = summary[["shape_id", "alpha", "beta"]].drop_duplicates()
    colors = dict(zip(METHODS, ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]))
    saved = []
    with plt.rc_context({"font.size": 10, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False}):
        figure, axes = plt.subplots(2, len(shapes), figsize=(4 * len(shapes), 8), squeeze=False,
                                    sharex=True, sharey="row", constrained_layout=True)
        for column, shape in enumerate(shapes.itertuples(index=False)):
            panel = summary.loc[summary["shape_id"].eq(shape.shape_id)]
            for method in METHODS:
                rows = panel.loc[panel["method"].eq(method)].sort_values("order")
                x = rows["order"].to_numpy()
                for index, metric in enumerate(("mean_mae", "median_mae")):
                    axes[index, column].plot(x, rows[metric], marker="o", color=colors[method], label=method)
                axes[0, column].fill_between(x, rows["mean_ci_lower"].to_numpy(), rows["mean_ci_upper"].to_numpy(),
                                            color=colors[method], alpha=.12)
            axes[0, column].set_title(f"Beta({shape.alpha:g}, {shape.beta:g})")
            for index in range(2):
                axes[index, column].set_yscale("log")
                axes[index, column].set_xticks(sorted(panel["order"].unique()))
                axes[index, column].grid(alpha=.2)
            axes[1, column].set_xlabel("Known AR order p")
        axes[0, 0].set_ylabel("Mean coefficient MAE")
        axes[1, 0].set_ylabel("Median coefficient MAE")
        handles = [Line2D([0], [0], color=colors[method], marker="o", label=method) for method in METHODS]
        figure.legend(handles=handles, loc="outside lower center", ncol=len(METHODS))
        figure.suptitle("Coefficient estimation across AR orders and Beta innovation shapes\n"
                         "Shading: paired 95% bootstrap intervals for the mean; failed fits remain reported in tables")
        for suffix, directory in (("pdf", pdf_directory), ("png", image_directory)):
            path = directory / f"01_coefficient_accuracy.{suffix}"
            figure.savefig(path, bbox_inches="tight", **({"dpi": 180} if suffix == "png" else {}))
            saved.append(path)
        plt.show()
        plt.close(figure)

        figure, axes = plt.subplots(1, len(shapes), figsize=(4 * len(shapes), 4.5), squeeze=False,
                                    sharey=True, constrained_layout=True)
        for column, shape in enumerate(shapes.itertuples(index=False)):
            axis = axes[0, column]
            axis.axhline(0, color="0.4", linestyle="--", linewidth=1)
            for comparator in METHODS[1:]:
                rows = paired.loc[paired["shape_id"].eq(shape.shape_id) & paired["comparator"].eq(comparator)].sort_values("order")
                x = rows["order"].to_numpy()
                axis.plot(x, rows["mean_mae_difference"], marker="o", color=colors[comparator], label=comparator)
                axis.fill_between(x, rows["ci_lower"].to_numpy(), rows["ci_upper"].to_numpy(), color=colors[comparator], alpha=.12)
            axis.set(title=f"Beta({shape.alpha:g}, {shape.beta:g})", xlabel="Known AR order p",
                     xticks=sorted(paired["order"].unique()))
            axis.grid(alpha=.2)
        axes[0, 0].set_ylabel("Mean MAE difference: minimum width − comparator")
        handles = [Line2D([0], [0], color=colors[method], marker="o", label=method) for method in METHODS[1:]]
        figure.legend(handles=handles, loc="outside lower center", ncol=len(handles))
        figure.suptitle("Paired coefficient-error comparisons\nNegative values favor minimum width; shading: 95% bootstrap intervals")
        for suffix, directory in (("pdf", pdf_directory), ("png", image_directory)):
            path = directory / f"02_paired_mae_differences.{suffix}"
            figure.savefig(path, bbox_inches="tight", **({"dpi": 180} if suffix == "png" else {}))
            saved.append(path)
        plt.show()
        plt.close(figure)
    return saved
