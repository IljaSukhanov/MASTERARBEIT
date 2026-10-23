"""Shared experiment runners and summaries for the AR notebooks."""

from pathlib import Path
from time import perf_counter
import os
import hashlib
import itertools
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy.signal import residuez
from threadpoolctl import threadpool_limits

from .classical_selectors import (
    conditional_gaussian_qml, horvath_linf_coefficients, yule_walker_coefficients,
)
from .coefficient_errors import COEFFICIENT_ERROR_METRIC, coefficient_mae
from .descent import (
    batched_geometric_width_curves, fit_ar3_mae_prefix, fit_ar3_warm_start_rows,
)
from .breakpoint_selectors import (
    all_population_selectors, descent_chain, width_selectors,
)
from .population_widths import population_ar3_widths
from .experiment_settings import (
    AR_DIRECTORY, AR3_MAE_CONFIGURATION, AR3_MAE_OUTPUT_DIRECTORY, WARM_START_VARIANTS,
)
from .functions import (
    ar_impulse_response, is_stationary, jones_ar_coefficients,
    simulate_uniform_ar, sliding_window_point_cloud,
)


def count_outcomes(selected_orders, *, true_order):
    return {
        "Correct": int((selected_orders == true_order).sum()),
        "Under": int((selected_orders < true_order).sum()),
        "Over": int((selected_orders > true_order).sum()),
    }


def run_mae_replication(replication, configuration, cache_directory, resume=True):
    """Fit all four methods to one independently seeded, nested sample path."""
    cache_path = Path(cache_directory) / f"replication_{replication:04d}.csv"
    expected_rows = len(configuration["T_values"]) * len(configuration["methods"])
    if resume and cache_path.exists():
        cached = pd.read_csv(cache_path)
        if (len(cached) == expected_rows
                and not cached.duplicated(["T", "method"]).any()
                and (cached["replication"] == replication).all()
                and set(cached["T"]) == set(configuration["T_values"])
                and set(cached["method"]) == set(configuration["methods"])
                and "coefficient_mae" in cached
                and "coefficient_error_metric" in cached
                and cached["coefficient_error_metric"].eq(COEFFICIENT_ERROR_METRIC).all()):
            return cached
        raise RuntimeError(f"Incomplete or inconsistent cache: {cache_path}")
    true_phi = np.asarray(configuration["phi"], dtype=float)
    order = len(true_phi)
    k = configuration["k"]
    common_K = configuration["K"]
    process_generator = np.random.default_rng(np.random.SeedSequence([
        configuration["process_seed"], replication,
    ]))
    rows = []
    with threadpool_limits(limits=1):
        series = simulate_uniform_ar(
            true_phi, sample_size=max(configuration["T_values"]),
            burnin=configuration["burnin"], half_width=configuration["half_width"],
            rng=process_generator,
        )
        for T in configuration["T_values"]:
            prefix = series[:T]
            for method in configuration["methods"]:
                start_time = perf_counter()
                extra = {"minimum_width": np.nan, "abs_u_0": np.nan,
                         "residual_variance": np.nan, "quasi_log_likelihood": np.nan,
                         "linf_radius": np.nan, "linf_duality_gap": np.nan}
                estimate = np.full(order, np.nan)
                status, error_message = "ok", ""
                try:
                    if method == "Gradient descent":
                        series_for_k = prefix[common_K - k:]
                        cloud = sliding_window_point_cloud(series_for_k, k)
                        _, widths, _, directions = batched_geometric_width_curves(
                            series_for_k, k, seed=configuration["optimizer_seed"],
                            replication=replication - 1,
                            starts=configuration["starts"], iterations=configuration["iterations"],
                            initial_step=configuration["initial_step"],
                            geometric_decay=configuration["geometric_decay"],
                            return_directions=True, use_convex_hull=configuration["use_convex_hull"],
                        )
                        direction = directions[k - 1]
                        extra["minimum_width"] = float(widths[k - 1])
                        extra["abs_u_0"] = float(abs(direction[0]))
                        if not np.isclose(np.linalg.norm(direction), 1.0, atol=1e-10):
                            raise RuntimeError("The descent direction is not a unit vector.")
                        if not np.isclose(np.ptp(cloud @ direction), widths[k - 1], atol=1e-10):
                            raise RuntimeError("The direction does not reproduce its width.")
                        if abs(direction[0]) < configuration["invalid_u_0_threshold"]:
                            raise ValueError("Cannot recover coefficients: near-zero u_0.")
                        estimate = -direction[1:] / direction[0]
                    elif method == "Yule-Walker":
                        estimate = yule_walker_coefficients(prefix, order)
                    elif method == "Conditional QML":
                        estimate, variance, log_likelihood = conditional_gaussian_qml(prefix, order)
                        extra["residual_variance"] = variance
                        extra["quasi_log_likelihood"] = log_likelihood
                    elif method == "Horvath L-infinity LP":
                        estimate, radius, duality_gap = horvath_linf_coefficients(
                            prefix, order, use_convex_hull=configuration["use_convex_hull"],
                        )
                        extra["linf_radius"] = radius
                        extra["linf_duality_gap"] = duality_gap
                    else:
                        raise ValueError(f"Unknown method: {method}")
                    if estimate.shape != true_phi.shape or not np.all(np.isfinite(estimate)):
                        raise ValueError("The coefficient estimate is invalid.")
                    errors = estimate - true_phi
                    mae = coefficient_mae(estimate, true_phi)
                    if not np.isfinite(mae):
                        raise ValueError("The coefficient MAE is not finite.")
                    estimated_stationary = bool(is_stationary(estimate)[0])
                except Exception as exc:
                    status, error_message = "failed", f"{type(exc).__name__}: {exc}"
                    errors = np.full(order, np.nan)
                    mae = np.nan
                    estimated_stationary = np.nan
                elapsed = perf_counter() - start_time
                row = {
                    "replication": replication, "T": T, "method": method,
                    "status": status, "error_message": error_message,
                    "coefficient_mae": mae,
                    "coefficient_error_metric": COEFFICIENT_ERROR_METRIC,
                    "estimated_stationary": estimated_stationary,
                    "runtime_seconds": elapsed, **extra,
                }
                for j in range(order):
                    row[f"estimated_phi_{j + 1}"] = float(estimate[j])
                    row[f"absolute_error_phi_{j + 1}"] = float(abs(errors[j]))
                rows.append(row)
    result = pd.DataFrame(rows)
    temporary_path = cache_path.with_suffix(f".tmp.{os.getpid()}")
    result.to_csv(temporary_path, index=False)
    temporary_path.replace(cache_path)
    return result


def fit_ar3_gradient_examples(*, ar_directory=AR_DIRECTORY, project_directory=None,
                              data_directory=None, source_directory=None, population_directory=None):
    AR_DIR = Path(ar_directory)
    ROOT = Path(project_directory) if project_directory is not None else AR_DIR.parents[1]
    DATA_DIR = Path(data_directory) if data_directory is not None else AR_DIR / "outputs/csv/ar3_gradient_figure_2_4_examples"
    SOURCE = Path(source_directory) if source_directory is not None else AR_DIR / "outputs/csv/ar3_order_selection_warm_starts"
    POP_SOURCE = Path(population_directory) if population_directory is not None else AR_DIR / "outputs/csv/ar3_population_breakpoint_selectors"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings = json.loads((SOURCE / "configuration.json").read_text())
    configuration = {
        "source_configuration": settings,
        "replications": list(range(1, 11)),
        "selection_of_examples": "First ten runs, without selection by performance",
        "optimizer": "batched_geometric_width_curves, padded previous direction",
        "warm_start_previous_direction": True,
        "reference_figure": str(ROOT / "Abschlussarbeit_2021/LaTeX/images/AR3_experiment.pdf"),
        "reference_notebook": str(AR_DIR / "xx_notebooks_figures" / "ar_thesis_figures_2_1_to_2_3.ipynb"),
        "figure_selector": "Original descent chain on raw widths",
        "population_direction_source": "Recomputed global truncated zonotope minima in ar_tools/population_widths.py",
        "population_direction_truncation_bound": 1e-8,
        "directional_population_width_truncation_target": 1e-9,
        "source_hashes": {name: hashlib.sha256((AR_DIR / name).read_bytes()).hexdigest()
                          for name in ["ar_tools/descent.py", "ar_tools/functions.py"]},
    }
    (DATA_DIR / "configuration.json").write_text(json.dumps(configuration, indent=2) + "\n")
    rows, summaries = [], []
    errors = {"maximum_cached_width_difference": 0.0, "maximum_cached_direction_difference": 0.0,
              "maximum_width_reconstruction_error": 0.0, "maximum_direction_norm_error": 0.0,
              "maximum_directional_population_tail_bound": 0.0,
              "maximum_population_cache_coefficient_roundoff_difference": 0.0}
    with threadpool_limits(limits=1):
        for rep in range(1, 11):
            series = pd.read_csv(SOURCE / "processes" / f"process_{rep:03d}.csv",
                                 float_precision="round_trip").X_t.to_numpy()
            saved = pd.read_csv(SOURCE / "replications" / f"replication_{rep:03d}.csv",
                                float_precision="round_trip")
            saved = saved[saved.variant == "Padded warm start"].sort_values("k")
            phi = np.array(json.loads(saved.coefficients.iloc[0]))
            _, widths, _, directions = batched_geometric_width_curves(
                series, 10, seed=settings["seed"], replication=rep - 1,
                starts=settings["starts"], iterations=settings["iterations"],
                initial_step=settings["initial_step"], geometric_decay=settings["geometric_decay"],
                return_directions=True, warm_start_previous_direction=True,
            )
            errors["maximum_cached_width_difference"] = max(errors["maximum_cached_width_difference"],
                float(np.max(np.abs(widths - saved.minimum_width.to_numpy()))))
            population = pd.read_csv(POP_SOURCE / "replications" / f"replication_{rep:03d}.csv",
                                     float_precision="round_trip").sort_values("k")
            cached_phi = np.array(json.loads(population.coefficients.iloc[0]))
            np.testing.assert_allclose(phi, cached_phi, rtol=0, atol=1e-14)
            errors["maximum_population_cache_coefficient_roundoff_difference"] = max(
                errors["maximum_population_cache_coefficient_roundoff_difference"], float(np.abs(phi - cached_phi).max()))
            population_fit = population_ar3_widths(phi, max_k=10)
            residue, poles, direct = residuez([1.0], np.r_[1.0, -phi])
            rho, constant = float(np.max(np.abs(poles))), float(np.abs(residue).sum())
            assert rho < 1 and len(direct) == 0
            history_count = max(10, int(np.ceil(np.log(1e-9 * (1 - rho)
                / (2 * np.sqrt(10) * constant)) / np.log(rho))) + 9)
            psi = ar_impulse_response(phi, history_count - 1)
            order = descent_chain(widths)[0]
            q = order + 1
            summaries.append({"replication": rep, "phi_1": phi[0], "phi_2": phi[1], "phi_3": phi[2],
                              "true_order": 3, "selected_order": order, "selected_dimension": q,
                              "selected_width": float(widths[q - 1]),
                              "descent_chain_raw_order": descent_chain(widths)[0],
                              "descent_chain_log_order": descent_chain(np.log(widths))[0],
                              "population_plateau": float(2 / np.linalg.norm(np.r_[1., -phi]))})
            for k, (width, direction) in enumerate(zip(widths, directions), 1):
                cloud = sliding_window_point_cloud(series, k)
                reconstructed = float(np.ptp(cloud @ direction))
                original_u = saved.iloc[k - 1][[f"u_{j}" for j in range(k)]].to_numpy(dtype=float)
                errors["maximum_width_reconstruction_error"] = max(errors["maximum_width_reconstruction_error"],
                                                                     abs(reconstructed - width))
                errors["maximum_cached_direction_difference"] = max(errors["maximum_cached_direction_difference"],
                                                                      float(np.abs(direction - original_u).max()))
                errors["maximum_direction_norm_error"] = max(errors["maximum_direction_norm_error"],
                                                               abs(float(np.linalg.norm(direction)) - 1))
                if k <= 3:
                    u_pop = population_fit["directions"][k - 1].copy()
                else:
                    u_pop = np.r_[1., -phi, np.zeros(k - 4)]
                u_pop /= np.linalg.norm(u_pop)
                orange = float(2 * np.abs(np.convolve(psi, direction)[:history_count]).sum())
                tail = float(2 * np.sqrt(k) * constant * rho ** (history_count - k + 1) / (1 - rho))
                errors["maximum_directional_population_tail_bound"] = max(
                    errors["maximum_directional_population_tail_bound"], tail)
                assert orange + tail >= width - 1e-10
                row = {"replication": rep, "k": k, "empirical_fitted_width": float(width),
                       "population_at_empirical_direction": orange,
                       "population_directional_tail_bound": tail,
                       "empirical_at_population_direction": float(np.ptp(cloud @ u_pop)),
                       "population_minimum_width": float(population_fit["widths"][k - 1]
                                                          + population_fit["tail_bounds"][k - 1] / 2),
                       "population_plateau": summaries[-1]["population_plateau"] if k >= 4 else np.nan,
                       "coefficients": json.dumps(phi.tolist()), "population_history_count": history_count}
                row.update({f"gradient_u_{j}": float(direction[j]) if j < k else np.nan for j in range(10)})
                row.update({f"population_u_{j}": float(u_pop[j]) if j < k else np.nan for j in range(10)})
                rows.append(row)
            print(f"Gradient example {rep}/10 reproduced", flush=True)
    assert errors["maximum_cached_width_difference"] < 1e-12
    assert errors["maximum_cached_direction_difference"] < 1e-12
    assert errors["maximum_directional_population_tail_bound"] <= 1e-9 * (1 + 1e-10)
    curves, summary = pd.DataFrame(rows), pd.DataFrame(summaries)
    curves.to_csv(DATA_DIR / "figure_data.csv", index=False)
    summary.to_csv(DATA_DIR / "selection_summary.csv", index=False)
    errors.update({"processes": 10, "fitted_widths_and_directions": 100,
                  "monotone_curves_within_1e-12": int(sum(np.all(np.diff(g.empirical_fitted_width) <= 1e-12)
                                                        for _, g in curves.groupby("replication"))),
                  "correct_descent_chain_selections": int((summary.selected_order == 3).sum())})
    (DATA_DIR / "verification.json").write_text(json.dumps(errors, indent=2) + "\n")
    return curves, summary


def run_ar3_mae_process(replication_index: int, *, configuration, cache_directory) -> pd.DataFrame:
    CONFIG = configuration
    CACHE_DIR = Path(cache_directory)
    cache_path = CACHE_DIR / f"process_{replication_index + 1:03d}.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        if (
            len(cached) == len(CONFIG["sample_sizes"])
            and cached["T"].tolist() == CONFIG["sample_sizes"]
            and cached["replication"].eq(replication_index + 1).all()
            and "coefficient_mae" in cached
            and "coefficient_error_metric" in cached
            and cached["coefficient_error_metric"].eq(COEFFICIENT_ERROR_METRIC).all()
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
            "coefficient_error_metric": COEFFICIENT_ERROR_METRIC,
        }
        for index, value in enumerate(true_phi, start=1):
            row[f"true_phi_{index}"] = float(value)
        try:
            result = fit_ar3_mae_prefix(series, replication_index, sample_size, configuration=CONFIG)
            estimate = result.pop("estimate")
            row.update(result)
            absolute_errors = np.abs(estimate - true_phi)
            for index, (value, error) in enumerate(zip(estimate, absolute_errors), start=1):
                row[f"estimated_phi_{index}"] = float(value)
                row[f"absolute_error_phi_{index}"] = float(error)
            row["coefficient_mae"] = coefficient_mae(estimate, true_phi)
            if not np.isfinite(row["coefficient_mae"]):
                raise ValueError("The coefficient MAE is not finite.")
        except Exception as exc:
            row["status"] = "failed"
            row["error_message"] = f"{type(exc).__name__}: {exc}"
            row["coefficient_mae"] = np.nan
        row["runtime_seconds"] = perf_counter() - start_time
        rows.append(row)

    frame = pd.DataFrame(rows)
    temporary_path = cache_path.with_suffix(f".tmp.{os.getpid()}")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(cache_path)
    return frame


def run_ar3_coefficient_mae_experiment(*, configuration=None,
                                       output_directory=AR3_MAE_OUTPUT_DIRECTORY, n_jobs=4) -> None:
    CONFIG = dict(AR3_MAE_CONFIGURATION if configuration is None else configuration)
    CONFIG["coefficient_error_metric"] = COEFFICIENT_ERROR_METRIC
    OUTPUT_DIR = Path(output_directory)
    CACHE_DIR = OUTPUT_DIR / "processes"
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
        completed = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
            delayed(run_ar3_mae_process)(index, configuration=CONFIG, cache_directory=CACHE_DIR) for index in range(CONFIG["replications"])
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
        losses = group["coefficient_mae"].to_numpy(dtype=float)
        failed_count = int((group["status"] != "ok").sum())
        summary_rows.append(
            {
                "T": sample_size,
                "processes": len(group),
                "failed_fits": failed_count,
                # A failed fit makes the unconditional MAE unavailable.
                "coefficient_mae": float(np.mean(losses)),
                "median_process_mae": float(np.median(losses)),
                "max_process_mae": float(np.max(losses)),
                "mean_runtime_seconds": float(group["runtime_seconds"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "mae_summary.csv", index=False)
    print(f"Finished {expected} fits; failed fits: {len(failures)}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.8g}"))
    print(f"Results: {OUTPUT_DIR}")


def run_ar3_warm_start_experiment(*, ar_directory=AR_DIRECTORY, data_directory=None,
                                  original_directory=None, image_directory=None,
                                  pdf_directory=None, variants=WARM_START_VARIANTS):
    AR_DIR = Path(ar_directory)
    DATA_DIR = Path(data_directory) if data_directory is not None else AR_DIR / "outputs/csv/ar3_order_selection_warm_starts"
    ORIGINAL_DIR = Path(original_directory) if original_directory is not None else AR_DIR / "outputs/csv/ar_order_minimum_width"
    IMAGE_DIR = Path(image_directory) if image_directory is not None else AR_DIR / "outputs/images/ar3_order_selection_warm_starts"
    PDF_DIR = Path(pdf_directory) if pdf_directory is not None else AR_DIR / "outputs/pdfs/ar3_order_selection_warm_starts"
    VARIANTS = variants
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "processes").mkdir(exist_ok=True)
    (DATA_DIR / "replications").mkdir(exist_ok=True)
    settings = json.loads((ORIGINAL_DIR / "configuration.json").read_text())
    original = pd.read_csv(ORIGINAL_DIR / "width_curves.csv")
    original = original[original.true_order == 3].copy()
    assert len(original) == 500 and original.replication.nunique() == 50
    configuration = {
        "true_order": 3, "replications": settings["replications_per_order"],
        "sample_size": settings["sample_size"], "burnin": settings["burnin"],
        "half_width": settings["half_width"], "k_values": settings["k_values"],
        "seed": settings["seeds"]["3"], "starts": settings["starts"],
        "iterations": settings["iterations"], "initial_step": settings["initial_step"],
        "geometric_decay": settings["geometric_decay"],
        "warm_start": "First start replaced by previous best direction padded with zero for every k >= 2",
        "observation_windows": "Original all-valid sliding windows; identical in both fresh fits",
        "source_hashes": {name: hashlib.sha256((AR_DIR / name).read_bytes()).hexdigest()
                          for name in ("ar_tools/functions.py", "ar_tools/descent.py")},
        "original_csv_sha256": hashlib.sha256((ORIGINAL_DIR / "width_curves.csv").read_bytes()).hexdigest(),
    }
    configuration_path = DATA_DIR / "configuration.json"
    if configuration_path.exists():
        if json.loads(configuration_path.read_text()) != configuration:
            raise RuntimeError("Existing rerun has different settings; preserve it in a separate directory.")
    else:
        configuration_path.write_text(json.dumps(configuration, indent=2) + "\n")

    # Check a flat prefix followed by a genuine drop, including the stationary law.
    assert descent_chain([5, 5, 5, 1, 1, 1])[0] == 3

    fitting_started = perf_counter()
    new_rows = []
    with threadpool_limits(limits=1):
        for replication in range(1, configuration["replications"] + 1):
            checkpoint = DATA_DIR / "replications" / f"replication_{replication:03d}.csv"
            series_path = DATA_DIR / "processes" / f"process_{replication:03d}.csv"
            if checkpoint.exists() and series_path.exists():
                cached = pd.read_csv(checkpoint)
                assert len(cached) == 20 and set(cached.variant) == set(VARIANTS[1:])
                new_rows.extend(cached.to_dict("records"))
                continue
            generator = np.random.default_rng(np.random.SeedSequence([
                configuration["seed"], replication - 1, 0,
            ]))
            phi = jones_ar_coefficients(3, rng=generator)
            old_group = original[original.replication == replication].sort_values("k")
            assert np.allclose(phi, json.loads(old_group.iloc[0].coefficients), rtol=0, atol=1e-14)
            series = simulate_uniform_ar(phi, configuration["sample_size"],
                                         configuration["burnin"], configuration["half_width"],
                                         rng=generator)
            pd.DataFrame({"t": np.arange(1, len(series) + 1), "X_t": series}).to_csv(series_path, index=False)
            pair_rows = fit_ar3_warm_start_rows(series, phi, settings, replication, warm=False, variants=VARIANTS)
            pair_rows += fit_ar3_warm_start_rows(series, phi, settings, replication, warm=True, variants=VARIANTS)
            pd.DataFrame(pair_rows).to_csv(checkpoint, index=False)
            new_rows.extend(pair_rows)
            if replication % 5 == 0:
                print(f"Completed {replication}/50 paired fits ({perf_counter() - fitting_started:.1f}s)", flush=True)

        original["variant"] = VARIANTS[0]
        width_curves = pd.concat([original, pd.DataFrame(new_rows)], ignore_index=True)
        width_curves = width_curves.sort_values(["variant", "replication", "k"])
        width_curves.to_csv(DATA_DIR / "width_curves.csv", index=False)
        selections, diagnostics, chain_rows, comparison_rows = [], [], [], []
        for (variant, replication), group in width_curves.groupby(["variant", "replication"]):
            m = group.sort_values("k").minimum_width.to_numpy()
            diff = np.diff(m)
            diagnostics.append({"variant": variant, "replication": replication,
                                "upward_steps": int(np.sum(diff > 1e-12)),
                                "positive_steps_at_machine_precision": int(np.sum((diff > 0) & (diff <= 1e-12))),
                                "maximum_increase": max(0.0, float(diff.max()))})
            selected = width_selectors(m)
            if variant == VARIANTS[2]:
                order, drops, remaining, alpha, stationary, odds, residual = descent_chain(m)
                selected["Descent chain"] = order
                for i in range(len(m)):
                    chain_rows.append({"replication": replication, "k": i + 1,
                                       "drop": drops[i], "remaining_reduction": remaining[i],
                                       "alpha": alpha[i], "stationary_probability": stationary[i],
                                       "odds": odds[i], "candidate": 0 < i < len(m) - 1,
                                       "stationarity_residual": residual})
            for method, order in selected.items():
                selections.append({"variant": variant, "replication": replication,
                                   "true_order": 3, "method": method,
                                   "selected_order": order, "correct": order == 3})
            if variant == VARIANTS[2]:
                fresh = width_curves[(width_curves.variant == VARIANTS[1])
                                     & (width_curves.replication == replication)].sort_values("k")
                old = original[original.replication == replication].sort_values("k")
                for k, (w, independent, historical) in enumerate(zip(m, fresh.minimum_width,
                                                                    old.minimum_width), start=1):
                    comparison_rows.append({"replication": replication, "k": k,
                                            "original_saved_width": historical,
                                            "independent_rerun_width": independent,
                                            "padded_warm_start_width": w,
                                            "warm_minus_independent": w - independent})

    selections = pd.DataFrame(selections)
    summary_rows = []
    for (variant, method), group in selections.groupby(["variant", "method"], sort=False):
        q = group.selected_order
        summary_rows.append({"variant": variant, "method": method, "correct": int((q == 3).sum()),
                             "under": int((q < 3).sum()), "over": int((q > 3).sum()), "total": len(q)})
    summary = pd.DataFrame(summary_rows)
    selections.to_csv(DATA_DIR / "selections.csv", index=False)
    summary.to_csv(DATA_DIR / "summary.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(DATA_DIR / "monotonicity_diagnostics.csv", index=False)
    pd.DataFrame(chain_rows).to_csv(DATA_DIR / "descent_chain_scores.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(DATA_DIR / "paired_width_comparison.csv", index=False)
    (DATA_DIR / "numerical_notes.json").write_text(json.dumps({
        "warm_monotonicity_check": "32 * machine epsilon * max(1, maximum absolute width)",
        "descent_chain": "Reject larger upward steps; set only machine-precision-sized drops within that tolerance to zero and compute remaining reduction by summing the resulting drops",
        "curve_postprocessing": "None; saved widths are the actual optimizer outputs",
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2) + "\n")

    # Verify signs and order indexing directly from the retained source curves.
    historical_orders = {"First difference": {}, "Second difference": {}}
    for replication, group in original.groupby("replication"):
        m = group.sort_values("k").minimum_width.to_numpy()
        historical_orders["First difference"][replication] = int(np.argmax((m[:-1] - m[1:])[:-1]) + 1)
        historical_orders["Second difference"][replication] = int(np.argmax(np.diff(m, n=2)) + 1)
    for label, orders in historical_orders.items():
        actual = selections[(selections.variant == VARIANTS[0]) & (selections.method == label)]
        actual = actual.set_index("replication").selected_order.sort_index()
        expected = pd.Series(orders).sort_index()
        assert np.array_equal(actual, expected)
    # Keep importing the plotting helpers from changing the caller's style.
    import matplotlib
    with matplotlib.rc_context():
        from .plotting_functions import plot_ar3_warm_start_results
    plot_ar3_warm_start_results(width_curves, summary, image_directory=IMAGE_DIR,
                                pdf_directory=PDF_DIR, variants=VARIANTS)
    print("\nCorrect selections out of 50:", flush=True)
    print(summary.pivot(index="method", columns="variant", values="correct").to_string(), flush=True)
    print("\nCurves with upward steps:", flush=True)
    diagnostic_frame = pd.DataFrame(diagnostics)
    print(diagnostic_frame.assign(nonmonotone=lambda d: d.upward_steps > 0)
          .groupby("variant").nonmonotone.sum().to_string(), flush=True)
    return DATA_DIR


def write_ar3_population_verification(widths, selections, stability, *, data_directory):
    DATA_DIR = Path(data_directory)
    verification = {
        "population_curves": int(widths.replication.nunique()),
        "width_values": len(widths),
        "selectors": int(selections.method.nunique()),
        "selection_results": len(selections),
        "population_plateau_analytical_and_identical_from_k4": bool(
            widths[widths.k >= 4].groupby("replication").population_width.nunique().eq(1).all()),
        "maximum_omitted_tail_bound": float(widths.tail_error_bound.max()),
        "maximum_angular_sweep_roundoff_residual": float(widths.maximum_sweep_roundoff_residual.max()),
        "maximum_pole_formula_error": float(widths.pole_formula_max_error.max()),
        "stable_at_all_eight_truncation_corners": bool(stability.stable_at_all_eight_corners.all()),
        "floating_point_operations_interval_certified": False,
    }
    (DATA_DIR / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")


def run_ar3_population_selector_experiment(*, ar_directory=AR_DIRECTORY, data_directory=None,
                                           image_directory=None, pdf_directory=None, tail_target=1e-8,
                                           variants=WARM_START_VARIANTS):
    AR_DIR = Path(ar_directory)
    DATA_DIR = Path(data_directory) if data_directory is not None else AR_DIR / "outputs/csv/ar3_population_breakpoint_selectors"
    IMAGE_DIR = Path(image_directory) if image_directory is not None else AR_DIR / "outputs/images/ar3_population_breakpoint_selectors"
    PDF_DIR = Path(pdf_directory) if pdf_directory is not None else AR_DIR / "outputs/pdfs/ar3_population_breakpoint_selectors"
    TAIL_TARGET, VARIANTS = tail_target, variants
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    checkpoints = DATA_DIR / "replications"
    checkpoints.mkdir(exist_ok=True)
    source = AR_DIR / "outputs/csv/ar_order_minimum_width/width_curves.csv"
    parameters = pd.read_csv(source)
    parameters = parameters[(parameters.true_order == 3) & (parameters.k == 1)].sort_values("replication")
    assert len(parameters) == 50
    configuration = {
        "true_order": 3, "replications": 50, "max_k": 10, "tail_target": TAIL_TARGET,
        "pre_plateau_method": "Global finite-zonotope facet enumeration by angular sweeps",
        "plateau_method": "Analytical 2 / sqrt(1 + phi_1^2 + phi_2^2 + phi_3^2) for k >= 4",
        "width_estimate": "Midpoint of truncated minimum and truncated minimum plus analytical tail bound",
        "floating_point": "Not interval certified; angular-sweep and pole-formula residuals recorded",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "width_algorithm_sha256": hashlib.sha256((AR_DIR / "ar_tools/population_widths.py").read_bytes()).hexdigest(),
    }
    config_path = DATA_DIR / "configuration.json"
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise RuntimeError("Existing population experiment has different settings; preserve it separately.")
    config_path.write_text(json.dumps(configuration, indent=2) + "\n")

    records = []
    start = perf_counter()
    with threadpool_limits(limits=1):
        for parameter in parameters.itertuples(index=False):
            rep = int(parameter.replication)
            checkpoint = checkpoints / f"replication_{rep:03d}.csv"
            if checkpoint.exists():
                frame = pd.read_csv(checkpoint, float_precision="round_trip")
                assert len(frame) == 10
                records.extend(frame.to_dict("records"))
                continue
            result = population_ar3_widths(np.array(json.loads(parameter.coefficients)), tail_target=TAIL_TARGET)
            rows = []
            for i, finite in enumerate(result["widths"]):
                tail = result["tail_bounds"][i]
                row = {"replication": rep, "true_order": 3, "k": i + 1,
                       "population_width": finite + tail / 2, "truncated_width": finite,
                       "truncation_upper_bound": finite + tail, "tail_error_bound": tail,
                       "method": "global truncated minimum" if i < 3 else "analytical plateau",
                       "coefficients": parameter.coefficients,
                       "generator_count": result["generator_count"],
                       "spectral_radius": result["spectral_radius"],
                       "pole_formula_max_error": result["pole_formula_max_error"],
                       "maximum_sweep_roundoff_residual": result["maximum_sweep_roundoff_residual"]}
                direction = result["directions"][i] if i < 3 else np.r_[1., -np.array(json.loads(parameter.coefficients))]
                direction /= np.linalg.norm(direction)
                row.update({f"u_{j}": float(direction[j]) if j < len(direction) else np.nan for j in range(4)})
                rows.append(row)
            pd.DataFrame(rows).to_csv(checkpoint, index=False)
            records.extend(rows)
            print(f"Population curve {rep}/50: {result['generator_count']} generators ({perf_counter() - start:.1f}s)", flush=True)

        widths = pd.DataFrame(records).sort_values(["replication", "k"])
        widths.to_csv(DATA_DIR / "population_width_curves.csv", index=False)
        selections, stability, chain_rows, raw_selections = [], [], [], []
        for rep, group in widths.groupby("replication"):
            group = group.sort_values("k")
            m = group.population_width.to_numpy()
            chosen = all_population_selectors(m)
            raw = chosen.copy()
            for method, q in raw.items():
                raw_selections.append({"replication": rep, "method": method, "selected_order": q})
            for method, q in chosen.items():
                selections.append({"replication": rep, "true_order": 3, "method": method,
                                   "selected_order": q, "correct": q == 3})
            # All eight corners of the analytical truncation intervals.
            choices = {method: set() for method in chosen}
            for corner in itertools.product((0, 1), repeat=3):
                perturbed = m.copy()
                perturbed[:3] = group.truncated_width.to_numpy()[:3] + np.array(corner) * group.tail_error_bound.to_numpy()[:3]
                for method, q in all_population_selectors(perturbed).items():
                    choices[method].add(q)
            for method, orders in choices.items():
                stability.append({"replication": rep, "method": method,
                                  "corner_orders": json.dumps(sorted(orders)),
                                  "stable_at_all_eight_corners": len(orders) == 1 and chosen[method] in orders})
            chain = descent_chain(m)
            for i in range(10):
                chain_rows.append({"replication": rep, "k": i + 1, "drop": chain[1][i],
                                   "remaining_reduction": chain[2][i], "stationary_probability": chain[4][i],
                                   "odds": chain[5][i], "candidate": 0 < i < 9})
            if int(rep) % 10 == 0:
                print(f"Selectors and truncation checks {rep}/50", flush=True)

    selections = pd.DataFrame(selections)
    summary = selections.groupby("method", sort=False).selected_order.agg(
        correct=lambda q: int((q == 3).sum()), under=lambda q: int((q < 3).sum()),
        over=lambda q: int((q > 3).sum()), total="count").reset_index()
    selections.to_csv(DATA_DIR / "selections.csv", index=False)
    pd.DataFrame(raw_selections).to_csv(DATA_DIR / "raw_float_selections.csv", index=False)
    summary.to_csv(DATA_DIR / "summary.csv", index=False)
    pd.DataFrame(stability).to_csv(DATA_DIR / "truncation_corner_checks.csv", index=False)
    pd.DataFrame(chain_rows).to_csv(DATA_DIR / "descent_chain_scores.csv", index=False)
    write_ar3_population_verification(widths, selections, pd.DataFrame(stability), data_directory=DATA_DIR)
    old_population = pd.read_csv(AR_DIR / "outputs/csv/ar_order_minimum_width/population_width_curves.csv")
    comparison = widths.merge(old_population[old_population.true_order == 3],
                              on=["true_order", "replication", "k"], suffixes=("", "_cached"))
    comparison["global_minus_cached"] = comparison.population_width - comparison.population_minimum_width
    comparison.to_csv(DATA_DIR / "cached_population_comparison.csv", index=False)
    empirical = pd.read_csv(AR_DIR / "outputs/csv/ar3_order_selection_warm_starts/summary.csv")
    empirical = empirical[empirical.variant == VARIANTS[2]].drop(columns="variant")
    paired = summary.merge(empirical, on="method", how="left", suffixes=("_population", "_warm_empirical"))
    paired.to_csv(DATA_DIR / "population_vs_warm_empirical.csv", index=False)
    (DATA_DIR / "coverage_notes.json").write_text(json.dumps({
        "tested": "Two difference rules and descent chain (3 total)",
        "birth_death_inverse_rule": "Undefined on population plateau: similarity matrix singular and reciprocal-drop rates infinite. No stationary-chain accuracy assigned.",
        "other_functionals": "BIC/PACF require time-series data; not evaluated by substituting geometric population widths into them.",
        "tail_corner_checks": "Sensitivity check, not a proof over the entire uncertainty interval",
        "descent_chain_source": "/Users/iljasukhanov/Downloads/elbow_point_selection_preview.pdf, Definitions 1.2.8-1.2.10 and Eq. (1.6), pp. 5-6",
    }, indent=2) + "\n")
    import matplotlib
    with matplotlib.rc_context():
        from .plotting_functions import plot_ar3_population_selector_results
    plot_ar3_population_selector_results(widths, summary, paired,
                                        image_directory=IMAGE_DIR, pdf_directory=PDF_DIR)
    print("\nPopulation results:", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\nMaximum tail bound:", widths.tail_error_bound.max(), flush=True)
    print("Maximum angular-sweep roundoff residual:", widths.maximum_sweep_roundoff_residual.max(), flush=True)
    print("Maximum absolute change versus cached population widths:", comparison.global_minus_cached.abs().max(), flush=True)
    print("Unstable truncation corners:", int((~pd.DataFrame(stability).stable_at_all_eight_corners).sum()), flush=True)
    return DATA_DIR
