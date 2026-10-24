"""Checkpointed Figure 3.5, localized, and exact Gaussian bandwidth experiments.

The Figure 3.5 and localized profiles share Gaussian samples, Fourier
divergence, and exact grid ISE/Q-hat controls. The exact Gaussian benchmark
uses independent per-repetition seeds and exact pair divergence. All profiles
use the central Perron solver and retain their original selection objectives.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .density import exact_oracle_ise_values
from .fourier import FOURIER_L2_METHOD_VERSION, beta2_divergence_matrix_fourier
from .perron import PERRON_SOLVER_VERSION, perron_eigenpair_from_divergence
from .similarities import beta2_divergence_matrix


RUNNER_VERSION = "checkpointed_beta2_distributions_v1"
SELECTOR_NAMES = ("spectral selections", "ISE-optimal grid bandwidths", "Q-hat minimizers")


def exact_beta2_qhat_values(samples: np.ndarray, h_grid: np.ndarray) -> np.ndarray:
    """Evaluate the original Figure 3.5 Q-hat objective on its candidate grid."""
    squared_differences = (samples[:, None] - samples[None, :]) ** 2
    values = np.empty(len(h_grid), dtype=float)
    for index, bandwidth in enumerate(h_grid):
        q = 2.0 * bandwidth**2
        overlap = np.exp(-squared_differences / (2.0 * q)) / np.sqrt(2.0 * np.pi * q)
        second_q_factor = (
            3.0 / (4.0 * q**2)
            - 3.0 * squared_differences / (2.0 * q**3)
            + squared_differences**2 / (4.0 * q**4)
        )
        values[index] = 2.0 * bandwidth**4 * np.mean(overlap * second_q_factor)
    return values


def _sample_bank(n: int, repetitions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array([rng.normal(size=n) for _ in range(repetitions)])


def _sample_hash(samples: np.ndarray) -> str:
    return hashlib.sha256(samples.tobytes()).hexdigest()


def _read_checkpoint(path, settings, grid, samples, *, allow_prefix=False):
    """Validate provenance and completed rows before accepting any cached data."""
    with np.load(path, allow_pickle=False) as saved:
        previous = json.loads(str(saved["settings_json"].item()))
        repetitions = int(previous["M"])
        if allow_prefix:
            compatible = (
                1 <= repetitions <= settings["M"]
                and {k: v for k, v in previous.items() if k != "M"}
                == {k: v for k, v in settings.items() if k != "M"}
            )
        else:
            compatible = previous == settings
        if not compatible:
            raise ValueError("Checkpoint experiment settings or method versions do not match.")
        if not np.array_equal(saved["h_grid"], grid):
            raise ValueError("Checkpoint bandwidth grid does not match.")
        expected_hash = _sample_hash(samples[:repetitions])
        if "samples_sha256" in saved:
            previous_hash = str(saved["samples_sha256"].item())
        else:
            previous_hash = json.loads((path.parent / "validation.json").read_text())["samples_sha256"]
        if previous_hash != expected_hash:
            raise ValueError("Checkpoint sample hash does not match.")
        indices = saved["selected_indices"].copy()
        completed = saved["completed"].copy()
        diagnostics = saved["solver_diagnostics"].copy()
        selected_ise = saved["selected_ise_values"].copy() if "selected_ise_values" in saved else None
    if (
        indices.shape != (repetitions, 2 if settings.get("sampling") == "seed_plus_repetition" else 3)
        or not np.issubdtype(indices.dtype, np.integer)
        or completed.shape != (repetitions,)
        or completed.dtype != np.dtype(bool)
        or diagnostics.shape != (repetitions, 3)
    ):
        raise ValueError("Checkpoint arrays have invalid shapes or types.")
    if np.any(indices[completed] < 0) or np.any(indices[completed] >= len(grid)):
        raise ValueError("Checkpoint contains out-of-range completed selections.")
    if not np.all(np.isfinite(diagnostics[completed])):
        raise ValueError("Checkpoint contains invalid completed eigenpair diagnostics.")
    if np.any(diagnostics[completed, 0] < 0) or np.any(diagnostics[completed, 1] <= 0) or np.any(diagnostics[completed, 2] < 0):
        raise ValueError("Checkpoint contains invalid residuals, gaps, or operation counts.")
    if settings.get("sampling") == "seed_plus_repetition":
        if selected_ise is None or selected_ise.shape != indices.shape or not np.all(np.isfinite(selected_ise[completed])):
            raise ValueError("Checkpoint contains invalid selected ISE values.")
        if np.any(selected_ise[completed, 1] > selected_ise[completed, 0] + 1e-12):
            raise ValueError("Checkpoint ISE optimum exceeds spectral-choice ISE.")
    return indices, completed, diagnostics, selected_ise


def _write_csv(path, fields, rows):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def run_beta2_distribution(
    h_grid,
    *,
    n,
    repetitions,
    lambda_value,
    seed,
    data_root,
    profile="figure_3_5",
    half_width=12.0,
    spatial_points=8192,
    workers=4,
    reuse=True,
    import_directories=(),
    control_directories=(),
    validate_fourier=True,
    binning=5,
    sigma2=1.0,
):
    """Run a histogram profile with configuration-specific checkpoints.

    Historical checkpoints are read only. Matching solver/method versions,
    settings, grid, and sample hashes are required for prefix imports. Exact
    ISE/Q-hat controls can be reused from a matching current baseline run.
    """
    for name, value in (("n", n), ("repetitions", repetitions), ("workers", workers), ("binning", binning)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    grid = np.asarray(h_grid, dtype=float)
    if grid.ndim != 1 or len(grid) < 2 or not np.all(np.isfinite(grid)) or np.any(grid <= 0) or np.any(np.diff(grid) <= 0):
        raise ValueError("h_grid must be finite, positive, and strictly increasing.")
    if profile not in {"figure_3_5", "localized", "exact_gaussian"}:
        raise ValueError("Choose figure_3_5, localized, or exact_gaussian.")
    exact_profile = profile == "exact_gaussian"
    if not np.isfinite(sigma2) or sigma2 <= 0 or (not exact_profile and sigma2 != 1):
        raise ValueError("Use positive sigma2; the Fourier profiles use sigma2=1.")
    if not np.isfinite(lambda_value) or lambda_value <= 0:
        raise ValueError("lambda_value must be finite and positive.")
    if not np.isfinite(half_width) or half_width <= 0:
        raise ValueError("half_width must be finite and positive.")
    if isinstance(spatial_points, (bool, np.bool_)) or not isinstance(spatial_points, (int, np.integer)) or spatial_points < 2:
        raise ValueError("spatial_points must be an integer of at least two.")
    settings = {
        "n": int(n), "N": len(grid), "M": int(repetitions),
        "h_min": float(grid[0]), "h_max": float(grid[-1]),
        "lambda": float(lambda_value), "seed": int(seed),
        "fft_half_width": float(half_width), "fft_spatial_points": int(spatial_points),
        "l2_method": FOURIER_L2_METHOD_VERSION, "perron_solver": PERRON_SOLVER_VERSION,
    }
    if profile == "localized":
        settings["lambda_rule"] = "N**(3/2)" if lambda_value == len(grid) ** 1.5 else "fixed"
    if exact_profile:
        settings.update(l2_method="exact_gaussian_pairs_v1", sigma2=float(sigma2), sampling="seed_plus_repetition")
    settings_json = json.dumps(settings, sort_keys=True)
    identity = json.dumps({"settings": settings, "grid": grid.tolist(), "runner": RUNNER_VERSION}, sort_keys=True)
    configuration_id = hashlib.sha256(identity.encode()).hexdigest()[:16]
    directory = Path(data_root) / profile / configuration_id
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / "checkpoint.npz"
    samples = (np.array([np.random.default_rng(seed + r).normal(0, np.sqrt(sigma2), size=n)
                         for r in range(repetitions)]) if exact_profile else _sample_bank(n, repetitions, seed))
    sample_hash = _sample_hash(samples)
    methods = ("L2", "ISE") if exact_profile else SELECTOR_NAMES
    indices = np.full((repetitions, len(methods)), -1, dtype=int)
    selected_ise = np.full(indices.shape, np.nan)
    completed = np.zeros(repetitions, dtype=bool)
    diagnostics = np.full((repetitions, 3), np.nan)
    validation = {
        "runner_version": RUNNER_VERSION, "samples_sha256": sample_hash,
        "checkpoint_imports": [], "reused_control_source": None,
    }
    if reuse and checkpoint_path.exists():
        indices, completed, diagnostics, old_ise = _read_checkpoint(checkpoint_path, settings, grid, samples)
        if exact_profile:
            selected_ise[:] = old_ise
    elif reuse:
        for source in map(Path, import_directories):
            if not (source / "checkpoint.npz").is_file():
                continue
            try:
                prior_indices, prior_completed, prior_diagnostics, prior_ise = _read_checkpoint(
                    source / "checkpoint.npz", settings, grid, samples, allow_prefix=True
                )
            except (ValueError, KeyError, OSError) as error:
                validation["checkpoint_imports"].append({"source": str(source), "status": "skipped", "reason": str(error)})
                continue
            rows = np.flatnonzero(prior_completed)
            new_rows = rows[~completed[rows]]
            indices[new_rows] = prior_indices[new_rows]
            diagnostics[new_rows] = prior_diagnostics[new_rows]
            completed[new_rows] = True
            if exact_profile:
                selected_ise[new_rows] = prior_ise[new_rows]
            validation["checkpoint_imports"].append({"source": str(source), "status": "imported", "repetitions": len(new_rows)})
    reused_repetitions = int(completed.sum())

    control_indices = None
    controls = ([*sorted((Path(data_root) / "figure_3_5").glob("*")), *map(Path, control_directories)] if reuse and not exact_profile else [])
    for source in controls:
        if not (source / "checkpoint.npz").is_file() or source == directory:
            continue
        control_settings = {k: v for k, v in settings.items() if k != "lambda_rule"}
        control_settings["lambda"] = 1.0
        try:
            cached, done, _, _ = _read_checkpoint(source / "checkpoint.npz", control_settings, grid, samples)
        except (ValueError, KeyError, OSError):
            continue
        if np.all(done):
            control_indices = cached[:, 1:]
            validation["reused_control_source"] = str(source)
            break

    if validate_fourier and not exact_profile:
        probe = samples[0, :min(n, 100)]
        probe_grid = grid[np.unique(np.linspace(0, len(grid) - 1, min(len(grid), 20), dtype=int))]
        exact = beta2_divergence_matrix(probe, probe_grid)
        approximate = beta2_divergence_matrix_fourier(probe, probe_grid, half_width, spatial_points)
        absolute_error = float(np.max(np.abs(exact - approximate)))
        relative_error = absolute_error / max(float(np.max(exact)), np.finfo(float).tiny)
        if relative_error >= 2e-5:
            raise ArithmeticError("Fourier divergence validation exceeds relative error 2e-5.")
        validation.update(fft_maximum_absolute_error=absolute_error, fft_relative_error=relative_error)

    def save():
        temporary = checkpoint_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, settings_json=settings_json, h_grid=grid,
                                samples_sha256=sample_hash, selected_indices=indices,
                                completed=completed, solver_diagnostics=diagnostics, selected_ise_values=selected_ise)
        temporary.replace(checkpoint_path)

    def select_one(repetition):
        sample = samples[repetition]
        divergence = (beta2_divergence_matrix(sample, grid) if exact_profile else
                      beta2_divergence_matrix_fourier(sample, grid, half_width, spatial_points))
        result = perron_eigenpair_from_divergence(divergence, lambda_value)
        if exact_profile:
            errors = exact_oracle_ise_values(sample, grid, sigma2=sigma2)
            choices = (result.selected_index, int(np.argmin(errors)))
            chosen_ise = errors[list(choices)]
        else:
            if control_indices is None:
                ise = int(np.argmin(exact_oracle_ise_values(sample, grid, sigma2=1.0)))
                qhat = int(np.argmin(exact_beta2_qhat_values(sample, grid)))
            else:
                ise, qhat = map(int, control_indices[repetition])
            choices, chosen_ise = (result.selected_index, ise, qhat), None
        return repetition, choices, (result.relative_residual, result.relative_gap, result.matvecs), chosen_ise

    print(f"{profile}: {reused_repetitions}/{repetitions} repetitions cached; {directory}", flush=True)
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(select_one, int(r)) for r in np.flatnonzero(~completed)]
            for future in as_completed(futures):
                repetition, choices, diagnostic, errors = future.result()
                indices[repetition] = choices
                diagnostics[repetition] = diagnostic
                if exact_profile:
                    selected_ise[repetition] = errors
                completed[repetition] = True
                total = int(completed.sum())
                if total % 10 == 0:
                    save()
                if total % 100 == 0 or total == repetitions:
                    print(f"{profile}: completed {total}/{repetitions}", flush=True)
    finally:
        save()
    counts = np.stack([np.bincount(indices[:, column], minlength=len(grid)) for column in range(len(methods))])
    validation.update(completed_repetitions=int(completed.sum()), reused_repetitions=reused_repetitions,
                      maximum_perron_relative_residual=float(diagnostics[:, 0].max()),
                      minimum_perron_relative_gap=float(diagnostics[:, 1].min()))
    (directory / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    (directory / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    if exact_profile:
        _write_csv(directory / "selections.csv", ["repetition", "seed", "method", "grid_index", "h", "ise"],
                   ([r, seed+r, name, int(indices[r, c]), grid[indices[r, c]], selected_ise[r, c]]
                    for r in range(repetitions) for c, name in enumerate(methods)))
        _write_csv(directory / "frequencies.csv", ["method", "grid_index", "h", "count", "relative_frequency"],
                   ([name, i, h, counts[c, i], counts[c, i]/repetitions]
                    for c, name in enumerate(methods) for i, h in enumerate(grid)))
    else:
        _write_csv(directory / "selections.csv",
                   ["repetition", "spectral_grid_index", "spectral_h", "ise_grid_index", "ise_h", "qhat_grid_index", "qhat_h"],
                   ([r, *[value for index in row for value in (int(index), grid[index])]] for r, row in enumerate(indices)))
        _write_csv(directory / "frequencies.csv", ["grid_index", "h", "spectral_count", "ise_count", "qhat_count"],
                   ([i, h, *counts[:, i]] for i, h in enumerate(grid)))
    _write_csv(directory / "solver_diagnostics.csv", ["repetition", "relative_residual", "relative_gap", "matvecs"],
               ([i, *row] for i, row in enumerate(diagnostics)))
    if exact_profile:
        _write_csv(directory / "summary.csv", ["method", "modal_h", "mean_h", "median_h", "modal_frequency", "mean_ise"],
                   ([name, grid[np.argmax(counts[c])], float(grid[indices[:, c]].mean()),
                     float(np.median(grid[indices[:, c]])), float(counts[c].max()/repetitions),
                     float(selected_ise[:, c].mean())] for c, name in enumerate(methods)))
    else:
        _write_csv(directory / "summary.csv", ["selector", "mean_h", "median_h", "modal_h", "repetitions"],
                   ([label, float(grid[indices[:, c]].mean()), float(np.median(grid[indices[:, c]])), grid[np.argmax(counts[c])], repetitions]
                    for c, label in enumerate(methods)))
    starts = np.arange(0, len(grid), binning)
    stops = np.minimum(starts + binning, len(grid))
    binned = np.add.reduceat(counts, starts, axis=1)
    _write_csv(directory / f"frequencies_bins{binning}.csv",
               ["bin_index", "first_grid_index", "last_grid_index", "h_first", "h_last", "spectral_count", "ise_count", *([] if exact_profile else ["qhat_count"])],
               ([b, first, stop - 1, grid[first], grid[stop - 1], *binned[:, b]] for b, (first, stop) in enumerate(zip(starts, stops))))
    return {"methods": methods, "selected_ise_values": selected_ise if exact_profile else None, "selected_indices": indices, "counts": counts, "solver_diagnostics": diagnostics,
            "run_directory": directory, "configuration_id": configuration_id,
            "settings": settings, "validation": validation}
