"""Shared spectral sample-size sweeps for the main, fast, and joint notebooks.

Cache identities include the actual sample prefixes, bandwidth grid,
frequency-quadrature settings, and method/solver versions. Lambda columns
are added to the same cache without recomputing completed selections.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from .fourier import FOURIER_L2_METHOD_VERSION, beta2_divergence_matrix_fourier
from .perron import PERRON_SOLVER_VERSION, perron_eigenpair_from_divergence


SPECTRAL_SWEEP_RUNNER_VERSION = "shared_spectral_sweep_v1"


def _sample_hash(samples):
    return hashlib.sha256(np.ascontiguousarray(samples).tobytes()).hexdigest()


def _atomic_checkpoint(path, **values):
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz.tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.savez_compressed(stream, **values)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def validate_l2_checkpoint(indices, completed, repetitions, number_of_lambdas, grid_size):
    """Validate the original fast notebook's exported checkpoint layout."""
    if indices.shape != (repetitions, number_of_lambdas):
        raise ValueError("Checkpoint shape differs from the current experiment.")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("Checkpoint selections must be integer indices.")
    if not 0 <= completed <= repetitions:
        raise ValueError("Invalid checkpoint completion count.")
    if np.any(indices[:completed] < 0) or np.any(indices[:completed] >= grid_size):
        raise ValueError("Checkpoint contains invalid selected indices.")


def _legacy_checkpoint(path, samples, grid, settings, sample_size):
    """Read a compatible fast-notebook checkpoint without changing its files."""
    with np.load(path, allow_pickle=False) as saved:
        config = json.loads(str(saved["configuration_json"].item()))
        previous = config["settings"]
        if (
            config.get("implementation_version") not in (2, 3)
            or config.get("l2_method") != FOURIER_L2_METHOD_VERSION
            or config.get("perron_solver") != PERRON_SOLVER_VERSION
            or config["sample_size"] != sample_size
            or previous["repetitions"] != len(samples)
            or previous["fft_half_width"] != settings["fft_half_width"]
            or previous["fft_spatial_points"] != settings["fft_spatial_points"]
        ):
            raise ValueError("L² checkpoint settings or method versions differ from the current configuration.")
        if not np.array_equal(saved["h_grid"], grid):
            raise ValueError("L² checkpoint bandwidth grid differs from the current grid.")
        expected_hash = _sample_hash(samples)
        if config.get("samples_sha256") is not None:
            actual_hash = config["samples_sha256"]
        else:
            # Earlier checkpoints did not store samples. Verify their documented
            # seeded sample bank before importing any selections.
            if "seed" not in previous:
                raise ValueError("The legacy checkpoint has no verifiable sample identity.")
            maximum = int(previous["maximum_sample_size"])
            regenerated = np.random.default_rng(previous["seed"]).normal(
                0.0, np.sqrt(previous.get("sigma2", 1.0)), size=(len(samples), maximum)
            )
            actual_hash = _sample_hash(regenerated[:, :sample_size])
        if actual_hash != expected_hash:
            raise ValueError("L² checkpoint sample hash differs from the current samples.")
        lambdas = np.asarray(config["lambda_values"], dtype=float)
        indices = saved["selected_indices"].copy()
        completed = int(saved["completed"].item())
    if lambdas.ndim != 1 or not len(lambdas) or not np.all(np.isfinite(lambdas)) or np.any(lambdas <= 0) or len(np.unique(lambdas)) != len(lambdas):
        raise ValueError("Checkpoint lambda values must be finite, positive, and unique.")
    validate_l2_checkpoint(indices, completed, len(samples), len(lambdas), len(grid))
    return config, lambdas, indices, completed


def run_l2_point(
    sample_size, full_samples, h_grid, settings, lambda_values, output_directory,
    *, reuse=True, cache_root=None, import_directories=(), diagnostics=False,
):
    """Select one sample-size point with a cache shared across notebooks.

    The existing per-notebook checkpoint is retained as an export. The shared
    cache keys on numerical inputs, so matching runs can reuse selections even
    when their notebook settings contain different reporting fields. A single
    divergence matrix serves every missing lambda for a given repetition.
    Request diagnostics to cache residual, relative spectral gap, and operation
    count alongside each selection. A selection-only cache is upgraded by
    recomputing and checking its eigenpair, rather than inventing diagnostics.
    """
    repetitions = settings["repetitions"]
    if isinstance(repetitions, (bool, np.bool_)) or not isinstance(repetitions, (int, np.integer)) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer.")
    bank = np.asarray(full_samples, dtype=float)
    if bank.shape != (repetitions, settings["maximum_sample_size"]) or not np.all(np.isfinite(bank)):
        raise ValueError("Sample array shape or values differ from the settings.")
    if isinstance(sample_size, (bool, np.bool_)) or not isinstance(sample_size, (int, np.integer)) or not 1 <= sample_size <= bank.shape[1]:
        raise ValueError("Sample size is outside the generated sample range.")
    grid = np.asarray(h_grid, dtype=float)
    if grid.ndim != 1 or len(grid) < 2 or not np.all(np.isfinite(grid)) or np.any(grid <= 0) or np.any(np.diff(grid) <= 0):
        raise ValueError("Bandwidths must be finite, positive, and increasing.")
    requested = np.asarray(lambda_values, dtype=float)
    if requested.ndim != 1 or not len(requested) or not np.all(np.isfinite(requested)) or np.any(requested <= 0) or len(np.unique(requested)) != len(requested):
        raise ValueError("Lambda values must be finite, positive, and unique.")
    half_width, points = settings["fft_half_width"], settings["fft_spatial_points"]
    if not np.isfinite(half_width) or half_width <= 0 or isinstance(points, (bool, np.bool_)) or not isinstance(points, (int, np.integer)) or points < 2 or points % 2:
        raise ValueError("Use a positive half-width and an even frequency-grid size of at least two.")
    samples = bank[:, :sample_size]
    sample_hash = _sample_hash(samples)
    identity = {
        "runner": SPECTRAL_SWEEP_RUNNER_VERSION, "sample_size": int(sample_size),
        "repetitions": int(repetitions), "samples_sha256": sample_hash,
        "h_grid": grid.tolist(), "fft_half_width": float(half_width), "fft_spatial_points": int(points),
        "l2_method": FOURIER_L2_METHOD_VERSION, "perron_solver": PERRON_SOLVER_VERSION,
    }
    identity_json = json.dumps(identity, sort_keys=True)
    fingerprint = hashlib.sha256(identity_json.encode()).hexdigest()[:16]
    shared_root = Path(cache_root) if cache_root is not None else Path(output_directory) / "shared_spectral_cache"
    cache_directory = shared_root / f"n{int(sample_size)}" / fingerprint
    cache_directory.mkdir(parents=True, exist_ok=True)
    cache_path = cache_directory / "checkpoint.npz"
    point_directory = Path(output_directory) / f"n{int(sample_size)}"
    point_directory.mkdir(parents=True, exist_ok=True)
    point_path = point_directory / "checkpoint.npz"
    point_config = {
        "settings": settings, "sample_size": int(sample_size), "lambda_values": requested.tolist(),
        "implementation_version": 3, "samples_sha256": sample_hash,
        "l2_method": FOURIER_L2_METHOD_VERSION, "perron_solver": PERRON_SOLVER_VERSION,
    }
    point_json = json.dumps(point_config, sort_keys=True)
    lambdas = requested.copy()
    indices = np.full((repetitions, len(lambdas)), -1, dtype=int)
    done = np.zeros(indices.shape, dtype=bool)
    solver_diagnostics = np.full((*indices.shape, 3), np.nan)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as saved:
            if str(saved["identity_json"].item()) != identity_json:
                raise ValueError("Shared spectral cache identity does not match its inputs.")
            lambdas = saved["lambda_values"].copy()
            indices = saved["selected_indices"].copy()
            done = saved["completed"].copy()
            solver_diagnostics = (saved["solver_diagnostics"].copy() if "solver_diagnostics" in saved
                                  else np.full((*indices.shape, 3), np.nan))
        if (lambdas.ndim != 1 or not len(lambdas) or not np.all(np.isfinite(lambdas))
                or np.any(lambdas <= 0) or len(np.unique(lambdas)) != len(lambdas)
                or indices.shape != (repetitions, len(lambdas)) or done.shape != indices.shape
                or done.dtype != np.dtype(bool) or not np.issubdtype(indices.dtype, np.integer)
                or np.any(indices[done] < 0) or np.any(indices[done] >= len(grid))):
            raise ValueError("Shared spectral cache contains invalid selections or completion flags.")
        if solver_diagnostics.shape != (*indices.shape, 3):
            raise ValueError("Shared spectral diagnostics have an invalid shape.")
        available = np.all(np.isfinite(solver_diagnostics), axis=2)
        if (np.any(available & ~done) or np.any(solver_diagnostics[available, 0] < 0)
                or np.any(solver_diagnostics[available, 1] <= 0)
                or np.any(solver_diagnostics[available, 2] < 0)):
            raise ValueError("Shared spectral diagnostics contain invalid results.")

    def add_columns(values):
        nonlocal lambdas, indices, done, solver_diagnostics
        for value in values:
            if value not in lambdas:
                lambdas = np.append(lambdas, value)
                indices = np.column_stack((indices, np.full(repetitions, -1, dtype=int)))
                done = np.column_stack((done, np.zeros(repetitions, dtype=bool)))
                solver_diagnostics = np.concatenate((solver_diagnostics, np.full((repetitions, 1, 3), np.nan)), axis=1)

    skipped_imports = []
    if reuse:
        sources = [point_path, *(Path(source) / f"n{int(sample_size)}" / "checkpoint.npz" for source in import_directories)]
        for source in dict.fromkeys(sources):
            if not source.is_file():
                continue
            try:
                config, old_lambdas, old_indices, completed = _legacy_checkpoint(source, samples, grid, settings, sample_size)
                if source == point_path and (
                    config["settings"] != settings or not np.array_equal(old_lambdas, requested)
                ):
                    raise ValueError("L² checkpoint settings differ from the current configuration.")
            except (ValueError, KeyError, OSError) as error:
                if source == point_path:
                    raise ValueError(f"Invalid local L² checkpoint: {error}") from error
                skipped_imports.append({"source": str(source), "reason": str(error)})
                continue
            add_columns(old_lambdas)
            for old_column, value in enumerate(old_lambdas):
                column = int(np.flatnonzero(lambdas == value)[0])
                if np.any(done[:completed, column] & (indices[:completed, column] != old_indices[:completed, old_column])):
                    raise ValueError("Imported selections conflict with the matching shared spectral cache.")
                indices[:completed, column] = old_indices[:completed, old_column]
                done[:completed, column] = True
    add_columns(requested)
    columns = np.array([int(np.flatnonzero(lambdas == value)[0]) for value in requested])
    if not reuse:
        indices[:, columns] = -1
        done[:, columns] = False
        solver_diagnostics[:, columns] = np.nan
    reused_selections = int(done[:, columns].sum())
    divergence_evaluations = 0

    def save():
        _atomic_checkpoint(cache_path, identity_json=identity_json, lambda_values=lambdas,
                           selected_indices=indices, completed=done, solver_diagnostics=solver_diagnostics)
        complete_rows = np.all(done[:, columns], axis=1)
        unfinished = np.flatnonzero(~complete_rows)
        completed = int(unfinished[0]) if len(unfinished) else repetitions
        _atomic_checkpoint(point_path, configuration_json=point_json, h_grid=grid,
                           selected_indices=indices[:, columns], completed=completed)

    try:
        for repetition in range(repetitions):
            missing = columns[~done[repetition, columns]]
            if diagnostics:
                missing = columns[(~done[repetition, columns]) |
                                  ~np.all(np.isfinite(solver_diagnostics[repetition, columns]), axis=1)]
            if not len(missing):
                continue
            divergence = beta2_divergence_matrix_fourier(
                samples[repetition], grid, half_width=half_width, spatial_points=points,
            )
            divergence_evaluations += 1
            for column in missing:
                result = perron_eigenpair_from_divergence(divergence, float(lambdas[column]))
                if done[repetition, column] and indices[repetition, column] != result.selected_index:
                    raise ArithmeticError("Recomputed eigenpair diagnostics disagree with the cached selection.")
                indices[repetition, column] = result.selected_index
                solver_diagnostics[repetition, column] = (result.relative_residual, result.relative_gap, result.matvecs)
                done[repetition, column] = True
            save()
            if repetition == 0 or (repetition + 1) % 100 == 0 or repetition + 1 == repetitions:
                print(f"n={sample_size}: completed repetition {repetition + 1}/{repetitions}", flush=True)
    finally:
        save()
    selected = indices[:, columns].copy()
    validate_l2_checkpoint(selected, repetitions, repetitions, len(requested), len(grid))
    counts = np.stack([np.bincount(selected[:, c], minlength=len(grid)) for c in range(len(requested))])
    print(f"n={sample_size}: reused {reused_selections} spectral selections; "
          f"computed {divergence_evaluations} divergence matrices", flush=True)
    return {
        "sample_size": int(sample_size), "selected_indices": selected,
        "counts": counts, "frequencies": counts / repetitions,
        "cache_directory": cache_directory, "point_directory": point_directory,
        "samples_sha256": sample_hash, "reused_selections": reused_selections,
        "divergence_evaluations": divergence_evaluations, "skipped_imports": skipped_imports,
        "solver_diagnostics": solver_diagnostics[:, columns].copy(),
    }
