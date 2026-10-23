"""Fixed 12 starts versus terminal-stationarity-triggered starts, without filtering.

Same original 50 AR(3) paths and ordered random initial directions as the earlier
coefficient-recovery study. Each attempt uses max_it=2000, improv_patience=200.
The conditional policy returns the first attempt whose minimum-width incumbent
is stationary. It never selects an earlier stationary iterate within an attempt.
At the 12-attempt cap, an all-nonstationary outcome is explicitly marked failed
and its unrestricted smallest-width result is retained for error accounting.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Code.Code_AR_processes.ar_tools.coefficient_errors import coefficient_mae
from Code.Code_AR_processes.ar_tools.descent import projected_subgradient_empirical_width
from Code.Code_AR_processes.ar_tools.functions import (
    is_stationary, jones_ar_coefficients, simulate_uniform_ar, sliding_window_point_cloud,
)
from Code.Code_AR_processes.ar_tools.coefficient_recovery_inputs import (
    AR_DIR, SOURCE_DATA, SOURCE_NOTEBOOK, SOURCE_NOTEBOOK_SHA256, INPUT_DATA, digest, json_default,
    load_process, source_settings,
)

NAME = "ar3_conditional_restarts_2000_200"
OUT = AR_DIR / "outputs/csv" / NAME / "absolute_steps" / "mae"
FIXED = "Fixed 12 starts"
CONDITIONAL = "Stationarity-triggered starts"
METHODS = [FIXED, CONDITIONAL]
MAX_IT = 2000
PATIENCE = 200
MAX_ATTEMPTS = 12


def replication_path(replication):
    """Resolve the renamed process-16 cache after Overleaf rejected its old path."""
    filename = "replication_016.json" if replication == 16 else f"process_{replication:03d}.json"
    return OUT / "replications" / filename


def choose_terminal_results(attempts):
    """Selection uses only the final best-width result of each completed attempt."""
    fixed = min(attempts, key=lambda a: a["minimum_width"])
    conditional = next((a for a in attempts if a["stationary"]), None)
    exhausted = conditional is None
    if exhausted:
        conditional = fixed
    used = MAX_ATTEMPTS if exhausted else conditional["attempt"]
    return fixed, conditional, used, exhausted


def fit_process(replication, settings, historical):
    index = replication - 1
    rng = np.random.default_rng(np.random.SeedSequence([settings["experiment_seed"], index, 0]))
    true_phi = jones_ar_coefficients(3, rng=rng)
    series = simulate_uniform_ar(true_phi, settings["sample_size"], burnin=settings["burnin"],
                                 half_width=1.0, rng=rng)
    archived_series, archived_phi, archived_directions = load_process(replication)
    # Restore exact shared inputs after independently checking seed identity.
    # Different numerical runtimes can otherwise change a few final bits.
    np.testing.assert_allclose(series, archived_series, atol=1e-10, rtol=1e-12)
    np.testing.assert_allclose(true_phi, archived_phi, atol=1e-14, rtol=0)
    series, true_phi = archived_series, archived_phi
    points = sliding_window_point_cloud(series, 4)
    true_u = np.r_[1., -true_phi]
    true_u /= np.linalg.norm(true_u)
    true_width = float(np.ptp(points @ true_u))
    old = historical.loc[historical.replication == replication].iloc[0]
    np.testing.assert_allclose(true_width, old.width_at_true_direction, atol=1e-10, rtol=0)
    series_hash = hashlib.sha256(series.tobytes()).hexdigest()
    eta0 = settings["initial_step"]
    rng = np.random.default_rng(np.random.SeedSequence([settings["experiment_seed"], index, 4, 20]))
    directions = rng.normal(size=(MAX_ATTEMPTS, 4))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    np.testing.assert_allclose(directions, archived_directions, atol=1e-14, rtol=0)
    directions = archived_directions

    attempts = []
    traces = {}
    for attempt, initial in enumerate(directions, 1):
        # Both policies use this same attempted trajectory. The conditional mode
        # consumes only the prefix of complete attempts up to its first success.
        run = projected_subgradient_empirical_width(
            points, iterations=MAX_IT, patience=PATIENCE, initial_step=eta0,
            initial_direction=initial, improvement_tolerance=1e-10,
            step_schedule="geometric", geometric_decay=settings["geometric_decay"],
            rng=np.random.SeedSequence([settings["experiment_seed"], index, 4, 21, attempt - 1]),
            record_directions=True, record_best_coefficients=True,
        )
        u = run["direction"]
        phi = run["coefficients"]
        if np.all(np.isfinite(phi)):
            stationary, eigenvalues = is_stationary(phi)
            rho = float(np.max(np.abs(eigenvalues)))
        else:
            stationary, rho = False, float("inf")
        attempts.append(dict(
            replication=replication, attempt=attempt, initial_direction=initial.tolist(),
            direction=u.tolist(), coefficients=phi.tolist(),
            minimum_width=float(run["minimum_width"]), stationary=bool(stationary),
            spectral_radius=rho, evaluated_states=len(run["width_history"]),
            termination_reason=run["termination_reason"],
            best_iteration=int(np.flatnonzero(run["best_update_history"])[-1]),
            coefficient_mae=float(coefficient_mae(phi, true_phi)),
        ))
        if replication in [1, 27, 35, 41]:
            traces[f"u_{attempt:02d}"] = run["direction_history"]
            traces[f"width_{attempt:02d}"] = run["width_history"]
            traces[f"best_width_{attempt:02d}"] = run["best_width_history"]
            traces[f"best_phi_{attempt:02d}"] = run["best_coefficient_history"]
            traces[f"update_{attempt:02d}"] = run["best_update_history"]
        assert len(run["width_history"]) <= MAX_IT
        if run["termination_reason"] == "patience":
            assert len(run["width_history"]) - 1 - attempts[-1]["best_iteration"] == PATIENCE
        else:
            assert len(run["width_history"]) == MAX_IT

    fixed, conditional, used, exhausted = choose_terminal_results(attempts)
    rows = []
    for method, winner, count in [(FIXED, fixed, MAX_ATTEMPTS), (CONDITIONAL, conditional, used)]:
        row = {key: value for key, value in winner.items() if key not in ["initial_direction", "direction", "coefficients", "attempt"]}
        row.update(method=method, selected_attempt=winner["attempt"],
                   attempts_used=count, restarts_used=count - 1,
                   evaluated_states=sum(a["evaluated_states"] for a in attempts[:count]),
                   status="no_stationary_result_within_cap" if method == CONDITIONAL and exhausted else "ok",
                   series_sha256=series_hash, initial_step=eta0,
                   true_spectral_radius=float(np.max(np.abs(is_stationary(true_phi)[1]))))
        for j in range(3):
            row[f"true_phi_{j+1}"] = float(true_phi[j])
            row[f"estimated_phi_{j+1}"] = winner["coefficients"][j]
        for j in range(4):
            row[f"u_{j}"] = winner["direction"][j]
        rows.append(row)
    if traces:
        np.savez_compressed(OUT / "traces" / f"process_{replication:03d}.npz", **traces,
                            true_phi=true_phi, initial_directions=directions)
    return dict(results=rows, attempts=attempts, path_check=dict(
        replication=replication, series_sha256=series_hash,
        true_width_difference=true_width-float(old.width_at_true_direction)))


def main():
    settings = source_settings()
    for name in ["replications", "traces"]:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    configuration = dict(coefficient_error_metric="mae_per_coefficient_v1",
        original_source=str(SOURCE_NOTEBOOK), original_source_sha256=SOURCE_NOTEBOOK_SHA256,
        input_data=str(INPUT_DATA), input_data_sha256=digest(INPUT_DATA),
        original_settings=settings, optimizer_sha256=digest(AR_DIR / "ar_tools/descent.py"),
        max_it=MAX_IT, improv_patience=PATIENCE, max_attempts=MAX_ATTEMPTS,
        improvement_tolerance=1e-10, methods=METHODS,
        restart_check="Only the completed attempt's best-width direction; never an intermediate candidate",
        stationarity="companion spectral radius < 1; undefined coefficients fail",
        restart_direction="Next random direction in the same original ordered bank of 12",
        schedule="Reset absolute eta0=0.2 and beta=0.997 on every attempt",
        conditional_stop="First stationary completed attempt",
        filtering=False, truth_used_for_decisions=False,
        cap_failure="Mark failure and retain unrestricted smallest-width result for MAE accounting",
        cost="Actual number of evaluated states in the complete attempts consumed by each policy",
    )
    path = OUT / "configuration.json"
    if path.exists() and json.loads(path.read_text()) != configuration:
        raise ValueError("Refusing to reuse a cache with different settings")
    path.write_text(json.dumps(configuration, indent=2) + "\n")
    historical = pd.read_csv(SOURCE_DATA / "known_p_empirical_direction_diagnostics.csv")
    records = []
    started = perf_counter()
    with threadpool_limits(limits=1):
        for replication in range(1, 51):
            cache = replication_path(replication)
            if cache.exists():
                record = json.loads(cache.read_text())
            else:
                record = fit_process(replication, settings, historical)
                cache.write_text(json.dumps(record, indent=2, default=json_default) + "\n")
            records.append(record)
            if replication == 1 or replication % 5 == 0:
                print(f"Completed {replication}/50 ({perf_counter()-started:.1f}s)", flush=True)
    results = pd.DataFrame([row for r in records for row in r["results"]])
    attempts = pd.DataFrame([row for r in records for row in r["attempts"]])
    results.to_csv(OUT / "per_process_results.csv", index=False)
    # Keep array columns in the replication JSON; the flat table stays convenient.
    attempts.drop(columns=["initial_direction", "direction", "coefficients"]).to_csv(OUT / "per_attempt_diagnostics.csv", index=False)
    pd.DataFrame([r["path_check"] for r in records]).to_csv(OUT / "source_reproduction.csv", index=False)
    rows = []
    for method, group in results.groupby("method", sort=False):
        assert len(group) == 50 and group.coefficient_mae.notna().all()
        rows.append(dict(method=method, processes=len(group), stationary_estimates=int(group.stationary.sum()),
            exhausted_without_stationary=int((group.status != "ok").sum()),
            mean_coefficient_mae=float(group.coefficient_mae.mean()),
            median_coefficient_mae=float(group.coefficient_mae.median()),
            maximum_coefficient_mae=float(group.coefficient_mae.max()),
            mae_above_1=int((group.coefficient_mae > 1).sum()),
            mean_attempts=float(group.attempts_used.mean()), total_attempts=int(group.attempts_used.sum()),
            mean_evaluated_states=float(group.evaluated_states.mean()),
            total_evaluated_states=int(group.evaluated_states.sum())))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary.csv", index=False)
    pivot = results.pivot(index="replication", columns="method", values="coefficient_mae")
    delta = pivot[CONDITIONAL] - pivot[FIXED]
    tolerance = 1e-10 + 1e-6 * np.maximum(abs(pivot[CONDITIONAL]), abs(pivot[FIXED]))
    comparison = dict(improved=int((delta < -tolerance).sum()),
                      worsened=int((delta > tolerance).sum()), tied=int((abs(delta) <= tolerance).sum()),
                      evaluated_state_saving_fraction=1-float(summary.iloc[1].total_evaluated_states / summary.iloc[0].total_evaluated_states))
    (OUT / "paired_comparison.json").write_text(json.dumps(comparison, indent=2)+"\n")
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(comparison, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()
