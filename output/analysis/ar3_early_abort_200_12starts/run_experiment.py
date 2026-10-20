"""Twelve starts in both modes; optionally abort a nonstationary run at 200.

This experiment changes the trajectory stopping rule, not the outer number of
attempts. An aborted attempt cannot win the final width comparison. Surviving
attempts are selected by width, without any further stationarity filter.
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
from Code.Code_AR_processes.ar_tools.descent import projected_subgradient_empirical_width
from Code.Code_AR_processes.ar_tools.functions import is_stationary, sliding_window_point_cloud
from Code.Code_AR_processes.ar_tools.coefficient_recovery_inputs import INPUT_DATA, load_process
from output.analysis.ar3_conditional_restarts_2000_200.run_experiment import (
    replication_path as source_replication_path,
)

AR_DIR = ROOT / "Code/Code_AR_processes"
NAME = "ar3_early_abort_200_12starts"
OUT = AR_DIR / "outputs/csv" / NAME
SOURCE = AR_DIR / "outputs/csv/ar3_conditional_restarts_2000_200"
FIXED = "12 starts, no early abort"
TRIGGERED = "12 starts, stationarity abort at 200"
METHODS = [FIXED, TRIGGERED]
MAX_IT, PATIENCE, CHECKPOINT, STARTS = 2000, 200, 200, 12
TOLERANCE, BETA = 1e-10, .997


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def coefficient_state(direction):
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        phi = -np.asarray(direction)[1:] / direction[0]
    if not np.all(np.isfinite(phi)):
        return phi, float("inf"), False
    stationary, values = is_stationary(phi)
    return phi, float(np.max(np.abs(values))), bool(stationary)


def checkpoint_descent(points, initial_direction, initial_step):
    """Experimental descent with a single checkpoint before update number 200.

    The sphere update and best-width/patience bookkeeping follow the existing
    public optimizer exactly. Every run is checked against that optimizer's
    complete trajectory (or its first 200 evaluated states) in fit_process.
    Initial state is evaluated iteration 1, as in the existing max_it count.
    """
    direction = np.asarray(initial_direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    directions = np.empty((MAX_IT, 4))
    widths, best_widths = np.empty(MAX_IT), np.empty(MAX_IT)
    updates = np.zeros(MAX_IT, dtype=bool)
    best_width, stale = np.inf, 0
    best_direction = direction.copy()
    termination, checkpoint = "iteration_limit", None
    for iteration in range(MAX_IT):
        directions[iteration] = direction
        projections = points @ direction
        imax, imin = int(np.argmax(projections)), int(np.argmin(projections))
        width = projections[imax] - projections[imin]
        if width < best_width - TOLERANCE:
            best_width = float(width)
            best_direction = direction.copy()
            stale = 0
            updates[iteration] = True
        else:
            stale += 1
        widths[iteration], best_widths[iteration] = width, best_width
        step = initial_step * BETA**iteration
        if iteration+1 == CHECKPOINT:
            phi, rho, stationary = coefficient_state(best_direction)
            checkpoint = dict(coefficients=phi.tolist(), spectral_radius=rho,
                              stationary=stationary, direction=best_direction.tolist(),
                              minimum_width=best_width)
            if not stationary:
                termination = "stationarity_abort"
                break
        if stale >= PATIENCE:
            termination = "patience"
            break
        subgradient = points[imax] - points[imin]
        tangent = subgradient - (direction @ subgradient) * direction
        candidate = direction - step*tangent
        norm = np.linalg.norm(candidate)
        # The public optimizer's random fallback is never reached in this
        # study. Fail explicitly if it would be needed rather than diverge.
        assert norm > np.finfo(float).eps
        direction = candidate/norm
    length = iteration+1
    return dict(direction=best_direction, minimum_width=best_width,
                direction_history=directions[:length], width_history=widths[:length],
                best_width_history=best_widths[:length], best_update_history=updates[:length],
                termination_reason=termination, checkpoint=checkpoint)


def candidate(run, start, true_phi):
    phi, rho, stationary = coefficient_state(run["direction"])
    return dict(start=start, direction=run["direction"].tolist(), coefficients=phi.tolist(),
                minimum_width=run["minimum_width"], spectral_radius=rho,
                stationary=stationary, coefficient_mse=float(np.mean((phi-true_phi)**2)),
                evaluated_states=len(run["width_history"]),
                termination_reason=run["termination_reason"],
                best_iteration=int(np.flatnonzero(run["best_update_history"])[-1]))


def fit_process(replication):
    source = json.loads(source_replication_path(replication).read_text())
    true_phi = np.array([source["results"][0][f"true_phi_{j}"] for j in range(1,4)])
    series, input_phi, _ = load_process(replication)
    np.testing.assert_array_equal(true_phi, input_phi)
    assert hashlib.sha256(series.tobytes()).hexdigest() == source["results"][0]["series_sha256"]
    points = sliding_window_point_cloud(series, 4)
    eta0 = source["results"][0]["initial_step"]
    baseline_candidates, policy_candidates, events = [], [], []
    traces = {}
    for saved in source["attempts"]:
        start = saved["attempt"]
        initial = np.array(saved["initial_direction"])
        baseline = projected_subgradient_empirical_width(
            points, iterations=MAX_IT, patience=PATIENCE, improvement_tolerance=TOLERANCE,
            initial_direction=initial, initial_step=eta0, step_schedule="geometric",
            geometric_decay=BETA, record_directions=True, record_best_coefficients=True)
        policy = checkpoint_descent(points, initial, eta0)
        full, early = candidate(baseline, start, true_phi), candidate(policy, start, true_phi)
        # Independently reproduce the previous complete attempt, then verify
        # the policy is an exact prefix, with no altered step schedule.
        np.testing.assert_array_equal(full["direction"], saved["direction"])
        assert full["minimum_width"] == saved["minimum_width"]
        assert full["evaluated_states"] == saved["evaluated_states"]
        length = early["evaluated_states"]
        np.testing.assert_array_equal(policy["direction_history"], baseline["direction_history"][:length])
        np.testing.assert_array_equal(policy["width_history"], baseline["width_history"][:length])
        np.testing.assert_array_equal(policy["best_width_history"], baseline["best_width_history"][:length])
        checkpoint = policy["checkpoint"]
        assert checkpoint is not None
        aborted = early["termination_reason"] == "stationarity_abort"
        assert aborted == (not checkpoint["stationary"])
        if aborted:
            assert length == CHECKPOINT
        else:
            assert length == full["evaluated_states"]
            np.testing.assert_array_equal(full["direction"], early["direction"])
        baseline_candidates.append(full)
        policy_candidates.append(early)
        events.append(dict(replication=replication, start=start, aborted=aborted,
            checkpoint_best_rho=checkpoint["spectral_radius"],
            checkpoint_best_width=checkpoint["minimum_width"],
            baseline_final_rho=full["spectral_radius"], baseline_final_stationary=full["stationary"],
            baseline_final_mse=full["coefficient_mse"],
            would_recover_stationarity=bool(aborted and full["stationary"]),
            baseline_evaluations=full["evaluated_states"], policy_evaluations=length,
            saved_evaluations=full["evaluated_states"]-length))
        # Store all traces so any checkpoint, discarded run, or late recovery
        # can be inspected independently without rerunning the experiment.
        for label, run in [("baseline",baseline),("policy",policy)]:
            for field in ["direction_history","width_history","best_width_history","best_update_history"]:
                traces[f"{label}_{start:02d}_{field}"] = run[field]
    baseline_winner = min(baseline_candidates, key=lambda row: row["minimum_width"])
    survivors = [row for row in policy_candidates if row["termination_reason"] != "stationarity_abort"]
    policy_winner = min(survivors or policy_candidates, key=lambda row: row["minimum_width"])
    rows = []
    for method, winner, options in [(FIXED,baseline_winner,baseline_candidates),(TRIGGERED,policy_winner,policy_candidates)]:
        row = {key:value for key,value in winner.items() if key not in ["direction","coefficients","evaluated_states","start"]}
        row.update(replication=replication, method=method, selected_start=winner["start"],
                   attempts=STARTS, restarts=STARTS-1,
                   aborted_attempts=0 if method == FIXED else STARTS-len(survivors),
                   completed_attempts=STARTS if method == FIXED else len(survivors),
                   evaluated_states=sum(item["evaluated_states"] for item in options),
                   status="all_attempts_aborted" if method == TRIGGERED and not survivors else "ok",
                   series_sha256=source["results"][0]["series_sha256"])
        for j in range(3):
            row[f"true_phi_{j+1}"] = float(true_phi[j])
            row[f"estimated_phi_{j+1}"] = winner["coefficients"][j]
        for j in range(4):
            row[f"u_{j}"] = winner["direction"][j]
        rows.append(row)
    for event in events:
        event["baseline_winning_start"] = event["start"] == baseline_winner["start"]
        event["policy_winning_start"] = event["start"] == policy_winner["start"]
    np.savez_compressed(OUT / "traces" / f"process_{replication:03d}.npz", **traces)
    return dict(results=rows, events=events, baseline_candidates=baseline_candidates,
                policy_candidates=policy_candidates)


def main():
    for subfolder in ["replications","traces"]:
        (OUT/subfolder).mkdir(parents=True,exist_ok=True)
    configuration = dict(max_it=MAX_IT, improv_patience=PATIENCE, checkpoint=CHECKPOINT,
        starts_per_method=STARTS, improvement_tolerance=TOLERANCE, beta=BETA,
        input_data=str(INPUT_DATA), input_data_sha256=digest(INPUT_DATA),
        source_configuration=str(SOURCE/"configuration.json"),
        source_configuration_sha256=digest(SOURCE/"configuration.json"),
        source_replication_sha256={str(i):digest(source_replication_path(i)) for i in range(1,51)},
        optimizer_sha256=digest(AR_DIR/"ar_tools/descent.py"),
        checkpoint_direction="Best-width incumbent after exactly 200 evaluated states",
        abort_condition="Undefined coefficients or companion spectral radius >= 1",
        after_pass="Continue identical trajectory and original patience/step schedule; no further checks",
        after_abort="Discard the entire attempt from final selection; start next original random direction",
        final_selection="Smallest width among nonaborted completed attempts; no final stationarity filter",
        all_aborted_fallback="Flag failed; retain lowest-width aborted prefix only for explicit MSE accounting",
        all_12_attempts_always_used=True, intermediate_stationary_filtering=False,
        truth_used_for_decisions=False, methods=METHODS)
    config_path=OUT/"configuration.json"
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise ValueError("Cached experiment settings changed")
    config_path.write_text(json.dumps(configuration,indent=2)+"\n")
    records=[]
    started=perf_counter()
    with threadpool_limits(limits=1):
        for replication in range(1,51):
            cache=OUT/"replications"/f"process_{replication:03d}.json"
            if cache.exists():
                record=json.loads(cache.read_text())
            else:
                record=fit_process(replication)
                cache.write_text(json.dumps(record,indent=2,default=json_default)+"\n")
            records.append(record)
            if replication==1 or replication%5==0:
                print(f"Completed {replication}/50 ({perf_counter()-started:.1f}s)",flush=True)
    results=pd.DataFrame([row for record in records for row in record["results"]])
    events=pd.DataFrame([row for record in records for row in record["events"]])
    results.to_csv(OUT/"per_process_results.csv",index=False)
    events.to_csv(OUT/"per_start_diagnostics.csv",index=False)
    summary=[]
    for method in METHODS:
        group=results[results.method==method]
        assert len(group)==50 and group.attempts.eq(12).all()
        summary.append(dict(method=method,processes=50,stationary_estimates=int(group.stationary.sum()),
            failed_no_survivor=int((group.status!="ok").sum()),
            mean_coefficient_mse=float(group.coefficient_mse.mean()),
            median_coefficient_mse=float(group.coefficient_mse.median()),
            maximum_coefficient_mse=float(group.coefficient_mse.max()),
            mse_above_1=int((group.coefficient_mse>1).sum()),
            total_attempts=int(group.attempts.sum()),total_aborted=int(group.aborted_attempts.sum()),
            total_evaluated_states=int(group.evaluated_states.sum())))
    summary=pd.DataFrame(summary)
    summary.to_csv(OUT/"summary.csv",index=False)
    paired=results.pivot(index="replication",columns="method",values="coefficient_mse")
    difference=paired[TRIGGERED]-paired[FIXED]
    tol=1e-10+1e-6*np.maximum(abs(paired[TRIGGERED]),abs(paired[FIXED]))
    comparison=dict(improved=int((difference < -tol).sum()),worsened=int((difference > tol).sum()),
        tied=int((abs(difference)<=tol).sum()),
        iteration_saving_fraction=1-float(summary.iloc[1].total_evaluated_states/summary.iloc[0].total_evaluated_states),
        aborted_starts=int(events.aborted.sum()),
        aborted_starts_that_would_recover=int(events.would_recover_stationarity.sum()),
        original_winners_aborted=int((events.baseline_winning_start & events.aborted).sum()),
        original_winners_aborted_but_would_recover=int((events.baseline_winning_start & events.would_recover_stationarity).sum()))
    (OUT/"paired_comparison.json").write_text(json.dumps(comparison,indent=2)+"\n")
    paired.to_csv(OUT/"paired_mse.csv")
    print(summary.to_string(index=False),flush=True)
    print(json.dumps(comparison,indent=2),flush=True)
    return summary


if __name__=="__main__":
    main()
