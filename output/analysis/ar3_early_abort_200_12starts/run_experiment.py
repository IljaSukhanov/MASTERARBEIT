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
from Code.Code_AR_processes.ar_tools.coefficient_errors import coefficient_mae
from Code.Code_AR_processes.ar_tools.descent import projected_subgradient_empirical_width
from Code.Code_AR_processes.ar_tools.functions import is_stationary, sliding_window_point_cloud
from Code.Code_AR_processes.ar_tools.coefficient_recovery_inputs import INPUT_DATA, load_process

AR_DIR = ROOT / "Code/Code_AR_processes"
NAME = "ar3_early_abort_200_12starts"
OUT = AR_DIR / "outputs/csv" / NAME / "absolute_steps" / "mae"
FIXED = "12 starts, no early abort"
TRIGGERED = "12 starts, stationarity abort at 200"
METHODS = [FIXED, TRIGGERED]
MAX_IT, PATIENCE, STARTS = 2000, 200, 12
CHECKPOINT = PATIENCE
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


def candidate(run, start, true_phi):
    phi, rho, stationary = coefficient_state(run["direction"])
    return dict(start=start, direction=run["direction"].tolist(), coefficients=phi.tolist(),
                minimum_width=run["minimum_width"], spectral_radius=rho,
                stationary=stationary, coefficient_mae=float(coefficient_mae(phi, true_phi)),
                evaluated_states=len(run["width_history"]),
                termination_reason=run["termination_reason"],
                best_iteration=int(np.flatnonzero(run["best_update_history"])[-1]))


def fit_process(replication):
    series, true_phi, directions = load_process(replication)
    points = sliding_window_point_cloud(series, 4)
    eta0 = .2
    baseline_candidates, policy_candidates, events = [], [], []
    traces = {}
    for start, initial in enumerate(directions, 1):
        baseline = projected_subgradient_empirical_width(
            points, iterations=MAX_IT, patience=PATIENCE, improvement_tolerance=TOLERANCE,
            initial_direction=initial, initial_step=eta0, step_schedule="geometric",
            geometric_decay=BETA, record_directions=True, record_best_coefficients=True)
        policy = projected_subgradient_empirical_width(
            points, iterations=MAX_IT, patience=PATIENCE, improvement_tolerance=TOLERANCE,
            initial_direction=initial, initial_step=eta0, step_schedule="geometric",
            geometric_decay=BETA, record_directions=True, record_best_coefficients=True,
            stationarity_check=True, max_restarts=0,
        )
        full, early = candidate(baseline, start, true_phi), candidate(policy, start, true_phi)
        # A checkpoint-aborted run must be an exact prefix of the same ordinary run.
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
            baseline_final_mae=full["coefficient_mae"],
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
                   series_sha256=hashlib.sha256(series.tobytes()).hexdigest())
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
    configuration = dict(coefficient_error_metric="mae_per_coefficient_v1", max_it=MAX_IT, improv_patience=PATIENCE, checkpoint=CHECKPOINT,
        starts_per_method=STARTS, improvement_tolerance=TOLERANCE, beta=BETA,
        input_data=str(INPUT_DATA), input_data_sha256=digest(INPUT_DATA),
        optimizer_sha256=digest(AR_DIR/"ar_tools/descent.py"),
        checkpoint_direction="Shared optimizer: best-width incumbent after exactly patience evaluated states",
        initial_step=.2,
        abort_condition="Undefined coefficients or companion spectral radius >= 1",
        after_pass="Continue identical trajectory and original patience/step schedule; no further checks",
        after_abort="Discard the entire attempt from final selection; start next original random direction",
        final_selection="Smallest width among nonaborted completed attempts; no final stationarity filter",
        all_aborted_fallback="Flag failed; retain lowest-width aborted prefix only for explicit MAE accounting",
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
            mean_coefficient_mae=float(group.coefficient_mae.mean()),
            median_coefficient_mae=float(group.coefficient_mae.median()),
            maximum_coefficient_mae=float(group.coefficient_mae.max()),
            mae_above_1=int((group.coefficient_mae>1).sum()),
            total_attempts=int(group.attempts.sum()),total_aborted=int(group.aborted_attempts.sum()),
            total_evaluated_states=int(group.evaluated_states.sum())))
    summary=pd.DataFrame(summary)
    summary.to_csv(OUT/"summary.csv",index=False)
    paired=results.pivot(index="replication",columns="method",values="coefficient_mae")
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
    paired.to_csv(OUT/"paired_mae.csv")
    print(summary.to_string(index=False),flush=True)
    print(json.dumps(comparison,indent=2),flush=True)
    return summary


if __name__=="__main__":
    main()
