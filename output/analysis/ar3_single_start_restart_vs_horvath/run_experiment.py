"""One starting direction with conditional resampling versus Horvath L-infinity.

The initial direction and subsequent draws follow the original study's random
stream, but another direction is drawn only after a stationarity abort at 200.
The first nonaborted attempt is returned, including a later nonstationary result.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import scipy
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from Code.Code_AR_processes.ar_tools.coefficient_errors import coefficient_mae
from Code.Code_AR_processes.ar_tools.descent import projected_subgradient_empirical_width
from Code.Code_AR_processes.ar_tools.classical_selectors import horvath_linf_coefficients
from Code.Code_AR_processes.ar_tools.functions import sliding_window_point_cloud, is_stationary
from Code.Code_AR_processes.ar_tools.coefficient_recovery_inputs import INPUT_DATA, load_process
from output.analysis.ar3_early_abort_200_12starts.run_experiment import (
    coefficient_state, json_default, digest,
    MAX_IT, PATIENCE, CHECKPOINT, BETA, TOLERANCE,
)

AR_DIR=ROOT/"Code/Code_AR_processes"
NAME="ar3_single_start_restart_vs_horvath"
OUT=AR_DIR/"outputs/csv"/NAME/"absolute_steps"/"mae"
GD="Single-start descent + abort at 200"
LP="Horvath L-infinity"
METHODS=[GD,LP]
TIMING_REPETITIONS=5
ATTEMPT_SAFETY_LIMIT=100


def fit_descent(series,replication):
    """Fit using the unified optimizer's stationarity restart option."""
    points=sliding_window_point_cloud(series,4)
    rng=np.random.default_rng(np.random.SeedSequence([20260913,replication-1,4,20]))
    direction=rng.normal(size=4)
    direction/=np.linalg.norm(direction)
    run=projected_subgradient_empirical_width(
        points, iterations=MAX_IT, patience=PATIENCE, improvement_tolerance=TOLERANCE,
        initial_direction=direction, initial_step=.2, step_schedule="geometric", geometric_decay=BETA,
        rng=rng, stationarity_check=True, max_restarts=ATTEMPT_SAFETY_LIMIT-1,
        record_directions=True, record_best_coefficients=True,
    )
    if run["restart_limit_reached"]:
        raise RuntimeError("Safety limit reached with no attempt passing the stationarity checkpoint")
    attempts=[dict(initial_direction=attempt["initial_direction"],run=attempt) for attempt in run["attempts"]]
    return dict(coefficients=coefficient_state(run["direction"])[0], attempts=attempts, initial_step=.2)


def fit_horvath(series):
    phi,radius,gap=horvath_linf_coefficients(series,3,use_convex_hull=True)
    return dict(coefficients=phi,radius=radius,duality_gap=gap)


def measure(series,replication):
    timing=[]
    first={}
    for repeat in range(TIMING_REPETITIONS):
        order=METHODS if (replication+repeat)%2 else METHODS[::-1]
        for method in order:
            started=perf_counter()
            fit=fit_descent(series,replication) if method==GD else fit_horvath(series)
            seconds=perf_counter()-started
            timing.append(dict(replication=replication,method=method,repeat=repeat+1,
                               elapsed_seconds=seconds))
            if method not in first:
                first[method]=fit
            else:
                np.testing.assert_array_equal(fit["coefficients"],first[method]["coefficients"])
                if method==GD:
                    assert len(fit["attempts"])==len(first[method]["attempts"])
    return first,timing


def fit_process(replication):
    series,true_phi,_=load_process(replication)
    points=sliding_window_point_cloud(series,4)
    series_hash=hashlib.sha256(series.tobytes()).hexdigest()
    fits,timing=measure(series,replication)
    ledger=[]
    traces={}
    gd=fits[GD]
    for index,item in enumerate(gd["attempts"]):
        run=item["run"]
        checkpoint=run["checkpoint"]
        aborted=run["termination_reason"]=="stationarity_abort"
        phi,rho,stationary=coefficient_state(run["direction"])
        assert aborted==(not checkpoint["stationary"])
        assert aborted==(index<len(gd["attempts"])-1)
        ledger.append(dict(replication=replication,attempt=index+1,aborted=aborted,
            evaluated_states=len(run["width_history"]),checkpoint_radius=checkpoint["spectral_radius"],
            final_radius=rho,final_stationary=stationary,
            termination_reason=run["termination_reason"],minimum_width=run["minimum_width"]))
        for field in ["direction_history","width_history","best_width_history","best_update_history"]:
            traces[f"attempt_{index+1:03d}_{field}"]=run[field]
    rows=[]
    for method in METHODS:
        fit=fits[method]
        phi=fit["coefficients"]
        errors=np.abs(phi-true_phi)
        residuals=points[:,0]-points[:,1:]@phi
        stationary,roots=is_stationary(phi)
        row=dict(replication=replication,method=method,status="ok",
            coefficient_mae=coefficient_mae(phi, true_phi),
            stationary=bool(stationary),spectral_radius=float(max(abs(roots))),
            normalized_directional_width=float(np.ptp(residuals)/np.sqrt(1+phi@phi)),
            maximum_absolute_residual=float(max(abs(residuals))),
            median_fit_seconds=float(np.median([r["elapsed_seconds"] for r in timing if r["method"]==method])),
            series_sha256=series_hash)
        if method==GD:
            row.update(attempts=len(ledger),restarts=len(ledger)-1,
                evaluated_states=sum(r["evaluated_states"] for r in ledger),
                initial_step=gd["initial_step"],
                termination_reason=ledger[-1]["termination_reason"],
                final_checkpoint_radius=ledger[-1]["checkpoint_radius"],lp_radius=np.nan,lp_duality_gap=np.nan)
        else:
            row.update(attempts=1,restarts=0,evaluated_states=np.nan,initial_step=np.nan,
                termination_reason="optimal_lp",final_checkpoint_radius=np.nan,
                lp_radius=fit["radius"],lp_duality_gap=fit["duality_gap"])
        for j in range(3):
            row[f"true_phi_{j+1}"]=float(true_phi[j])
            row[f"estimated_phi_{j+1}"]=float(phi[j])
            row[f"absolute_error_phi_{j+1}"]=float(errors[j])
        rows.append(row)
    np.savez_compressed(OUT/"traces"/f"process_{replication:03d}.npz",**traces)
    return dict(results=rows,attempts=ledger,timing=timing)


def main():
    assert (MAX_IT,PATIENCE,CHECKPOINT,BETA,TOLERANCE)==(2000,200,200,.997,1e-10)
    for subfolder in ["replications","traces"]:
        (OUT/subfolder).mkdir(parents=True,exist_ok=True)
    configuration=dict(coefficient_error_metric="mae_per_coefficient_v1", methods=METHODS,processes=50,known_order=3,sample_size=2000,
        same_original_coefficient_recovery_paths=True,max_it=MAX_IT,improvement_patience=PATIENCE,
        checkpoint=CHECKPOINT,improvement_tolerance=TOLERANCE,beta=BETA,
        step="Absolute initial_step=0.2, reset per attempted direction",
        initial_directions="One draw from the original process-specific random stream; next draw only after abort",
        gd_return="First attempt that passes checkpoint and finishes; no more starts after that",
        gd_stationarity_check="Best-width incumbent after 200 evaluated states; reject undefined or radius >= 1",
        final_stationarity_filter=False,intermediate_stationary_filter=False,
        maximum_attempts_safety_limit=ATTEMPT_SAFETY_LIMIT,
        horvath="Existing no-intercept L-infinity LP, use_convex_hull=True; full-data residual and duality checks",
        timing_repetitions=TIMING_REPETITIONS,timing_warmup=3,
        timing_scope="End-to-end fit from in-memory series; includes cloud/draws/all discarded GD attempts and LP hull/constraints/validation; excludes I/O, accuracy scoring and plotting",
        timing_order="Alternate method order in repeated measurements; median per process",
        thread_limit="BLAS libraries limited to one thread; HiGHS uses the existing helper's solver defaults",
        truth_used_for_decisions=False,numpy_version=np.__version__,scipy_version=scipy.__version__,
        python_version=platform.python_version(),platform=platform.platform(),
        optimizer_sha256=digest(AR_DIR/"ar_tools/descent.py"),
        horvath_implementation_sha256=digest(AR_DIR/"ar_tools/classical_selectors.py"),
        input_data=str(INPUT_DATA),input_data_sha256=digest(INPUT_DATA))
    config_path=OUT/"configuration.json"
    if config_path.exists() and json.loads(config_path.read_text())!=configuration:
        raise ValueError("Refusing to reuse results from a different experiment configuration")
    config_path.write_text(json.dumps(configuration,indent=2)+"\n")
    records=[]
    started=perf_counter()
    with threadpool_limits(limits=1):
        pending=any(not (OUT/"replications"/f"process_{i:03d}.json").exists() for i in range(1,51))
        if pending:
            warm,_,_=load_process(1)
            for _ in range(3):
                fit_descent(warm,1)
                fit_horvath(warm)
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
    results=pd.DataFrame([r for record in records for r in record["results"]])
    attempts=pd.DataFrame([r for record in records for r in record["attempts"]])
    timing=pd.DataFrame([r for record in records for r in record["timing"]])
    results.to_csv(OUT/"per_process_results.csv",index=False)
    attempts.to_csv(OUT/"descent_attempts.csv",index=False)
    timing.to_csv(OUT/"timing_measurements.csv",index=False)
    summary=[]
    for method in METHODS:
        f=results[results.method==method]
        assert len(f)==50 and f.coefficient_mae.notna().all()
        summary.append(dict(method=method,processes=50,stationary_estimates=int(f.stationary.sum()),
            mean_coefficient_mae=float(f.coefficient_mae.mean()),median_coefficient_mae=float(f.coefficient_mae.median()),
            maximum_coefficient_mae=float(f.coefficient_mae.max()),mae_above_1=int((f.coefficient_mae>1).sum()),
            mean_attempts=float(f.attempts.mean()),maximum_attempts=int(f.attempts.max()),
            total_restarts=int(f.restarts.sum()),mean_evaluated_states=float(f.evaluated_states.mean()),
            mean_fit_ms=float(1000*f.median_fit_seconds.mean()),median_fit_ms=float(1000*f.median_fit_seconds.median())))
    summary=pd.DataFrame(summary)
    summary.to_csv(OUT/"summary.csv",index=False)
    pivot=results.pivot(index="replication",columns="method",values="coefficient_mae")
    difference=pivot[GD]-pivot[LP]
    tolerance=1e-10+1e-6*np.maximum(abs(pivot[GD]),abs(pivot[LP]))
    comparison=dict(descent_better=int((difference < -tolerance).sum()),
        horvath_better=int((difference > tolerance).sum()),tied=int((abs(difference)<=tolerance).sum()),
        descent_attempts=int(len(attempts)),descent_aborts=int(attempts.aborted.sum()),
        processes_with_restart=int(results[(results.method==GD)&(results.restarts>0)].shape[0]),
        maximum_attempts=int(attempts.groupby("replication").size().max()),
        safety_limit_reached=False)
    (OUT/"paired_comparison.json").write_text(json.dumps(comparison,indent=2)+"\n")
    pivot.to_csv(OUT/"paired_mae.csv")
    print(summary.to_string(index=False),flush=True)
    print(json.dumps(comparison,indent=2),flush=True)
    return summary


if __name__=="__main__":
    main()
