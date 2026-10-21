"""Audit actual conditional starts and independently solve the full-data LP."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linprog

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from Code.Code_AR_processes.ar_tools.coefficient_errors import coefficient_mae
from Code.Code_AR_processes.ar_tools.functions import sliding_window_point_cloud
from output.analysis.ar3_single_start_restart_vs_horvath.run_experiment import (
    AR_DIR,OUT,NAME,GD,LP,
)
from Code.Code_AR_processes.ar_tools.coefficient_recovery_inputs import load_process
from output.analysis.ar3_early_abort_200_12starts.verify_and_report import incumbent_indices
from output.analysis.ar3_conditional_restarts_2000_200.verify_and_report import markdown_table


def validate():
    results=pd.read_csv(OUT/"per_process_results.csv",float_precision="round_trip")
    timing=pd.read_csv(OUT/"timing_measurements.csv",float_precision="round_trip")
    ledger=pd.read_csv(OUT/"descent_attempts.csv",float_precision="round_trip")
    assert len(results)==100 and results.groupby("method").size().eq(50).all()
    assert len(timing)==500 and timing.groupby(["replication","method"]).size().eq(5).all()
    assert (timing.elapsed_seconds>0).all()
    max_lp_difference,max_width_difference=0.,0.
    for replication in range(1,51):
        series,_,_=load_process(replication)
        points=sliding_window_point_cloud(series,4)
        group=results[results.replication==replication].set_index("method")
        truth=group.loc[GD,[f"true_phi_{j}" for j in range(1,4)]].to_numpy(dtype=float)
        for method in [GD,LP]:
            row=group.loc[method]
            phi=row[[f"estimated_phi_{j}" for j in range(1,4)]].to_numpy(dtype=float)
            np.testing.assert_allclose(coefficient_mae(phi, truth),row.coefficient_mae,rtol=1e-12)
            rho=float(max(abs(np.roots(np.r_[1.,-phi]))))
            np.testing.assert_allclose(rho,row.spectral_radius,rtol=1e-10,atol=1e-12)
            assert bool(rho<1)==row.stationary
            raw=timing[(timing.replication==replication)&(timing.method==method)].elapsed_seconds
            assert float(np.median(raw))==row.median_fit_seconds
        gd=group.loc[GD]
        attempts=ledger[ledger.replication==replication].sort_values("attempt")
        assert len(attempts)==gd.attempts
        assert attempts.aborted.iloc[:-1].all() and not attempts.aborted.iloc[-1]
        assert gd.restarts==len(attempts)-1
        assert gd.evaluated_states==int(attempts.evaluated_states.sum())
        with np.load(OUT/"traces"/f"process_{replication:03d}.npz") as trace:
            for event in attempts.itertuples():
                u=trace[f"attempt_{event.attempt:03d}_direction_history"]
                w=trace[f"attempt_{event.attempt:03d}_width_history"]
                indices,updates=incumbent_indices(w)
                checkpoint=u[indices[199]]
                phi=-checkpoint[1:]/checkpoint[0]
                rho=float(max(abs(np.roots(np.r_[1.,-phi]))))
                assert event.aborted==bool(rho>=1)
                assert len(u)==event.evaluated_states
                if event.aborted:
                    assert len(u)==200
                else:
                    assert len(u)<=2000
                    if event.termination_reason=="patience":
                        assert len(u)-1-updates[-1]==200
                    final_u=u[indices[-1]]
                    selected_phi=gd[[f"estimated_phi_{j}" for j in range(1,4)]].to_numpy(dtype=float)
                    np.testing.assert_array_equal(-final_u[1:]/final_u[0],selected_phi)
                    width_error=abs(float(np.ptp(points@final_u))-event.minimum_width)
                    max_width_difference=max(max_width_difference,width_error)
                    assert width_error<1e-10
        # Independent LP construction: both signs of y-X phi, using all 1997
        # rows and no convex-hull reduction, no free residual centre.
        y,x=points[:,0],points[:,1:]
        matrix=np.column_stack([np.vstack([x,-x]),-np.ones(2*len(y))])
        bounds=np.r_[y,-y]
        reference=linprog(np.r_[np.zeros(3),1.],A_ub=matrix,b_ub=bounds,
            bounds=[(None,None)]*3+[(0.,None)],method="highs",
            options={"primal_feasibility_tolerance":1e-9,"dual_feasibility_tolerance":1e-9})
        assert reference.success
        lp=group.loc[LP]
        phi=lp[[f"estimated_phi_{j}" for j in range(1,4)]].to_numpy(dtype=float)
        objective_difference=abs(reference.fun-lp.lp_radius)
        max_lp_difference=max(max_lp_difference,objective_difference)
        assert objective_difference<1e-8
        np.testing.assert_allclose(reference.x[:3],phi,rtol=1e-8,atol=1e-8)
        np.testing.assert_allclose(max(abs(y-x@phi)),lp.lp_radius,rtol=0,atol=1e-8)
        assert lp.lp_duality_gap<1e-8
    validation=dict(processes=50,estimates_checked=100,actual_descent_attempts_checked=len(ledger),
        no_extra_draw_without_abort=True,first_nonaborted_result_returned=True,
        stationarity_roots_independently_checked=True,all_50_lps_independently_solved_on_full_data=True,
        maximum_lp_objective_difference=max_lp_difference,maximum_selected_width_error=max_width_difference,
        all_timing_medians_checked=True,timing_samples_checked=len(timing),no_processes_dropped=True)
    (OUT/"validation.json").write_text(json.dumps(validation,indent=2)+"\n")
    return validation


def plot_and_report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results=pd.read_csv(OUT/"per_process_results.csv")
    summary=pd.read_csv(OUT/"summary.csv").set_index("method")
    comparison=json.loads((OUT/"paired_comparison.json").read_text())
    gd=results[results.method==GD].set_index("replication")
    lp=results[results.method==LP].set_index("replication")
    paired=pd.DataFrame(dict(descent_mae=gd.coefficient_mae,horvath_mae=lp.coefficient_mae,
        descent_fit_ms=1000*gd.median_fit_seconds,horvath_fit_ms=1000*lp.median_fit_seconds,
        descent_attempts=gd.attempts,descent_stationary=gd.stationary,horvath_stationary=lp.stationary))
    paired.to_csv(OUT/"paired_process_comparison.csv")
    per_coefficient_maes=[]
    for method in [GD,LP]:
        f=results[results.method==method]
        for j in range(1,4):
            per_coefficient_maes.append(dict(method=method,coefficient=f"phi_{j}",
                mae=float(f[f"absolute_error_phi_{j}"].mean())))
    pd.DataFrame(per_coefficient_maes).to_csv(OUT/"coefficient_mae.csv",index=False)
    image_dir=AR_DIR/"outputs/images"/NAME/"mae"
    pdf_dir=AR_DIR/"outputs/pdfs"/NAME/"mae"
    image_dir.mkdir(parents=True,exist_ok=True)
    pdf_dir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout="constrained")
    for ax,x,y,title,xlabel,ylabel in [
        (axes[0],paired.horvath_mae,paired.descent_mae,"Paired coefficient recovery",
         r"Horváth $L^\infty$: coefficient MAE","Your descent: coefficient MAE"),
        (axes[1],paired.horvath_fit_ms,paired.descent_fit_ms,"Fit time including preparation and restarts",
         r"Horváth $L^\infty$: milliseconds","Your descent: milliseconds")]:
        colors=np.where(paired.descent_attempts>1,"#ce653a","#247b86")
        ax.scatter(x,y,c=colors,s=34,alpha=.85)
        low=float(min(x.min(),y.min()))*.7
        high=float(max(x.max(),y.max()))*1.5
        ax.plot([low,high],[low,high],color=".5",linestyle="--",linewidth=1)
        ax.set(xscale="log",yscale="log",xlim=(low,high),ylim=(low,high),
               xlabel=xlabel,ylabel=ylabel,title=title)
        ax.grid(alpha=.12)
    from matplotlib.lines import Line2D
    axes[0].legend(handles=[Line2D([],[],marker="o",linestyle="",color="#247b86",label="First direction sufficed"),
                           Line2D([],[],marker="o",linestyle="",color="#ce653a",label="Triggered resampling")],
                   loc="lower right",fontsize=8)
    fig.suptitle("Same 50 AR(3) paths: one initial direction; resample only after a stationarity abort at 200",fontsize=12)
    fig.savefig(image_dir/"comparison.png",dpi=180,bbox_inches="tight")
    fig.savefig(pdf_dir/"comparison.pdf",bbox_inches="tight")
    plt.close(fig)
    a,b=summary.loc[GD],summary.loc[LP]
    speed_ratio=float(a.mean_fit_ms/b.mean_fit_ms)
    paired_speed_ratio=float(np.median(paired.descent_fit_ms/paired.horvath_fit_ms))
    wins=paired[paired.descent_mae<paired.horvath_mae]
    worst=paired.sort_values("descent_mae",ascending=False).head(5)
    final_nonstationary=gd[~gd.stationary]
    report=rf"""# One initial descent direction with conditional restarts versus Horváth L-infinity

## Exact policy used

Use the same original 50 coefficient-recovery AR(3) paths, each with 2000 observations after 1000 burn-in observations and Uniform[-1,1] innovations. The known order is 3. Both methods receive the same observations, with no true coefficients or innovation half-width supplied to fitting.

The descent uses **one initial random direction**. After 200 evaluated iterations, inspect the recovered coefficients of its best-width incumbent. If undefined or with companion spectral radius at least 1, discard that attempt and only then draw a new random direction. Reset the geometric schedule and patience for the new attempt. Each attempt uses `max_it=2000`, improvement patience 200, tolerance `1e-10`, and `gamma_t=0.2*0.997**t`.

Return the first attempt that passes the checkpoint and reaches its usual stopping point. Do not run additional directions to improve that result. Do not apply a final stationarity filter or restart if it becomes nonstationary after the single checkpoint. This implements the requested resampling-only-on-trigger rule exactly. There is an administrative 100-attempt safety guard; no process reached it (maximum actual attempts: {comparison['maximum_attempts']}). There is no 12-start allocation or cap.

The random stream is the original study's process-specific stream. Directions are generated one at a time, only when needed. The first 12 possible draws agree exactly with the earlier bank, but unused directions are not evaluated or drawn during the actual fit. Initial state is evaluated iteration 1; the check reads stored index 199. Both methods use the newest-first point cloud with 1997 rows.

## Horváth benchmark

The benchmark is the repository's existing **no-intercept** Horváth L-infinity fit:

$$\min_{{\phi\in\mathbb{{R}}^3,\ h\geq0}}h\quad\text{{subject to}}\quad|X_t-\phi_1X_{{t-1}}-\phi_2X_{{t-2}}-\phi_3X_{{t-3}}|\leq h.$$

Here only h is constrained nonnegative; the three coefficients are unrestricted. The residual centre is fixed at zero. This differs from the earlier free-centre Chebyshev benchmark. The implementation uses HiGHS and its default convex-hull constraint reduction, followed by residual feasibility checks on all observations and a primal/dual objective check. It imposes no stationarity constraint. Source: [classical_selectors.py, horvath_linf_coefficients]({AR_DIR}/ar_tools/classical_selectors.py:204). Original reference: [Horváth (1989), AR models with uniformly distributed noise, pp. 396–401, publication record](https://dml.cz/handle/10338.dmlcz/104367). The exact benchmark formulation above is verified from the workspace implementation; the paper's full PDF could not be retrieved through the browsing tool during this comparison. Solver interface: [SciPy linprog documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html).

Your method still minimizes normalized directional width on the sphere, whereas the LP minimizes maximum absolute residual. The comparison below evaluates coefficient recovery against the same true coefficients, rather than treating these different optimization objectives as numerically interchangeable. Sources: [descent implementation]({ROOT}/output/analysis/ar3_early_abort_200_12starts/run_experiment.py) and the LP implementation linked above.

## Results across all 50 processes

| Measure | Your descent with conditional resampling | Horváth L-infinity |
|---|---:|---:|
| Mean coefficient MAE | {a.mean_coefficient_mae:.9g} | {b.mean_coefficient_mae:.9g} |
| Median coefficient MAE | {a.median_coefficient_mae:.9g} | {b.median_coefficient_mae:.9g} |
| Maximum coefficient MAE | {a.maximum_coefficient_mae:.9g} | {b.maximum_coefficient_mae:.9g} |
| Stationary returned estimates | {int(a.stationary_estimates)}/50 | {int(b.stationary_estimates)}/50 |
| Mean fit time | {a.mean_fit_ms:.3f} ms | {b.mean_fit_ms:.3f} ms |
| Median fit time | {a.median_fit_ms:.3f} ms | {b.median_fit_ms:.3f} ms |

Horváth has lower coefficient MAE on {comparison['horvath_better']} paths, your descent on {comparison['descent_better']}, with {comparison['tied']} ties (comparison tolerance: absolute `1e-10` plus relative `1e-6`). MAE is the average of the three absolute coefficient errors, then averaged over all 50 processes for the aggregate. No processes were excluded.

Your method used {comparison['descent_attempts']} actual attempts: 50 initial directions plus {comparison['descent_aborts']} triggered resamples. {comparison['processes_with_restart']} processes needed resampling; {50-comparison['processes_with_restart']} finished from their first direction. Mean attempts were {a.mean_attempts:.3g}, maximum {comparison['maximum_attempts']}, and mean evaluated iterations including aborted attempts were {a.mean_evaluated_states:.2f}.

The ratio of mean per-process fit times is {speed_ratio:.3f}:1 (descent / LP); the median of the paired time ratios is {paired_speed_ratio:.3f}:1. These are local implementation measurements, not claims about intrinsic algorithmic complexity or performance on other data.

On these 50 paths, Horváth performed better in both aggregate coefficient accuracy and measured fit time. The first run that passes the 200-iteration stationarity checkpoint can still yield a less accurate coefficient vector. Passing also does not guarantee the returned vector remains stationary: the descent's nonstationary returned estimate is process {', '.join(map(str,final_nonstationary.index))}. The 200-iteration trigger's two limitations were previously documented in [the checkpoint examples]({AR_DIR}/outputs/csv/ar3_early_abort_200_12starts/checkpoint_counterexamples.md).

Processes where descent has lower coefficient MAE:

{markdown_table(wins)}

Largest descent errors:

{markdown_table(worst)}

## Timing protocol and limits

Each method was warmed up three times, then timed five times per path with the method order alternating. Report the median fit time per path, then the mean and median of those 50 medians. Each timing starts with the in-memory series. Descent includes point-cloud construction, conditional random draws, all aborted attempts, the surviving attempt, the checkpoint, and trace recording. Horváth includes its cloud/hull construction, LP assembly and solve, and full-data feasibility/optimality checks. File I/O, saving traces, true-error scoring, and plotting are excluded from both. Repeated timing fits use the same deterministic random stream; they do not add independent starting-direction experiments.

BLAS libraries were limited to one thread; HiGHS uses the existing estimator's solver defaults. Environment and versions are saved in the configuration. Only the requested settings and original 50 paths were tested. No additional tuning was performed.

## Files and verification

[Configuration]({OUT}/configuration.json), [summary]({OUT}/summary.csv), [all 100 results]({OUT}/per_process_results.csv), [actual descent attempts]({OUT}/descent_attempts.csv), [500 timing samples]({OUT}/timing_measurements.csv), [paired comparison]({OUT}/paired_process_comparison.csv), [coefficient MAE]({OUT}/coefficient_mae.csv), [validation]({OUT}/validation.json), [plot PDF]({pdf_dir}/comparison.pdf).

The experiment reproduces each consumed direction's previous trajectory exactly and verifies it returns the first nonaborted attempt. An independent audit checks all actual abort decisions, stopping counts, coefficient errors and root diagnostics; solves all 50 LPs independently on the complete observation set; verifies hull-reduced estimates and full-data objective/feasibility; and recomputes all timing medians. See [run_experiment.py]({ROOT}/output/analysis/{NAME}/run_experiment.py) and [verification code]({ROOT}/output/analysis/{NAME}/verify_and_report.py).
"""
    (OUT/"findings.md").write_text(report)
    return image_dir/"comparison.png"


def main():
    validation=validate()
    image=plot_and_report()
    print(json.dumps(validation,indent=2))
    print(f"Figure: {image}")


if __name__=="__main__":
    main()
