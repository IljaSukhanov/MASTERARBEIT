"""Independent checkpoint/selection audit and scientific figures."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Code.Code_AR_processes.ar_tools.coefficient_errors import coefficient_mae
from Code.Code_AR_processes.ar_tools.functions import sliding_window_point_cloud
from output.analysis.ar3_early_abort_200_12starts.run_experiment import (
    AR_DIR, OUT, NAME, FIXED, TRIGGERED, CHECKPOINT, TOLERANCE,
)
from Code.Code_AR_processes.ar_tools.coefficient_recovery_inputs import radii, load_process
from output.analysis.ar3_conditional_restarts_2000_200.verify_and_report import markdown_table


def incumbent_indices(widths):
    best, index = np.inf, -1
    indices = []
    updates = []
    for t,width in enumerate(widths):
        if width < best-TOLERANCE:
            best, index = width, t
            updates.append(t)
        indices.append(index)
    return np.array(indices), np.array(updates)


def validate():
    maximum_width_error, recovered_aborts, baseline_evaluations, policy_evaluations = 0., 0, 0, 0
    for replication in range(1,51):
        record=json.loads((OUT/"replications"/f"process_{replication:03d}.json").read_text())
        series, _, _ = load_process(replication)
        points=sliding_window_point_cloud(series,4)
        truth=np.array([record["results"][0][f"true_phi_{j}"] for j in range(1,4)])
        assert len(record["events"])==len(record["baseline_candidates"])==len(record["policy_candidates"])==12
        with np.load(OUT/"traces"/f"process_{replication:03d}.npz") as trace:
            for i,event in enumerate(record["events"]):
                start=i+1
                base=trace[f"baseline_{start:02d}_direction_history"]
                policy=trace[f"policy_{start:02d}_direction_history"]
                w=trace[f"baseline_{start:02d}_width_history"]
                indices, updates=incumbent_indices(w)
                checkpoint_direction=base[indices[CHECKPOINT-1]]
                phi=-checkpoint_direction[1:]/checkpoint_direction[0]
                rho=float(np.max(abs(np.roots(np.r_[1.,-phi]))))
                np.testing.assert_allclose(rho,event["checkpoint_best_rho"],rtol=1e-10,atol=1e-12)
                assert event["aborted"]==bool(rho>=1)
                expected_length=CHECKPOINT if event["aborted"] else len(base)
                assert len(policy)==expected_length
                np.testing.assert_array_equal(policy,base[:expected_length])
                baseline_evaluations+=len(base)
                policy_evaluations+=len(policy)
                for label,candidates in [("baseline",record["baseline_candidates"]),("policy",record["policy_candidates"])]:
                    candidate=candidates[i]
                    directions=trace[f"{label}_{start:02d}_direction_history"]
                    widths=trace[f"{label}_{start:02d}_width_history"]
                    accepted,last=incumbent_indices(widths)
                    np.testing.assert_array_equal(np.flatnonzero(trace[f"{label}_{start:02d}_best_update_history"]),last)
                    u=np.array(candidate["direction"])
                    np.testing.assert_array_equal(u,directions[accepted[-1]])
                    assert len(widths)==candidate["evaluated_states"]
                    if candidate["termination_reason"]=="patience":
                        assert len(widths)-1-last[-1]==200
                    elif candidate["termination_reason"]=="iteration_limit":
                        assert len(widths)==2000
                    else:
                        assert label=="policy" and len(widths)==200
                    error=abs(float(np.ptp(points@u))-candidate["minimum_width"])
                    maximum_width_error=max(maximum_width_error,error)
                    assert error<1e-10
                    phi=-u[1:]/u[0]
                    np.testing.assert_allclose(phi,candidate["coefficients"],rtol=1e-12,atol=1e-14)
                    np.testing.assert_allclose(coefficient_mae(phi, truth),candidate["coefficient_mae"],rtol=1e-12)
                    root=float(np.max(abs(np.roots(np.r_[1.,-phi]))))
                    assert bool(root<1)==candidate["stationary"]
                recovered_aborts+=int(event["aborted"] and record["baseline_candidates"][i]["stationary"])
        for result in record["results"]:
            triggered=result["method"]==TRIGGERED
            candidates=record["policy_candidates"] if triggered else record["baseline_candidates"]
            eligible=[c for c in candidates if c["termination_reason"]!="stationarity_abort"]
            selected=min(eligible or candidates,key=lambda c:c["minimum_width"])
            assert result["selected_start"]==selected["start"]
            assert result["minimum_width"]==selected["minimum_width"]
            assert result["evaluated_states"]==sum(c["evaluated_states"] for c in candidates)
            assert result["attempts"]==12
    validation=dict(processes=50,policy_results_checked=100,attempts_per_policy_checked=600,
        all_600_prefixes_match_public_optimizer=True,
        checkpoint_roots_independently_checked=True,all_1200_candidate_widths_and_coefficients_checked=True,
        all_1200_trace_stopping_counts_checked=True,all_final_selections_and_iteration_costs_checked=True,
        maximum_candidate_width_error=maximum_width_error,
        baseline_evaluations=baseline_evaluations,policy_evaluations=policy_evaluations,
        aborted_runs_that_would_recover=recovered_aborts)
    (OUT/"validation.json").write_text(json.dumps(validation,indent=2)+"\n")
    return validation


def plot_and_report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results=pd.read_csv(OUT/"per_process_results.csv")
    events=pd.read_csv(OUT/"per_start_diagnostics.csv")
    summary=pd.read_csv(OUT/"summary.csv").set_index("method")
    comparison=json.loads((OUT/"paired_comparison.json").read_text())
    fixed=results[results.method==FIXED].set_index("replication")
    triggered=results[results.method==TRIGGERED].set_index("replication")
    paired=pd.DataFrame(dict(fixed_mae=fixed.coefficient_mae,triggered_mae=triggered.coefficient_mae,
        fixed_stationary=fixed.stationary,triggered_stationary=triggered.stationary,
        fixed_evaluations=fixed.evaluated_states,triggered_evaluations=triggered.evaluated_states,
        aborted_attempts=triggered.aborted_attempts))
    paired.to_csv(OUT/"paired_process_comparison.csv")
    image_dir=AR_DIR/"outputs/images"/NAME/"mae"
    pdf_dir=AR_DIR/"outputs/pdfs"/NAME/"mae"
    image_dir.mkdir(parents=True,exist_ok=True)
    pdf_dir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout="constrained")
    ax=axes[0]
    for mask,color,label in [(paired.fixed_stationary,"#247b86","Ordinary result stationary"),
                              (~paired.fixed_stationary,"#ce653a","Ordinary result nonstationary")]:
        ax.scatter(paired.fixed_mae[mask],paired.triggered_mae[mask],color=color,s=40,label=label,alpha=.85)
    low=float(paired[["fixed_mae","triggered_mae"]].min().min())*.5
    high=float(paired[["fixed_mae","triggered_mae"]].max().max())*2
    ax.plot([low,high],[low,high],linestyle="--",color=".5",linewidth=1)
    ax.set(xscale="log",yscale="log",xlim=(low,high),ylim=(low,high),
        xlabel="No early abort: coefficient MAE",ylabel="Abort at 200: coefficient MAE",title="Paired coefficient recovery")
    ax.legend(loc="upper left",fontsize=8)
    ax=axes[1]
    ax.plot(paired.index,paired.fixed_evaluations,".-",color="#5470aa",label="No early abort",linewidth=.8)
    ax.plot(paired.index,paired.triggered_evaluations,".-",color="#ce653a",label="Stationarity abort at 200",linewidth=.8)
    ax.set(xlabel="Process",ylabel="Total evaluated iterations",ylim=(0,None),title="Same 12 attempts in both modes")
    ax.legend(fontsize=8)
    fig.suptitle("50 AR(3) paths: max_it = 2000, patience = 200, stationarity checkpoint = 200",fontsize=12)
    fig.savefig(image_dir/"01_comparison.png",dpi=180)
    fig.savefig(pdf_dir/"01_comparison.pdf")
    plt.close(fig)

    # These cases illustrate catching failure, aborting a later recovery, and
    # passing at 200 but becoming nonstationary afterwards, respectively.
    examples=[(35,int(fixed.loc[35].selected_start),"Aborted failing direction"),
              (8,int(fixed.loc[8].selected_start),"Aborted direction that would recover"),
              (43,int(triggered.loc[43].selected_start),"Passed at 200, became nonstationary later")]
    fig,axes=plt.subplots(1,3,figsize=(14,4.4),layout="constrained")
    for ax,(replication,start,title) in zip(axes,examples):
        with np.load(OUT/"traces"/f"process_{replication:03d}.npz") as trace:
            u=trace[f"baseline_{start:02d}_direction_history"]
            w=trace[f"baseline_{start:02d}_width_history"]
            indices,_=incumbent_indices(w)
            rho,_=radii(u[indices])
            length=len(trace[f"policy_{start:02d}_width_history"])
        record=events[(events.replication==replication)&(events.start==start)].iloc[0]
        if record.aborted:
            ax.plot(np.arange(1,len(rho)+1),rho,color=".55",linestyle="--",label="Without abort (counterfactual)",linewidth=1)
            ax.plot(np.arange(1,length+1),rho[:length],color="#ce653a",label="Actually followed",linewidth=1.2)
        else:
            ax.plot(np.arange(1,len(rho)+1),rho,color="#247b86",label="Followed after passing check",linewidth=1.2)
        truth=np.array([fixed.loc[replication,f"true_phi_{j}"] for j in range(1,4)])
        true_rho=float(max(abs(np.roots(np.r_[1.,-truth]))))
        ax.axhline(1,color=".4",linestyle=":",label="Stationarity boundary")
        ax.axhline(true_rho,color="black",linestyle="--",linewidth=.8,label="True parameter radius")
        ax.axvline(CHECKPOINT,color="#a7503a",linestyle=":",linewidth=1)
        ax.scatter([CHECKPOINT],[rho[CHECKPOINT-1]],color="#a7503a",s=25,zorder=4)
        ax.set(xlabel="Evaluated iteration",ylabel="Best estimate's companion radius",
               title=f"Process {replication}, start {start}\n{title}",yscale="symlog")
        if float(np.max(rho)) < 2:
            ax.set_yscale("linear")
        else:
            ax.set_yscale("symlog",linthresh=1)
        ax.set_xlim(0,len(rho))
        ax.legend(fontsize=7,loc="best")
    fig.savefig(image_dir/"02_checkpoint_examples.png",dpi=180)
    fig.savefig(pdf_dir/"02_checkpoint_examples.pdf")
    plt.close(fig)

    changed=paired[abs(paired.triggered_mae-paired.fixed_mae)>1e-10]
    later_unstable=events[(~events.aborted)&(~events.baseline_final_stationary)]
    late_winners=triggered[~triggered.stationary]
    aborted_recoveries=events[events.would_recover_stationarity]
    aborted_recoveries.to_csv(OUT/"aborted_runs_that_would_recover.csv",index=False)
    later_unstable.to_csv(OUT/"passed_checkpoint_but_nonstationary_final.csv",index=False)
    a,b=summary.loc[FIXED],summary.loc[TRIGGERED]
    report=f"""# Twelve attempts with and without a stationarity check at iteration 200

## Frozen experiment

Use the same 50 coefficient-recovery AR(3) paths, same ordered bank of 12 random starting directions for each path, same geometric schedule `gamma_t=0.2*0.997**t`, `max_it=2000`, improvement patience 200, and width-improvement tolerance `1e-10`.

Both modes use **all 12 attempts** (the initial attempt plus 11 restarts). The comparison changes one attempt's stopping rule:

- Ordinary mode: each attempt stops at the iteration cap or after 200 consecutive evaluations without sufficient width improvement.
- Triggered mode: after exactly 200 evaluated states, recover phi from that attempt's **best-width incumbent**, and abort if the coefficients are undefined or the companion spectral radius is at least 1. Otherwise continue exactly the original trajectory, patience counter, and geometric schedule. There are no subsequent stationarity checks in this experiment.
- Aborted attempts are discarded from the final comparison. Among completed attempts, choose the smallest width **without a final stationarity filter**. Do not retain the best stationary intermediate iterate. If all 12 attempts were aborted, flag failure and retain the smallest-width aborted prefix only for explicit error accounting; this did not occur.
- A new attempt uses the next original random direction and resets its own step schedule and patience. The true coefficients never enter the abort or selection decision.

Iteration counts follow the existing optimizer: the initial state is evaluated iteration 1, and the check occurs after evaluation 200 (zero-based stored index 199), before another gradient update. Both modes have the same number of attempts and per-attempt cap; actual computation differs because early abortion is the purpose of the trigger. The reported savings concern evaluated iterations, not measured wall-clock speed.

This is a new exploratory test of the user's 200-iteration checkpoint, rather than the previous end-of-run stop-at-first-stationary policy or intermediate-iterate filtering experiment. The checkpoint was not tuned after seeing these results.

Implementation: [run_experiment.py]({ROOT}/output/analysis/{NAME}/run_experiment.py). Checkpoint stopping uses the shared optimizer with `stationarity_check=True, max_restarts=0`; the twelve-start outer policy controls subsequent attempts. Every triggered trajectory is checked against the existing [public optimizer]({AR_DIR}/ar_tools/descent.py): a rejected attempt is its exact first 200 states and a surviving attempt is its exact full trajectory.

The parameter check uses companion spectral radius below 1; equivalently the autoregressive polynomial has its roots outside the unit circle. See [statsmodels ArmaProcess.isstationary](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_process.ArmaProcess.isstationary.html) and the reciprocal-root equivalence in [is_invertible, Notes](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.tools.is_invertible.html#notes).

## Results on all 50 paths

| Measure | Ordinary 12 starts | 12 starts with abort at 200 |
|---|---:|---:|
| Stationary returned estimates | {int(a.stationary_estimates)}/50 | {int(b.stationary_estimates)}/50 |
| Mean coefficient MAE | {a.mean_coefficient_mae:.9g} | {b.mean_coefficient_mae:.9g} |
| Median coefficient MAE | {a.median_coefficient_mae:.9g} | {b.median_coefficient_mae:.9g} |
| Maximum coefficient MAE | {a.maximum_coefficient_mae:.9g} | {b.maximum_coefficient_mae:.9g} |
| Processes with MAE > 1 | {int(a.mae_above_1)} | {int(b.mae_above_1)} |
| Total attempts | {int(a.total_attempts)} | {int(b.total_attempts)} |
| Attempts aborted at 200 | 0 | {int(b.total_aborted)} |
| Total evaluated iterations | {int(a.total_evaluated_states):,} | {int(b.total_evaluated_states):,} |

MAE is the mean of the three absolute coefficient errors for a process, then averaged across all 50 processes for the aggregate. None of the 50 paths was removed.

The triggered rule improved coefficient MAE on {comparison['improved']} paths, worsened it on {comparison['worsened']}, and tied on {comparison['tied']} (absolute tolerance `1e-10` plus relative tolerance `1e-6`). It saved {100*comparison['iteration_saving_fraction']:.2f}% of iteration evaluations.

The mean error reduction is driven by avoiding the four original MAE-above-1 cases. The median is unchanged. These are observed results on the selected 50 paths, not a general guarantee.

Changed estimates:

{markdown_table(changed)}

## What happens inside the trigger

Of {comparison['aborted_starts']} aborted attempts, **{comparison['aborted_starts_that_would_recover']} would later return a stationary best estimate** if allowed to continue. The rule aborted {comparison['original_winners_aborted']} original winning starts, including {comparison['original_winners_aborted_but_would_recover']} whose eventual best estimate would be stationary. These counterfactual continuation outcomes are diagnostics only; they were unavailable to the policy at the checkpoint.

A stationarity failure at 200 is therefore a heuristic for rejecting an unpromising direction. It is not sufficient evidence that the trajectory cannot recover. Conversely, passing the check does not ensure the later best-width incumbent will remain stationary: {len(later_unstable)} passed attempts ended nonstationary. The triggered policy returned a nonstationary estimate on process {', '.join(map(str,late_winners.index))}: its selected start passed at 200 and later changed incumbent.

[Checkpoint examples]({image_dir}/02_checkpoint_examples.png) show an aborted failing direction, an aborted direction that would recover, and the later nonstationarity after a passed checkpoint. [Aborted but later recovered attempts]({OUT}/aborted_runs_that_would_recover.csv) and [passed but later nonstationary attempts]({OUT}/passed_checkpoint_but_nonstationary_final.csv) provide the complete diagnostics.

## Verification and outputs

[Configuration]({OUT}/configuration.json), [summary]({OUT}/summary.csv), [100 policy results]({OUT}/per_process_results.csv), [600 start diagnostics]({OUT}/per_start_diagnostics.csv), [paired comparison]({OUT}/paired_comparison.json), [independent validation]({OUT}/validation.json), [comparison PDF]({pdf_dir}/01_comparison.pdf), [checkpoint PDF]({pdf_dir}/02_checkpoint_examples.pdf).

The independent audit checks all 600 prefix decisions using polynomial roots, all 1200 saved candidate widths and coefficient errors, all 1200 trajectories' stopping counts, all selected estimates, and both modes' iteration costs. The implementation separately verified all 600 policy trajectories against the public optimizer and reproduced all 600 previous baseline results exactly. See [verification code]({ROOT}/output/analysis/{NAME}/verify_and_report.py).
"""
    (OUT/"findings.md").write_text(report)
    return image_dir


def main():
    validation=validate()
    image_dir=plot_and_report()
    print(json.dumps(validation,indent=2))
    print(f"Figures: {image_dir}")


if __name__=="__main__":
    main()
