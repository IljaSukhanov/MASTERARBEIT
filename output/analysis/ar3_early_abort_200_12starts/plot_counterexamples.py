"""Detailed iteration plots for the two failure modes of a check at 200.

Read existing saved trajectories only; do not change fits or rerun descent.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from output.analysis.ar3_early_abort_200_12starts.run_experiment import AR_DIR, OUT, NAME, CHECKPOINT
from output.analysis.ar3_early_abort_200_12starts.verify_and_report import incumbent_indices
from Code.Code_AR_processes.ar_tools.coefficient_recovery_inputs import radii


def load_example(replication, start):
    record=json.loads((OUT/"replications"/f"process_{replication:03d}.json").read_text())
    truth=np.array([record["results"][0][f"true_phi_{j}"] for j in range(1,4)])
    with np.load(OUT/"traces"/f"process_{replication:03d}.npz") as trace:
        u=trace[f"baseline_{start:02d}_direction_history"]
        widths=trace[f"baseline_{start:02d}_width_history"]
        best_widths=trace[f"baseline_{start:02d}_best_width_history"]
        policy_length=len(trace[f"policy_{start:02d}_width_history"])
        indices,_=incumbent_indices(widths)
        best_u=u[indices]
        rho,phi=radii(best_u)
        np.testing.assert_array_equal(trace[f"policy_{start:02d}_direction_history"],u[:policy_length])
    # Independently verify all plotted roots with characteristic polynomials.
    independent=np.array([max(abs(np.roots(np.r_[1.,-p]))) for p in phi])
    np.testing.assert_allclose(rho,independent,rtol=1e-10,atol=1e-12)
    event=record["events"][start-1]
    np.testing.assert_allclose(rho[CHECKPOINT-1],event["checkpoint_best_rho"],rtol=1e-12)
    late=np.arange(len(rho))>=CHECKPOINT
    recovery=np.flatnonzero(late&(rho<1))
    failure=np.flatnonzero(late&(rho>=1))
    true_rho=float(max(abs(np.roots(np.r_[1.,-truth]))))
    data=pd.DataFrame(dict(replication=replication,start=start,
        iteration=np.arange(1,len(rho)+1),best_width=best_widths,
        best_spectral_radius=rho,stationary=rho<1,true_spectral_radius=true_rho,
        followed_by_early_abort_policy=np.arange(len(rho))<policy_length,
        coefficient_mse=np.mean((phi-truth)**2,axis=1)))
    for j in range(3):
        data[f"best_phi_{j+1}"]=phi[:,j]
    info=dict(replication=replication,start=start,aborted=event["aborted"],
        checkpoint_iteration=CHECKPOINT,checkpoint_radius=float(rho[CHECKPOINT-1]),
        final_iteration=len(rho),final_radius=float(rho[-1]),true_radius=true_rho,
        first_stationary_iteration_after_checkpoint=int(recovery[0]+1) if len(recovery) else None,
        first_nonstationary_iteration_after_checkpoint=int(failure[0]+1) if len(failure) else None,
        checkpoint_coefficient_mse=float(data.coefficient_mse.iloc[CHECKPOINT-1]),
        final_coefficient_mse=float(data.coefficient_mse.iloc[-1]))
    return data,info


def annotate(ax,text,xy,position,color=".2"):
    ax.annotate(text,xy=xy,xytext=position,textcoords="data",fontsize=9,color=color,
        arrowprops=dict(arrowstyle="->",linewidth=.8,color=color),
        bbox=dict(boxstyle="round,pad=.25",facecolor="white",edgecolor="none",alpha=.95))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image_dir=AR_DIR/"outputs/images"/NAME
    pdf_dir=AR_DIR/"outputs/pdfs"/NAME
    image_dir.mkdir(parents=True,exist_ok=True)
    pdf_dir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    all_data,examples=[],[]
    cases=[(8,11,"03_aborted_run_that_recovers","Aborted at 200, but would recover"),
           (43,2,"04_passed_run_that_becomes_nonstationary","Passed at 200, then became nonstationary")]
    for replication,start,stem,title in cases:
        data,info=load_example(replication,start)
        all_data.append(data)
        examples.append(info)
        x=data.iteration.to_numpy()
        rho=data.best_spectral_radius.to_numpy()
        checkpoint_rho=info["checkpoint_radius"]
        final_rho=info["final_radius"]
        aborted=info["aborted"]
        fig,axes=plt.subplots(1,2,figsize=(11.4,4.7))
        fig.subplots_adjust(left=.075,right=.98,bottom=.24,top=.79,wspace=.28)
        fig.suptitle(f"Process {replication}, start {start}: {title}",fontsize=13,y=.98)
        for ax in axes:
            if aborted:
                ax.plot(x[:CHECKPOINT],rho[:CHECKPOINT],color="#cf6638",linewidth=1.5,
                        label="Followed before abortion")
                ax.plot(x[CHECKPOINT-1:],rho[CHECKPOINT-1:],color="#3976a5",linestyle="--",
                        linewidth=1.5,label="Continuation without abortion")
            else:
                ax.plot(x,rho,color="#247c87",linewidth=1.5,label="Followed throughout")
            ax.axhline(1,color=".35",linestyle=":",linewidth=1.3,label="Stationarity boundary (1)")
            ax.axhline(info["true_radius"],color="black",linestyle="--",linewidth=.9,
                       label=f"True parameter radius ({info['true_radius']:.4f})")
            ax.axvline(CHECKPOINT,color="#9b4771",linestyle=":",linewidth=1)
            ax.scatter([CHECKPOINT],[checkpoint_rho],color="#9b4771",s=32,zorder=5)
            ax.set(xlabel="Evaluated iteration",ylabel="Best estimate's companion spectral radius")
            ax.grid(axis="y",alpha=.12)
        full,zoom=axes
        full.set(title="Full saved trajectory",xlim=(0,len(x)+35))
        if aborted:
            full.set_yscale("symlog",linthresh=1)
            full.scatter([len(x)],[final_rho],color="#3976a5",s=22,zorder=4)
            annotate(full,f"Final radius = {final_rho:.4f}",(len(x),final_rho),(1100,1.7))
            zoom.set(title="Detail around the checkpoint and recovery",xlim=(150,600),ylim=(.92,1.25),
                     yticks=[.95,1,1.05,1.1,1.15,1.2,1.25])
            recovery=info["first_stationary_iteration_after_checkpoint"]
            assert recovery==371 and bool(data.stationary.iloc[recovery-1:].all())
            recovery_rho=rho[recovery-1]
            zoom.scatter([recovery],[recovery_rho],color="#3976a5",s=28,zorder=5)
            annotate(zoom,f"Abort at 200\nradius = {checkpoint_rho:.4f}",(200,checkpoint_rho),(255,1.18),"#9b4771")
            annotate(zoom,"Stationary from 371 onward",(recovery,recovery_rho),(375,1.065),"#3976a5")
        else:
            full.set_ylim(.92,1.055)
            full.scatter([len(x)],[final_rho],color="#247c87",s=22,zorder=4)
            annotate(full,f"Final radius = {final_rho:.4f}",(len(x),final_rho),(315,1.03))
            zoom.set(title="Detail around passing and the later crossing",xlim=(175,320),ylim=(.92,1.055))
            failure=info["first_nonstationary_iteration_after_checkpoint"]
            assert failure==256 and bool((~data.stationary.iloc[failure-1:]).all())
            fail_rho=rho[failure-1]
            zoom.scatter([failure],[fail_rho],color="#247c87",s=28,zorder=5)
            annotate(zoom,f"Pass at 200\nradius = {checkpoint_rho:.4f}",(200,checkpoint_rho),(205,.945),"#9b4771")
            annotate(zoom,"Nonstationary from 256",(failure,fail_rho),(190,1.047),"#247c87")
        handles,labels=full.get_legend_handles_labels()
        fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,.005),ncol=2,
                   frameon=False,fontsize=9)
        fig.savefig(image_dir/f"{stem}.png",dpi=180,bbox_inches="tight")
        fig.savefig(pdf_dir/f"{stem}.pdf",bbox_inches="tight")
        plt.close(fig)
        info["png"]=str(image_dir/f"{stem}.png")
        info["pdf"]=str(pdf_dir/f"{stem}.pdf")
    pd.concat(all_data,ignore_index=True).to_csv(OUT/"checkpoint_counterexamples_history.csv",index=False)
    (OUT/"checkpoint_counterexamples.json").write_text(json.dumps(examples,indent=2)+"\n")
    report=f"""# Detailed examples of the 200-iteration checkpoint

The plotted quantity is the companion spectral radius of the **best-width incumbent's recovered coefficients** at each evaluated iteration. The boundary used by this experiment is radius 1. This parameter criterion and companion-root equivalence are documented in [statsmodels stationarity](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_process.ArmaProcess.isstationary.html) and [is_invertible, Notes](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.tools.is_invertible.html#notes). These curves describe the recorded experiment, not a claim about every possible AR fit.

## Aborted although it would recover: process 8, start 11

The radius is {examples[0]['checkpoint_radius']:.9f} at evaluated iteration 200, so the rule aborts. The saved ordinary trajectory becomes stationary at iteration {examples[0]['first_stationary_iteration_after_checkpoint']} and remains stationary thereafter, ending at radius {examples[0]['final_radius']:.9f} at iteration {examples[0]['final_iteration']}. Its final coefficient MSE is {examples[0]['final_coefficient_mse']:.9g}. Only the solid orange segment was followed by the early-abort policy; the dashed blue continuation shows the ordinary run without that abortion.

![Aborted but later recovered]({examples[0]['png']})

[PDF]({examples[0]['pdf']})

## Passed but later became nonstationary: process 43, start 2

The radius is {examples[1]['checkpoint_radius']:.9f} at iteration 200, so the attempt continues. At iteration {examples[1]['first_nonstationary_iteration_after_checkpoint']} a new best-width incumbent is nonstationary. The best incumbent remains nonstationary through the final evaluated iteration {examples[1]['final_iteration']}, with radius {examples[1]['final_radius']:.9f}. This attempt is the triggered method's selected result for process 43. The entire teal curve was followed by the policy.

![Passed but later nonstationary]({examples[1]['png']})

[PDF]({examples[1]['pdf']})

The plots also show each true parameter radius as a black dashed horizontal line. Full and zoomed views use the same data; the full recovery plot uses a symmetric-log scale (linear below 1), while the other views use linear scales. Initial state is evaluated iteration 1, so the checkpoint is stored index 199. The final coefficient MSE of the nonstationary second example is {examples[1]['final_coefficient_mse']:.9g}; parameter validity and coefficient error are distinct recorded diagnostics here.

Sources: [all plotted iterations]({OUT}/checkpoint_counterexamples_history.csv), [exact event values]({OUT}/checkpoint_counterexamples.json), [record 8]({OUT}/replications/process_008.json), [record 43]({OUT}/replications/process_043.json), [plot implementation]({ROOT}/output/analysis/{NAME}/plot_counterexamples.py). The plotted roots were independently cross-checked against characteristic-polynomial roots at every saved iteration. Existing fits and checkpoint decisions were not changed.
"""
    (OUT/"checkpoint_counterexamples.md").write_text(report)
    print(json.dumps(examples,indent=2))
    return examples


if __name__=="__main__":
    main()
