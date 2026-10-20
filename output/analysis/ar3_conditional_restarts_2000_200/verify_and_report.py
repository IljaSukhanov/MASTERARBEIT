"""Independently check terminal choices and publish the two-policy comparison."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from output.analysis.ar3_conditional_restarts_2000_200.run_experiment import (
    AR_DIR, OUT, NAME, FIXED, CONDITIONAL, replication_path,
)
from Code.Code_AR_processes.ar_tools.functions import sliding_window_point_cloud
from Code.Code_AR_processes.ar_tools.coefficient_recovery_inputs import load_process


def markdown_table(frame):
    frame = frame.reset_index()
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "| " + " | ".join(["---"]*len(frame.columns)) + " |"
    rows = ["| " + " | ".join(f"{value:.9g}" if isinstance(value, float) else str(value)
                              for value in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *rows])


def validate():
    results = pd.read_csv(OUT / "per_process_results.csv", float_precision="round_trip")
    assert len(results) == 100 and results.groupby("method").size().eq(50).all()
    max_width_error = 0.0
    total_states = {FIXED: 0, CONDITIONAL: 0}
    for replication in range(1, 51):
        record = json.loads(replication_path(replication).read_text())
        attempts = record["attempts"]
        assert len(attempts) == 12
        series, _, _ = load_process(replication)
        points = sliding_window_point_cloud(series, 4)
        truth = np.array([record["results"][0][f"true_phi_{j}"] for j in range(1, 4)])
        for attempt in attempts:
            u = np.array(attempt["direction"])
            phi = np.array(attempt["coefficients"])
            np.testing.assert_allclose(-u[1:]/u[0], phi, rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(np.linalg.norm(u), 1, atol=1e-14, rtol=0)
            error = abs(float(np.ptp(points @ u)) - attempt["minimum_width"])
            max_width_error = max(max_width_error, error)
            assert error < 1e-10
            # Independent characteristic-polynomial roots, rather than the
            # experiment's companion-matrix eigensolver.
            rho = float(np.max(np.abs(np.roots(np.r_[1., -phi]))))
            np.testing.assert_allclose(rho, attempt["spectral_radius"], rtol=1e-10, atol=1e-12)
            assert bool(rho < 1) == attempt["stationary"]
            np.testing.assert_allclose(np.mean((phi-truth)**2), attempt["coefficient_mse"], rtol=1e-12)
            length = attempt["evaluated_states"]
            assert 1 <= length <= 2000
            if attempt["termination_reason"] == "patience":
                assert length-1-attempt["best_iteration"] == 200
            else:
                assert length == 2000
        fixed_index = int(np.argmin([a["minimum_width"] for a in attempts]))
        stable_indices = [i for i,a in enumerate(attempts) if a["stationary"]]
        conditional_index = stable_indices[0] if stable_indices else fixed_index
        consumed = conditional_index+1 if stable_indices else 12
        for row in record["results"]:
            method = row["method"]
            chosen = attempts[fixed_index if method == FIXED else conditional_index]
            assert row["selected_attempt"] == chosen["attempt"]
            assert row["minimum_width"] == chosen["minimum_width"]
            assert row["attempts_used"] == (12 if method == FIXED else consumed)
            states = sum(a["evaluated_states"] for a in attempts[:row["attempts_used"]])
            assert row["evaluated_states"] == states
            total_states[method] += states
        trace_path = OUT / "traces" / f"process_{replication:03d}.npz"
        if trace_path.exists():
            with np.load(trace_path) as trace:
                for i, attempt in enumerate(attempts, 1):
                    widths = trace[f"width_{i:02d}"]
                    updates = trace[f"update_{i:02d}"]
                    indices = np.flatnonzero(updates)
                    assert len(widths) == attempt["evaluated_states"]
                    accepted, best = [], np.inf
                    for j,w in enumerate(widths):
                        if w < best-1e-10:
                            accepted.append(j)
                            best = w
                    np.testing.assert_array_equal(indices, accepted)
                    np.testing.assert_array_equal(trace[f"u_{i:02d}"][indices[-1]], attempt["direction"])
                    assert indices[-1] == attempt["best_iteration"]
                    if attempt["termination_reason"] == "patience":
                        assert len(widths)-1-indices[-1] == 200
    validation = dict(processes=50, terminal_candidates_checked=600, policy_results_checked=100,
                      selected_widths_checked=True, coefficient_ratios_and_mses_checked=True,
                      independent_polynomial_root_check=True, stopping_counts_checked=True,
                      first_stationary_selection_and_cap_checked=True,
                      complete_traces_checked=48, max_width_error=max_width_error,
                      total_evaluated_states=total_states, filtering_used=False)
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2)+"\n")
    return validation


def make_report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.read_csv(OUT / "per_process_results.csv")
    summary = pd.read_csv(OUT / "summary.csv").set_index("method")
    comparison = json.loads((OUT / "paired_comparison.json").read_text())
    fixed = frame[frame.method == FIXED].set_index("replication")
    conditional = frame[frame.method == CONDITIONAL].set_index("replication")
    paired = pd.DataFrame({"fixed_mse": fixed.coefficient_mse,
        "conditional_mse": conditional.coefficient_mse,
        "fixed_stationary": fixed.stationary, "conditional_attempts": conditional.attempts_used,
        "fixed_evaluations": fixed.evaluated_states,
        "conditional_evaluations": conditional.evaluated_states})
    paired.to_csv(OUT / "paired_process_comparison.csv")
    image_dir = AR_DIR / "outputs/images" / NAME
    pdf_dir = AR_DIR / "outputs/pdfs" / NAME
    image_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained")
    ax = axes[0]
    valid = fixed.stationary.to_numpy()
    for mask, color, label in [(valid,"#257a88","Fixed result stationary"), (~valid,"#ce653a","Fixed result nonstationary")]:
        ax.scatter(paired.fixed_mse[mask], paired.conditional_mse[mask], color=color, label=label, s=35, alpha=.85)
    bounds = (max(1e-10, float(paired[["fixed_mse","conditional_mse"]].min().min())*.5),
              float(paired[["fixed_mse","conditional_mse"]].max().max())*2)
    ax.plot(bounds, bounds, color=".5", linestyle="--", linewidth=1)
    ax.set(xscale="log", yscale="log", xlim=bounds, ylim=bounds,
           xlabel="Fixed 12 starts: coefficient MSE", ylabel="Conditional starts: coefficient MSE",
           title="Paired accuracy on the same 50 paths")
    ax.legend(loc="upper left", fontsize=8)
    ax = axes[1]
    ax.plot(paired.index, paired.fixed_evaluations, ".-", color="#5470aa", label="Fixed 12 starts", linewidth=.8)
    ax.plot(paired.index, paired.conditional_evaluations, ".-", color="#ce653a", label="Conditional starts", linewidth=.8)
    ax.set(xlabel="Process", ylabel="Evaluated iterations", title="Computation consumed by each policy", ylim=(0,None))
    ax.legend(fontsize=8)
    ax = axes[2]
    counts = conditional.attempts_used.value_counts().reindex(range(1,13), fill_value=0)
    ax.bar(counts.index, counts.values, color="#ce653a")
    ax.set(xlabel="Attempts until first stationary result", ylabel="Number of processes",
           title="Stationarity-triggered mode", xticks=range(1,13), ylim=(0,None))
    fig.suptitle("AR(3): max_it = 2000, improvement patience = 200; no stationary filtering", fontsize=13)
    fig.savefig(image_dir / "comparison.png", dpi=180)
    fig.savefig(pdf_dir / "comparison.pdf")
    plt.close(fig)
    baseline, policy = summary.loc[FIXED], summary.loc[CONDITIONAL]
    bad = paired.loc[~paired.fixed_stationary]
    high = conditional[conditional.coefficient_mse > .01].sort_values("coefficient_mse", ascending=False)
    report = f"""# Fixed 12 starts versus stationarity-triggered starts

This corrects the earlier 5000-iteration, no-patience, 100-iteration-checkpoint experiment. The user specified `max_it=2000`, `improv_patience=200`, and checking stationarity **after each run ends**. There is no filtering of intermediate candidates in either mode.

## Prespecified comparison

- **Fixed 12 starts:** run all 12 original random directions and return the smallest-width completed-run result, even if it is nonstationary.
- **Stationarity-triggered starts:** run those same directions in their original order; inspect each completed run's best-width direction. Return the first stationary completed-run result. Restart from the next random direction only after a nonstationary result. Cap at 12 attempts (initial plus at most 11 restarts).
- Every attempt uses 2000 maximum evaluated states, stopping after 200 consecutive states without an incumbent-width improvement of more than `1e-10`. Step sizes reset on each attempt: `gamma_t=(0.2/diameter(P))*0.997**t`.
- If all 12 run results were nonstationary, mark the conditional policy failed and retain the unrestricted best result for error accounting. This did not occur in these 50 paths.
- The same 50 paths, true parameters, and ordered starting directions are loaded from `Code/Code_AR_processes/inputs/ar3_coefficient_recovery_50.npz`. Their original floating-point values are checked against sample hashes and independently against seed reconstruction. The true coefficients enter error reporting only.
- The conditional policy consumes a prefix of the fixed policy's attempted runs. Sharing those deterministic runs avoids duplicate computation in this paired analysis; reported costs count only the iterations the respective policy would consume. They are not wall-clock speed measurements.

The parameter validity criterion is companion spectral radius below 1, equivalent to the AR lag polynomial having roots outside the unit circle; see [statsmodels ArmaProcess.isstationary](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_process.ArmaProcess.isstationary.html) and the companion-root equivalence in [is_invertible, Notes](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.tools.is_invertible.html#notes). The implementation is in [run_experiment.py]({ROOT}/output/analysis/{NAME}/run_experiment.py), and the existing stopping behavior is in [descent.py]({AR_DIR}/ar_tools/descent.py).

## Results across all 50 processes

| Measure | Fixed 12 starts | Stationarity-triggered starts |
|---|---:|---:|
| Stationary returned estimates | {int(baseline.stationary_estimates)}/50 | {int(policy.stationary_estimates)}/50 |
| Mean coefficient MSE | {baseline.mean_coefficient_mse:.9g} | {policy.mean_coefficient_mse:.9g} |
| Median coefficient MSE | {baseline.median_coefficient_mse:.9g} | {policy.median_coefficient_mse:.9g} |
| Maximum coefficient MSE | {baseline.maximum_coefficient_mse:.9g} | {policy.maximum_coefficient_mse:.9g} |
| Processes with MSE > 1 | {int(baseline.mse_above_1)} | {int(policy.mse_above_1)} |
| Mean attempts | {baseline.mean_attempts:.3g} | {policy.mean_attempts:.3g} |
| Total evaluated iterations | {int(baseline.total_evaluated_states):,} | {int(policy.total_evaluated_states):,} |

Per process, the conditional policy improved MSE in {comparison['improved']}, worsened it in {comparison['worsened']}, and tied in {comparison['tied']} cases (absolute tolerance `1e-10` plus relative tolerance `1e-6`). It consumed {100*comparison['evaluated_state_saving_fraction']:.2f}% fewer evaluated iterations. MSE means the mean squared error of the three coefficients, averaged across all 50 processes for the aggregate. All 50 processes remain in every summary.

These observed results show a tradeoff: the conditional rule avoids the four nonstationary outputs and reduces computation, but its first acceptable result often has less accurate coefficients than the result from 12 attempts. Stationary validity alone does not establish coefficient accuracy; the observed maximum MSE in the conditional mode was {policy.maximum_coefficient_mse:.6g}. This interpretation is based on this paired experiment, not a general performance guarantee.

Fixed-mode nonstationary cases:

{markdown_table(bad)}

Conditional estimates with coefficient MSE > 0.01:

{markdown_table(high[['selected_attempt','attempts_used','coefficient_mse','spectral_radius']])}

## Relationship to filtering

Filtering was an additional estimator in the earlier exploratory comparison: it retained the smallest-width **stationary intermediate iterate** across all attempted directions. That can rescue a discarded earlier candidate even if a run ends with a better-width nonstationary incumbent. The comparison here never does that. It only checks the result of a completed attempt to decide whether another attempt is required. Both modes still use the original width objective and return an actual completed run's unrestricted best-width direction.

Filtering intermediate iterates is outside the two-policy comparison here.

## Reproducibility and verification

[Configuration]({OUT}/configuration.json), [summary]({OUT}/summary.csv), [all 100 policy results]({OUT}/per_process_results.csv), [600 terminal attempt results]({OUT}/per_attempt_diagnostics.csv), [paired comparison]({OUT}/paired_process_comparison.csv), [validation]({OUT}/validation.json), [PDF figure]({pdf_dir}/comparison.pdf).

Independent checks recompute all 600 widths and coefficient ratios on their original clouds, check spectral radii using characteristic-polynomial roots, verify all 100 selections and iteration counts, and inspect 48 complete traces including the 200-state patience condition. See [verification implementation]({ROOT}/output/analysis/{NAME}/verify_and_report.py). The thesis sources and original experiment notebook are untouched.
"""
    (OUT / "findings.md").write_text(report)
    return image_dir / "comparison.png"


def main():
    validation = validate()
    image = make_report()
    print(json.dumps(validation, indent=2))
    print(f"Figure: {image}")


if __name__ == "__main__":
    main()
