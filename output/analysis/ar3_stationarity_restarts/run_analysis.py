"""Audit stationarity and a delayed restart on the saved AR(3) failure.

This is an exploratory same-path experiment, not a replacement for the
shared optimizer. It uses only estimated coefficients for restart decisions.
"""

from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Code.Code_AR_processes.ar_tools.descent import projected_subgradient_empirical_width
from Code.Code_AR_processes.ar_tools.functions import (
    is_stationary, simulate_uniform_ar, sliding_window_point_cloud,
)
from Code.Code_AR_processes.ar_tools.lp_widths import exact_empirical_width_low_dimension

OUT = Path(__file__).resolve().parent
TRUE_PHI = np.array([0.5, -0.2, 0.1])
TOTAL_EVALUATIONS = 5000
CHECKPOINT = 100  # Exploratory budget choice, not a calibrated statistical cutoff.
GAMMA0, BETA = 0.1, 0.997
CAP_BOUND = 1 / np.sqrt(20.0)


def spectral_radii(coefficients):
    """Companion eigenvalue criterion, using rho < 1 with no extra margin."""
    phi = np.atleast_2d(np.asarray(coefficients, dtype=float))
    if phi.shape[1] != 3 or not np.all(np.isfinite(phi)):
        raise ValueError("This audit expects finite AR(3) coefficient rows.")
    matrices = np.zeros((len(phi), 3, 3))
    matrices[:, 0, :] = phi
    matrices[:, 1, 0] = 1
    matrices[:, 2, 1] = 1
    return np.max(np.abs(np.linalg.eigvals(matrices)), axis=1)


def run(points, initial_direction, evaluations):
    return projected_subgradient_empirical_width(
        points, iterations=evaluations, initial_direction=initial_direction,
        initial_step=GAMMA0, geometric_decay=BETA, step_schedule="geometric",
        patience=evaluations + 1, improvement_tolerance=0.0,
        record_directions=True, record_best_coefficients=True,
    )


def diagnostics(result):
    directions = result["direction_history"]
    phi = -directions[:, 1:] / directions[:, :1]
    best_phi = result["best_coefficient_history"]
    rho = spectral_radii(phi)
    best_rho = spectral_radii(best_phi)
    # Independent scalar helper agrees with the batched implementation.
    for index in np.unique([0, len(phi) // 2, len(phi) - 1]):
        valid, eigenvalues = is_stationary(phi[index])
        assert valid == bool(rho[index] < 1)
        np.testing.assert_allclose(rho[index], np.max(abs(eigenvalues)), rtol=1e-12)
    return pd.DataFrame({
        "iteration": np.arange(len(phi)),
        "current_width": result["width_history"],
        "best_width": result["best_width_history"],
        "current_rho": rho, "best_rho": best_rho,
        "current_stationary": rho < 1, "best_stationary": best_rho < 1,
        "best_abs_u0": 1 / np.sqrt(1 + np.sum(best_phi**2, axis=1)),
        "best_coefficient_error": np.linalg.norm(best_phi - TRUE_PHI, axis=1),
    })


def summary_row(method, result, global_width, *, evaluations, restarts=0):
    phi = result["coefficients"]
    rho = float(spectral_radii(phi)[0])
    return dict(
        method=method, evaluated_states=evaluations, restarts=restarts,
        minimum_width=result["minimum_width"],
        relative_width_gap_percent=100 * (result["minimum_width"] / global_width - 1),
        phi_1=phi[0], phi_2=phi[1], phi_3=phi[2],
        coefficient_l2_error=float(np.linalg.norm(phi - TRUE_PHI)),
        spectral_radius=rho, stationary=rho < 1,
        abs_u0=float(abs(result["direction"][0])),
    )


def main():
    series = simulate_uniform_ar(TRUE_PHI, 1000, burnin=1000, rng=20261018)
    points = sliding_window_point_cloud(series, 4)
    bad_u = np.random.default_rng(20261019).normal(size=4)
    bad_u /= np.linalg.norm(bad_u)
    anchor_u = np.array([1.0, 0.0, 0.0, 0.0])

    with threadpool_limits(limits=1):
        global_width, global_u, _ = exact_empirical_width_low_dimension(points)
        failed = run(points, bad_u, TOTAL_EVALUATIONS)
        prefix = run(points, bad_u, CHECKPOINT)
        # Reproducing the prefix verifies that the checkpoint uses no future states.
        np.testing.assert_array_equal(prefix["direction_history"], failed["direction_history"][:CHECKPOINT])
        prefix_rho = float(spectral_radii(prefix["coefficients"])[0])
        restart_triggered = not np.isfinite(prefix_rho) or prefix_rho >= 1
        assert restart_triggered
        restarted = run(points, anchor_u, TOTAL_EVALUATIONS - CHECKPOINT)
        anchor = run(points, anchor_u, TOTAL_EVALUATIONS)

        failed_d = diagnostics(failed)
        restarted_d = diagnostics(restarted)
        anchor_d = diagnostics(anchor)

        # Restrict accepted output to causal stationary estimates: choose the
        # smallest ORIGINAL width among valid visited states across both stages.
        # Preserve the unconstrained winner separately so this change is explicit.
        all_directions = np.vstack([prefix["direction_history"], restarted["direction_history"]])
        all_widths = np.r_[prefix["width_history"], restarted["width_history"]]
        all_phi = -all_directions[:, 1:] / all_directions[:, :1]
        all_rho = spectral_radii(all_phi)
        valid_indices = np.flatnonzero(all_rho < 1)
        assert len(valid_indices)
        chosen = int(valid_indices[np.argmin(all_widths[valid_indices])])
        unrestricted_index = int(np.argmin(all_widths))
        # In THIS example the stationary filter does not change the winner.
        assert chosen == unrestricted_index
        policy_result = dict(coefficients=all_phi[chosen], direction=all_directions[chosen],
                             minimum_width=float(all_widths[chosen]))
        np.testing.assert_allclose(np.ptp(points @ policy_result["direction"]),
                                   policy_result["minimum_width"], atol=1e-12, rtol=0)

        audit_rows = []
        recovering = None
        recovering_d = None
        for seed in range(12):
            direction = np.random.default_rng(seed).normal(size=4)
            direction /= np.linalg.norm(direction)
            result = run(points, direction, TOTAL_EVALUATIONS)
            diag = diagnostics(result)
            first_stable = np.flatnonzero(diag.best_stationary.to_numpy())
            audit_rows.append(dict(
                initialization_seed=seed,
                initial_rho=float(diag.best_rho.iloc[0]),
                initial_stationary=bool(diag.best_stationary.iloc[0]),
                first_stationary_best_iteration=int(first_stable[0]) if len(first_stable) else None,
                checkpoint_best_rho=float(diag.best_rho.iloc[CHECKPOINT - 1]),
                final_rho=float(diag.best_rho.iloc[-1]),
                final_stationary=bool(diag.best_stationary.iloc[-1]),
                final_width=result["minimum_width"],
                final_coefficient_l2_error=float(diag.best_coefficient_error.iloc[-1]),
            ))
            if seed == 3:
                recovering, recovering_d = result, diag
        assert recovering is not None
        assert not recovering_d.best_stationary.iloc[0]
        assert recovering_d.best_stationary.iloc[1]
        assert recovering_d.best_stationary.iloc[-1]

    # The necessary AR(3) cap must hold for every observed stationary state.
    good_u0 = np.abs(anchor["direction_history"][:, 0])
    assert np.all(good_u0[anchor_d.current_stationary] > CAP_BOUND)

    rows = [
        summary_row("Original failed start", failed, global_width, evaluations=TOTAL_EVALUATIONS),
        summary_row("Stationarity restart after 100 states", policy_result, global_width,
                    evaluations=TOTAL_EVALUATIONS, restarts=1),
        summary_row("Zero-coefficient start from the beginning", anchor, global_width,
                    evaluations=TOTAL_EVALUATIONS),
    ]
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "restart_results.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(OUT / "same_path_initialization_audit.csv", index=False)
    failed_d.assign(run="Failed start").to_csv(OUT / "failed_stationarity_history.csv", index=False)
    recovering_d.to_csv(OUT / "recovering_initialization_history.csv", index=False)
    policy_d = pd.concat([
        failed_d.iloc[:CHECKPOINT].assign(phase=0, local_iteration=lambda x: x.iteration),
        restarted_d.assign(phase=1, local_iteration=lambda x: x.iteration,
                            iteration=lambda x: x.iteration + CHECKPOINT),
    ], ignore_index=True)
    policy_d["best_width_across_stages"] = np.minimum.accumulate(policy_d.current_width)
    policy_d.to_csv(OUT / "restart_history.csv", index=False)

    # Concrete stable-but-large coefficient vector, from (lambda - 0.9)^3.
    large_stable_phi = np.array([2.7, -2.43, 0.729])
    large_stable_rho = float(spectral_radii(large_stable_phi)[0])
    assert large_stable_rho < 1

    with plt.rc_context({"font.size": 10, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
        # Zero spectral radii are omitted only from these logarithmic plots.
        for label, frame, color in [
            ("Failed start", failed_d, "#b34638"),
            ("Zero-start control", anchor_d, "#2a8c7b"),
        ]:
            axes[0].semilogy(frame.iteration, np.where(frame.best_rho > 0, frame.best_rho, np.nan),
                             color=color, lw=1.5, label=label)
        axes[0].axhline(1, color="#222222", ls=":", label="Causal stability boundary")
        axes[0].set(title="Stationarity detects this failure", xlabel="Evaluated iteration",
                    ylabel="Spectral radius of best estimate")
        axes[0].legend(frameon=False, fontsize=8)

        early = recovering_d.iloc[:11]
        axes[1].plot(early.iteration, early.best_rho, "o-", color="#315a9a", lw=1.5,
                     label="Random initialization seed 3")
        axes[1].axhline(1, color="#222222", ls=":")
        axes[1].set(title="An unstable initial estimate can recover", xlabel="Evaluated iteration",
                    ylabel="Spectral radius of best estimate", xticks=np.arange(0, 11, 2))
        axes[1].legend(frameon=False, fontsize=8)

        axes[2].semilogy(failed_d.iteration,
                         np.maximum(failed_d.best_width - global_width, 1e-15),
                         color="#b34638", label="No restart")
        axes[2].semilogy(policy_d.iteration,
                         np.maximum(policy_d.best_width_across_stages - global_width, 1e-15),
                         color="#315a9a", label="Restart from zero coefficients at state 100")
        axes[2].axvline(CHECKPOINT, color="#777777", ls=":")
        axes[2].set(title="Same total budget: 5000 evaluated states", xlabel="Cumulative evaluated states",
                    ylabel="Best width minus global numerical reference")
        axes[2].legend(frameon=False, fontsize=8)
        for ax in axes:
            ax.grid(alpha=0.2)
        fig.suptitle("AR(3) stationarity diagnostic and an exploratory restart | same fixed sample path")
        fig.savefig(OUT / "stationarity_restart_diagnostic.pdf", bbox_inches="tight")
        fig.savefig(OUT / "stationarity_restart_diagnostic.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    configuration = dict(
        true_phi=TRUE_PHI.tolist(), process_seed=20261018, optimizer_seed=20261019,
        T=1000, burnin=1000, gamma0=GAMMA0, beta=BETA,
        evaluated_states=TOTAL_EVALUATIONS, restart_checkpoint=CHECKPOINT,
        maximum_restarts=1, restart_direction=anchor_u.tolist(),
        reset_geometric_schedule_on_restart=True,
        restart_decision_uses_true_coefficients=False,
        stationarity_criterion="companion spectral radius < 1",
        AR3_necessary_abs_u0_lower_bound=CAP_BOUND,
        same_path_audit_initialization_seeds=list(range(12)),
        global_numerical_width=global_width,
        checkpoint_failed_best_rho=prefix_rho,
        results=results.to_dict(orient="records"),
        all_failed_current_states_nonstationary=bool(np.all(~failed_d.current_stationary)),
        all_control_current_states_stationary=bool(np.all(anchor_d.current_stationary)),
        stable_but_large_example=large_stable_phi.tolist(),
        stable_but_large_example_rho=large_stable_rho,
        validation="prefix equality; scalar/batched roots; width reproduction; AR3 cap; recovering initialization",
    )
    (OUT / "configuration_and_results.json").write_text(json.dumps(configuration, indent=2) + "\n")

    report = [
        "# Stationarity as a restart diagnostic for the failed AR(3) run",
        "This is our same-path exploratory experiment. It does not replace the shared optimizer or establish a generally calibrated restart rule.",
        "## What failed, and does changing the start help?",
        f"The original failed estimate has spectral radius {results.spectral_radius.iloc[0]:.8f}. "
        "All 5000 visited current estimates in that run violate the causal stationarity condition. "
        f"The zero-start control ends at radius {results.spectral_radius.iloc[2]:.8f}, and all its visited estimates are stationary. "
        "Sources: [failed trajectory](failed_stationarity_history.csv), [comparison](restart_results.csv). "
        "With data and step settings fixed, this is direct evidence of initialization sensitivity. "
        "It does not prove initialization is the only cause or certify a local minimum.",
        "## Model-based meaning of a bad estimate",
        "For a causal stationary AR(p), the zeros of 1-phi_1 z-...-phi_p z^p lie outside the unit circle. "
        "Equivalently the companion eigenvalues have modulus below one. "
        "Primary documentation: [statsmodels ArmaProcess.isstationary](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_process.ArmaProcess.isstationary.html); "
        "[companion and reciprocal-root criterion, Notes](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.tools.is_invertible.html#notes). "
        "Workspace implementation: [functions.py](../../../Code/Code_AR_processes/ar_tools/functions.py), is_stationary. "
        "This check uses estimated coefficients and the model assumption; it does not require knowing the true vector. "
        "Passing it does not establish coefficient accuracy or global optimality.",
        "## A necessary AR(3) coefficient and direction bound: our derivation",
        "Write the AR(3) characteristic polynomial as "
        "lambda^3-phi_1 lambda^2-phi_2 lambda-phi_3 = product_i(lambda-lambda_i), with |lambda_i|<1. "
        "Matching coefficients and applying the triangle inequality gives |phi_1|<3, |phi_2|<3 and |phi_3|<1. "
        "Consequently ||phi||_2^2<19, and for a unit direction that recovers phi, "
        "|u_0|=1/sqrt(1+||phi||_2^2)>1/sqrt(20)=0.2236067977. "
        "This is a necessary condition, not a sufficient stationarity test. "
        "It provides a model-based rejection threshold without choosing an arbitrary coefficient-size cutoff. "
        "The full root test is stronger. The derivation is ours from the root factorization documented in the Notes cited above.",
        f"Individual coefficients larger than one can be stationary: (2.7,-2.43,0.729) has characteristic polynomial "
        f"(lambda-0.9)^3 and numerically measured radius {large_stable_rho:.8f}. "
        "Thus a rule rejecting every coefficient outside [-1,1] would be incorrect. "
        "This example follows by expansion of that polynomial; numerical check: [script](run_analysis.py).",
        "## Tested restart, same total evaluation budget",
        "At the checkpoint after 100 evaluated states, inspect the best recovered estimate. "
        "If undefined/nonfinite or nonstationary, restart from e_0=(1,0,0,0), corresponding to phi=0, "
        "and reset the geometric step schedule. Spend the remaining 4900 states on that run. "
        "Keep the smallest original width among visited stationary estimates, retaining the unconstrained winner separately. "
        "This explicitly restricts acceptable outputs to stationary models; in this example the stationary and unrestricted winners coincide. "
        "The true coefficients enter diagnostics only. The checkpoint 100 is an exploratory budget choice, not a theoretically justified threshold.",
        "| Method | Evaluated states | Width | Spectral radius | Coefficient L2 error |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in results.itertuples():
        report.append(f"| {row.method} | {row.evaluated_states} | {row.minimum_width:.10f} | "
                      f"{row.spectral_radius:.8f} | {row.coefficient_l2_error:.8f} |")
    report.extend([
        "\nSource: [restart results](restart_results.csv), [restart trajectory](restart_history.csv), "
        "[settings and checks](configuration_and_results.json). "
        "The restart solves this observed failure to approximately the same empirical optimum as the zero-start control, within the same evaluation budget. "
        "It is not a guarantee for other data or starts.",
        "## Why not reject at the first unstable estimate?",
        f"On this same path, random initialization seed 3 begins with radius {recovering_d.best_rho.iloc[0]:.8f}, "
        f"becomes stationary at evaluated state 1, and finishes with error {recovering_d.best_coefficient_error.iloc[-1]:.8f}. "
        "An immediate rejection would discard that successful trajectory. "
        "Sources: [recovering initialization history](recovering_initialization_history.csv), "
        "[12 fixed-seed initialization audit](same_path_initialization_audit.csv). "
        "These 12 starts share a single sample path and do not establish a population failure rate.",
        "## Suggested first implementation",
        "Always include the known stationary phi=0 start. Run additional starts if budget allows. "
        "For each run retain a best stationary candidate separately from the unrestricted best. "
        "At a run endpoint, or a chosen checkpoint after persistent lack of valid progress, restart if the best estimate is nonstationary or undefined. "
        "Reset the local geometric schedule and cap the restart count or total budget. "
        "Do not interpret stationarity as an accuracy certificate, and do not use a positive stability margin without stating the additional assumption it imposes near the unit-root boundary. "
        "This is our proposed optimization heuristic based on the model check and the experiment, not a published theorem.",
        "A stronger alternative is to parameterize only stationary coefficients and optimize the original normalized width there. "
        "A documented stationary parameter transform is "
        "[statsmodels constrain_stationary_univariate](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.tools.constrain_stationary_univariate.html), "
        "which cites Monahan (1984), Biometrika 71(2), pp. 403-404. "
        "Adapting that parameterization to this width objective would change the update geometry and has not been tested in this audit.",
    ])
    # Keep the comparison table together and separate all surrounding paragraphs.
    table_start = next(i for i, line in enumerate(report) if line.startswith("| Method"))
    table_end = table_start
    while table_end < len(report) and report[table_end].startswith("|"):
        table_end += 1
    report = report[:table_start] + ["\n".join(report[table_start:table_end])] + report[table_end:]
    text = "\n\n".join(report) + "\n"
    (OUT / "findings.md").write_text(text)
    print(results.to_string(index=False))
    print(f"Report: {OUT / 'findings.md'}")


if __name__ == "__main__":
    main()
