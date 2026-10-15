"""Separate bound calibration, fit quality, and finite-sample order ambiguity.

Truth is used only for diagnostic witnesses and oracle comparisons. This
does not change any estimator, saved gradient run, or thesis source.
"""
import json

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits

from check import INPUT, MAXIMUM_ORDER, OUT, canonical_direction


def minimax_radius(response, lags, q):
    if q == 0:
        return float(np.max(np.abs(response)))
    design = lags[:, :q]
    fit = linprog(
        np.r_[np.zeros(q), 1.0],
        A_ub=np.vstack([
            np.column_stack([-design, -np.ones(len(response))]),
            np.column_stack([design, -np.ones(len(response))]),
        ]),
        b_ub=np.r_[-response, response],
        bounds=[(None, None)] * q + [(0, None)], method="highs",
        options={"primal_feasibility_tolerance": 1e-9},
    )
    if not fit.success:
        raise RuntimeError(fit.message)
    radius = float(fit.x[-1])
    if np.max(np.abs(response - design @ fit.x[:-1])) > radius + 1e-8:
        raise ArithmeticError("Minimax radius residual verification failed")
    return radius


def main():
    results = pd.read_csv(OUT / "selections.csv")
    canonical = results[results.method == "canonical_maxabs"].set_index("replication")
    known = results[results.method == "known_radius_benchmark"].set_index("replication")
    rows = []
    with threadpool_limits(limits=1):
        for replication in range(1, 51):
            series = pd.read_csv(INPUT / "processes" / f"process_{replication:03d}.csv").X_t.to_numpy()
            fits = pd.read_csv(INPUT / "replications" / f"replication_{replication:03d}.csv")
            fit = fits[(fits.variant == "Padded warm start") & (fits.k == MAXIMUM_ORDER + 1)].iloc[0]
            u = fit[[f"u_{j}" for j in range(MAXIMUM_ORDER + 1)]].to_numpy(dtype=float)
            v = canonical_direction(u)
            cloud = np.column_stack([
                series[MAXIMUM_ORDER-j:len(series)-j]
                for j in range(MAXIMUM_ORDER + 1)
            ])
            response, lags = cloud[:, 0], cloud[:, 1:]
            phi = np.array(json.loads(fit.coefficients))
            truth = np.r_[1.0, -phi]
            truth /= np.linalg.norm(truth)
            candidates = [
                np.pad(truth, (shift, MAXIMUM_ORDER + 1 - len(truth) - shift))
                for shift in range(MAXIMUM_ORDER + 2 - len(truth))
            ]
            witness_width = min(float(np.ptp(cloud @ candidate)) for candidate in candidates)
            achieved_width = float(np.ptp(cloud @ u))
            true_residuals = response - lags[:, :3] @ phi
            oracle_radius = float(np.max(np.abs(true_residuals)))
            r2 = minimax_radius(response, lags, 2)
            r3 = minimax_radius(response, lags, 3)
            estimate = float(canonical.loc[replication, "estimated_radius"])
            selected = int(canonical.loc[replication, "selected_order"])
            coefficient_error = np.linalg.norm(-v[1:] / v[0] - np.pad(phi, (0, MAXIMUM_ORDER - 3)))
            rows.append(dict(
                replication=replication, selected_order=selected,
                known_radius_selected_order=int(known.loc[replication, "selected_order"]),
                estimated_radius=estimate, true_innovation_sample_maximum=oracle_radius,
                order_2_minimax_radius=r2, order_3_minimax_radius=r3,
                true_coefficient_bound_accepts_order_2=bool(r2 <= oracle_radius + 1e-8),
                achieved_width=achieved_width, truth_witness_width=witness_width,
                certified_width_excess=achieved_width - witness_width,
                relative_width_excess=(achieved_width - witness_width) / witness_width,
                canonical_coefficient_l2_error=float(coefficient_error),
                phi_1=phi[0], phi_2=phi[1], phi_3=phi[2],
            ))
    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(OUT / "failure_diagnostics.csv", index=False)
    columns = ["replication", "selected_order", "known_radius_selected_order", "estimated_radius",
               "order_2_minimax_radius", "order_3_minimax_radius", "relative_width_excess",
               "canonical_coefficient_l2_error", "phi_3"]
    print("Failed runs:", flush=True)
    print(diagnostics[diagnostics.selected_order != 3][columns].to_string(index=False), flush=True)
    new_under = diagnostics[(diagnostics.selected_order < 3) & (diagnostics.known_radius_selected_order == 3)]
    aggregate = dict(
        correct=int((diagnostics.selected_order == 3).sum()),
        under=int((diagnostics.selected_order < 3).sum()),
        over=int((diagnostics.selected_order > 3).sum()),
        extra_underselections_vs_known_radius=len(new_under),
        extra_underselections_with_certified_suboptimal_gradient_width=int((new_under.certified_width_excess > 1e-8).sum()),
        oracle_true_coefficient_bound_accepts_order_2=int(diagnostics.true_coefficient_bound_accepts_order_2.sum()),
        all_50_with_certified_suboptimal_gradient_width=int((diagnostics.certified_width_excess > 1e-8).sum()),
        failed_median_relative_width_excess=float(diagnostics[diagnostics.selected_order != 3].relative_width_excess.median()),
        correct_median_relative_width_excess=float(diagnostics[diagnostics.selected_order == 3].relative_width_excess.median()),
    )
    (OUT / "failure_summary.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    main()
