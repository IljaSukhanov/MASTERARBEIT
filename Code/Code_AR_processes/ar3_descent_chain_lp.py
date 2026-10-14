"""Test preview Algorithm 2 on the same saved AR(3) series.

Uses common response times t=10,...,2000, three prescribed starts, and LP
tangent steps. Audit dimensions 1–4 independently by global convex-hull
calculations. Raw and log-width descent chains are reported at caps 5,8,10;
cap 5 is only a diagnostic assuming an order upper bound of three.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Code.Code_AR_processes.ar_tools.functions import sliding_window_point_cloud
from Code.Code_AR_processes.ar_tools.lp_widths import lp_width_curve, exact_empirical_width_low_dimension
from Code.Code_AR_processes.ar_tools.breakpoint_selectors import descent_chain

AR_DIR = Path(__file__).resolve().parent
SOURCE = AR_DIR / "outputs/csv/ar3_order_selection_warm_starts"
DATA_DIR = AR_DIR / "outputs/csv/ar3_descent_chain_lp"
IMAGE_DIR = AR_DIR / "outputs/images/ar3_descent_chain_lp"
PDF_DIR = AR_DIR / "outputs/pdfs/ar3_descent_chain_lp"


def earlier_bend_certificate(prefix_widths, terminal_upper):
    """Original deterministic diagnostic, not a changed selection rule.

    Prefix entries must be global minima through the true break dimension.
    terminal_upper is any feasible width in the last dimension. If an earlier
    candidate wins here, it also wins for every smaller terminal minimum:
    A/(a-b)>B/(c-b) iff A(c-b)-B(a-b)>0. Such a win implies A>B,
    so decreasing b can only increase this positive margin.
    """
    m = np.asarray(prefix_widths, dtype=float)
    b = float(terminal_upper)
    if len(m) < 3 or not np.all(np.diff(m) <= 0) or b >= m[-1]:
        raise ValueError("Need a monotone prefix and terminal upper bound below it.")
    drops = -np.diff(m)
    earlier_scores = drops[:-1] / (m[1:-1] - b)
    index = int(np.argmax(earlier_scores))
    a, A, c, B = m[index + 1], drops[index], m[-1], drops[-1]
    margin = A * (c - b) - B * (a - b)
    return {"earlier_order": index + 1,
            "earlier_score_at_terminal_upper": float(earlier_scores[index]),
            "true_score_at_terminal_upper": float(B / (c - b)),
            "cross_product_margin": float(margin),
            "earlier_drop": float(A), "true_drop": float(B),
            "earlier_bend_wins_for_all_smaller_terminal_widths": bool(margin > 1e-10)}


def save_audit_diagnostics(widths):
    certificates, comparisons = [], []
    for rep, group in widths.groupby("replication"):
        group = group.sort_values("k")
        values = group.minimum_width.to_numpy().copy()
        values[:4] = group.global_audit_width.to_numpy()[:4]
        assert np.all(np.diff(values) <= 1e-10)
        for transform in ["identity", "log"]:
            z = values if transform == "identity" else np.log(values)
            comparisons.append({"replication": rep, "transform": transform,
                                "selected_order_after_global_prefix_replacement": descent_chain(z)[0]})
            certificates.append({"replication": rep, "transform": transform,
                                 "terminal_upper_width": float(z[-1]),
                                 **earlier_bend_certificate(z[:4], z[-1])})
    pd.DataFrame(certificates).to_csv(DATA_DIR / "earlier_bend_certificates.csv", index=False)
    pd.DataFrame(comparisons).to_csv(DATA_DIR / "global_prefix_selections.csv", index=False)


def make_plots(widths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    population = pd.read_csv(AR_DIR / "outputs/csv/ar3_population_breakpoint_selectors/population_width_curves.csv")
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for column, rep in enumerate([1, 2, 5]):
        group = widths[widths.replication == rep].sort_values("k")
        m = group.minimum_width.to_numpy()
        pop = population[population.replication == rep].sort_values("k")
        k = np.arange(1, 11)
        axes[0, column].plot(k, m, "o-", color="#0072B2", label="Empirical LP fit")
        axes[0, column].plot(k, pop.population_width, "--", color="#777777", label="Population width")
        axes[0, column].scatter(k[:4], group.global_audit_width.iloc[:4], s=55,
                                facecolors="none", edgecolors="#D55E00", label="Global hull audit")
        axes[0, column].axvline(4, color="#009E73", alpha=.5)
        axes[0, column].set_title(f"Replication {rep}")
        axes[0, column].set_ylabel("Minimum width")
        axes[0, column].set_xticks([1, 2, 4, 6, 8, 10])
        for label, z, color in [("Raw widths", m, "#0072B2"), ("Log widths", np.log(m), "#D55E00")]:
            chain = descent_chain(z)
            axes[1, column].plot(k[1:-1], chain[3][1:-1], "o-", color=color,
                                 label=f"{label}: selected order {chain[0]}")
        axes[1, column].axvline(4, color="#009E73", alpha=.5, label="True break dimension 4")
        axes[1, column].set_ylim(0, 1.04)
        axes[1, column].set_xlabel("Embedding dimension k")
        axes[1, column].set_ylabel("Chain visit probability αₖ")
        axes[1, column].legend(loc="lower right", fontsize=9, frameon=False)
    axes[0, 0].legend(fontsize=9, frameon=False)
    fig.suptitle("Original descent chain: corrected optimization helps, finite-sample failures remain", fontsize=14)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "lp_descent_chain_examples.png", dpi=180)
    fig.savefig(PDF_DIR / "lp_descent_chain_examples.pdf")
    plt.close(fig)


def run_experiment():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "replications").mkdir(exist_ok=True)
    configuration = {
        "replications": 50, "true_order": 3, "sample_size": 2000,
        "max_fitted_k": 10, "common_times": "t=10,...,2000 for all fitted dimensions",
        "lp_tolerance": 1e-9, "max_lp_iterations_per_start": 50,
        "starts": ["previous direction padded with zero", "previous direction shifted by zero", "first coordinate axis"],
        "source_process_folder": str(SOURCE / "processes"),
        "algorithm_sha256": hashlib.sha256((AR_DIR / "ar_tools/lp_widths.py").read_bytes()).hexdigest(),
        "caps": [5, 8, 10], "cap_5_note": "Assumes an order upper bound of 3; not an automatic data-driven cap choice",
        "transformations": ["identity", "log"],
        "global_audit": "Convex hull of P-P for dimensions 1–4, floating-point rather than interval-certified",
    }
    config_path = DATA_DIR / "configuration.json"
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise RuntimeError("Preserve the existing differently configured LP experiment separately.")
    config_path.write_text(json.dumps(configuration, indent=2) + "\n")
    start = perf_counter()
    rows = []
    with threadpool_limits(limits=1):
        for rep in range(1, 51):
            checkpoint = DATA_DIR / "replications" / f"replication_{rep:03d}.csv"
            if checkpoint.exists():
                cached = pd.read_csv(checkpoint, float_precision="round_trip")
                assert len(cached) == 10
                rows.extend(cached.to_dict("records"))
                continue
            series = pd.read_csv(SOURCE / "processes" / f"process_{rep:03d}.csv",
                                 float_precision="round_trip").X_t.to_numpy()
            source_fit = pd.read_csv(SOURCE / "replications" / f"replication_{rep:03d}.csv")
            phi = np.array(json.loads(source_fit.iloc[0].coefficients))
            widths, directions, diagnostics = lp_width_curve(series)
            cloud = sliding_window_point_cloud(series, 10)
            result_rows = []
            for i, (width, direction, diagnostic) in enumerate(zip(widths, directions, diagnostics)):
                k = i + 1
                exact, vertices = np.nan, np.nan
                if k <= 4:
                    exact, _, vertices = exact_empirical_width_low_dimension(cloud[:, :k])
                oracle_width = np.nan
                if k >= 4:
                    q = np.r_[1.0, -phi, np.zeros(k - 4)]
                    q /= np.linalg.norm(q)
                    oracle_width = float(np.ptp(cloud[:, :k] @ q))
                row = {"replication": rep, "k": k, "minimum_width": width,
                       "coefficients": json.dumps(phi.tolist()), "global_audit_width": exact,
                       "lp_minus_global_audit": width - exact,
                       "hull_vertices": vertices, "known_direction_width": oracle_width,
                       **diagnostic}
                row.update({f"u_{j}": float(direction[j]) if j < k else np.nan for j in range(10)})
                result_rows.append(row)
            pd.DataFrame(result_rows).to_csv(checkpoint, index=False)
            rows.extend(result_rows)
            print(f"LP fit and global low-dimension audit {rep}/50 ({perf_counter() - start:.1f}s)", flush=True)

    widths = pd.DataFrame(rows).sort_values(["replication", "k"])
    widths.to_csv(DATA_DIR / "width_curves.csv", index=False)
    selections, score_rows = [], []
    for rep, group in widths.groupby("replication"):
        m = group.sort_values("k").minimum_width.to_numpy()
        for cap in configuration["caps"]:
            for transform in configuration["transformations"]:
                values = m[:cap] if transform == "identity" else np.log(m[:cap])
                chain = descent_chain(values)
                selections.append({"replication": rep, "cap": cap, "transform": transform,
                                   "selected_order": chain[0], "true_order": 3})
                for i in range(cap):
                    score_rows.append({"replication": rep, "cap": cap, "transform": transform,
                                       "k": i + 1, "drop": chain[1][i], "remaining_reduction": chain[2][i],
                                       "alpha": chain[3][i], "stationary_probability": chain[4][i],
                                       "odds": chain[5][i], "candidate": 0 < i < cap - 1})
    selections = pd.DataFrame(selections)
    summary = selections.groupby(["cap", "transform"]).selected_order.agg(
        correct=lambda q: int((q == 3).sum()), under=lambda q: int((q < 3).sum()),
        over=lambda q: int((q > 3).sum()), total="count").reset_index()
    selections.to_csv(DATA_DIR / "selections.csv", index=False)
    summary.to_csv(DATA_DIR / "summary.csv", index=False)
    pd.DataFrame(score_rows).to_csv(DATA_DIR / "descent_chain_scores.csv", index=False)
    save_audit_diagnostics(widths)
    physical_errors, norm_errors = [], []
    for rep, group in widths.groupby("replication"):
        series = pd.read_csv(SOURCE / "processes" / f"process_{rep:03d}.csv",
                             float_precision="round_trip").X_t.to_numpy()
        cloud = sliding_window_point_cloud(series, 10)
        for row in group.itertuples():
            direction = np.array([getattr(row, f"u_{j}") for j in range(row.k)])
            physical_errors.append(abs(np.ptp(cloud[:, :row.k] @ direction) - row.minimum_width))
            norm_errors.append(abs(np.linalg.norm(direction) - 1))
    assert max(physical_errors) < 1e-10
    assert max(norm_errors) < 1e-12
    verification = {
        "curves": 50, "monotonicity_violations_above_1e-12": int(sum(
            np.any(np.diff(g.sort_values("k").minimum_width) > 1e-12) for _, g in widths.groupby("replication"))),
        "maximum_lp_gap_global_k2_to_k4": float(widths[widths.k.between(2, 4)].lp_minus_global_audit.max()),
        "k4_global_audit_matches_within_1e-8": int((widths[widths.k == 4].lp_minus_global_audit.abs() < 1e-8).sum()),
        "k4_widths_above_known_feasible_direction_by_1e-8": int(((widths[widths.k == 4].minimum_width
                                                               - widths[widths.k == 4].known_direction_width) > 1e-8).sum()),
        "total_lp_solves": int(widths.lp_solves.sum()),
        "starts_hitting_iteration_limit": int(widths.iteration_limit_starts.sum()),
        "maximum_saved_direction_width_reconstruction_error": float(max(physical_errors)),
        "maximum_saved_direction_unit_norm_error": float(max(norm_errors)),
    }
    (DATA_DIR / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print("\nLP descent-chain results:", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(verification, indent=2), flush=True)
    return DATA_DIR


if __name__ == "__main__":
    run_experiment()
    make_plots(pd.read_csv(DATA_DIR / "width_curves.csv", float_precision="round_trip"))
