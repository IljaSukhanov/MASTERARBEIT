"""Compare previously discussed width-based selectors on population AR(3) curves."""

from __future__ import annotations

import hashlib
import itertools
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

from Code.Code_AR_processes.ar_tools.population_widths import population_ar3_widths
from Code.Code_AR_processes.ar3_order_selection_warm_starts import (
    VARIANTS,
)
from Code.Code_AR_processes.ar_tools.breakpoint_selectors import descent_chain, width_selectors

AR_DIR = Path(__file__).resolve().parent
NAME = "ar3_population_breakpoint_selectors"
DATA_DIR = AR_DIR / "outputs/csv" / NAME
IMAGE_DIR = AR_DIR / "outputs/images" / NAME
PDF_DIR = AR_DIR / "outputs/pdfs" / NAME
TAIL_TARGET = 1e-8


def _sse(values):
    # A repeated analytical plateau has zero variance exactly; avoid roundoff
    # from taking the mean of repeated identical floating-point values.
    if np.ptp(values) == 0:
        return 0.0
    residual = values - values.mean()
    return float(residual @ residual)


def _ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else (np.inf if numerator > 0 else 0.0)


def all_population_selectors(widths):
    m = np.asarray(widths, dtype=float)
    results = width_selectors(m)
    results["Descent chain"] = descent_chain(m)[0]
    drops = m[:-1] - m[1:]
    denominator = drops[:-1] + drops[1:]
    drift = np.full(len(m) - 2, -np.inf)
    np.divide(drops[:-1] - drops[1:], denominator, out=drift, where=denominator > 0)
    results["Local drift"] = int(np.argmax(drift) + 1)
    for method in ("Drop-to-tail energy", "Tail SSE ratio", "Tail variance ratio"):
        scores = []
        for q in range(1, len(m) - 1):
            if method == "Drop-to-tail energy":
                numerator, denominator = drops[q - 1] ** 2, float((drops[q:] ** 2).sum())
            else:
                previous, tail = m[q - 1:], m[q:]
                numerator, denominator = _sse(previous), _sse(tail)
                if method == "Tail variance ratio":
                    numerator /= len(previous)
                    denominator /= len(tail)
            scores.append(_ratio(numerator, denominator))
        results[method] = int(np.argmax(scores) + 1)
    results["Minimum width — first tied minimum"] = int(np.argmin(m[1:]) + 1)
    return results


def write_verification(widths, selections, stability):
    verification = {
        "population_curves": int(widths.replication.nunique()),
        "width_values": len(widths),
        "selectors": int(selections.method.nunique()),
        "selection_results": len(selections),
        "population_plateau_analytical_and_identical_from_k4": bool(
            widths[widths.k >= 4].groupby("replication").population_width.nunique().eq(1).all()),
        "maximum_omitted_tail_bound": float(widths.tail_error_bound.max()),
        "maximum_angular_sweep_roundoff_residual": float(widths.maximum_sweep_roundoff_residual.max()),
        "maximum_pole_formula_error": float(widths.pole_formula_max_error.max()),
        "stable_at_all_eight_truncation_corners": bool(stability.stable_at_all_eight_corners.all()),
        "floating_point_operations_interval_certified": False,
    }
    (DATA_DIR / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")


def make_plots(widths, summary, comparison):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    labels = summary.method
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(np.arange(len(summary)), summary.correct, color="#0072B2")
    ax.bar_label(bars, padding=3, fontsize=11)
    ax.set_yticks(np.arange(len(summary)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 54)
    ax.set_xticks(np.arange(0, 51, 10))
    ax.set_xlabel("Correct order selections out of 50 AR(3) population curves")
    ax.set_title("Breakpoint selectors supplied with population widths", pad=15)
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    fig.savefig(IMAGE_DIR / "population_selector_accuracy.png", dpi=180, bbox_inches="tight")
    fig.savefig(PDF_DIR / "population_selector_accuracy.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(14, 9))
    for row, rep in enumerate([1, 2, 5]):
        m = widths[widths.replication == rep].sort_values("k").population_width.to_numpy()
        axes[row, 0].plot(np.arange(1, 11), m, "o-", color="#0072B2", ms=4)
        axes[row, 0].set_yscale("log")
        axes[row, 0].set_ylabel(f"Replication {rep}\nWidth (log scale)")
        second = np.diff(m, n=2)
        peak = int(np.argmax(second) + 2)
        axes[row, 1].bar(np.arange(2, 10), second,
                         color=["#D55E00" if k == peak else "#0072B2" for k in range(2, 10)])
        axes[row, 1].set_ylabel("Change in slope")
        axes[row, 1].text(0.98, 0.94, f"Selected order: {peak - 1}",
                          transform=axes[row, 1].transAxes, ha="right", va="top")
        result = descent_chain(m)
        axes[row, 2].plot(np.arange(2, 10), result[4][1:-1], "o-", color="#0072B2", ms=4)
        axes[row, 2].set_ylabel("Stationary probability")
        axes[row, 2].text(0.98, 0.94, f"Selected order: {result[0]}",
                          transform=axes[row, 2].transAxes, ha="right", va="top")
        for ax in axes[row]:
            ax.axvline(4, color="#009E73", ls="--", lw=1.1)
            ax.grid(axis="y", alpha=0.15)
            ax.set_xticks([2, 4, 6, 8])
    for ax, title in zip(axes[0], ["Population widths", "Second differences", "Descent-chain interior weights"]):
        ax.set_title(title, pad=12)
    for ax in axes[-1]:
        ax.set_xlabel("Embedding dimension k")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "population_examples.png", dpi=180, bbox_inches="tight")
    fig.savefig(PDF_DIR / "population_examples.pdf", bbox_inches="tight")
    plt.close(fig)


def run_experiment():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    checkpoints = DATA_DIR / "replications"
    checkpoints.mkdir(exist_ok=True)
    source = AR_DIR / "outputs/csv/ar_order_minimum_width/width_curves.csv"
    parameters = pd.read_csv(source)
    parameters = parameters[(parameters.true_order == 3) & (parameters.k == 1)].sort_values("replication")
    assert len(parameters) == 50
    configuration = {
        "true_order": 3, "replications": 50, "max_k": 10, "tail_target": TAIL_TARGET,
        "pre_plateau_method": "Global finite-zonotope facet enumeration by angular sweeps",
        "plateau_method": "Analytical 2 / sqrt(1 + phi_1^2 + phi_2^2 + phi_3^2) for k >= 4",
        "width_estimate": "Midpoint of truncated minimum and truncated minimum plus analytical tail bound",
        "floating_point": "Not interval certified; angular-sweep and pole-formula residuals recorded",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "width_algorithm_sha256": hashlib.sha256((AR_DIR / "ar_tools/population_widths.py").read_bytes()).hexdigest(),
    }
    config_path = DATA_DIR / "configuration.json"
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise RuntimeError("Existing population experiment has different settings; preserve it separately.")
    config_path.write_text(json.dumps(configuration, indent=2) + "\n")

    records = []
    start = perf_counter()
    with threadpool_limits(limits=1):
        for parameter in parameters.itertuples(index=False):
            rep = int(parameter.replication)
            checkpoint = checkpoints / f"replication_{rep:03d}.csv"
            if checkpoint.exists():
                frame = pd.read_csv(checkpoint, float_precision="round_trip")
                assert len(frame) == 10
                records.extend(frame.to_dict("records"))
                continue
            result = population_ar3_widths(np.array(json.loads(parameter.coefficients)), tail_target=TAIL_TARGET)
            rows = []
            for i, finite in enumerate(result["widths"]):
                tail = result["tail_bounds"][i]
                row = {"replication": rep, "true_order": 3, "k": i + 1,
                       "population_width": finite + tail / 2, "truncated_width": finite,
                       "truncation_upper_bound": finite + tail, "tail_error_bound": tail,
                       "method": "global truncated minimum" if i < 3 else "analytical plateau",
                       "coefficients": parameter.coefficients,
                       "generator_count": result["generator_count"],
                       "spectral_radius": result["spectral_radius"],
                       "pole_formula_max_error": result["pole_formula_max_error"],
                       "maximum_sweep_roundoff_residual": result["maximum_sweep_roundoff_residual"]}
                direction = result["directions"][i] if i < 3 else np.r_[1., -np.array(json.loads(parameter.coefficients))]
                direction /= np.linalg.norm(direction)
                row.update({f"u_{j}": float(direction[j]) if j < len(direction) else np.nan for j in range(4)})
                rows.append(row)
            pd.DataFrame(rows).to_csv(checkpoint, index=False)
            records.extend(rows)
            print(f"Population curve {rep}/50: {result['generator_count']} generators ({perf_counter() - start:.1f}s)", flush=True)

        widths = pd.DataFrame(records).sort_values(["replication", "k"])
        widths.to_csv(DATA_DIR / "population_width_curves.csv", index=False)
        selections, stability, chain_rows, raw_selections = [], [], [], []
        for rep, group in widths.groupby("replication"):
            group = group.sort_values("k")
            m = group.population_width.to_numpy()
            chosen = all_population_selectors(m)
            raw = chosen.copy()
            for method, q in raw.items():
                raw_selections.append({"replication": rep, "method": method, "selected_order": q})
            for method, q in chosen.items():
                selections.append({"replication": rep, "true_order": 3, "method": method,
                                   "selected_order": q, "correct": q == 3})
            # All eight corners of the analytical truncation intervals.
            choices = {method: set() for method in chosen}
            for corner in itertools.product((0, 1), repeat=3):
                perturbed = m.copy()
                perturbed[:3] = group.truncated_width.to_numpy()[:3] + np.array(corner) * group.tail_error_bound.to_numpy()[:3]
                for method, q in all_population_selectors(perturbed).items():
                    choices[method].add(q)
            for method, orders in choices.items():
                stability.append({"replication": rep, "method": method,
                                  "corner_orders": json.dumps(sorted(orders)),
                                  "stable_at_all_eight_corners": len(orders) == 1 and chosen[method] in orders})
            chain = descent_chain(m)
            for i in range(10):
                chain_rows.append({"replication": rep, "k": i + 1, "drop": chain[1][i],
                                   "remaining_reduction": chain[2][i], "stationary_probability": chain[4][i],
                                   "odds": chain[5][i], "candidate": 0 < i < 9})
            if int(rep) % 10 == 0:
                print(f"Selectors and truncation checks {rep}/50", flush=True)

    selections = pd.DataFrame(selections)
    summary = selections.groupby("method", sort=False).selected_order.agg(
        correct=lambda q: int((q == 3).sum()), under=lambda q: int((q < 3).sum()),
        over=lambda q: int((q > 3).sum()), total="count").reset_index()
    selections.to_csv(DATA_DIR / "selections.csv", index=False)
    pd.DataFrame(raw_selections).to_csv(DATA_DIR / "raw_float_selections.csv", index=False)
    summary.to_csv(DATA_DIR / "summary.csv", index=False)
    pd.DataFrame(stability).to_csv(DATA_DIR / "truncation_corner_checks.csv", index=False)
    pd.DataFrame(chain_rows).to_csv(DATA_DIR / "descent_chain_scores.csv", index=False)
    write_verification(widths, selections, pd.DataFrame(stability))
    old_population = pd.read_csv(AR_DIR / "outputs/csv/ar_order_minimum_width/population_width_curves.csv")
    comparison = widths.merge(old_population[old_population.true_order == 3],
                              on=["true_order", "replication", "k"], suffixes=("", "_cached"))
    comparison["global_minus_cached"] = comparison.population_width - comparison.population_minimum_width
    comparison.to_csv(DATA_DIR / "cached_population_comparison.csv", index=False)
    empirical = pd.read_csv(AR_DIR / "outputs/csv/ar3_order_selection_warm_starts/summary.csv")
    empirical = empirical[empirical.variant == VARIANTS[2]].drop(columns="variant")
    paired = summary.merge(empirical, on="method", how="left", suffixes=("_population", "_warm_empirical"))
    paired.to_csv(DATA_DIR / "population_vs_warm_empirical.csv", index=False)
    (DATA_DIR / "coverage_notes.json").write_text(json.dumps({
        "tested": "Four derivative/spectral rules, descent chain, local drift, three cumulative-tail ratios, and first tied minimum width (10 total)",
        "birth_death_inverse_rule": "Undefined on population plateau: similarity matrix singular and reciprocal-drop rates infinite. No stationary-chain accuracy assigned.",
        "other_functionals": "BIC/PACF and residual/implied-innovation-width rules require series or a different functional; not evaluated by substituting geometric population widths into them.",
        "tail_corner_checks": "Sensitivity check, not a proof over the entire uncertainty interval",
        "descent_chain_source": "/Users/iljasukhanov/Downloads/elbow_point_selection_preview.pdf, Definitions 1.2.8-1.2.10 and Eq. (1.6), pp. 5-6",
    }, indent=2) + "\n")
    make_plots(widths, summary, paired)
    print("\nPopulation results:", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\nMaximum tail bound:", widths.tail_error_bound.max(), flush=True)
    print("Maximum angular-sweep roundoff residual:", widths.maximum_sweep_roundoff_residual.max(), flush=True)
    print("Maximum absolute change versus cached population widths:", comparison.global_minus_cached.abs().max(), flush=True)
    print("Unstable truncation corners:", int((~pd.DataFrame(stability).stable_at_all_eight_corners).sum()), flush=True)
    return DATA_DIR


if __name__ == "__main__":
    run_experiment()
