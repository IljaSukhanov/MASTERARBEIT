"""Paired AR(3) rerun: independent starts versus padded previous directions.

Original cached results are read only. Every regenerated series, fitted direction,
width curve, selector result, and displayed figure is saved separately.
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Code.Code_AR_processes.ar_tools.descent import batched_geometric_width_curves
from Code.Code_AR_processes.ar_tools.functions import (
    jones_ar_coefficients,
    simulate_uniform_ar,
    sliding_window_point_cloud,
)
from Code.Code_AR_processes.ar_tools.breakpoint_selectors import descent_chain, width_selectors

NAME = "ar3_order_selection_warm_starts"
AR_DIR = Path(__file__).resolve().parent
DATA_DIR = AR_DIR / "outputs/csv" / NAME
IMAGE_DIR = AR_DIR / "outputs/images" / NAME
PDF_DIR = AR_DIR / "outputs/pdfs" / NAME
ORIGINAL_DIR = AR_DIR / "outputs/csv/ar_order_minimum_width"
VARIANTS = ("Original saved", "Independent rerun", "Padded warm start")


def _fit_rows(series, phi, settings, replication, *, warm):
    start = perf_counter()
    _, widths, _, directions = batched_geometric_width_curves(
        series, max(settings["k_values"]), seed=settings["seeds"]["3"],
        replication=replication - 1, starts=settings["starts"],
        iterations=settings["iterations"], initial_step=settings["initial_step"],
        geometric_decay=settings["geometric_decay"], return_directions=True,
        warm_start_previous_direction=warm,
    )
    elapsed = perf_counter() - start
    variant = VARIANTS[2] if warm else VARIANTS[1]
    rows = []
    for k, (width, direction) in enumerate(zip(widths, directions), start=1):
        cloud = sliding_window_point_cloud(series, k)
        actual_width = float(np.ptp(cloud @ direction))
        assert abs(actual_width - width) < 1e-10
        assert abs(np.linalg.norm(direction) - 1) < 1e-12
        candidate_width = np.nan
        if warm and k >= 2:
            padded = np.r_[directions[k - 2], 0.0]
            candidate_width = float(np.ptp(cloud @ padded))
            assert width <= candidate_width + 1e-10
            assert candidate_width <= widths[k - 2] + 1e-10
        row = {
            "variant": variant, "true_order": 3, "replication": replication,
            "k": k, "minimum_width": float(width),
            "coefficients": json.dumps(phi.tolist()),
            "padded_candidate_width": candidate_width,
            "curve_runtime_seconds": elapsed,
        }
        row.update({f"u_{j}": float(direction[j]) if j < k else np.nan
                    for j in range(max(settings["k_values"]))})
        rows.append(row)
    if warm:
        tolerance = 32 * np.finfo(float).eps * max(1.0, float(np.abs(widths).max()))
        assert np.all(np.diff(widths) <= tolerance), (replication, np.diff(widths))
    return rows


def _save_figure(fig, name):
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMAGE_DIR / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(PDF_DIR / f"{name}.pdf", bbox_inches="tight")


def make_plots(width_curves, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 12, "axes.spines.top": False,
                         "axes.spines.right": False})
    labels = {
        "First difference": "First difference",
        "Second difference": "Second difference",
        "Perron first difference": "Perron first difference",
        "Perron second difference": "Perron slope change",
        "Descent chain": "Descent chain",
    }
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    locations = np.arange(len(labels))
    for offset, variant, color in [(-0.17, VARIANTS[1], "#888888"),
                                   (0.17, VARIANTS[2], "#0072B2")]:
        block = summary[summary.variant == variant].set_index("method")
        values = [float(block.loc[method, "correct"]) if method in block.index else np.nan
                  for method in labels]
        bars = ax.barh(locations + offset, values, height=0.30, color=color, label=variant)
        ax.bar_label(bars, labels=[str(int(v)) if np.isfinite(v) else "" for v in values],
                     padding=3, fontsize=11)
    ax.set_yticks(locations, labels.values())
    ax.invert_yaxis()
    ax.set_xlim(0, 53)
    ax.set_xticks(np.arange(0, 51, 10))
    ax.set_xlabel("Correct AR(3) selections out of 50")
    ax.set_title("Same 50 series and search budget: independent versus padded starts", pad=14)
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False)
    _save_figure(fig, "selector_accuracy_comparison")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, replication in zip(axes, [1, 2, 5]):
        for variant, color, linestyle in [(VARIANTS[0], "#999999", ":"),
                                         (VARIANTS[1], "#D55E00", "--"),
                                         (VARIANTS[2], "#0072B2", "-")]:
            rows = width_curves[(width_curves.variant == variant)
                                & (width_curves.replication == replication)].sort_values("k")
            ax.plot(rows.k, rows.minimum_width, linestyle=linestyle, marker="o",
                    ms=3, lw=1.6, color=color, label=variant)
        ax.axvline(4, color="#009E73", ls="--", lw=1, label="True-order reference k=4")
        ax.set(title=f"Replication {replication}", xlabel="Embedding dimension k")
        ax.set_xticks([1, 2, 4, 6, 8, 10])
        ax.grid(axis="y", alpha=0.15)
    axes[0].set_ylabel("Width")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.015), fontsize=11)
    fig.subplots_adjust(bottom=0.22, wspace=0.28)
    _save_figure(fig, "example_width_curves")
    plt.close(fig)


def run_experiment():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "processes").mkdir(exist_ok=True)
    (DATA_DIR / "replications").mkdir(exist_ok=True)
    settings = json.loads((ORIGINAL_DIR / "configuration.json").read_text())
    original = pd.read_csv(ORIGINAL_DIR / "width_curves.csv")
    original = original[original.true_order == 3].copy()
    assert len(original) == 500 and original.replication.nunique() == 50
    configuration = {
        "true_order": 3, "replications": settings["replications_per_order"],
        "sample_size": settings["sample_size"], "burnin": settings["burnin"],
        "half_width": settings["half_width"], "k_values": settings["k_values"],
        "seed": settings["seeds"]["3"], "starts": settings["starts"],
        "iterations": settings["iterations"], "initial_step": settings["initial_step"],
        "geometric_decay": settings["geometric_decay"],
        "warm_start": "First start replaced by previous best direction padded with zero for every k >= 2",
        "observation_windows": "Original all-valid sliding windows; identical in both fresh fits",
        "source_hashes": {name: hashlib.sha256((AR_DIR / name).read_bytes()).hexdigest()
                          for name in ("ar_tools/functions.py", "ar_tools/descent.py")},
        "original_csv_sha256": hashlib.sha256((ORIGINAL_DIR / "width_curves.csv").read_bytes()).hexdigest(),
    }
    configuration_path = DATA_DIR / "configuration.json"
    if configuration_path.exists():
        if json.loads(configuration_path.read_text()) != configuration:
            raise RuntimeError("Existing rerun has different settings; preserve it in a separate directory.")
    else:
        configuration_path.write_text(json.dumps(configuration, indent=2) + "\n")

    # Check a flat prefix followed by a genuine drop, including the stationary law.
    assert descent_chain([5, 5, 5, 1, 1, 1])[0] == 3

    fitting_started = perf_counter()
    new_rows = []
    with threadpool_limits(limits=1):
        for replication in range(1, configuration["replications"] + 1):
            checkpoint = DATA_DIR / "replications" / f"replication_{replication:03d}.csv"
            series_path = DATA_DIR / "processes" / f"process_{replication:03d}.csv"
            if checkpoint.exists() and series_path.exists():
                cached = pd.read_csv(checkpoint)
                assert len(cached) == 20 and set(cached.variant) == set(VARIANTS[1:])
                new_rows.extend(cached.to_dict("records"))
                continue
            generator = np.random.default_rng(np.random.SeedSequence([
                configuration["seed"], replication - 1, 0,
            ]))
            phi = jones_ar_coefficients(3, rng=generator)
            old_group = original[original.replication == replication].sort_values("k")
            assert np.allclose(phi, json.loads(old_group.iloc[0].coefficients), rtol=0, atol=1e-14)
            series = simulate_uniform_ar(phi, configuration["sample_size"],
                                         configuration["burnin"], configuration["half_width"],
                                         rng=generator)
            pd.DataFrame({"t": np.arange(1, len(series) + 1), "X_t": series}).to_csv(series_path, index=False)
            pair_rows = _fit_rows(series, phi, settings, replication, warm=False)
            pair_rows += _fit_rows(series, phi, settings, replication, warm=True)
            pd.DataFrame(pair_rows).to_csv(checkpoint, index=False)
            new_rows.extend(pair_rows)
            if replication % 5 == 0:
                print(f"Completed {replication}/50 paired fits ({perf_counter() - fitting_started:.1f}s)", flush=True)

        original["variant"] = VARIANTS[0]
        width_curves = pd.concat([original, pd.DataFrame(new_rows)], ignore_index=True)
        width_curves = width_curves.sort_values(["variant", "replication", "k"])
        width_curves.to_csv(DATA_DIR / "width_curves.csv", index=False)
        selections, diagnostics, chain_rows, comparison_rows = [], [], [], []
        for (variant, replication), group in width_curves.groupby(["variant", "replication"]):
            m = group.sort_values("k").minimum_width.to_numpy()
            diff = np.diff(m)
            diagnostics.append({"variant": variant, "replication": replication,
                                "upward_steps": int(np.sum(diff > 1e-12)),
                                "positive_steps_at_machine_precision": int(np.sum((diff > 0) & (diff <= 1e-12))),
                                "maximum_increase": max(0.0, float(diff.max()))})
            selected = width_selectors(m)
            if variant == VARIANTS[2]:
                order, drops, remaining, alpha, stationary, odds, residual = descent_chain(m)
                selected["Descent chain"] = order
                for i in range(len(m)):
                    chain_rows.append({"replication": replication, "k": i + 1,
                                       "drop": drops[i], "remaining_reduction": remaining[i],
                                       "alpha": alpha[i], "stationary_probability": stationary[i],
                                       "odds": odds[i], "candidate": 0 < i < len(m) - 1,
                                       "stationarity_residual": residual})
            for method, order in selected.items():
                selections.append({"variant": variant, "replication": replication,
                                   "true_order": 3, "method": method,
                                   "selected_order": order, "correct": order == 3})
            if variant == VARIANTS[2]:
                fresh = width_curves[(width_curves.variant == VARIANTS[1])
                                     & (width_curves.replication == replication)].sort_values("k")
                old = original[original.replication == replication].sort_values("k")
                for k, (w, independent, historical) in enumerate(zip(m, fresh.minimum_width,
                                                                    old.minimum_width), start=1):
                    comparison_rows.append({"replication": replication, "k": k,
                                            "original_saved_width": historical,
                                            "independent_rerun_width": independent,
                                            "padded_warm_start_width": w,
                                            "warm_minus_independent": w - independent})

    selections = pd.DataFrame(selections)
    summary_rows = []
    for (variant, method), group in selections.groupby(["variant", "method"], sort=False):
        q = group.selected_order
        summary_rows.append({"variant": variant, "method": method, "correct": int((q == 3).sum()),
                             "under": int((q < 3).sum()), "over": int((q > 3).sum()), "total": len(q)})
    summary = pd.DataFrame(summary_rows)
    selections.to_csv(DATA_DIR / "selections.csv", index=False)
    summary.to_csv(DATA_DIR / "summary.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(DATA_DIR / "monotonicity_diagnostics.csv", index=False)
    pd.DataFrame(chain_rows).to_csv(DATA_DIR / "descent_chain_scores.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(DATA_DIR / "paired_width_comparison.csv", index=False)
    (DATA_DIR / "numerical_notes.json").write_text(json.dumps({
        "warm_monotonicity_check": "32 * machine epsilon * max(1, maximum absolute width)",
        "descent_chain": "Reject larger upward steps; set only machine-precision-sized drops within that tolerance to zero and compute remaining reduction by summing the resulting drops",
        "curve_postprocessing": "None; saved widths are the actual optimizer outputs",
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2) + "\n")

    # Reproduce the saved derivative selections to verify signs and order indexing.
    old_derivatives = pd.read_csv(AR_DIR / "outputs/csv/ar3_implied_support_width/derivative_selector_results.csv")
    for label, column in [("First difference", "first_difference_order"),
                          ("Second difference", "second_difference_order")]:
        actual = selections[(selections.variant == VARIANTS[0]) & (selections.method == label)]
        actual = actual.set_index("replication").selected_order.sort_index()
        expected = old_derivatives.set_index("replication")[column].sort_index()
        assert np.array_equal(actual, expected)
    make_plots(width_curves, summary)
    print("\nCorrect selections out of 50:", flush=True)
    print(summary.pivot(index="method", columns="variant", values="correct").to_string(), flush=True)
    print("\nCurves with upward steps:", flush=True)
    diagnostic_frame = pd.DataFrame(diagnostics)
    print(diagnostic_frame.assign(nonmonotone=lambda d: d.upward_steps > 0)
          .groupby("variant").nonmonotone.sum().to_string(), flush=True)
    return DATA_DIR


if __name__ == "__main__":
    run_experiment()
