"""Paired distributions of interior local Q_r minima, M=1000 and n=200.

Uses the exact Gaussian-pair energy and log-bandwidth derivatives in
kde_tools.qhat and kde_tools.continuous_minimizers. Endpoints are excluded.
When several local minima exist, select the one with the lowest Q_r value;
save every candidate and report zero/multiple-minimum counts separately.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

KDE_DIR = Path(__file__).resolve().parents[3]
if str(KDE_DIR) not in sys.path:
    sys.path.insert(0, str(KDE_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from kde_tools.qhat import _energy, GaussianQhatObjective, minimize_qhat
from kde_tools.energy_minimization import (energy_search_id, energy_search_settings,
                                            minimize_bandwidth_energy, search_config, DEFAULT_ENERGY_SEARCH)
from kde_tools.pdf_export import enable_automatic_pdf_saving

NAME = "local_energy_minimizers_n200"
DATA_DIR = KDE_DIR / "outputs" / "data" / NAME / ("unified_" + energy_search_id())
IMAGE_DIR = KDE_DIR / "outputs" / "images" / NAME
M, N_DATA, SEED, BANK_WIDTH = 1000, 200, 20260830, 1000
METHODS = ("r0", "r13over14", "r1")
R_VALUES = (0.0, 13.0/14.0, 1.0)
R_LABELS = ("0", "13/14", "1")
SCAN_POINTS, WORKERS = DEFAULT_ENERGY_SEARCH.scan_points, 4
VERSION = "unified_local_minima_v2_" + energy_search_id()


def transformed(base: np.ndarray, z: float, r: float) -> np.ndarray:
    q, g, k = base
    return np.exp(-2*r*z)*np.array([q, g-2*r*q, k-4*r*g+4*r*r*q])


def roots_from_scan(d2, n, z, base, r):
    """Compatibility adapter; all root searches use the shared engine."""
    cached = {float(t): transformed(row, float(t), r) for t, row in zip(z, base)}
    def exact(t):
        t = float(t)
        if t not in cached:
            cached[t] = transformed(_energy(d2, n, t), t, r)
        return cached[t]
    report = minimize_bandwidth_energy(exact, float(np.exp(z[0])), float(np.exp(z[-1])),
        config=search_config(scan_points=len(z), check_resolution=False, scan_backend="exact"))
    return report["local_minima"]


def same_roots(a, b):
    return len(a) == len(b) and all(abs(np.log(x["h"]/y["h"])) < 1e-7 for x, y in zip(a, b))


def sample_minima(sample_id, samples):
    objective = GaussianQhatObjective(samples)
    positive = objective.d2[objective.d2 > 0]
    lower = float(np.sqrt(positive.min()/200.0)) if len(positive) else 1e-6
    upper = float(5*np.sqrt(positive.max())) if len(positive) else 10.0
    rows, candidates, analyses = [], [], {}
    for method, r in zip(METHODS, R_VALUES):
        report = minimize_qhat(objective, lower, upper, r=r,
            config=search_config(scan_backend="exact", scan_points=SCAN_POINTS))
        analyses[method] = report
        points = report["local_minima"]
        best = report["best_local_minimum"]
        rows.append(dict(sample_id=sample_id, method=method, r=r,
                         local_minimum_count=len(points), selected_h=best["h"] if best else np.nan,
                         selected_energy=best["energy"] if best else np.nan,
                         search_lower=lower, search_upper=upper,
                         interval_minimum_h=report["global_minimum"]["h"],
                         interval_minimum_energy=report["global_minimum"]["energy"],
                         interval_minimum_boundary=report["global_minimum"]["boundary"]))
        candidates.extend(dict(sample_id=sample_id, method=method, r=r,
                               candidate_id=i, selected=point is best, **point)
                          for i, point in enumerate(points))
    resolution = dict(sample_id=sample_id,
                      scan_points=max(a["diagnostics"]["scan_points"] for a in analyses.values()),
                      refinement_needed=any(a["diagnostics"]["refinements"] > 1 for a in analyses.values()),
                      max_log_h_discrepancy=max((a["diagnostics"]["maximum_log_h_discrepancy"] or 0.0) for a in analyses.values()),
                      resolution_stable=all(a["diagnostics"]["resolution_stable"] for a in analyses.values()),
                      analyses_json=json.dumps(analyses, sort_keys=True))
    return rows, candidates, resolution


def validate_reference_cases() -> dict:
    from kde_tools.experiment_inputs import load_reference_sample
    sample, _ = load_reference_sample()
    rows, candidates, _ = sample_minima(-1, sample)
    expected = [0.1764236320307353, 0.5614864810366537, 0.5898122919784192]
    errors = []
    for row, target in zip(rows, expected):
        if row["local_minimum_count"] != 1 or abs(row["selected_h"]-target) > 5e-7:
            raise ArithmeticError("Previously plotted local minimum was not recovered.")
        errors.append(abs(row["selected_h"]-target))
    gaussian_rows, _, _ = sample_minima(-2, np.zeros(N_DATA))
    if any(row["local_minimum_count"] for row in gaussian_rows):
        raise ArithmeticError("A monotone single-Gaussian energy produced a false minimum.")
    return {"maximum_previous_sample_h_error": max(errors),
            "monotone_single_gaussian_has_no_local_minimum": True,
            "reference_candidate_count": len(candidates)}


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Retain the sample-bank construction of the earlier repeated-sampling
    # experiment, so its n=200 sample prefixes can be compared directly.
    full_bank = np.random.default_rng(SEED).normal(size=(M, BANK_WIDTH))
    samples = np.ascontiguousarray(full_bank[:, :N_DATA])
    settings = {
        "version": VERSION, "energy_search": energy_search_settings(), "M": M, "n": N_DATA, "seed": SEED,
        "distribution": "standard normal", "paired_across_r": True,
        "generated_bank_shape": [M, BANK_WIDTH], "used_prefix_columns": N_DATA,
        "sample_bank_sha256": hashlib.sha256(samples.tobytes()).hexdigest(),
        "r_values": list(R_VALUES), "methods": list(METHODS), "scan_points": SCAN_POINTS,
        "selection": "lowest energy among all detected interior local minima; no endpoint fallback",
        "search_bounds": "sample-adaptive bounds with negative-gradient Gaussian tails",
        "resolution_check": "shared engine: compare nested scans using recorded controls",
        "qhat_source_sha256": hashlib.sha256((KDE_DIR/"kde_tools/qhat.py").read_bytes()).hexdigest(),
    }
    path = DATA_DIR / "settings.json"
    if path.exists() and json.loads(path.read_text()) != settings:
        raise RuntimeError("Saved local-minimum experiment settings differ.")
    path.write_text(json.dumps(settings, indent=2)+"\n")
    np.save(DATA_DIR / "samples.npy", samples)
    reference_validation = validate_reference_cases()
    result_path, candidate_path = DATA_DIR/"results.csv", DATA_DIR/"candidates.csv"
    resolution_path = DATA_DIR/"resolution_checks.csv"
    rows = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    candidates = pd.read_csv(candidate_path).to_dict("records") if candidate_path.exists() else []
    resolutions = pd.read_csv(resolution_path).to_dict("records") if resolution_path.exists() else []
    completed = {int(row["sample_id"]) for row in rows}
    if any(sum(int(row["sample_id"]) == i for row in rows) != len(METHODS) for i in completed):
        raise RuntimeError("Incomplete checkpoint.")

    def save():
        for target, records in [(result_path, rows), (candidate_path, candidates),
                                (resolution_path, resolutions)]:
            temporary = target.with_suffix(".tmp")
            pd.DataFrame(records).to_csv(temporary, index=False)
            temporary.replace(target)

    pending = [i for i in range(M) if i not in completed]
    start = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for sample_id, result in zip(pending, pool.map(lambda i: sample_minima(i, samples[i]), pending)):
                new_rows, new_candidates, resolution = result
                rows.extend(new_rows)
                candidates.extend(new_candidates)
                resolutions.append(resolution)
                if len(resolutions) % 25 == 0:
                    save()
                    print(f"Completed {len(resolutions)}/{M} samples", flush=True)
    finally:
        save()
    results = pd.DataFrame(rows).sort_values(["sample_id", "r"]).reset_index(drop=True)
    candidate_frame = pd.DataFrame(candidates).sort_values(["sample_id", "r", "h"]).reset_index(drop=True)
    summaries = []
    for method, r in zip(METHODS, R_VALUES):
        part = results[results.method == method]
        chosen = part.selected_h.dropna().to_numpy()
        if len(part) != M:
            raise RuntimeError("The sample bank is incomplete.")
        summaries.append({"method": method, "r": r, "samples": M,
                          "with_local_minimum": len(chosen),
                          "without_local_minimum": int((part.local_minimum_count == 0).sum()),
                          "multiple_local_minima": int((part.local_minimum_count > 1).sum()),
                          "total_local_minima": int(part.local_minimum_count.sum()),
                          "mean_h": float(chosen.mean()), "median_h": float(np.median(chosen)),
                          "sd_h": float(chosen.std(ddof=1)),
                          "q05_h": float(np.quantile(chosen, .05)),
                          "q95_h": float(np.quantile(chosen, .95))})
    summary = pd.DataFrame(summaries)
    summary.to_csv(DATA_DIR/"summary.csv", index=False)
    validation = {
        "reference_cases": reference_validation,
        "all_samples_checked_at_two_scan_resolutions": len(resolutions) == M,
        "samples_requiring_extra_refinement": sum(bool(x["refinement_needed"]) for x in resolutions),
        "maximum_log_h_discrepancy": max(float(x["max_log_h_discrepancy"]) for x in resolutions),
        "maximum_stationarity_newton_correction": float(candidate_frame.newton_correction.max()),
        "all_detected_candidates_have_positive_curvature": bool((candidate_frame.log_curvature > 0).all()),
        "elapsed_seconds_this_invocation": time.perf_counter()-start,
    }
    (DATA_DIR/"validation.json").write_text(json.dumps(validation, indent=2)+"\n")
    print(summary.to_string(index=False), flush=True)
    print("Validation:", validation, flush=True)
    return results, summary, validation


def plot_results(results: pd.DataFrame, summary: pd.DataFrame) -> Path:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.1), sharex=True, sharey=True)
    bins = np.linspace(0.0, float(np.ceil(results.selected_h.max()/0.05)*0.05), 41)
    colors = ("#0072B2", "#009E73", "#D55E00")
    histogram_rows = []
    for ax, method, r_label, color in zip(axes, METHODS, R_LABELS, colors):
        chosen = results.loc[results.method == method, "selected_h"].dropna().to_numpy()
        info = summary[summary.method == method].iloc[0]
        counts, _ = np.histogram(chosen, bins=bins)
        ax.hist(chosen, bins=bins, color=color, alpha=.8, edgecolor="white", linewidth=.5)
        ax.axvline(np.median(chosen), color="#333333", linestyle="--", linewidth=1.2)
        ax.set(title=rf"$r={r_label}$", xlabel=r"interior local minimizer $h$", xlim=(bins[0], bins[-1]))
        ax.text(.97, .96,
                f"with a minimum: {int(info.with_local_minimum)}/{M}\n"
                f"no minimum: {int(info.without_local_minimum)}\n"
                f"multiple minima: {int(info.multiple_local_minima)}\n"
                f"median h: {np.median(chosen):.4f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .92})
        ax.grid(axis="x", visible=False)
        for left, right, count in zip(bins[:-1], bins[1:], counts):
            histogram_rows.append({"method": method, "bin_left": left, "bin_right": right, "count": int(count)})
        assert counts.sum() == len(chosen)
    axes[0].set_ylabel("number of samples")
    fig.suptitle(r"Interior local energy minimizers: 1,000 paired samples, $n=200$", fontsize=16)
    fig.text(.5, .018,
             "Standard-normal data; Gaussian kernel. Multiple minima: choose the lowest-energy interior minimum.\n"
             "Samples without an interior minimum are excluded from the histogram and counted above.",
             ha="center", fontsize=9.5, color="#444444")
    fig.tight_layout(rect=(0, .105, 1, .94), w_pad=2)
    path = IMAGE_DIR / "local_energy_minimizers_n200.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.show()  # Shared automatic PDF export.
    plt.close(fig)
    pd.DataFrame(histogram_rows).to_csv(DATA_DIR/"histogram.csv", index=False)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    enable_automatic_pdf_saving(NAME)
    with threadpool_limits(limits=1):
        if args.plot_only:
            results = pd.read_csv(DATA_DIR/"results.csv")
            summary = pd.read_csv(DATA_DIR/"summary.csv")
        else:
            results, summary, _ = run_experiment()
        print("PNG:", plot_results(results, summary), flush=True)
