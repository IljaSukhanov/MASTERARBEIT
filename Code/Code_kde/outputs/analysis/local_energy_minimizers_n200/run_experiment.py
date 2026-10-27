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
from scipy.optimize import brentq
from scipy.spatial.distance import pdist
from threadpoolctl import threadpool_limits

from kde_tools.qhat import _energy
from kde_tools.pdf_export import enable_automatic_pdf_saving

NAME = "local_energy_minimizers_n200"
DATA_DIR = KDE_DIR / "outputs" / "data" / NAME
IMAGE_DIR = KDE_DIR / "outputs" / "images" / NAME
M, N_DATA, SEED, BANK_WIDTH = 1000, 200, 20260830, 1000
METHODS = ("r0", "r13over14", "r1")
R_VALUES = (0.0, 13.0/14.0, 1.0)
R_LABELS = ("0", "13/14", "1")
SCAN_POINTS, WORKERS = 1025, 4
VERSION = "exact_local_minima_curvature_augmented_v1"


def transformed(base: np.ndarray, z: float, r: float) -> np.ndarray:
    q, g, k = base
    return np.exp(-2*r*z)*np.array([q, g-2*r*q, k-4*r*g+4*r*r*q])


def roots_from_scan(d2: np.ndarray, n: int, z: np.ndarray, base: np.ndarray, r: float) -> list[dict]:
    """Refine negative-to-positive energy-gradient crossings exactly.

    Curvature roots augment the grid, exposing shallow gradient excursions
    that could contain a nearby minimum/maximum pair between scan nodes.
    """
    def exact(t):
        return transformed(_energy(d2, n, t), t, r)

    curvature = base[:, 2]-4*r*base[:, 1]+4*r*r*base[:, 0]
    nodes = {float(t): row for t, row in zip(z, base)}
    for i in np.flatnonzero(curvature[:-1]*curvature[1:] < 0):
        t = brentq(lambda t: exact(t)[2], z[i], z[i+1], xtol=2e-13)
        nodes[float(t)] = _energy(d2, n, t)
    zz = np.array(sorted(nodes))
    gradients = np.array([nodes[t][1]-2*r*nodes[t][0] for t in zz])
    result = []
    for i in np.flatnonzero((gradients[:-1] < 0) & (gradients[1:] > 0)):
        t = float(brentq(lambda t: exact(t)[1], zz[i], zz[i+1], xtol=2e-13))
        value = exact(t)
        if value[0] <= 0 or value[2] <= 0 or abs(value[1]/value[2]) > 1e-8:
            raise ArithmeticError("Local minimum failed exact stationarity/curvature checks.")
        if result and abs(t-np.log(result[-1]["h"])) < 1e-8:
            continue
        result.append({"h": float(np.exp(t)), "energy": float(value[0]),
                       "log_derivative": float(value[1]), "log_curvature": float(value[2]),
                       "newton_correction": float(abs(value[1]/value[2]))})
    return result


def same_roots(a: list[dict], b: list[dict]) -> bool:
    return len(a) == len(b) and all(abs(np.log(x["h"]/y["h"])) < 1e-7 for x, y in zip(a, b))


def sample_minima(sample_id: int, samples: np.ndarray) -> tuple[list[dict], list[dict], dict]:
    n = len(samples)
    d2 = pdist(samples[:, None], metric="sqeuclidean")
    positive = d2[d2 > 0]
    if not len(positive):
        # Coincident observations give a single Gaussian, whose Q_r decreases
        # strictly for all r used here. This is also a validation case.
        lower, upper = 1e-6, 10.0
        minima = [[] for _ in R_VALUES]
        resolution = {"scan_points": 0, "refinement_needed": False, "max_log_h_discrepancy": 0.0}
    else:
        # For h <= lower every nonzero pair has u=d^2/(4h^2)>=50.
        # Its positive gradient contribution is bounded by
        # (2*50^3+10.5*50)*exp(-50), far below the negative diagonal term.
        # For h >= upper all u<=0.01: p(u)>0 and g(u)<0, so Q_r'<0.
        # Therefore the tails outside these sample-specific bounds contain
        # no stationary points for n=200 and r in [0,1].
        lower = float(np.sqrt(positive.min()/200.0))
        upper = float(5*np.sqrt(positive.max()))
        z = np.linspace(np.log(lower), np.log(upper), SCAN_POINTS)
        base = np.array([_energy(d2, n, t) for t in z])
        minima = [roots_from_scan(d2, n, z, base, r) for r in R_VALUES]
        coarse = [roots_from_scan(d2, n, z[::2], base[::2], r) for r in R_VALUES]
        resolution = {"scan_points": SCAN_POINTS, "refinement_needed": False,
                      "max_log_h_discrepancy": 0.0}
        while not all(same_roots(a, b) for a, b in zip(coarse, minima)):
            resolution["refinement_needed"] = True
            if len(z) >= 8193:
                raise ArithmeticError(f"Local minima did not stabilize for sample {sample_id}.")
            coarse = minima
            z = np.linspace(np.log(lower), np.log(upper), 2*len(z)-1)
            base = np.array([_energy(d2, n, t) for t in z])
            minima = [roots_from_scan(d2, n, z, base, r) for r in R_VALUES]
            resolution["scan_points"] = len(z)
        resolution["max_log_h_discrepancy"] = max(
            [abs(np.log(a["h"]/b["h"])) for aa, bb in zip(coarse, minima)
             for a, b in zip(aa, bb)] or [0.0]
        )
    rows, candidates = [], []
    for method, r, points in zip(METHODS, R_VALUES, minima):
        best = min(range(len(points)), key=lambda i: (points[i]["energy"], points[i]["h"])) if points else None
        row = {"sample_id": sample_id, "method": method, "r": r,
               "local_minimum_count": len(points),
               "selected_h": points[best]["h"] if best is not None else np.nan,
               "selected_energy": points[best]["energy"] if best is not None else np.nan,
               "search_lower": lower, "search_upper": upper}
        rows.append(row)
        for i, point in enumerate(points):
            candidates.append({"sample_id": sample_id, "method": method, "r": r,
                               "candidate_id": i, "selected": i == best, **point})
    return rows, candidates, {"sample_id": sample_id, **resolution}


def validate_reference_cases() -> dict:
    source = KDE_DIR / "outputs/data/linear_grid_joint_n_lambda_limit/sample.csv"
    sample = np.loadtxt(source, skiprows=1)
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
        "version": VERSION, "M": M, "n": N_DATA, "seed": SEED,
        "distribution": "standard normal", "paired_across_r": True,
        "generated_bank_shape": [M, BANK_WIDTH], "used_prefix_columns": N_DATA,
        "sample_bank_sha256": hashlib.sha256(samples.tobytes()).hexdigest(),
        "r_values": list(R_VALUES), "methods": list(METHODS), "scan_points": SCAN_POINTS,
        "selection": "lowest energy among all detected interior local minima; no endpoint fallback",
        "search_bounds": "sample-adaptive bounds with negative-gradient Gaussian tails",
        "resolution_check": "compare 513 and 1025 nodes for every sample; refine further on disagreement",
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
