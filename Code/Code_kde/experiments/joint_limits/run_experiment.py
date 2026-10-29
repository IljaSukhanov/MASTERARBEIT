"""Repeat current Figure 3.6 with a chosen bandwidth grid (default thesis r=1).

The fixed sample, interval, N values, lambda rules, Fourier approximation,
and Perron solver match notebooks/experiment_1_joint_N_lambda_limit.ipynb.
Only the bandwidth grid changes. No thesis files are written.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time

KDE_DIR = Path(__file__).resolve().parents[2]
if str(KDE_DIR) not in sys.path:
    sys.path.insert(0, str(KDE_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from kde_tools.qhat import minimize_qhat
from kde_tools.energy_minimization import energy_search_settings
from kde_tools.bandwidths import bandwidth_grid_linear_in_psi_beta
from kde_tools.fourier import (
    DEFAULT_FREQUENCY_CUTOFF,
    DEFAULT_FREQUENCY_INTERVALS,
    FOURIER_L2_METHOD_VERSION,
    beta2_divergence_matrix_fourier,
    gaussian_kde_l2_gram_fourier,
)
from kde_tools.pdf_export import enable_automatic_pdf_saving
from kde_tools.perron import PERRON_SOLVER_VERSION, perron_eigenpair_from_divergence
from kde_tools.similarities import beta2_divergence_matrix, squared_l2_distance_from_gram

NOTEBOOK_NAME = "linear_grid_joint_N_lambda_limit"
DATA_DIR = KDE_DIR / "outputs" / "data" / "linear_grid_joint_n_lambda_limit"
IMAGE_DIR = KDE_DIR / "outputs" / "images" / "linear_grid_joint_n_lambda_limit"
SEED, N_DATA = 20260823, 200
H_MIN, H_MAX = 0.01, 2.0
N_VALUES = np.arange(100, 3001, 100)
EXPONENTS = (0.5, 1.0, 1.5, 0.0, 2.0, 3.0)
COLORS = ("#0072B2", "#009E73", "#CC79A7", "#E69F00", "#7A7A7A", "#D55E00")
LABELS = (r"$N^{1/2}$", r"$N$", r"$N^{3/2}$", r"$1$", r"$N^2$", r"$N^3$")
R, GRID_BETA, R_MATH = 1.0, 0.0, "1"
GRID_DESCRIPTION = "Geometric bandwidth grid replaced by linear grid."
PLOT_TITLE = r"Linear bandwidth grid ($r=1$): joint $(N,\lambda_N)$ experiment"


def configure_grid(r: str = "1") -> str:
    """Choose a thesis r and return its isolated experiment/PDF name."""
    global R, GRID_BETA, R_MATH, GRID_DESCRIPTION, PLOT_TITLE
    global NOTEBOOK_NAME, DATA_DIR, IMAGE_DIR
    fraction = Fraction(r)
    if not 0 < fraction <= 1:
        raise ValueError("Require 0 < r <= 1.")
    R = float(fraction)
    GRID_BETA = 2.0 * (1.0 - R)
    R_MATH = str(fraction)
    if fraction == 1:
        NOTEBOOK_NAME = "linear_grid_joint_N_lambda_limit"
        GRID_DESCRIPTION = "Geometric bandwidth grid replaced by linear grid."
        PLOT_TITLE = r"Linear bandwidth grid ($r=1$): joint $(N,\lambda_N)$ experiment"
    else:
        tag = f"r{fraction.numerator}over{fraction.denominator}"
        NOTEBOOK_NAME = f"{tag}_joint_N_lambda_limit"
        GRID_DESCRIPTION = f"Bandwidth grid uses thesis r={fraction}, with all other settings fixed."
        PLOT_TITLE = rf"Bandwidth grid $r={R_MATH}$: joint $(N,\lambda_N)$ experiment"
    directory_name = NOTEBOOK_NAME.lower()
    DATA_DIR = KDE_DIR / "outputs" / "data" / directory_name
    IMAGE_DIR = KDE_DIR / "outputs" / "images" / directory_name
    return NOTEBOOK_NAME


def bandwidth_grid(size: int) -> np.ndarray:
    return bandwidth_grid_linear_in_psi_beta(H_MIN, H_MAX, beta=GRID_BETA, grid_size=size)


def linear_metric(h: float, squared_differences: np.ndarray) -> float:
    """Exact (1/2)||partial_h f_hat_h||_2^2, the r=1 local coefficient.

    This is local_beta2_log_metric in the original notebook divided by h**2.
    The factor 1/2 matches the divergence and thesis Proposition prop:main_thm.
    """
    q = 2.0 * float(h)**2
    d = squared_differences
    overlap = np.exp(-d / (2.0 * q)) / np.sqrt(2.0 * np.pi * q)
    return float(2.0 * h**2 * np.mean(overlap * (
        3.0 / (4.0 * q**2) - 3.0 * d / (2.0 * q**3) + d**2 / (4.0 * q**4)
    )))


def grid_metric(h: float, squared_differences: np.ndarray) -> float:
    """Thesis Q_r(h) = (1/2) h**(2*(1-r)) ||partial_h f_hat_h||_2**2."""
    return float(h**GRID_BETA * linear_metric(h, squared_differences))


def run_experiment() -> tuple[pd.DataFrame, dict, np.ndarray, np.ndarray]:
    global DATA_DIR
    from kde_tools.energy_minimization import unified_energy_run_directory
    if not DATA_DIR.name.startswith("unified_"):
        DATA_DIR = unified_energy_run_directory(DATA_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    samples = np.random.default_rng(SEED).normal(size=N_DATA)
    settings = {
        "seed": SEED, "sample_size": N_DATA,
        "sample_sha256": hashlib.sha256(samples.tobytes()).hexdigest(),
        "h_min": H_MIN, "h_max": H_MAX, "r": R, "grid_beta": GRID_BETA,
        "N_values": N_VALUES.tolist(), "lambda_exponents": list(EXPONENTS),
        "frequency_cutoff": DEFAULT_FREQUENCY_CUTOFF,
        "frequency_intervals": DEFAULT_FREQUENCY_INTERVALS,
        "l2_method": FOURIER_L2_METHOD_VERSION, "perron_solver": PERRON_SOLVER_VERSION,
        "energy_search": energy_search_settings(),
        "source_notebook": "notebooks/experiment_1_joint_N_lambda_limit.ipynb",
        "only_experiment_change": GRID_DESCRIPTION,
    }
    settings_path = DATA_DIR / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise RuntimeError("Saved settings differ; preserve that run before changing settings.")
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    np.savetxt(DATA_DIR / "sample.csv", samples, delimiter=",", header="x", comments="")

    squared_differences = (samples[:, None] - samples[None, :])**2
    metric_h = np.geomspace(H_MIN, H_MAX, 2001)
    metric_q = np.array([grid_metric(h, squared_differences) for h in metric_h])
    energy_analysis = minimize_qhat(samples, H_MIN, H_MAX, r=R)
    best = energy_analysis["global_minimum"]
    candidates = energy_analysis["endpoints"]+energy_analysis["local_minima"]
    reference = {
        "h_reference": best["h"], "q_reference": best["energy"],
        "reference_is_endpoint": bool(best["boundary"]),
        "candidate_h": [point["h"] for point in candidates],
        "candidate_q": [point["energy"] for point in candidates],
        "reference_method": "shared bandwidth-energy minimization engine",
        "analysis": energy_analysis,
    }
    (DATA_DIR / "reference.json").write_text(json.dumps(reference, indent=2) + "\n")
    metric_column = "q_r1" if R == 1.0 else "q_r"
    pd.DataFrame({"h": metric_h, metric_column: metric_q}).to_csv(DATA_DIR / "metric.csv", index=False)
    print("Reference:", reference, flush=True)

    validation_grid = bandwidth_grid(25)
    exact = beta2_divergence_matrix(samples, validation_grid)
    approximate = beta2_divergence_matrix_fourier(samples, validation_grid)
    absolute_error = float(np.max(np.abs(exact - approximate)))
    validation = {"maximum_absolute_divergence_error_25": absolute_error}
    if absolute_error > 1e-9:
        raise RuntimeError(f"Fourier validation failed: {absolute_error}")
    for exponent in EXPONENTS:
        exact_result = perron_eigenpair_from_divergence(exact, 25**exponent)
        approximate_result = perron_eigenpair_from_divergence(approximate, 25**exponent)
        if exact_result.selected_index != approximate_result.selected_index:
            raise RuntimeError("Exact and Fourier bandwidth selections differ on validation grid.")

    result_path = DATA_DIR / "results.csv"
    rows = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    completed = {int(row["N"]) for row in rows}
    if any(sum(int(row["N"]) == n for row in rows) != len(EXPONENTS) for n in completed):
        raise RuntimeError("Incomplete checkpoint; inspect results.csv before resuming.")
    start = time.perf_counter()
    for N in N_VALUES:
        N = int(N)
        if N in completed:
            continue
        grid = bandwidth_grid(N)
        np.testing.assert_allclose(
            grid**R, np.linspace(H_MIN**R, H_MAX**R, N), rtol=2e-14, atol=1e-15,
        )
        divergence = beta2_divergence_matrix_fourier(samples, grid)
        current_rows = []
        for exponent in EXPONENTS:
            result = perron_eigenpair_from_divergence(divergence, N**exponent)
            current_rows.append({
                "N": N, "alpha": exponent, "lambda": N**exponent,
                "selected_index": result.selected_index,
                "selected_h": float(grid[result.selected_index]),
                "relative_residual": result.relative_residual,
                "relative_gap": result.relative_gap,
            })
        rows.extend(current_rows)
        pd.DataFrame(rows).to_csv(result_path, index=False)
        print(f"N={N}: " + ", ".join(
            f"alpha={row['alpha']:g}: h={row['selected_h']:.6f}" for row in current_rows
        ), flush=True)

    results = pd.DataFrame(rows).sort_values(["alpha", "N"]).reset_index(drop=True)
    if len(results) != len(N_VALUES) * len(EXPONENTS):
        raise RuntimeError("The sweep is incomplete.")

    # Refine the frequency quadrature at the finest bandwidth grid, checking
    # every displayed regime, especially the sensitive high-lambda choices.
    grid = bandwidth_grid(int(N_VALUES[-1]))
    refined_gram = gaussian_kde_l2_gram_fourier(
        samples, grid, frequency_points=2 * DEFAULT_FREQUENCY_INTERVALS + 1,
    )
    refined_divergence = 0.5 * squared_l2_distance_from_gram(refined_gram)
    validation["refined_finest_grid_selections"] = []
    for exponent in EXPONENTS:
        refined = perron_eigenpair_from_divergence(refined_divergence, len(grid)**exponent)
        baseline = results[(results.N == len(grid)) & (results.alpha == exponent)].iloc[0]
        index_shift = abs(refined.selected_index - int(baseline.selected_index))
        validation["refined_finest_grid_selections"].append({
            "alpha": exponent, "selected_h": float(grid[refined.selected_index]),
            "index_shift": index_shift,
        })
        if index_shift != 0:
            raise RuntimeError("Frequency refinement changed a selected bandwidth.")
    validation["maximum_relative_eigenpair_residual"] = float(results.relative_residual.max())
    validation["elapsed_seconds_this_invocation"] = time.perf_counter() - start
    validation["all_checks_passed"] = True
    (DATA_DIR / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print("Validation:", validation, flush=True)
    return results, reference, metric_h, metric_q


def plot_results(results: pd.DataFrame, reference: dict) -> Path:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10.5, 6.3))
    for exponent, label, color in zip(EXPONENTS, LABELS, COLORS):
        subset = results[results.alpha == exponent]
        ax.plot(
            subset.N, subset.selected_h, label=label, color=color,
            marker="o", markersize=4, linewidth=2 if 0 < exponent < 2 else 1.6,
            linestyle="-" if 0 < exponent < 2 else "--",
        )
    ax.axhline(
        reference["h_reference"], color="black", linestyle=":", linewidth=2,
        label=(rf"$\arg\min\widehat Q_{{{R_MATH}}}(h)={reference['h_reference']:g}$"
               + (" (upper endpoint)" if reference["h_reference"] == H_MAX else "")),
    )
    ax.set(
        xlabel=r"number of bandwidths $N$",
        ylabel=r"selected bandwidth $\widehat h_{\mathrm{spec},N}$",
        title=PLOT_TITLE,
    )
    ax.set_xticks([100, 500, 1000, 1500, 2000, 2500, 3000])
    ax.legend(
        title=r"$\lambda_N$", ncol=2, loc="lower right",
        bbox_to_anchor=(0.99, 0.17), frameon=True, fontsize=10,
    )
    fig.text(
        0.5, 0.015,
        r"Same fixed $n=200$ standard-normal sample as Figure 3.6; $h\in[0.01,2]$; seed 20260823",
        ha="center", fontsize=10, color="#444444",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    image_path = IMAGE_DIR / f"{NOTEBOOK_NAME}.png"
    fig.savefig(image_path, dpi=180, bbox_inches="tight")
    plt.show()  # Shared automatic exporter saves the corresponding PDF.
    plt.close(fig)
    return image_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--r", default="1", help="Thesis grid parameter, e.g. 1 or 13/14.")
    args = parser.parse_args()
    configure_grid(args.r)
    enable_automatic_pdf_saving(NOTEBOOK_NAME)
    with threadpool_limits(limits=2):
        if args.plot_only:
            results = pd.read_csv(DATA_DIR / "results.csv")
            reference = json.loads((DATA_DIR / "reference.json").read_text())
        else:
            results, reference, _, _ = run_experiment()
        print("PNG:", plot_results(results, reference), flush=True)
