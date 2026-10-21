"""Replay the six failed Experiment 3 fits and plot best-estimate rho traces.

Calls the shared optimizer with its original checkpoint/restart settings.
Plots the actual evaluated prefixes; no continuation beyond rejection is run.
Run from the repository root with a noninteractive Matplotlib backend.
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Code.Code_AR_processes.ar_tools.descent import projected_subgradient_empirical_width
from Code.Code_AR_processes.ar_tools.functions import is_stationary
from Code.Code_AR_processes.ar_tools.plotting_functions import save_figure
from Code.Code_AR_processes.ar_tools.tuning import _ar_realization_cases


AR_DIR = ROOT / "Code/Code_AR_processes"
EXPERIMENT = "ar_tuning_experiments/jones_stationarity_failure_examples"
IMAGE_DIR = AR_DIR / "outputs/images" / EXPERIMENT
PDF_DIR = AR_DIR / "outputs/pdfs" / EXPERIMENT
FAILED_MODELS = (6, 16, 27, 70, 98, 99)
GAMMA0, HALF_LIFE = 0.2, 50
PATIENCE, ITERATIONS, MAX_RESTARTS = 200, 2_000, 100
OPTIMIZER_SEED = 20261019
ATTEMPTS_TO_PLOT = (0, 1, 100)
COLORS = ("#2166ac", "#d97914", "#8b46a1")
LINESTYLES = ("-", "--", "-.")


def spectral_radius_history(coefficients):
    """Use the same companion-matrix helper as the checkpoint itself."""
    return np.array([
        float(np.max(np.abs(is_stationary(phi)[1])))
        if np.isfinite(phi).all() else np.nan
        for phi in coefficients
    ])


def replay_failures():
    direction = np.random.default_rng(OPTIMIZER_SEED).normal(size=4)
    direction /= np.linalg.norm(direction)
    cases = _ar_realization_cases(
        None, [500], direction, replications=100,
        coefficient_seed=20261023, process_seed=20261024, burnin=2_000,
    )
    examples = []
    with threadpool_limits(limits=1):
        for case in cases:
            if case["replication"] not in FAILED_MODELS:
                continue
            result = projected_subgradient_empirical_width(
                case["points"], iterations=ITERATIONS, patience=PATIENCE,
                initial_step=GAMMA0, geometric_decay=np.exp2(-1.0 / HALF_LIFE),
                step_schedule="geometric", improvement_tolerance=0.0,
                initial_direction=case["initial_direction"],
                rng=np.random.SeedSequence([OPTIMIZER_SEED, 0, 4, 21, 0]),
                stationarity_check=True, max_restarts=MAX_RESTARTS,
                record_best_coefficients=True,
            )
            assert result["restart_limit_reached"]
            assert result["attempt_count"] == MAX_RESTARTS + 1
            assert result["total_evaluations"] == (MAX_RESTARTS + 1) * PATIENCE
            assert all(not attempt["checkpoint"]["stationary"] for attempt in result["attempts"])
            curves = []
            for index in ATTEMPTS_TO_PLOT:
                attempt = result["attempts"][index]
                rho = spectral_radius_history(attempt["best_coefficient_history"])
                assert len(rho) == PATIENCE
                assert rho[-1] >= 1 or not np.isfinite(rho[-1])
                np.testing.assert_allclose(
                    rho[-1], attempt["checkpoint"]["spectral_radius"], rtol=1e-12,
                )
                curves.append(rho)
            examples.append(dict(case=case, curves=curves, result=result))
            print(f"Model {case['replication']}: 101/101 rejected; "
                  f"plotted checkpoint rho = {[round(rho[-1], 4) for rho in curves]}", flush=True)
    assert tuple(example["case"]["replication"] for example in examples) == FAILED_MODELS
    return examples


def format_axes(ax, curves):
    finite = np.concatenate([rho[np.isfinite(rho)] for rho in curves])
    upper = max(2.0, float(np.max(finite)) * 1.4)
    ax.set_yscale("symlog", linthresh=1.0, linscale=0.65)
    ax.set_ylim(0, upper)
    ticks = [0, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000]
    ax.set_yticks([tick for tick in ticks if tick < upper])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.axhspan(0, 1, color="#e3f2e5", zorder=0)
    ax.axhline(1, color="#2e7544", ls="--", lw=1.35, zorder=3)
    ax.axvline(PATIENCE, color="#b34638", ls=":", lw=1.3, zorder=3)
    ax.set_xlim(1, PATIENCE + 7)
    ax.set_xticks([1, 50, 100, 150, 200])
    ax.grid(axis="both", alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    examples = replay_failures()
    for directory in (IMAGE_DIR, PDF_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 11, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(3, 2, figsize=(13.5, 10.8), constrained_layout=True)
        t = np.arange(1, PATIENCE + 1)
        for ax, example in zip(axes.flat, examples):
            case, curves = example["case"], example["curves"]
            format_axes(ax, curves)
            for rho, color, linestyle in zip(curves, COLORS, LINESTYLES):
                ax.step(t, rho, where="post", color=color, ls=linestyle, lw=1.6, alpha=0.9)
                ax.plot(PATIENCE, rho[-1], marker="x", color=color, markersize=8,
                        markeredgewidth=2, zorder=5)
            ax.set_title(
                f"Model {case['replication']}  |  true rho = {case['spectral_radius']:.3f}\n"
                "101 of 101 attempts rejected at the checkpoint",
                fontsize=11.5,
            )
        for ax in axes[-1, :]:
            ax.set_xlabel("Evaluated iteration within the attempt")
        for ax in axes[:, 0]:
            ax.set_ylabel(r"Best-width estimate: $\widehat\rho_t$")
        handles = [Line2D([0], [0], color=color, ls=linestyle, lw=2, label=label)
                   for color, linestyle, label in zip(COLORS, LINESTYLES,
                       ["Initial attempt", "First restart", "Final restart (number 100)"])]
        handles += [Line2D([0], [0], color="#2e7544", ls="--", lw=1.4,
                           label="Threshold rho = 1 (green region below)"),
                    Line2D([0], [0], color="#b34638", ls=":", lw=1.4,
                           label="Checkpoint: 200 evaluations")]
        fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=10)
        fig.suptitle(
            "Failed stationarity checks in Experiment 3\n"
            r"$\gamma_0=0.2$, $H=50$, $T=500$; coefficients recovered from the best-width direction"
            "\nVertical axis is linear from 0 to 1 and logarithmic above 1.",
            fontsize=14,
        )
        save_figure(fig, "04_failed_rho_vs_iteration", pdf_directory=PDF_DIR, image_directory=IMAGE_DIR)
        plt.close(fig)

        # Show one attempt that briefly had an admissible best estimate.
        example = examples[0]
        candidate_traces = [spectral_radius_history(attempt["best_coefficient_history"])
                            for attempt in example["result"]["attempts"]]
        attempt_index = next(index for index, rho in enumerate(candidate_traces) if np.any(rho < 1))
        rho = candidate_traces[attempt_index]
        first_pass_index = int(np.flatnonzero(rho < 1)[0])
        first_failure = first_pass_index + int(np.flatnonzero(rho[first_pass_index:] >= 1)[0]) + 1
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.7), constrained_layout=True)
        for ax, last in zip(axes, [PATIENCE, min(PATIENCE, max(30, first_failure + 10))]):
            format_axes(ax, [rho[:last]])
            ax.step(t[:last], rho[:last], where="post", color=COLORS[0], lw=2)
            if last == PATIENCE:
                ax.plot(PATIENCE, rho[-1], "x", color="#b34638", markersize=9, markeredgewidth=2)
            ax.set_xlim(1, last + (7 if last == PATIENCE else 0))
            if last < PATIENCE:
                ax.set_xticks([1, 5, 10, 20, 30] if last == 30 else np.linspace(1, last, 5, dtype=int))
            ax.set(xlabel="Evaluated iteration within the attempt",
                   ylabel=r"Best-width estimate: $\widehat\rho_t$",
                   title="Complete rejected attempt" if last == PATIENCE else "Early iterations")
        fig.suptitle(
            f"Model 6, attempt {attempt_index + 1}: rho was below 1 earlier, but failed at the checkpoint\n"
            f"Checkpoint rho = {rho[-1]:.2f}; same gamma0 = 0.2 and H = 50",
            fontsize=13,
        )
        save_figure(fig, "05_temporary_pass_then_failed_checkpoint", pdf_directory=PDF_DIR, image_directory=IMAGE_DIR)
        plt.close(fig)
        print(f"Detail example: model 6, attempt {attempt_index + 1}; "
              f"rho returned to >= 1 at evaluated iteration {first_failure}; final rho {rho[-1]:.6f}")
    print(f"PNG figures: {IMAGE_DIR}")
    print(f"PDF figures: {PDF_DIR}")


if __name__ == "__main__":
    main()
