"""Plot the tracked best-estimate spectral radii from the saved AR(3) runs."""

from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Code.Code_AR_processes.ar_tools.functions import is_stationary
from output.analysis.ar3_stationarity_restarts.run_analysis import spectral_radii

OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "Code/Code_AR_processes/outputs/csv/ar3_failed_coefficient_tracking/mae"
COLORS = {"failed": "#b34638", "working": "#167d77"}


def main():
    configuration = json.loads((SOURCE / "configuration.json").read_text())
    history = pd.read_csv(SOURCE / "iteration_history.csv")
    failed = history[history.start == "Failed random start"].reset_index(drop=True)
    working = history[history.start == "Zero-start control"].reset_index(drop=True)
    np.testing.assert_array_equal(failed.iteration, working.iteration)
    phi_columns = [f"best_phi_{j}" for j in range(1, 4)]
    failed_rho = spectral_radii(failed[phi_columns].to_numpy())
    working_rho = spectral_radii(working[phi_columns].to_numpy())
    stationary, true_eigenvalues = is_stationary(configuration["true_phi"])
    true_rho = float(np.max(np.abs(true_eigenvalues)))
    assert stationary
    np.testing.assert_allclose(true_rho, 0.5, rtol=0, atol=1e-14)
    previous = pd.read_csv(OUT / "failed_stationarity_history.csv")
    np.testing.assert_allclose(failed_rho, previous.best_rho, rtol=1e-11, atol=1e-10)
    assert np.all(failed_rho > 1)
    assert np.all(working_rho < 1)
    assert working_rho[0] == 0
    assert np.all(np.isfinite(failed_rho)) and np.all(np.isfinite(working_rho))

    curves = pd.DataFrame({
        "iteration": failed.iteration,
        "failed_best_estimate_rho": failed_rho,
        "working_best_estimate_rho": working_rho,
        "true_rho": true_rho,
        "causal_stationarity_boundary": 1.0,
    })
    curves.to_csv(OUT / "stationarity_iteration_curves.csv", index=False)

    def add_references(ax, vertical=False):
        reference = ax.axvline if vertical else ax.axhline
        reference(true_rho, color="#222222", lw=1.5, ls="--",
                  label=rf"True coefficients: $\rho={true_rho:g}$", zorder=5)
        reference(1, color="#777777", lw=1.2, ls=":",
                  label=r"Stationarity boundary: $\rho=1$", zorder=4)

    def format_measure(ax, vertical=False):
        # Include rho=0 at the working initial direction without clipping it.
        # Values up to one are linear; values above one are logarithmic.
        if vertical:
            ax.set_xscale("symlog", linthresh=1.0, linscale=1.0)
            ax.set_xlim(0, float(np.max(failed_rho)) * 1.2)
            ax.set_xticks([0, 0.5, 1, 10, 100])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        else:
            ax.set_yscale("symlog", linthresh=1.0, linscale=1.0)
            ax.set_ylim(0, float(np.max(failed_rho)) * 1.2)
            ax.set_yticks([0, 0.5, 1, 10, 100])
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))

    measure_label = r"Stationarity measure $\rho(C(\widehat\phi_t))$"
    with plt.rc_context({"font.size": 10, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), constrained_layout=True)
        for ax, last, title in zip(axes, [4999, 100], ["Complete run", "First 100 iterations"]):
            selection = curves.iteration <= last
            t = curves.loc[selection, "iteration"]
            ax.step(t, failed_rho[selection], where="post", color=COLORS["failed"],
                    lw=1.6, label="Failed random start")
            ax.step(t, working_rho[selection], where="post", color=COLORS["working"],
                    lw=1.7, label=r"Working start: $u_0=(1,0,0,0)$")
            add_references(ax)
            format_measure(ax)
            ax.set(xlabel="Evaluated iteration (0 = initial direction)",
                   ylabel=measure_label, title=title, xlim=(0, last))
            ax.grid(alpha=0.2)
        axes[0].legend(loc="upper right", frameon=False, fontsize=9)
        fig.suptitle(
            "Failed and working starting directions: stationarity of the tracked best estimate\n"
            "Same AR(3) sample | true phi = (0.5, -0.2, 0.1) | gamma0 = 0.1, beta = 0.997",
            fontsize=12,
        )
        fig.supxlabel("Measure axis: linear from 0 to 1, logarithmic above 1; the initial zero is included.",
                      fontsize=9, color="#555555")
        fig.savefig(OUT / "stationarity_vs_iteration.pdf", bbox_inches="tight")
        fig.savefig(OUT / "stationarity_vs_iteration.png", dpi=170, bbox_inches="tight")
        plt.close(fig)

        # Swapping the axes gives the user-requested VERTICAL true-value line.
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), constrained_layout=True)
        for ax, last, title in zip(axes, [4999, 100], ["Complete run", "First 100 iterations"]):
            selection = curves.iteration <= last
            t = curves.loc[selection, "iteration"].to_numpy()
            ax.plot(failed_rho[selection], t, color=COLORS["failed"],
                    lw=1.6, label="Failed random start", drawstyle="steps-pre")
            ax.plot(working_rho[selection], t, color=COLORS["working"],
                    lw=1.7, label=r"Working start: $u_0=(1,0,0,0)$", drawstyle="steps-pre")
            add_references(ax, vertical=True)
            format_measure(ax, vertical=True)
            ax.set(xlabel=measure_label, ylabel="Evaluated iteration (0 = initial direction)",
                   title=title, ylim=(0, last))
            ax.grid(alpha=0.2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=2,
                   frameon=False, fontsize=9)
        fig.suptitle(
            "Iteration versus stationarity measure | vertical line marks the true value rho = 0.5\n"
            "Tracked best-width coefficients | measure axis: linear from 0 to 1, logarithmic above 1",
            fontsize=12,
        )
        fig.savefig(OUT / "iteration_vs_stationarity_vertical_reference.pdf", bbox_inches="tight")
        fig.savefig(OUT / "iteration_vs_stationarity_vertical_reference.png", dpi=170, bbox_inches="tight")
        plt.close(fig)

    metadata = dict(
        source=str(SOURCE / "iteration_history.csv"),
        measure="spectral radius of companion matrix of the best-width coefficient estimate",
        true_phi=configuration["true_phi"], true_spectral_radius=true_rho,
        true_companion_eigenvalues=[dict(real=float(z.real), imag=float(z.imag)) for z in true_eigenvalues],
        failed_initial_direction=configuration["initial_directions"]["Failed random start"],
        working_initial_direction=configuration["initial_directions"]["Zero-start control"],
        failed_final_rho=float(failed_rho[-1]), working_final_rho=float(working_rho[-1]),
        iterations=len(curves), restart_applied=False,
        measure_axis="symlog, linear threshold 1, zero included without clipping",
        validation="failed rho matches saved audit; true rho=0.5; both runs aligned and finite",
    )
    (OUT / "stationarity_iteration_plot_settings.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"True rho: {true_rho:.10f}; failed final: {failed_rho[-1]:.8f}; working final: {working_rho[-1]:.8f}")
    print("Saved both axis orientations as PDF and PNG, with all-iteration data and settings.")


if __name__ == "__main__":
    main()
