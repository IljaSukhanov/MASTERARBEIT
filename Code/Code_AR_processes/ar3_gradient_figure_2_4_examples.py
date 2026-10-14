"""Ten Figure 2.4-style examples using the project's gradient width solver.

The first ten saved AR(3) processes are used without selecting by outcome.
Warm-start gradient fits are rerun with their original settings. The original
descent-chain selector provides the red marker.
Output is a ten-page PDF plus individual PNGs, data, and preview contact sheets.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import residuez
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Code.Code_AR_processes.ar_tools.descent import batched_geometric_width_curves
from Code.Code_AR_processes.ar_tools.functions import ar_impulse_response, sliding_window_point_cloud
from Code.Code_AR_processes.ar_tools.population_widths import population_ar3_widths
from Code.Code_AR_processes.ar_tools.breakpoint_selectors import descent_chain

NAME = "ar3_gradient_figure_2_4_examples"
AR_DIR = Path(__file__).resolve().parent
SOURCE = AR_DIR / "outputs/csv/ar3_order_selection_warm_starts"
POP_SOURCE = AR_DIR / "outputs/csv/ar3_population_breakpoint_selectors"
DATA_DIR = AR_DIR / "outputs/csv" / NAME
IMAGE_DIR = AR_DIR / "outputs/images" / NAME
PDF_PATH = ROOT / "output/pdf/ar3_gradient_figure_2_4_ten_examples.pdf"


def fit_examples():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings = json.loads((SOURCE / "configuration.json").read_text())
    configuration = {
        "source_configuration": settings,
        "replications": list(range(1, 11)),
        "selection_of_examples": "First ten runs, without selection by performance",
        "optimizer": "batched_geometric_width_curves, padded previous direction",
        "warm_start_previous_direction": True,
        "reference_figure": str(ROOT / "Abschlussarbeit_2021/LaTeX/images/AR3_experiment.pdf"),
        "reference_notebook": str(AR_DIR / "ar_thesis_figures_2_1_to_2_3.ipynb"),
        "figure_selector": "Original descent chain on raw widths",
        "population_direction_source": "Recomputed global truncated zonotope minima in ar_tools/population_widths.py",
        "population_direction_truncation_bound": 1e-8,
        "directional_population_width_truncation_target": 1e-9,
        "source_hashes": {name: hashlib.sha256((AR_DIR / name).read_bytes()).hexdigest()
                          for name in ["ar_tools/descent.py", "ar_tools/functions.py"]},
    }
    (DATA_DIR / "configuration.json").write_text(json.dumps(configuration, indent=2) + "\n")
    rows, summaries = [], []
    errors = {"maximum_cached_width_difference": 0.0, "maximum_cached_direction_difference": 0.0,
              "maximum_width_reconstruction_error": 0.0, "maximum_direction_norm_error": 0.0,
              "maximum_directional_population_tail_bound": 0.0,
              "maximum_population_cache_coefficient_roundoff_difference": 0.0}
    with threadpool_limits(limits=1):
        for rep in range(1, 11):
            series = pd.read_csv(SOURCE / "processes" / f"process_{rep:03d}.csv",
                                 float_precision="round_trip").X_t.to_numpy()
            saved = pd.read_csv(SOURCE / "replications" / f"replication_{rep:03d}.csv",
                                float_precision="round_trip")
            saved = saved[saved.variant == "Padded warm start"].sort_values("k")
            phi = np.array(json.loads(saved.coefficients.iloc[0]))
            _, widths, _, directions = batched_geometric_width_curves(
                series, 10, seed=settings["seed"], replication=rep - 1,
                starts=settings["starts"], iterations=settings["iterations"],
                initial_step=settings["initial_step"], geometric_decay=settings["geometric_decay"],
                return_directions=True, warm_start_previous_direction=True,
            )
            errors["maximum_cached_width_difference"] = max(errors["maximum_cached_width_difference"],
                float(np.max(np.abs(widths - saved.minimum_width.to_numpy()))))
            population = pd.read_csv(POP_SOURCE / "replications" / f"replication_{rep:03d}.csv",
                                     float_precision="round_trip").sort_values("k")
            cached_phi = np.array(json.loads(population.coefficients.iloc[0]))
            np.testing.assert_allclose(phi, cached_phi, rtol=0, atol=1e-14)
            errors["maximum_population_cache_coefficient_roundoff_difference"] = max(
                errors["maximum_population_cache_coefficient_roundoff_difference"], float(np.abs(phi - cached_phi).max()))
            population_fit = population_ar3_widths(phi, max_k=10)
            residue, poles, direct = residuez([1.0], np.r_[1.0, -phi])
            rho, constant = float(np.max(np.abs(poles))), float(np.abs(residue).sum())
            assert rho < 1 and len(direct) == 0
            history_count = max(10, int(np.ceil(np.log(1e-9 * (1 - rho)
                / (2 * np.sqrt(10) * constant)) / np.log(rho))) + 9)
            psi = ar_impulse_response(phi, history_count - 1)
            order = descent_chain(widths)[0]
            q = order + 1
            summaries.append({"replication": rep, "phi_1": phi[0], "phi_2": phi[1], "phi_3": phi[2],
                              "true_order": 3, "selected_order": order, "selected_dimension": q,
                              "selected_width": float(widths[q - 1]),
                              "descent_chain_raw_order": descent_chain(widths)[0],
                              "descent_chain_log_order": descent_chain(np.log(widths))[0],
                              "population_plateau": float(2 / np.linalg.norm(np.r_[1., -phi]))})
            for k, (width, direction) in enumerate(zip(widths, directions), 1):
                cloud = sliding_window_point_cloud(series, k)
                reconstructed = float(np.ptp(cloud @ direction))
                original_u = saved.iloc[k - 1][[f"u_{j}" for j in range(k)]].to_numpy(dtype=float)
                errors["maximum_width_reconstruction_error"] = max(errors["maximum_width_reconstruction_error"],
                                                                     abs(reconstructed - width))
                errors["maximum_cached_direction_difference"] = max(errors["maximum_cached_direction_difference"],
                                                                      float(np.abs(direction - original_u).max()))
                errors["maximum_direction_norm_error"] = max(errors["maximum_direction_norm_error"],
                                                               abs(float(np.linalg.norm(direction)) - 1))
                if k <= 3:
                    u_pop = population_fit["directions"][k - 1].copy()
                else:
                    u_pop = np.r_[1., -phi, np.zeros(k - 4)]
                u_pop /= np.linalg.norm(u_pop)
                orange = float(2 * np.abs(np.convolve(psi, direction)[:history_count]).sum())
                tail = float(2 * np.sqrt(k) * constant * rho ** (history_count - k + 1) / (1 - rho))
                errors["maximum_directional_population_tail_bound"] = max(
                    errors["maximum_directional_population_tail_bound"], tail)
                assert orange + tail >= width - 1e-10
                row = {"replication": rep, "k": k, "empirical_fitted_width": float(width),
                       "population_at_empirical_direction": orange,
                       "population_directional_tail_bound": tail,
                       "empirical_at_population_direction": float(np.ptp(cloud @ u_pop)),
                       "population_minimum_width": float(population_fit["widths"][k - 1]
                                                          + population_fit["tail_bounds"][k - 1] / 2),
                       "population_plateau": summaries[-1]["population_plateau"] if k >= 4 else np.nan,
                       "coefficients": json.dumps(phi.tolist()), "population_history_count": history_count}
                row.update({f"gradient_u_{j}": float(direction[j]) if j < k else np.nan for j in range(10)})
                row.update({f"population_u_{j}": float(u_pop[j]) if j < k else np.nan for j in range(10)})
                rows.append(row)
            print(f"Gradient example {rep}/10 reproduced", flush=True)
    assert errors["maximum_cached_width_difference"] < 1e-12
    assert errors["maximum_cached_direction_difference"] < 1e-12
    assert errors["maximum_directional_population_tail_bound"] <= 1e-9 * (1 + 1e-10)
    curves, summary = pd.DataFrame(rows), pd.DataFrame(summaries)
    curves.to_csv(DATA_DIR / "figure_data.csv", index=False)
    summary.to_csv(DATA_DIR / "selection_summary.csv", index=False)
    errors.update({"processes": 10, "fitted_widths_and_directions": 100,
                  "monotone_curves_within_1e-12": int(sum(np.all(np.diff(g.empirical_fitted_width) <= 1e-12)
                                                        for _, g in curves.groupby("replication"))),
                  "correct_descent_chain_selections": int((summary.selected_order == 3).sum())})
    (DATA_DIR / "verification.json").write_text(json.dumps(errors, indent=2) + "\n")
    return curves, summary


def make_figures(curves, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    with PdfPages(PDF_PATH, metadata={"Title": "Ten AR(3) gradient-descent width examples",
                                     "Subject": "Figure 2.4-style plots on the first ten saved processes"}) as pdf:
        for rep, g in curves.groupby("replication"):
            result = summary[summary.replication == rep].iloc[0]
            fig, ax = plt.subplots(figsize=(10.5, 6.6))
            fig.subplots_adjust(left=.085, right=.98, top=.90, bottom=.26)
            ax.plot(g.k, g.empirical_fitted_width, "o-", color="#1764ab", lw=2.2, ms=6,
                    label=r"Empirical fitted width $\widehat w_k(\widehat u_k)$", zorder=4)
            plateau = g[g.k >= 4]
            ax.plot(plateau.k, plateau.population_plateau, "--", color="black", lw=2.5,
                    label=r"Population minimum $m_k=m^*$, $k\geq4$", zorder=6)
            ax.plot(g.k, g.population_at_empirical_direction, "s--", color="#e66101", lw=2, ms=5,
                    label=r"Population width $w_k(\widehat u_k)$", zorder=3)
            ax.plot(g.k, g.empirical_at_population_direction, "x-", color="#16834a", lw=2, ms=7,
                    label=r"Empirical width $\widehat w_k(u_k^{\min})$", zorder=5)
            ax.axvline(result.selected_dimension, color="#c51b2b", ls="-.", lw=1.7,
                       label=r"Selected descent-chain breakpoint", zorder=1)
            ax.scatter([result.selected_dimension], [result.selected_width],
                       s=120, color="#c51b2b", zorder=7)
            ax.axvline(4, color="0.55", ls=":", lw=1.2, label="True breakpoint k = 4", zorder=0)
            ax.text(.975, .965, rf"$\phi_1={result.phi_1:.3f},\ \phi_2={result.phi_2:.3f},\ \phi_3={result.phi_3:.3f}$"
                    + "\n" + r"$T=2{,}000,\ K=10$" + "\n"
                    + rf"$\widehat q_{{DC}}={int(result.selected_dimension)},\ \widehat p_{{DC}}={int(result.selected_order)}$",
                    transform=ax.transAxes, ha="right", va="top", fontsize=11,
                    bbox={"boxstyle": "round,pad=.45", "facecolor": "white", "edgecolor": "0.75", "alpha": .95})
            ax.set(title=f"AR(3), run {rep:02d}: population and empirical widths",
                   xlabel=r"Embedding dimension $k$", ylabel="Directional width", xticks=np.arange(1, 11),
                   xlim=(.75, 10.45))
            ax.grid(axis="y", color=".88", lw=.8)
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.53, .055),
                       ncol=2, frameon=False, fontsize=10.5)
            fig.text(.5, .022, "Padded-start projected gradient descent: 3 starts, 500 iterations, step 0.2 × 0.99ⁿ",
                     ha="center", fontsize=9.5, color=".35")
            pdf.savefig(fig)
            fig.savefig(IMAGE_DIR / f"ar3_gradient_example_{rep:02d}.png", dpi=180)
            plt.close(fig)
    print(f"Saved ten-page PDF: {PDF_PATH}", flush=True)
    make_previews()


def make_previews():
    """Contact-sheet previews of the ten complete exported scientific plots."""
    from PIL import Image, ImageDraw, ImageFont

    card_width, card_height, gap = 900, 566, 18
    for first in [1, 6]:
        preview = Image.new("RGB", (2 * card_width + 3 * gap, 3 * card_height + 4 * gap), "white")
        for index, rep in enumerate(range(first, first + 5)):
            with Image.open(IMAGE_DIR / f"ar3_gradient_example_{rep:02d}.png") as page:
                page = page.convert("RGB")
                page.thumbnail((card_width, card_height), Image.Resampling.LANCZOS)
                x = gap + (index % 2) * (card_width + gap)
                y = gap + (index // 2) * (card_height + gap)
                preview.paste(page, (x, y))
        draw = ImageDraw.Draw(preview)
        font = ImageFont.load_default(size=26)
        draw.multiline_text((card_width + 2 * gap + 55, 2 * (card_height + gap) + 150),
            f"AR(3) runs {first:02d}-{first + 4:02d}\n\nFull-size figures in the ten-page PDF.\n\nFirst ten saved runs, in order.\nPadded-start gradient descent.\nRed marker: original descent chain.",
            fill="#444444", font=font, spacing=12)
        preview.save(IMAGE_DIR / f"ar3_gradient_examples_{first:02d}_to_{first + 4:02d}.png")


if __name__ == "__main__":
    data, selections = fit_examples()
    make_figures(data, selections)
