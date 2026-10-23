"""Plotting utilities for autoregressive processes and their embeddings."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .breakpoint_selectors import descent_chain
from .experiment_settings import (
    AR_DIRECTORY, GRADIENT_IMAGE_DIRECTORY, GRADIENT_PDF_PATH,
    POPULATION_IMAGE_DIRECTORY, POPULATION_PDF_DIRECTORY,
    WARM_START_IMAGE_DIRECTORY, WARM_START_PDF_DIRECTORY, WARM_START_VARIANTS,
)

__all__ = [
    "plot_time_series",
    "plot_embedding_k2",
    "plot_embedding_k3",
    "save_figure",
    "plot_ar3_gradient_examples",
    "make_ar3_gradient_previews",
    "save_ar3_warm_start_figure",
    "plot_ar3_warm_start_results",
    "plot_ar3_population_selector_results",
]

plt.style.use("seaborn-v0_8-whitegrid")


def plot_time_series(
    x,
    phi,
    max_plot_points=10000,
    *,
    innovation_label=r"Uniform$[-1,1]$ innovations",
):
    fig, ax = plt.subplots(figsize=(12, 3.6))
    plot_indices = np.linspace(
        0,
        len(x) - 1,
        min(len(x), max_plot_points),
        dtype=int,
    )
    ax.plot(plot_indices, x[plot_indices], lw=0.9, color="#315a9a")
    ax.axhline(
        np.mean(x),
        color="black",
        ls="--",
        lw=1,
        alpha=0.7,
        label="sample mean",
    )
    ax.set(
        title=f"AR({len(phi)}) process — {innovation_label}",
        xlabel="time $t$",
        ylabel="$X_t$",
    )
    robust_limits = np.quantile(x, [0.005, 0.995])
    if np.all(np.isfinite(robust_limits)) and robust_limits[0] < robust_limits[1]:
        margin = 0.08 * (robust_limits[1] - robust_limits[0])
        ax.set_ylim(robust_limits[0] - margin, robust_limits[1] + margin)
    ax.legend()
    plt.show()


def plot_embedding_k2(
    cloud,
    phi,
    intercept,
    innovation_half_width=1.0,
    *,
    innovation_bounds=None,
):
    if innovation_bounds is None:
        innovation_lower = -innovation_half_width
        innovation_upper = innovation_half_width
    else:
        innovation_bounds = np.asarray(innovation_bounds, dtype=float)
        if (
            innovation_bounds.shape != (2,)
            or not np.all(np.isfinite(innovation_bounds))
            or innovation_bounds[0] >= innovation_bounds[1]
        ):
            raise ValueError(
                "innovation_bounds must contain finite lower and upper bounds"
            )
        innovation_lower, innovation_upper = innovation_bounds

    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    time_color = np.arange(len(cloud))
    points = ax.scatter(
        cloud[:, 0],
        cloud[:, 1],
        c=time_color,
        cmap="viridis",
        s=10,
        alpha=0.55,
        label=r"$Y_t^{(2)}$",
    )
    fig.colorbar(points, ax=ax, label="window time index $t$")

    lo, hi = np.quantile(cloud[:, 0], [0.005, 0.995])
    grid = np.linspace(lo, hi, 300)
    if len(phi) <= 1:
        middle = intercept + phi[0] * grid
        ax.plot(
            grid,
            middle,
            color="black",
            lw=2.2,
            label="mean-innovation AR hyperplane",
        )
        ax.plot(
            grid,
            middle + innovation_upper,
            color="#d1495b",
            lw=2,
            ls="--",
            label="support boundary hyperplanes",
        )
        ax.plot(
            grid,
            middle + innovation_lower,
            color="#d1495b",
            lw=2,
            ls="--",
        )
    else:
        ax.text(
            0.02,
            0.98,
            "Exact AR hyperplane unavailable: k=2 requires p≤1",
            transform=ax.transAxes,
            va="top",
            color="#a33",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="#a33"),
        )

    ax.set(
        title=r"Forward sliding-window points $Y_t^{(2)}$",
        xlabel="$X_t$",
        ylabel="$X_{t+1}$",
    )
    x_limits = np.quantile(cloud[:, 0], [0.005, 0.995])
    y_limits = np.quantile(cloud[:, 1], [0.005, 0.995])
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best")
    plt.show()


def plot_embedding_k3(
    cloud,
    phi,
    intercept,
    innovation_half_width=1.0,
    *,
    innovation_bounds=None,
):
    import plotly.graph_objects as go

    if innovation_bounds is None:
        innovation_lower = -innovation_half_width
        innovation_upper = innovation_half_width
    else:
        innovation_bounds = np.asarray(innovation_bounds, dtype=float)
        if (
            innovation_bounds.shape != (2,)
            or not np.all(np.isfinite(innovation_bounds))
            or innovation_bounds[0] >= innovation_bounds[1]
        ):
            raise ValueError(
                "innovation_bounds must contain finite lower and upper bounds"
            )
        innovation_lower, innovation_upper = innovation_bounds

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=cloud[:, 0],
            y=cloud[:, 1],
            z=cloud[:, 2],
            mode="markers",
            name="Y<sub>t</sub><sup>(3)</sup>",
            marker=dict(
                size=2.5,
                color=np.arange(len(cloud)),
                colorscale="Viridis",
                opacity=0.55,
                colorbar=dict(title="time index t"),
            ),
        )
    )

    if len(phi) <= 2:
        q0 = np.quantile(cloud[:, 0], [0.01, 0.99])
        q1 = np.quantile(cloud[:, 1], [0.01, 0.99])
        g0, g1 = np.meshgrid(np.linspace(*q0, 28), np.linspace(*q1, 28))
        middle = intercept + phi[0] * g1
        if len(phi) >= 2:
            middle = middle + phi[1] * g0
        fig.add_trace(
            go.Surface(
                x=g0,
                y=g1,
                z=middle,
                name="mean-innovation AR hyperplane",
                showscale=False,
                opacity=0.34,
                colorscale=[[0, "#222222"], [1, "#222222"]],
                hovertemplate="mean-innovation AR hyperplane<extra></extra>",
            )
        )
        for offset, label in [
            (innovation_upper, "upper support boundary"),
            (innovation_lower, "lower support boundary"),
        ]:
            fig.add_trace(
                go.Surface(
                    x=g0,
                    y=g1,
                    z=middle + offset,
                    name=label,
                    showscale=False,
                    opacity=0.24,
                    colorscale=[[0, "#d1495b"], [1, "#d1495b"]],
                    hovertemplate=label + "<extra></extra>",
                )
            )
    else:
        fig.add_annotation(
            text="Exact AR hyperplane unavailable: k=3 requires p≤2",
            x=0.01,
            y=0.99,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color="#a33"),
            bgcolor="white",
        )

    fig.update_layout(
        title=(
            "Forward sliding-window points Y<sub>t</sub><sup>(3)</sup> "
            "— drag to rotate"
        ),
        width=900,
        height=700,
        scene=dict(
            xaxis_title="$X_t$",
            yaxis_title="$X_{t+1}$",
            zaxis_title="$X_{t+2}$",
            xaxis=dict(
                range=np.quantile(cloud[:, 0], [0.005, 0.995]).tolist()
            ),
            yaxis=dict(
                range=np.quantile(cloud[:, 1], [0.005, 0.995]).tolist()
            ),
            zaxis=dict(
                range=np.quantile(cloud[:, 2], [0.005, 0.995]).tolist()
            ),
            aspectmode="data",
        ),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, b=0, t=50),
    )
    fig.show()


def save_figure(figure, name, *, pdf_directory, image_directory):
    figure.savefig(Path(pdf_directory) / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(Path(image_directory) / f"{name}.png", dpi=160, bbox_inches="tight")


def plot_ar3_gradient_examples(curves, summary, *,
                               image_directory=GRADIENT_IMAGE_DIRECTORY,
                               pdf_path=GRADIENT_PDF_PATH):
    IMAGE_DIR, PDF_PATH = Path(image_directory), Path(pdf_path)
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
    make_ar3_gradient_previews(image_directory=IMAGE_DIR)


def make_ar3_gradient_previews(*, image_directory=GRADIENT_IMAGE_DIRECTORY):
    """Contact-sheet previews of the ten complete exported scientific plots."""
    IMAGE_DIR = Path(image_directory)
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


def save_ar3_warm_start_figure(fig, name, *, image_directory, pdf_directory):
    IMAGE_DIR, PDF_DIR = Path(image_directory), Path(pdf_directory)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMAGE_DIR / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(PDF_DIR / f"{name}.pdf", bbox_inches="tight")


def plot_ar3_warm_start_results(width_curves, summary, *,
                                image_directory=WARM_START_IMAGE_DIRECTORY,
                                pdf_directory=WARM_START_PDF_DIRECTORY, variants=WARM_START_VARIANTS):
    IMAGE_DIR, PDF_DIR = Path(image_directory), Path(pdf_directory)
    VARIANTS = variants
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 12, "axes.spines.top": False,
                         "axes.spines.right": False})
    labels = {
        "First difference": "First difference",
        "Second difference": "Second difference",
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
    save_ar3_warm_start_figure(fig, "selector_accuracy_comparison", image_directory=IMAGE_DIR, pdf_directory=PDF_DIR)
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
    save_ar3_warm_start_figure(fig, "example_width_curves", image_directory=IMAGE_DIR, pdf_directory=PDF_DIR)
    plt.close(fig)


def plot_ar3_population_selector_results(widths, summary, comparison, *,
                                         image_directory=POPULATION_IMAGE_DIRECTORY,
                                         pdf_directory=POPULATION_PDF_DIRECTORY):
    IMAGE_DIR, PDF_DIR = Path(image_directory), Path(pdf_directory)
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
