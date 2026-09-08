"""Plotting utilities for autoregressive processes and their embeddings."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

__all__ = ["plot_time_series", "plot_embedding_k2", "plot_embedding_k3"]

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
