from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.axes import Axes


##########################
### Perron vector plot ###
##########################

def plot_perron_vector(
    h_grid: np.ndarray,
    vector: np.ndarray,
    beta: float = 2.0,  # ### CHANGED
    plot_type: str = "perron",
        ax: Axes | None = None,
        label: str | None = None,
        maximum_label: str | None = None,
) -> Axes:
    vector = np.asarray(vector, dtype=float)
    if plot_type == "perron":
        plotted_values = vector
        y_label = "Perron-vector component"
    elif plot_type == "doob":
        plotted_values = vector**2
        plotted_values /= np.sum(plotted_values)
        y_label = "Doob stationary probability"
    else:
        raise ValueError('plot_type must be "perron" or "doob".')

    maximum_index = int(np.argmax(plotted_values))
    standalone = ax is None
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(h_grid, plotted_values, marker="o", label=label)
    ax.axvline(
        h_grid[maximum_index],
        color="red",
        linestyle="--",
        label=maximum_label,
    )
    exponent = 1.0 - beta / 2.0  # ### CHANGED
    if np.isclose(exponent, 0.0, rtol=0.0, atol=1e-12):  # ### CHANGED
        ax.set_xscale("log")  # ### CHANGED
    elif np.isclose(beta, 0.0, rtol=0.0, atol=1e-12):  # ### CHANGED
        ax.set_xscale("linear")  # ### CHANGED
    else:  # ### CHANGED
        def forward(h: np.ndarray) -> np.ndarray:  # ### CHANGED
            h = np.asarray(h, dtype=float)  # ### CHANGED
            return (h**exponent - 1.0) / exponent  # ### CHANGED

        def inverse(value: np.ndarray) -> np.ndarray:  # ### CHANGED
            value = np.asarray(value, dtype=float)  # ### CHANGED
            return (1.0 + exponent * value) ** (1.0 / exponent)  # ### CHANGED

        ax.set_xscale(  # ### CHANGED
            "function",  # ### CHANGED
            functions=(forward, inverse),  # ### CHANGED
        )  # ### CHANGED
    ax.set_xlabel("Bandwidth h")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.3)

    if standalone:
        plt.show()

    return ax


###################################
### Bandwidth-distribution plot ###
###################################

def plot_bandwidth_distributions(
    h_grid: np.ndarray,
    distributions: Mapping[str, np.ndarray] | np.ndarray,
    reference_lines: Sequence[tuple[str, float, str]] = (),
    colors: Mapping[str, str] | None = None,
    opacity: float | None = None,
    tick_count: int = 7,
    y_max: float | None = None,
    width: int = 1050,
    height: int = 550,
    legend_title: str | None = None,
) -> go.Figure:
    """Plot discrete bandwidth-selection distributions as Plotly bars.

    The bars are positioned by grid index, so candidates are equally spaced on
    the displayed axis whenever ``h_grid`` is linear in ``psi_beta``.  The
    corresponding bandwidth values are retained as tick labels and hover data.
    Every supplied count or weight vector is normalized to relative frequency.

    ``reference_lines`` contains ``(label, grid_index, color)`` triples.
    """
    h_grid = np.asarray(h_grid, dtype=float)
    if h_grid.ndim != 1 or len(h_grid) < 2:
        raise ValueError("h_grid must be a one-dimensional array of length two or more.")
    if not np.all(np.isfinite(h_grid)) or np.any(h_grid <= 0.0):
        raise ValueError("h_grid must contain only finite, positive values.")

    if isinstance(distributions, Mapping):
        distribution_items = list(distributions.items())
    else:
        distribution_items = [("selected bandwidth", distributions)]
    if not distribution_items:
        raise ValueError("At least one bandwidth distribution is required.")

    relative_frequencies: dict[str, np.ndarray] = {}
    for label, values in distribution_items:
        values = np.asarray(values, dtype=float)
        if values.shape != h_grid.shape:
            raise ValueError(
                f'Distribution "{label}" must have the same length as h_grid.'
            )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(
                f'Distribution "{label}" must contain finite, nonnegative values.'
            )
        total = float(np.sum(values))
        if total <= 0.0:
            raise ValueError(f'Distribution "{label}" must have positive total mass.')
        relative_frequencies[str(label)] = values / total

    if isinstance(tick_count, (bool, np.bool_)) or int(tick_count) != tick_count:
        raise ValueError("tick_count must be an integer.")
    tick_count = int(tick_count)
    if tick_count < 2:
        raise ValueError("tick_count must be at least two.")

    number_of_distributions = len(relative_frequencies)
    if opacity is None:
        if number_of_distributions == 1:
            opacity = 0.85
        elif number_of_distributions <= 3:
            opacity = 0.32
        else:
            opacity = 0.24
    opacity = float(opacity)
    if not np.isfinite(opacity) or not 0.0 < opacity <= 1.0:
        raise ValueError("opacity must lie in (0, 1].")

    if colors is None:
        colors = {}
    default_colors = [
        "#1f77b4"
        if number_of_distributions == 1
        else f"hsl({360.0 * position / number_of_distributions:.1f},65%,45%)"
        for position in range(number_of_distributions)
    ]

    reference_lines = list(reference_lines)
    show_bar_legend = number_of_distributions > 1 or bool(reference_lines)
    figure = go.Figure()
    maximum_relative_frequency = 0.0

    for (label, frequencies), default_color in zip(
        relative_frequencies.items(), default_colors
    ):
        nonzero_indices = np.flatnonzero(frequencies)
        maximum_relative_frequency = max(
            maximum_relative_frequency,
            float(np.max(frequencies)),
        )
        figure.add_trace(
            go.Bar(
                x=nonzero_indices,
                y=frequencies[nonzero_indices],
                customdata=h_grid[nonzero_indices],
                width=0.85,
                marker=dict(
                    color=colors.get(label, default_color),
                    line=dict(color="white", width=0.35),
                ),
                opacity=opacity,
                name=label,
                showlegend=show_bar_legend,
                hovertemplate=(
                    f"{label}"
                    "<br>grid index=%{x}"
                    "<br>bandwidth h=%{customdata:.6g}"
                    "<br>relative frequency=%{y:.1%}<extra></extra>"
                ),
            )
        )

    if y_max is None:
        y_max = 1.05 * maximum_relative_frequency
    y_max = float(y_max)
    if not np.isfinite(y_max) or y_max <= 0.0:
        raise ValueError("y_max must be finite and positive.")
    if y_max < maximum_relative_frequency:
        raise ValueError("y_max cannot be smaller than a plotted relative frequency.")

    for label, grid_index, color in reference_lines:
        grid_index = float(grid_index)
        if not np.isfinite(grid_index) or not 0.0 <= grid_index <= len(h_grid) - 1:
            raise ValueError("Reference-line indices must lie on the bandwidth grid.")
        figure.add_trace(
            go.Scatter(
                x=[grid_index, grid_index],
                y=[0.0, y_max],
                mode="lines",
                line=dict(color=color, width=2, dash="dash"),
                name=label,
                hoverinfo="skip",
            )
        )

    tick_indices = np.unique(
        np.linspace(0, len(h_grid) - 1, tick_count, dtype=int)
    )
    tick_labels = [f"{h_grid[index]:.3g}" for index in tick_indices]
    figure.update_xaxes(
        title_text="selected bandwidth h",
        range=[-0.5, len(h_grid) - 0.5],
        tickmode="array",
        tickvals=tick_indices,
        ticktext=tick_labels,
    )
    figure.update_yaxes(
        title_text="relative frequency",
        range=[0.0, y_max],
        tickformat=".0%",
    )
    figure.update_layout(
        width=int(width),
        height=int(height),
        template="plotly_white",
        barmode="overlay",
        legend=dict(
            title=legend_title,
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.18,
            yanchor="top",
        ),
        margin=dict(t=40, b=150),
    )
    return figure
