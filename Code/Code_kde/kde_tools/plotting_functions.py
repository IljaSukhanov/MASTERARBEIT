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
        marker_size: float = 3.5,
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

    ax.plot(
        h_grid,
        plotted_values,
        marker="o",
        markersize=marker_size,
        label=label,
    )
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


#####################################
### Bandwidth-grid stability plot ###
#####################################

def plot_bandwidth_grid_stability(
    grid_sizes: np.ndarray,
    selected_bandwidths: np.ndarray,
    selected_indices: np.ndarray,
    show_envelope_fits: bool = True,
    tick_values: np.ndarray | None = None,
    width: int = 1050,
    height: int = 550,
) -> go.Figure:
    """Plot how the selected bandwidth changes with bandwidth-grid size.

    If requested, upper and lower local peaks are fitted in log-bandwidth
    coordinates using ``log(h) = c + b / (N - 1)`` with a shared limiting
    value ``c``.  The fitted curves start at the first complete upper/lower
    peak pair so extrapolation does not compress the vertical scale.
    """
    grid_sizes = np.asarray(grid_sizes)
    selected_bandwidths = np.asarray(selected_bandwidths, dtype=float)
    selected_indices = np.asarray(selected_indices)

    if grid_sizes.ndim != 1 or len(grid_sizes) < 2:
        raise ValueError("grid_sizes must be a one-dimensional array of length two or more.")
    if np.any(grid_sizes < 2) or not np.all(np.isfinite(grid_sizes)):
        raise ValueError("grid_sizes must contain finite values of at least two.")
    if not np.all(grid_sizes == np.floor(grid_sizes)):
        raise ValueError("grid_sizes must contain integers.")
    grid_sizes = grid_sizes.astype(int)
    if np.any(np.diff(grid_sizes) <= 0):
        raise ValueError("grid_sizes must be strictly increasing.")

    if selected_bandwidths.shape != grid_sizes.shape:
        raise ValueError("selected_bandwidths must have the same shape as grid_sizes.")
    if not np.all(np.isfinite(selected_bandwidths)) or np.any(
        selected_bandwidths <= 0.0
    ):
        raise ValueError("selected_bandwidths must contain finite, positive values.")

    if selected_indices.shape != grid_sizes.shape:
        raise ValueError("selected_indices must have the same shape as grid_sizes.")
    if not np.all(np.isfinite(selected_indices)) or not np.all(
        selected_indices == np.floor(selected_indices)
    ):
        raise ValueError("selected_indices must contain finite integers.")
    selected_indices = selected_indices.astype(int)
    if np.any(selected_indices < 0) or np.any(selected_indices >= grid_sizes):
        raise ValueError("Each selected index must lie on its corresponding grid.")

    if not isinstance(show_envelope_fits, (bool, np.bool_)):
        raise ValueError("show_envelope_fits must be boolean.")

    normalized_indices = selected_indices / (grid_sizes - 1)
    customdata = np.column_stack((selected_indices, normalized_indices))

    middle_bandwidths = selected_bandwidths[1:-1]
    upper_peak_positions = np.flatnonzero(
        (middle_bandwidths > selected_bandwidths[:-2])
        & (middle_bandwidths >= selected_bandwidths[2:])
    ) + 1
    lower_peak_positions = np.flatnonzero(
        (middle_bandwidths <= selected_bandwidths[:-2])
        & (middle_bandwidths < selected_bandwidths[2:])
    ) + 1
    can_fit_envelopes = (
        show_envelope_fits
        and len(upper_peak_positions) > 0
        and len(lower_peak_positions) > 0
        and len(upper_peak_positions) + len(lower_peak_positions) >= 3
    )

    figure = go.Figure(
        go.Scatter(
            x=grid_sizes,
            y=selected_bandwidths,
            customdata=customdata,
            mode="lines+markers",
            line=dict(color="#1f77b4", width=2.5),
            marker=dict(size=7),
            name="selected bandwidth",
            showlegend=can_fit_envelopes,
            hovertemplate=(
                "N=%{x}"
                "<br>chosen bandwidth h=%{y:.6g}"
                "<br>chosen grid index=%{customdata[0]:.0f}"
                "<br>normalized index=%{customdata[1]:.6f}"
                "<extra></extra>"
            ),
        )
    )

    if can_fit_envelopes:
        upper_inverse_spacing = 1.0 / (
            grid_sizes[upper_peak_positions] - 1.0
        )
        lower_inverse_spacing = 1.0 / (
            grid_sizes[lower_peak_positions] - 1.0
        )
        envelope_design = np.vstack(
            (
                np.column_stack(
                    (
                        np.ones_like(upper_inverse_spacing),
                        upper_inverse_spacing,
                        np.zeros_like(upper_inverse_spacing),
                    )
                ),
                np.column_stack(
                    (
                        np.ones_like(lower_inverse_spacing),
                        np.zeros_like(lower_inverse_spacing),
                        lower_inverse_spacing,
                    )
                ),
            )
        )
        envelope_response = np.log(
            np.concatenate(
                (
                    selected_bandwidths[upper_peak_positions],
                    selected_bandwidths[lower_peak_positions],
                )
            )
        )
        limit_log_h, upper_slope, lower_slope = np.linalg.lstsq(
            envelope_design,
            envelope_response,
            rcond=None,
        )[0]

        fit_start_position = max(
            int(upper_peak_positions[0]),
            int(lower_peak_positions[0]),
        )
        fit_grid_sizes = grid_sizes[fit_start_position:]
        inverse_spacing = 1.0 / (fit_grid_sizes - 1.0)
        upper_fit = np.exp(limit_log_h + upper_slope * inverse_spacing)
        lower_fit = np.exp(limit_log_h + lower_slope * inverse_spacing)

        figure.add_trace(
            go.Scatter(
                x=fit_grid_sizes,
                y=upper_fit,
                mode="lines",
                line=dict(color="#d62728", width=2, dash="dash"),
                name="upper-envelope fit",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=fit_grid_sizes,
                y=lower_fit,
                mode="lines",
                line=dict(color="#2ca02c", width=2, dash="dash"),
                name="lower-envelope fit",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=grid_sizes[upper_peak_positions],
                y=selected_bandwidths[upper_peak_positions],
                mode="markers",
                marker=dict(color="#d62728", size=8),
                name="upper peaks",
                showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=grid_sizes[lower_peak_positions],
                y=selected_bandwidths[lower_peak_positions],
                mode="markers",
                marker=dict(color="#2ca02c", size=8),
                name="lower peaks",
                showlegend=False,
            )
        )

    if tick_values is None:
        tick_positions = np.unique(
            np.linspace(0, len(grid_sizes) - 1, min(10, len(grid_sizes)), dtype=int)
        )
        tick_values = grid_sizes[tick_positions]
    else:
        tick_values = np.asarray(tick_values)
        if tick_values.ndim != 1 or not np.all(np.isfinite(tick_values)):
            raise ValueError("tick_values must be a one-dimensional array of finite values.")

    figure.update_xaxes(
        title_text="number of bandwidth candidates N",
        tickmode="array",
        tickvals=tick_values,
    )
    figure.update_yaxes(
        title_text="chosen bandwidth h",
        tickformat=".4f",
    )
    figure.update_layout(
        width=int(width),
        height=int(height),
        template="plotly_white",
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=1.02,
            yanchor="bottom",
        ),
        margin=dict(t=80, b=100),
    )
    return figure
