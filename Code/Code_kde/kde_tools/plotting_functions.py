from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


##########################
### Perron vector plot ###
##########################

def plot_perron_vector(
        h_grid: np.ndarray,
        vector: np.ndarray,
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
    ax.set_xscale("log")
    ax.set_xlabel("Bandwidth h")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.3)

    if standalone:
        plt.show()

    return ax
