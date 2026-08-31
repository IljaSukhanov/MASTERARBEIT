"""Plotting helpers for the KDE experiments."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def plot_perron_vector(
    h_grid: np.ndarray,
    vector: np.ndarray,
    ax: Axes | None = None,
    label: str | None = None,
    maximum_label: str | None = None,
) -> Axes:
    """Plot a Perron vector and mark its maximum bandwidth."""
    maximum_index = int(np.argmax(vector))
    standalone = ax is None
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(h_grid, vector, marker="o", label=label)
    ax.axvline(
        h_grid[maximum_index],
        color="red",
        linestyle="--",
        label=maximum_label,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Bandwidth h")
    ax.set_ylabel("Perron-vector component")
    ax.grid(alpha=0.3)

    if standalone:
        plt.show()

    return ax
