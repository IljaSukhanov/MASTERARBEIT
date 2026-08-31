"""Plotting helpers for the KDE experiments."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_perron_vector(
    h_grid: np.ndarray,
    vector: np.ndarray,
) -> None:
    """Plot a Perron vector and mark its maximum bandwidth."""
    maximum_index = int(np.argmax(vector))

    plt.plot(h_grid, vector, marker="o")
    plt.axvline(
        h_grid[maximum_index],
        color="red",
        linestyle="--",
    )
    plt.xscale("log")
    plt.xlabel("Bandwidth h")
    plt.ylabel("Perron-vector component")
    plt.grid(alpha=0.3)
    plt.show()
