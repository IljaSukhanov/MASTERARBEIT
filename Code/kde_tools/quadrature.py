"""Shared trapezoidal integration for KDE numerical checks."""

from __future__ import annotations

import numpy as np


def trapezoidal_integral(values: np.ndarray, x_grid: np.ndarray, axis: int=-1) -> np.ndarray:
    trapezoid = getattr(np, 'trapezoid', None)
    if trapezoid is None:
        trapezoid = np.trapz
    return trapezoid(y=values, x=x_grid, axis=axis)
