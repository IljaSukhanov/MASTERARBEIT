"""Shared scaled-Beta innovation sampling for the AR code.

With no endpoints, use Z = L * (U - alpha / (alpha + beta)), L=2 by
default. Explicit endpoints instead use Z = a + (b - a) * U, which need
not have mean zero. L is the support width, never an extra multiplier.

Reference: NIST/SEMATECH e-Handbook, section 1.3.6.6.17 (density and mean),
https://itl.nist.gov/div898/handbook/eda/section3/eda366h.htm.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["beta_innovation_bounds", "sample_beta_innovations"]


def _finite_scalar(value: float, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_, str, bytes)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite scalar")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not np.isfinite(number) or (positive and number <= 0):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"{name} must be a {qualifier} scalar")
    return number


def beta_innovation_bounds(
    alpha: float = 1.0,
    beta: float = 1.0,
    a: float | None = None,
    b: float | None = None,
    L: float | None = None,
) -> tuple[float, float]:
    """Resolve support endpoints for centered or fixed-support innovations.

    Omit both a and b for centered innovations of width L (default 2):
    a = -L * alpha / (alpha + beta), b = L * beta / (alpha + beta).
    Supply both a and b for a fixed interval; its mean may be nonzero.
    If L is also supplied, it must agree with b - a. Partial endpoints,
    invalid shapes, and inconsistent or nonfinite widths are rejected.

    None distinguishes an omitted L from an explicitly requested width.
    """
    alpha = _finite_scalar(alpha, "alpha", positive=True)
    beta = _finite_scalar(beta, "beta", positive=True)
    width = None if L is None else _finite_scalar(L, "L", positive=True)

    if (a is None) != (b is None):
        raise ValueError("Supply both a and b, or omit both for centered innovations")
    if a is None:
        width = 2.0 if width is None else width
        # Scale before adding to avoid overflow in alpha + beta.
        shape_scale = max(alpha, beta)
        scaled_alpha, scaled_beta = alpha / shape_scale, beta / shape_scale
        shape_sum = scaled_alpha + scaled_beta
        lower = -width * (scaled_alpha / shape_sum)
        upper = width * (scaled_beta / shape_sum)
    else:
        lower = _finite_scalar(a, "a")
        upper = _finite_scalar(b, "b")

    support_width = upper - lower
    if not lower < upper or not np.isfinite(support_width):
        raise ValueError("Require finite ordered endpoints a < b and a finite width")
    if width is not None and not np.isclose(
        support_width, width, rtol=1e-12, atol=0.0,
    ):
        raise ValueError("L must equal b - a; the endpoints and width are not independent")
    return lower, upper


def sample_beta_innovations(
    size: int | tuple[int, ...] | None = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    a: float | None = None,
    b: float | None = None,
    L: float | None = None,
    rng: np.random.Generator | int | None = None,
) -> NDArray[np.float64] | float:
    """Sample scaled-Beta innovations; defaults are Uniform[-1, 1].

    Without a,b, the distribution has mean zero and width L (default 2).
    With explicit a,b, sample on exactly that interval without recentering.
    See beta_innovation_bounds for endpoint and width validation. Centering
    uses the population mean, not the sample mean: draws stay independent.

    size and rng follow NumPy's Generator conventions. A seed reproduces
    a draw; an existing Generator is advanced in place. The alpha=beta=1
    case uses Generator.uniform to preserve the existing seeded AR paths.
    """
    lower, upper = beta_innovation_bounds(alpha, beta, a, b, L)
    generator = np.random.default_rng(rng)
    if alpha == 1.0 and beta == 1.0:
        return generator.uniform(lower, upper, size=size)
    return lower + (upper - lower) * generator.beta(alpha, beta, size=size)
