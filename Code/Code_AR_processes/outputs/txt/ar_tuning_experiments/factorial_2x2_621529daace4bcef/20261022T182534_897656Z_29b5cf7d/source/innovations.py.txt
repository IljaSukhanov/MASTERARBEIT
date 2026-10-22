"""Mean-zero, variance-1/3 Beta innovations shared by all AR generators.

For U ~ Beta(alpha, beta), use Z = a + (b-a)*U, where
a = -sqrt(alpha*(alpha+beta+1)/(3*beta)) and
b = sqrt(beta*(alpha+beta+1)/(3*alpha)). Both shapes default to 1,
recovering Uniform[-1,1]. The endpoints and width are derived parameters.

Reference: NIST/SEMATECH e-Handbook, section 1.3.6.6.17 (mean and variance),
https://itl.nist.gov/div898/handbook/eda/section3/eda366h.htm.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["beta_innovation_bounds", "sample_beta_innovations"]


def _finite_scalar(value: float, name: str, *, positive: bool = False) -> float:
    if (
        isinstance(value, (bool, np.bool_, complex, np.complexfloating, str, bytes))
        or not np.isscalar(value)
    ):
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
    """Return the unique ordered support with mean zero and variance 1/3.

    L = (alpha+beta)*sqrt((alpha+beta+1)/(3*alpha*beta)),
    a = -L*alpha/(alpha+beta), b = L*beta/(alpha+beta).

    Optional legacy a,b,L arguments are consistency checks only. They must
    match these derived values, so no caller can override the normalization.
    Omit them in new code; alpha and beta are the distribution parameters.
    """
    alpha = _finite_scalar(alpha, "alpha", positive=True)
    beta = _finite_scalar(beta, "beta", positive=True)
    if (a is None) != (b is None):
        raise ValueError("Supply both a and b as checks, or omit both")
    if alpha == 1.0 and beta == 1.0:
        # Keep the uniform baseline endpoints and seeded draws exactly unchanged.
        lower, upper = -1.0, 1.0
    else:
        # Logarithms avoid overflowing intermediate products of valid shapes.
        log_alpha, log_beta = np.log(alpha), np.log(beta)
        log_total_plus_one = np.logaddexp(np.logaddexp(log_alpha, log_beta), 0.0)
        log_scale = 0.5 * (log_total_plus_one - np.log(3.0))
        with np.errstate(over="ignore", under="ignore"):
            lower = -float(np.exp(log_scale + 0.5 * (log_alpha - log_beta)))
            upper = float(np.exp(log_scale + 0.5 * (log_beta - log_alpha)))

    support_width = upper - lower
    if not lower < 0 < upper or not np.isfinite(support_width):
        raise ValueError("The normalized Beta support is not representable with finite floats")
    for name, requested, derived in (("a", a, lower), ("b", b, upper), ("L", L, support_width)):
        if requested is None:
            continue
        value = _finite_scalar(requested, name, positive=(name == "L"))
        if not np.isclose(value, derived, rtol=1e-12, atol=0.0):
            raise ValueError(
                f"{name} is determined by alpha,beta for mean-zero, variance-1/3 "
                f"innovations; expected {derived:g}, got {value:g}"
            )
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
    """Sample Beta innovations with population mean zero and variance 1/3.

    alpha=beta=1 gives Uniform[-1,1]. Every other shape uses the support
    derived by beta_innovation_bounds, preserving the same two moments.
    Legacy a,b,L arguments only check the derived values; they cannot
    select another mean or variance. Normalization uses population moments,
    not sample moments: the draws remain independent.

    size and rng follow NumPy's Generator conventions. A seed reproduces
    a draw; an existing Generator is advanced in place. The alpha=beta=1
    case uses Generator.uniform to preserve the existing seeded AR paths.
    """
    lower, upper = beta_innovation_bounds(alpha, beta, a, b, L)
    generator = np.random.default_rng(rng)
    if alpha == 1.0 and beta == 1.0:
        return generator.uniform(lower, upper, size=size)
    return lower + (upper - lower) * generator.beta(alpha, beta, size=size)
