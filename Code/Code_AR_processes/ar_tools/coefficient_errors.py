"""One coefficient-accuracy convention for all AR experiments.

For an AR(p) fit, MAE = sum_j abs(estimated_phi_j - true_phi_j) / p.
Monte Carlo summaries average these per-fit MAEs, without dropping failed fits.
This is the uniform average over coefficient outputs; compare scikit-learn's
mean_absolute_error documentation, Parameters: multioutput='uniform_average':
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html
"""

import numpy as np


COEFFICIENT_ERROR_METRIC = "mae_per_coefficient_v1"


def coefficient_mae(estimated, truth):
    """Average absolute error along the final (coefficient) axis.

    Leading axes may broadcast, e.g. (replications, p) against (p,).
    An undefined/nonfinite coefficient makes that fit's MAE unavailable (NaN).
    """
    estimated = np.asarray(estimated, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if (estimated.ndim == 0 or truth.ndim == 0 or estimated.shape[-1] == 0
            or estimated.shape[-1] != truth.shape[-1]):
        raise ValueError("Use nonempty coefficient vectors with the same AR order.")
    with np.errstate(invalid="ignore", over="ignore"):
        absolute_errors = np.abs(estimated - truth)
        result = np.sum(absolute_errors / estimated.shape[-1], axis=-1)
        result = np.where(np.isfinite(absolute_errors).all(axis=-1)
                          & np.isfinite(result), result, np.nan)
    return float(result) if result.ndim == 0 else result
