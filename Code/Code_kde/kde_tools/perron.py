"""Central symmetric Perron solver for all KDE experiments.

Exponential similarities are assembled from shifted divergence exponents,
before underflow can turn the original similarity into the identity. The
transformation is described in Section 3.5.1 of the thesis. SciPy's eigh
requests the two largest algebraic eigenpairs, so negative eigenvalues near the
closest-pair limit cannot be mistaken for the Perron eigenvalue.

Algorithm reference:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import LinAlgError, eigh


PERRON_SOLVER_VERSION = "stabilized_eigh_v1"
PERRON_TOLERANCE = 1e-12


class PerronConvergenceError(RuntimeError):
    """The eigensolver did not produce a sufficiently accurate eigenpair."""


class PerronDegeneracyError(RuntimeError):
    """The largest eigenvalue is not numerically distinguishable as simple."""


@dataclass(frozen=True)
class PerronResult:
    """A sum-normalized Perron vector and its numerical diagnostics.

    Eigenvalues and spectral_gap refer to the supplied matrix. relative_gap
    and relative_residual refer to its centered/scaled solver matrix, so the
    identity diagonal cannot hide poor accuracy. For divergence-based calls,
    the supplied matrix is B = exp(lambda*d_min) * (S - I), not S.
    matvecs is retained for compatibility and is zero for the direct eigh solve.
    """

    vector: np.ndarray
    eigenvalue: float
    second_eigenvalue: float | None
    spectral_gap: float
    relative_gap: float
    relative_residual: float
    matvecs: int
    minimum_divergence: float | None = None

    @property
    def selected_index(self) -> int:
        return perron_peak_index(self.vector)


def perron_peak_index(vector: np.ndarray) -> int:
    """Choose the first peak when weights differ only at roundoff level.

    The tie window is 64 machine epsilons times the largest absolute weight.
    This is a selection convention, not an alteration of the eigenvector.
    In the closest-pair limit it consistently selects the lower grid index.
    """
    if np.iscomplexobj(vector):
        raise ValueError("vector must be real.")
    weights = np.asarray(vector, dtype=float)
    if weights.ndim != 1 or not len(weights) or not np.all(np.isfinite(weights)):
        raise ValueError("vector must be a nonempty finite one-dimensional array.")
    resolution = 64.0 * np.finfo(float).eps * float(np.max(np.abs(weights)))
    return int(np.flatnonzero(float(np.max(weights)) - weights <= resolution)[0])


def _symmetric_nonnegative_matrix(matrix: np.ndarray, name: str) -> np.ndarray:
    if np.iscomplexobj(matrix):
        raise ValueError(f"{name} must be real.")
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or not len(values):
        raise ValueError(f"{name} must be a nonempty square matrix.")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative.")
    if not np.allclose(values, values.T, rtol=1e-12, atol=1e-12):
        raise ValueError(f"{name} must be symmetric.")
    # Remove roundoff asymmetry without mutating the caller's array.
    return 0.5 * values + 0.5 * values.T


def perron_eigenpair(
    matrix: np.ndarray,
    *,
    tolerance: float = PERRON_TOLERANCE,
    max_iterations: int | None = None,
) -> PerronResult:
    """Solve a real symmetric nonnegative matrix with the shared backend.

    SciPy's eigh computes the two largest algebraic eigenpairs for N >= 2.
    Nonconvergence, a failed residual check, and a numerically repeated
    largest eigenvalue are explicit errors. tolerance controls the residual
    check; max_iterations is validated and retained for compatibility but
    does not affect this direct solver.
    """
    values = _symmetric_nonnegative_matrix(matrix, "matrix")
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and strictly positive.")
    if max_iterations is not None and (
        isinstance(max_iterations, (bool, np.bool_))
        or not isinstance(max_iterations, (int, np.integer))
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer or None.")
    size = len(values)
    if size == 1:
        return PerronResult(
            np.ones(1), float(values[0, 0]), None, np.inf, np.inf, 0.0, 0
        )

    diagonal_shift = float(np.min(np.diag(values)))
    values.flat[:: size + 1] -= diagonal_shift
    scale = float(np.max(values))
    if scale == 0.0:
        raise PerronDegeneracyError(
            "The matrix is a scalar multiple of the identity and has no "
            "unique Perron vector. For exp(-lambda*D), pass D and lambda "
            "to perron_eigenpair_from_divergence before exponentiating."
        )
    values /= scale
    try:
        eigenvalues, eigenvectors = eigh(
            values, subset_by_index=[size - 2, size - 1], check_finite=False
        )
    except LinAlgError as error:
        raise PerronConvergenceError(
            f"eigh failed for a {size}-by-{size} matrix: {error}"
        ) from error
    leading = float(eigenvalues[-1])
    second = float(eigenvalues[-2])
    gap = leading - second
    resolution = 64.0 * np.finfo(float).eps * max(1.0, abs(leading), abs(second))
    if gap <= resolution:
        raise PerronDegeneracyError(
            "The two largest eigenvalues are numerically indistinguishable; "
            "a unique Perron vector cannot be reliably selected."
        )
    vector = eigenvectors[:, -1].copy()
    if vector.sum() < 0.0:
        vector *= -1.0
    negative_tolerance = max(1e-14, 1e-10 * float(np.max(np.abs(vector))))
    if np.min(vector) < -negative_tolerance:
        raise PerronConvergenceError("The leading eigenvector is not nonnegative.")
    np.maximum(vector, 0.0, out=vector)
    vector /= vector.sum()
    row_sums = values.sum(axis=1)
    if np.all(row_sums == row_sums[0]):
        vector[:] = 1.0 / size
    relative_residual = float(
        np.linalg.norm(values @ vector - leading * vector)
        / (abs(leading) * np.linalg.norm(vector))
    )
    residual_limit = max(20.0 * tolerance, 200.0 * np.finfo(float).eps * size)
    if not np.isfinite(relative_residual) or relative_residual > residual_limit:
        raise PerronConvergenceError(
            f"Perron relative residual {relative_residual:.3e} exceeds "
            f"{residual_limit:.3e}."
        )
    return PerronResult(
        vector=vector,
        eigenvalue=diagonal_shift + scale * leading,
        second_eigenvalue=diagonal_shift + scale * second,
        spectral_gap=scale * gap,
        relative_gap=gap / abs(leading),
        relative_residual=relative_residual,
        matvecs=0,
    )


def perron_eigenpair_from_divergence(
    divergence: np.ndarray,
    lambda_value: float = 1.0,
    *,
    tolerance: float = PERRON_TOLERANCE,
    max_iterations: int | None = None,
) -> PerronResult:
    """Solve exp(-lambda*D) from D using shifted off-diagonal exponents."""
    values = _symmetric_nonnegative_matrix(divergence, "divergence")
    if not np.allclose(np.diag(values), 0.0, rtol=0.0, atol=1e-12):
        raise ValueError("divergence must have zero diagonal.")
    lambda_value = float(lambda_value)
    if not np.isfinite(lambda_value) or lambda_value < 0.0:
        raise ValueError("lambda_value must be finite and nonnegative.")
    if len(values) == 1:
        return replace(
            perron_eigenpair(
                np.ones((1, 1)), tolerance=tolerance, max_iterations=max_iterations
            ),
            minimum_divergence=0.0,
        )
    np.fill_diagonal(values, np.inf)
    minimum = float(np.min(values))
    values -= minimum
    # Set the diagonal before multiplication to support lambda == 0.
    np.fill_diagonal(values, 0.0)
    with np.errstate(over="ignore", under="ignore"):
        values *= -lambda_value
        np.exp(values, out=values)
    np.fill_diagonal(values, 0.0)
    result = perron_eigenpair(
        values, tolerance=tolerance, max_iterations=max_iterations
    )
    return replace(result, minimum_divergence=minimum)


def perron_vector_from_divergence(
    divergence: np.ndarray,
    lambda_value: float = 1.0,
    *,
    tolerance: float = PERRON_TOLERANCE,
    max_iterations: int | None = None,
) -> np.ndarray:
    """Return the shared solver's sum-normalized vector for exp(-lambda*D)."""
    return perron_eigenpair_from_divergence(
        divergence, lambda_value, tolerance=tolerance, max_iterations=max_iterations
    ).vector


def perron_vector(
    matrix: np.ndarray,
    *,
    tolerance: float = PERRON_TOLERANCE,
    max_iterations: int | None = None,
) -> np.ndarray:
    """Compatibility entry point for an already assembled symmetric matrix."""
    return perron_eigenpair(
        matrix, tolerance=tolerance, max_iterations=max_iterations
    ).vector


def doob_mass_from_perron(vector: np.ndarray) -> np.ndarray:
    """Return the symmetric Doob mass v**2 / sum(v**2) from one Perron vector."""
    vector = np.asarray(vector, dtype=float)
    if (
        vector.ndim != 1
        or not len(vector)
        or not np.all(np.isfinite(vector))
        or np.any(vector < 0.0)
        or np.max(vector) <= 0.0
    ):
        raise ValueError("vector must be finite, nonnegative, and nonzero.")
    mass = (vector / np.max(vector)) ** 2
    return mass / mass.sum()
