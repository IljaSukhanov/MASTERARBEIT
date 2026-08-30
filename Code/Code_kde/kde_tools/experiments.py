"""Single-repetition runners used by the spectral KDE notebooks."""

from __future__ import annotations

import numpy as np

from .density import (
    exact_oracle_ise_values,
    gaussian_kde_values,
)
from .quadrature import (
    quadrature_grid_for_kdes,
    quadrature_similarity_matrices,
    select_quadrature_similarity_bandwidths,
)
from .similarities import (
    select_spectral_bandwidth,
    spectral_choice_from_similarity,
)


def exact_beta2_bandwidth_selection_counts(
    n: int,
    h_grid: np.ndarray,
    M: int,
    sigma2: float = 1.0,
    tau: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Count exact beta-2 spectral bandwidth selections over ``M`` samples."""
    n = int(n)
    h_grid = np.asarray(h_grid, dtype=float)
    M = int(M)
    sigma2 = float(sigma2)
    tau = float(tau)
    rng = np.random.default_rng(seed)
    counts = np.zeros(len(h_grid), dtype=int)

    for _ in range(M):
        samples = rng.normal(0.0, np.sqrt(sigma2), size=n)
        selection = select_spectral_bandwidth(
            samples=samples,
            h_grid=h_grid,
            gamma=tau,
        )
        counts[int(selection["selected_index"])] += 1

    return counts


def quadrature_beta2_bandwidth_selection_counts(
    n: int,
    h_grid: np.ndarray,
    M: int,
    m_x: int = 1_001,
    sigma2: float = 1.0,
    tau: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Count quadrature beta-2 spectral selections over ``M`` samples."""
    n = int(n)
    h_grid = np.asarray(h_grid, dtype=float)
    M = int(M)
    sigma2 = float(sigma2)
    tau = float(tau)
    if isinstance(m_x, (bool, np.bool_)) or not isinstance(
        m_x,
        (int, np.integer),
    ):
        raise ValueError("m_x must be an integer of at least two.")
    if m_x < 2:
        raise ValueError("m_x must be an integer of at least two.")

    rng = np.random.default_rng(seed)
    counts = np.zeros(len(h_grid), dtype=int)

    for _ in range(M):
        samples = rng.normal(0.0, np.sqrt(sigma2), size=n)
        result = select_quadrature_similarity_bandwidths(
            samples=samples,
            h_grid=h_grid,
            similarity_name="exponential_l2_similarity",
            m_x=int(m_x),
            gamma=tau,
        )
        counts[int(result["choice"]["selected_index"])] += 1

    return counts


def run_one_spectral_vs_oracle_experiment(
    seed: int,
    n: int,
    sigma2: float,
    h_grid: np.ndarray,
    gamma: float = 1.0,
) -> dict[str, float | int]:
    """Compare the exact squared-L2 spectral selector with a normal-target oracle."""
    rng = np.random.default_rng(seed)
    samples = rng.normal(0.0, np.sqrt(sigma2), size=n)
    ise = exact_oracle_ise_values(samples, h_grid, sigma2)
    oracle_index = int(np.argmin(ise))
    spectral = select_spectral_bandwidth(
        samples,
        h_grid,
        gamma=gamma,
    )
    spectral_index = int(spectral["selected_index"])
    return {
        "seed": seed,
        "oracle_index": oracle_index,
        "spectral_index": spectral_index,
        "oracle_h": float(h_grid[oracle_index]),
        "spectral_h": float(h_grid[spectral_index]),
        "oracle_l2_error": float(np.sqrt(max(ise[oracle_index], 0.0))),
        "spectral_l2_error": float(np.sqrt(max(ise[spectral_index], 0.0))),
        "gamma": float(gamma),
    }


def run_one_consistency_experiment(
    seed: int,
    n: int,
    sigma2: float,
    h_grid: np.ndarray,
    m_x: int,
    gamma: float = 1.0,
) -> list[dict[str, float | int | str]]:
    """Run the oracle, Hellinger, and JS selectors for one consistency replicate."""
    rng = np.random.default_rng(seed)
    samples = rng.normal(0.0, np.sqrt(sigma2), size=n)
    h_grid = np.asarray(h_grid, dtype=float)
    ise = exact_oracle_ise_values(samples, h_grid, sigma2)
    oracle_index = int(np.argmin(ise))
    x_grid = quadrature_grid_for_kdes(samples, h_grid, m_x=m_x)
    densities = gaussian_kde_values(samples, h_grid, x_grid)
    similarities = {
        name: quadrature_similarity_matrices(
            densities,
            x_grid,
            similarity_name=name,
        )
        for name in ("hellinger_affinity", "js_similarity")
    }
    hellinger_index = int(
        spectral_choice_from_similarity(similarities["hellinger_affinity"])[
            "selected_index"
        ]
    )
    js_index = int(
        spectral_choice_from_similarity(similarities["js_similarity"])["selected_index"]
    )

    rows = []
    for method, index in (
        ("Oracle ISE", oracle_index),
        ("Hellinger affinity", hellinger_index),
        ("JS similarity", js_index),
    ):
        rows.append(
            {
                "n": int(n),
                "seed": int(seed),
                "method": method,
                "selected_index": index,
                "selected_h": float(h_grid[index]),
                "squared_integrated_error": float(ise[index]),
            }
        )
    return rows


def summarize(values: np.ndarray) -> dict[str, float]:
    """Return the summary statistics shared by the command-line experiments."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("values must be a nonempty one-dimensional array.")
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }
