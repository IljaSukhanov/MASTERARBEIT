"""Single-repetition runners used by the spectral KDE notebooks."""

from __future__ import annotations

import numpy as np

from .bandwidths import (
    bandwidth_grid_for_gaussian_experiment,
    direct_plugin_bandwidth,
    least_squares_cv_bandwidth_index,
    nearest_bandwidth_grid_index,
    silverman_rule_of_thumb_bandwidth,
)
from .density import (
    exact_mixture_ise_values,
    exact_oracle_ise_values,
    gaussian_kde_values,
    normalize_density_rows,
    quadrature_grid_for_gaussian_mixture_kdes,
    quadrature_grid_for_kdes,
    sample_gaussian_mixture,
)
from .similarities import (
    quadrature_similarity_matrices,
    select_spectral_bandwidth,
    spectral_choice_from_similarity,
)


def _spectral_indices(similarities: dict[str, np.ndarray | float]) -> dict[str, int]:
    return {
        name: int(spectral_choice_from_similarity(similarities[name])["selected_index"])
        for name in ("tv_overlap", "hellinger_affinity", "rbf_l2", "js_similarity")
    }


def run_one_spectral_vs_oracle_experiment(
    seed: int,
    n: int,
    sigma2: float,
    h_grid: np.ndarray,
    gamma: float | str = "auto",
) -> dict[str, float | int]:
    """Compare the exact squared-L2 spectral selector with a normal-target oracle."""
    rng = np.random.default_rng(seed)
    samples = rng.normal(0.0, np.sqrt(sigma2), size=n)
    ise = exact_oracle_ise_values(samples, h_grid, sigma2)
    oracle_index = int(np.argmin(ise))
    spectral = select_spectral_bandwidth(samples, h_grid, gamma=gamma)
    spectral_index = int(spectral["selected_index"])
    return {
        "seed": seed,
        "oracle_index": oracle_index,
        "spectral_index": spectral_index,
        "oracle_h": float(h_grid[oracle_index]),
        "spectral_h": float(h_grid[spectral_index]),
        "oracle_l2_error": float(np.sqrt(max(ise[oracle_index], 0.0))),
        "spectral_l2_error": float(np.sqrt(max(ise[spectral_index], 0.0))),
        "gamma": float(spectral["gamma"]),
    }


def run_one_all5_bandwidth_experiment(
    seed: int,
    n: int,
    sigma2: float,
    h_grid: np.ndarray,
    m_x: int,
    gamma: float | str = "auto",
) -> dict[str, float | int]:
    """Run the oracle and four spectral selectors for one normal sample."""
    rng = np.random.default_rng(seed)
    samples = rng.normal(0.0, np.sqrt(sigma2), size=n)
    ise = exact_oracle_ise_values(samples, h_grid, sigma2)
    oracle_index = int(np.argmin(ise))
    x_grid = quadrature_grid_for_kdes(samples, h_grid, sigma2=sigma2, m_x=m_x)
    densities = normalize_density_rows(
        gaussian_kde_values(samples, h_grid, x_grid), x_grid
    )
    similarities = quadrature_similarity_matrices(
        densities, x_grid, gamma=gamma, renormalize=False
    )
    indices = _spectral_indices(similarities)
    result: dict[str, float | int] = {
        "seed": seed,
        "oracle_index": oracle_index,
        "oracle_h": float(h_grid[oracle_index]),
        "oracle_l2_error": float(np.sqrt(max(ise[oracle_index], 0.0))),
        "gamma": float(similarities["gamma"]),
    }
    key_prefix = {
        "tv_overlap": "tv",
        "hellinger_affinity": "hellinger",
        "rbf_l2": "rbf_l2",
        "js_similarity": "js",
    }
    for name, index in indices.items():
        prefix = key_prefix[name]
        result[f"{prefix}_index"] = index
        result[f"{prefix}_h"] = float(h_grid[index])
        result[f"{prefix}_l2_error"] = float(np.sqrt(max(ise[index], 0.0)))
    return result


def run_one_consistency_experiment(
    seed: int,
    n: int,
    sigma2: float,
    n_h: int,
    m_x: int,
    gamma: float | str = "auto",
) -> list[dict[str, float | int | str]]:
    """Run the oracle, Hellinger, and JS selectors for one consistency replicate."""
    rng = np.random.default_rng(seed)
    samples = rng.normal(0.0, np.sqrt(sigma2), size=n)
    h_grid = bandwidth_grid_for_gaussian_experiment(n, sigma2, n_h=n_h)
    ise = exact_oracle_ise_values(samples, h_grid, sigma2)
    oracle_index = int(np.argmin(ise))
    x_grid = quadrature_grid_for_kdes(samples, h_grid, sigma2=sigma2, m_x=m_x)
    densities = normalize_density_rows(
        gaussian_kde_values(samples, h_grid, x_grid), x_grid
    )
    similarities = quadrature_similarity_matrices(
        densities, x_grid, gamma=gamma, renormalize=False
    )
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


def run_one_mixture_bandwidth_experiment(
    seed: int,
    n: int,
    h_grid: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    sigmas: np.ndarray,
    m_x: int,
    gamma: float | str = "auto",
) -> dict[str, float | int]:
    """Run oracle, spectral, plug-in, LSCV, and Silverman selectors once."""
    rng = np.random.default_rng(seed)
    samples = sample_gaussian_mixture(rng, n, weights, means, sigmas)
    ise = exact_mixture_ise_values(samples, h_grid, weights, means, sigmas)
    oracle_index = int(np.argmin(ise))
    x_grid = quadrature_grid_for_gaussian_mixture_kdes(
        samples, h_grid, weights, means, sigmas, m_x=m_x
    )
    densities = normalize_density_rows(
        gaussian_kde_values(samples, h_grid, x_grid), x_grid
    )
    similarities = quadrature_similarity_matrices(
        densities, x_grid, gamma=gamma, renormalize=False
    )
    indices = _spectral_indices(similarities)

    plugin_raw_h = direct_plugin_bandwidth(samples, x_grid)
    silverman_raw_h = silverman_rule_of_thumb_bandwidth(samples)
    plugin_index = nearest_bandwidth_grid_index(plugin_raw_h, h_grid)
    silverman_index = nearest_bandwidth_grid_index(silverman_raw_h, h_grid)
    lscv_index, lscv_scores = least_squares_cv_bandwidth_index(samples, h_grid)

    result: dict[str, float | int] = {
        "seed": seed,
        "oracle_index": oracle_index,
        "oracle_h": float(h_grid[oracle_index]),
        "oracle_l2_error": float(np.sqrt(max(ise[oracle_index], 0.0))),
        "plugin_index": plugin_index,
        "plugin_h": float(h_grid[plugin_index]),
        "plugin_raw_h": float(plugin_raw_h),
        "plugin_l2_error": float(np.sqrt(max(ise[plugin_index], 0.0))),
        "lscv_index": lscv_index,
        "lscv_h": float(h_grid[lscv_index]),
        "lscv_score": float(lscv_scores[lscv_index]),
        "lscv_l2_error": float(np.sqrt(max(ise[lscv_index], 0.0))),
        "silverman_index": silverman_index,
        "silverman_h": float(h_grid[silverman_index]),
        "silverman_raw_h": float(silverman_raw_h),
        "silverman_l2_error": float(np.sqrt(max(ise[silverman_index], 0.0))),
        "gamma": float(similarities["gamma"]),
    }
    key_prefix = {
        "tv_overlap": "tv",
        "hellinger_affinity": "hellinger",
        "rbf_l2": "rbf_l2",
        "js_similarity": "js",
    }
    for name, index in indices.items():
        prefix = key_prefix[name]
        result[f"{prefix}_index"] = index
        result[f"{prefix}_h"] = float(h_grid[index])
        result[f"{prefix}_l2_error"] = float(np.sqrt(max(ise[index], 0.0)))
    return result


def summarize(values: np.ndarray) -> dict[str, float]:
    """Return the summary statistics shared by the command-line experiments."""
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "sd": float(np.std(values, ddof=1)),
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }
