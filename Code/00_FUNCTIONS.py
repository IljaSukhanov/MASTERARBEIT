"""Reusable functions for the AR scale-selection notebook.

This module is loaded by ``01_AR_SCALE_SELECTION.ipynb``.  The numerical and
GUDHI implementation lives here so the notebook contains only controls,
execution, tables, and plots.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable
import importlib.util
import sys

import gudhi
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


_FUNCTIONALS_PATH = Path(__file__).with_name("001_SIMPLICIAL_COMPLEX_FUNCTIONALS.py")
_FUNCTIONALS_SPEC = importlib.util.spec_from_file_location(
    "simplicial_complex_functionals",
    _FUNCTIONALS_PATH,
)
simplicial_functionals = importlib.util.module_from_spec(_FUNCTIONALS_SPEC)
sys.modules[_FUNCTIONALS_SPEC.name] = simplicial_functionals
_FUNCTIONALS_SPEC.loader.exec_module(simplicial_functionals)

beta_0 = simplicial_functionals.beta_0
edge_count = simplicial_functionals.edge_count
log_component_edge_functional = simplicial_functionals.log_component_edge_functional

@dataclass(frozen=True)
class PipelineConfig:
    p: int = 1
    k_max: int = 5
    n_observations: int = 500
    train_fraction: float = 0.60
    burn_in: int = 500
    innovation_sd: float = 1.0
    seed: int = 20260707
    pacf_first: float = 0.75
    pacf_decay: float = -0.50
    weight_type: str = "l1"  # 'l1' or 'covariance'
    tau: float = 2.0
    max_simplex_dimension: int = 2
    alpha: float = 1.0
    xi: float = 1.0
    tilde_delta: float = 0.0
    similarity_mode: str = "union"  # 'union' or 'overlap'

    def __post_init__(self):
        if self.p < 1:
            raise ValueError("p must be a positive integer.")
        if self.k_max < self.p:
            raise ValueError("k_max must be at least p.")
        if self.weight_type not in {"l1", "covariance"}:
            raise ValueError("weight_type must be 'l1' or 'covariance'.")
        if self.tau <= 0:
            raise ValueError("tau must be strictly positive.")


def pacf_to_ar(partial_autocorrelations: Iterable[float]) -> np.ndarray:
    """Convert partial autocorrelations to stationary AR coefficients."""
    phi = np.empty(0, dtype=float)
    for kappa in np.asarray(tuple(partial_autocorrelations), dtype=float):
        if abs(kappa) >= 1:
            raise ValueError("Every partial autocorrelation must lie in (-1, 1).")
        previous = phi.copy()
        phi = np.empty(previous.size + 1)
        if previous.size:
            phi[:-1] = previous - kappa * previous[::-1]
        phi[-1] = kappa
    return phi


def automatic_ar_coefficients(order: int, first: float = 0.75, decay: float = -0.50) -> np.ndarray:
    pacf = first * decay ** np.arange(order)
    return pacf_to_ar(pacf)


def companion_radius(phi: np.ndarray) -> float:
    companion = np.zeros((len(phi), len(phi)))
    companion[0] = phi
    if len(phi) > 1:
        companion[1:, :-1] = np.eye(len(phi) - 1)
    return float(np.max(np.abs(np.linalg.eigvals(companion))))


def simulate_ar(phi: np.ndarray, n: int, burn_in: int, innovation_sd: float, rng: np.random.Generator) -> np.ndarray:
    phi = np.asarray(phi, dtype=float)
    radius = companion_radius(phi)
    if radius >= 1:
        raise ValueError(f"AR coefficients are not stationary (spectral radius={radius:.4f}).")
    p = len(phi)
    eps = rng.normal(0.0, innovation_sd, n + burn_in + p)
    x = np.zeros_like(eps)
    for t in range(p, x.size):
        x[t] = phi @ x[t - np.arange(1, p + 1)] + eps[t]
    return x[burn_in + p:]


def aligned_windows(x: np.ndarray, width: int) -> np.ndarray:
    if len(x) < width:
        raise ValueError(f"Need at least {width} observations, received {len(x)}.")
    return np.lib.stride_tricks.sliding_window_view(np.asarray(x), width)


def fit_completion_matrix(train: np.ndarray, k: int, k_max: int) -> np.ndarray:
    full = aligned_windows(train, k_max)
    if k == k_max:
        return np.empty((0, k))
    y, z = full[:, :k], full[:, k:]
    coefficients, *_ = np.linalg.lstsq(y, z, rcond=None)
    return coefficients.T


def lift_test_cloud(test: np.ndarray, k: int, k_max: int, completion: np.ndarray) -> np.ndarray:
    y = aligned_windows(test, k_max)[:, :k]
    predicted = y @ completion.T
    return np.concatenate((y, predicted), axis=1)


def build_lifted_clouds(train: np.ndarray, test: np.ndarray, k_max: int):
    completions, clouds = {}, {}
    for k in range(1, k_max + 1):
        completions[k] = fit_completion_matrix(train, k, k_max)
        clouds[k] = lift_test_cloud(test, k, k_max, completions[k])
        assert clouds[k].shape[1] == k_max
    return completions, clouds


def l1_weight_matrices(clouds: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    """Observed L1 dissimilarities between all lifted test points."""
    return {k: squareform(pdist(points, metric="cityblock")) for k, points in clouds.items()}


def estimate_autocovariance(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Biased sample autocovariance, using denominator n for every lag."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    if max_lag >= n:
        raise ValueError(
            f"Cannot estimate lag {max_lag} from only {n} training observations. "
            "Increase the training fraction or reduce the test length."
        )
    gamma = np.empty(max_lag + 1)
    for h in range(max_lag + 1):
        gamma[h] = x[h:] @ x[:n - h] / n
    return gamma


def lifted_covariance_scores(A_hat: np.ndarray, gamma: np.ndarray, k: int, max_h: int) -> np.ndarray:
    """Frobenius scores ||D_h^(k)||_F for h=0,...,max_h."""
    indices = np.arange(k)
    G = np.eye(k) + A_hat.T @ A_hat
    scores = np.empty(max_h + 1)
    for h in range(max_h + 1):
        lags = np.abs(h + indices[None, :] - indices[:, None])
        if lags.max() >= len(gamma):
            raise ValueError("More autocovariance lags must be estimated.")
        C_h = gamma[lags]
        squared_norm = np.trace(C_h.T @ G @ C_h @ G)
        scores[h] = np.sqrt(max(0.0, float(squared_norm)))
    return scores


def covariance_weight_matrices(
    train: np.ndarray,
    completions: Dict[int, np.ndarray],
    n_vertices: int,
    k_max: int,
) -> Dict[int, np.ndarray]:
    """Plug-in covariance dissimilarities based only on training-data estimates."""
    max_h = n_vertices - 1
    gamma = estimate_autocovariance(train, max_h + k_max - 1)
    temporal_lags = np.abs(np.subtract.outer(np.arange(n_vertices), np.arange(n_vertices)))
    weights = {}
    for k in range(1, k_max + 1):
        scores = lifted_covariance_scores(completions[k], gamma, k, max_h)
        if not np.isfinite(scores[0]) or scores[0] <= 0:
            raise ValueError(f"The lag-zero covariance score is not positive for k={k}.")
        dissimilarity_by_lag = 1.0 - np.clip(scores / scores[0], 0.0, 1.0)
        matrix = dissimilarity_by_lag[temporal_lags]
        np.fill_diagonal(matrix, 0.0)
        weights[k] = matrix
    return weights


def build_weight_matrices(
    train: np.ndarray,
    completions: Dict[int, np.ndarray],
    clouds: Dict[int, np.ndarray],
    config: PipelineConfig,
) -> Dict[int, np.ndarray]:
    if config.weight_type == "l1":
        return l1_weight_matrices(clouds)
    n_vertices = next(iter(clouds.values())).shape[0]
    return covariance_weight_matrices(train, completions, n_vertices, config.k_max)


def weight_axis_label(weight_type: str) -> str:
    if weight_type == "l1":
        return r"$w_k(s,t)=\|p_s^{(k)}-p_t^{(k)}\|_1$"
    return r"$w_k^{\mathrm{cov}}(s,t)$"


def gudhi_flag_complex(weight_matrix: np.ndarray, tau: float, max_dimension: int = 2) -> gudhi.SimplexTree:
    n = weight_matrix.shape[0]
    complex_ = gudhi.SimplexTree()
    for vertex in range(n):
        complex_.insert([vertex], filtration=0.0)
    rows, cols = np.where(np.triu(weight_matrix <= tau, k=1))
    for i, j in zip(rows, cols):
        complex_.insert([int(i), int(j)], filtration=float(weight_matrix[i, j]))
    complex_.expansion(max_dimension)
    complex_.make_filtration_non_decreasing()
    return complex_


def union_complex(left: gudhi.SimplexTree, right: gudhi.SimplexTree) -> gudhi.SimplexTree:
    union = gudhi.SimplexTree()
    for tree in (left, right):
        for simplex, filtration in tree.get_filtration():
            union.insert(simplex, filtration=filtration)
    union.make_filtration_non_decreasing()
    return union


def scale_similarity_matrix(complexes: Dict[int, gudhi.SimplexTree], config: PipelineConfig):
    ks = sorted(complexes)
    individual = {
        k: log_component_edge_functional(
            complexes[k],
            config.alpha,
            config.xi,
            config.tilde_delta,
        )
        for k in ks
    }
    sigma = np.empty((len(ks), len(ks)))
    for a, i in enumerate(ks):
        for b in range(a, len(ks)):
            j = ks[b]
            f_union = log_component_edge_functional(
                union_complex(complexes[i], complexes[j]),
                config.alpha,
                config.xi,
                config.tilde_delta,
            )
            value = f_union if config.similarity_mode == "union" else individual[i] + individual[j] - f_union
            sigma[a, b] = sigma[b, a] = value
    if config.similarity_mode not in {"union", "overlap"}:
        raise ValueError("similarity_mode must be 'union' or 'overlap'.")
    return sigma, individual


def principal_scale(sigma: np.ndarray, ks: Iterable[int]):
    eigenvalues, eigenvectors = np.linalg.eigh((sigma + sigma.T) / 2)
    vector = eigenvectors[:, -1]
    if vector.sum() < 0:
        vector = -vector
    if np.min(vector) > -1e-10:
        vector = np.maximum(vector, 0)
    vector = vector / np.sum(np.abs(vector))
    ks = np.asarray(tuple(ks))
    return int(ks[np.argmax(vector)]), eigenvalues, vector


def run_pipeline(order: int, config: PipelineConfig, seed_offset: int = 0):
    phi = automatic_ar_coefficients(order, config.pacf_first, config.pacf_decay)
    rng = np.random.default_rng(config.seed + seed_offset)
    x = simulate_ar(phi, config.n_observations, config.burn_in, config.innovation_sd, rng)
    split = int(config.train_fraction * len(x))
    train, test = x[:split], x[split:]
    completions, clouds = build_lifted_clouds(train, test, config.k_max)
    weights = build_weight_matrices(train, completions, clouds, config)
    tau = float(config.tau)
    complexes = {k: gudhi_flag_complex(w, tau, config.max_simplex_dimension) for k, w in weights.items()}
    sigma, functionals = scale_similarity_matrix(complexes, config)
    k_star, eigenvalues, eigenvector = principal_scale(sigma, sorted(complexes))
    counts = {
        k: (beta_0(complexes[k]), edge_count(complexes[k]))
        for k in sorted(complexes)
    }
    return {
        "order": order, "phi": phi, "series": x, "train": train, "test": test,
        "completions": completions, "clouds": clouds, "weights": weights,
        "weight_type": config.weight_type, "tau": tau, "complexes": complexes, "counts": counts,
        "functionals": functionals, "sigma": sigma, "eigenvalues": eigenvalues,
        "eigenvector": eigenvector, "k_star": k_star,
    }


def scan_tau_single_process(
    order: int,
    config: PipelineConfig,
    tau_values: Iterable[float],
    seed: int = 20260801,
):
    """Scan tau for one simulated AR process and retain every curve."""
    tau_values = np.asarray(tuple(tau_values), dtype=float)
    if tau_values.ndim != 1 or tau_values.size == 0 or np.any(tau_values <= 0):
        raise ValueError("tau_values must be a nonempty sequence of positive numbers.")

    phi = automatic_ar_coefficients(order, config.pacf_first, config.pacf_decay)
    rng = np.random.default_rng(seed)
    x = simulate_ar(phi, config.n_observations, config.burn_in, config.innovation_sd, rng)
    split = int(config.train_fraction * len(x))
    train, test = x[:split], x[split:]
    completions, clouds = build_lifted_clouds(train, test, config.k_max)
    weights = build_weight_matrices(train, completions, clouds, config)
    ks = np.arange(1, config.k_max + 1)
    n_vertices = next(iter(weights.values())).shape[0]
    possible_edges = n_vertices * (n_vertices - 1) / 2

    records = []
    curves = {}
    for tau_value in tau_values:
        complexes = {
            k: gudhi_flag_complex(weight_matrix, float(tau_value), max_dimension=1)
            for k, weight_matrix in weights.items()
        }
        sigma, functionals = scale_similarity_matrix(complexes, config)
        k_star, eigenvalues, eigenvector = principal_scale(sigma, ks)
        leading = float(eigenvalues[-1])
        relative_gap = (
            float((eigenvalues[-1] - eigenvalues[-2]) / abs(leading))
            if len(eigenvalues) > 1 and leading != 0
            else np.nan
        )
        counts = {
            k: (beta_0(complexes[k]), edge_count(complexes[k]))
            for k in ks
        }
        beta_0_curve = np.array([counts[k][0] for k in ks], dtype=float)
        edge_curve = np.array([counts[k][1] for k in ks], dtype=float)
        beta_contribution = config.alpha * np.log(beta_0_curve)
        edge_contribution = config.xi * np.log(np.maximum(1.0, edge_curve))
        mean_edge_density = float(np.mean(edge_curve / possible_edges))
        functional_curve = np.array([functionals[k] for k in ks])
        if not np.allclose(
            functional_curve,
            config.tilde_delta + beta_contribution + edge_contribution,
        ):
            raise RuntimeError("Functional decomposition is inconsistent.")
        curves[float(tau_value)] = {
            "functional": functional_curve,
            "perron": eigenvector,
            "beta_0": beta_0_curve,
            "edges": edge_curve,
            "beta_contribution": beta_contribution,
            "edge_contribution": edge_contribution,
            "sigma": sigma,
            "k_star": k_star,
        }
        records.append({
            "tau": float(tau_value),
            "k_star": k_star,
            "correct": k_star == order,
            "relative_spectral_gap": relative_gap,
            "mean_edge_density": mean_edge_density,
        })

    return {
        "order": order,
        "phi": phi,
        "series": x,
        "completions": completions,
        "clouds": clouds,
        "weights": weights,
        "weight_type": config.weight_type,
        "results": pd.DataFrame.from_records(records),
        "curves": curves,
    }

__all__ = ['PipelineConfig', 'pacf_to_ar', 'automatic_ar_coefficients', 'companion_radius', 'simulate_ar', 'aligned_windows', 'fit_completion_matrix', 'lift_test_cloud', 'build_lifted_clouds', 'l1_weight_matrices', 'estimate_autocovariance', 'lifted_covariance_scores', 'covariance_weight_matrices', 'build_weight_matrices', 'weight_axis_label', 'gudhi_flag_complex', 'union_complex', 'scale_similarity_matrix', 'principal_scale', 'run_pipeline', 'scan_tau_single_process']
