"""Reusable numerical functions for archived AR experiments."""

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd

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
    weight_type: str = "covariance"

    def __post_init__(self):
        if self.p < 1:
            raise ValueError("p must be a positive integer.")
        if self.k_max < self.p:
            raise ValueError("k_max must be at least p.")
        if self.weight_type != "covariance":
            raise ValueError("Only 'covariance' weights remain in this module.")


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


def simulate_ar_with_distribution(
    phi: np.ndarray,
    n: int,
    burn_in: int,
    innovation_distribution: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate a stationary AR process with N(0,1) or U[-1,1] innovations."""
    phi = np.asarray(phi, dtype=float)
    radius = companion_radius(phi)
    if radius >= 1:
        raise ValueError(
            f"AR coefficients are not stationary (spectral radius={radius:.4f})."
        )

    distribution = innovation_distribution.strip().lower()
    size = n + burn_in + len(phi)
    if distribution == "gaussian":
        innovations = rng.normal(0.0, 1.0, size)
    elif distribution == "uniform":
        innovations = rng.uniform(-1.0, 1.0, size)
    else:
        raise ValueError("innovation_distribution must be 'gaussian' or 'uniform'.")

    p = len(phi)
    x = np.zeros(size, dtype=float)
    for t in range(p, size):
        x[t] = phi @ x[t - np.arange(1, p + 1)] + innovations[t]
    return x[burn_in + p:]


def create_2d_trajectory_animation(
    points: np.ndarray,
    interval_ms: int = 40,
):
    """Create an accumulating animation of a two-dimensional trajectory."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("points must have shape (n, 2) with n >= 2.")
    if interval_ms <= 0:
        raise ValueError("interval_ms must be positive.")

    lower = points.min(axis=0)
    upper = points.max(axis=0)
    span = np.maximum(upper - lower, 1e-8)
    margin = 0.08 * span

    figure, axis = plt.subplots(figsize=(7, 7))
    axis.set_xlim(lower[0] - margin[0], upper[0] + margin[0])
    axis.set_ylim(lower[1] - margin[1], upper[1] + margin[1])
    axis.set_aspect("equal", adjustable="box")
    axis.set(
        title="Ordered 2D sliding-window trajectory",
        xlabel=r"$X_t$",
        ylabel=r"$X_{t+1}$",
    )
    axis.grid(alpha=0.25)

    trajectory, = axis.plot([], [], color="tab:blue", linewidth=1.2, alpha=0.8)
    visited = axis.scatter([], [], s=18, color="tab:blue", alpha=0.55)
    current, = axis.plot([], [], marker="o", color="tab:red", markersize=7)
    time_label = axis.text(0.02, 0.97, "", transform=axis.transAxes, va="top")

    def initialize():
        trajectory.set_data([], [])
        visited.set_offsets(np.empty((0, 2)))
        current.set_data([], [])
        time_label.set_text("")
        return trajectory, visited, current, time_label

    def update(frame):
        shown = points[:frame]
        trajectory.set_data(shown[:, 0], shown[:, 1])
        visited.set_offsets(shown)
        current.set_data([shown[-1, 0]], [shown[-1, 1]])
        time_label.set_text(f"embedded point {frame}/{len(points)}")
        return trajectory, visited, current, time_label

    animation = FuncAnimation(
        figure,
        update,
        frames=range(1, len(points) + 1),
        init_func=initialize,
        interval=interval_ms,
        blit=True,
        repeat=False,
    )
    return figure, animation


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
    n_vertices = next(iter(clouds.values())).shape[0]
    return covariance_weight_matrices(train, completions, n_vertices, config.k_max)


def weight_axis_label(weight_type: str) -> str:
    return r"$w_k^{\mathrm{cov}}(s,t)$"


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


__all__ = ['PipelineConfig', 'pacf_to_ar', 'automatic_ar_coefficients', 'companion_radius', 'simulate_ar', 'simulate_ar_with_distribution', 'create_2d_trajectory_animation', 'aligned_windows', 'fit_completion_matrix', 'lift_test_cloud', 'build_lifted_clouds', 'estimate_autocovariance', 'lifted_covariance_scores', 'covariance_weight_matrices', 'build_weight_matrices', 'weight_axis_label', 'principal_scale']
