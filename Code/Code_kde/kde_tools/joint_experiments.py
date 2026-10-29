"""Reusable functions for experiment_1_joint_N_lambda_limit. Experiment settings stay in the notebook."""

from __future__ import annotations
from kde_tools.energy_minimization import minimize_bandwidth_energy
from kde_tools.qhat import expected_gaussian_qhat_log_energy
import matplotlib.pyplot as plt
import numpy as np


def local_beta2_log_metric(bandwidth: float, squared_differences: np.ndarray) -> float:
    """Exact coefficient of (Delta log h)^2 for this KDE family."""
    h = float(bandwidth)
    q = 2.0 * h**2
    d = np.asarray(squared_differences, dtype=float)
    overlap = np.exp(-d / (2.0 * q)) / np.sqrt(2.0 * np.pi * q)
    second_q_factor = 3.0 / (4.0 * q**2) - 3.0 * d / (2.0 * q**3) + d**2 / (4.0 * q**4)
    return float(2.0 * h**4 * np.mean(overlap * second_q_factor))


def plot_selected_bandwidth_paths(
    *,
    show_continuous_reference: bool,
    h_max,
    h_min,
    regime_colors,
    regime_order,
    h_infinity,
    metric_grid,
    metric_values,
    plot_ticks,
    results,
    squared_differences,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for regime in regime_order:
        subset = results[results["regime"] == regime]
        admissible = "admissible" in regime
        ax.plot(
            subset["N"],
            subset["selected_h"],
            marker="o",
            linewidth=2.2 if admissible else 1.6,
            linestyle="-" if admissible else "--",
            color=regime_colors[regime],
            label=regime,
        )
    if show_continuous_reference:
        ax.axhline(
            h_infinity,
            color="black",
            linestyle=":",
            linewidth=2.2,
            label=f"continuous $h_\\infty={h_infinity:.4f}$",
        )
    ax.set(
        xlabel="number of bandwidths $N$",
        ylabel="selected bandwidth $h_{\\mathrm{spec},N}$",
        title=(
            "Finite selector versus continuous reference"
            if show_continuous_reference
            else "Selected bandwidth across grid resolutions"
        ),
    )
    ax.set_xticks(plot_ticks, labels=plot_ticks)
    ax.legend(ncol=2, frameon=True)
    fig.tight_layout()
    if show_continuous_reference:
        criterion_inset = ax.inset_axes([0.19, 0.15, 0.34, 0.4], zorder=5)
        criterion_inset.set_facecolor("white")
        criterion_inset.plot(metric_grid, metric_values, color="#244c66", linewidth=1.4)
        minimum_energy = local_beta2_log_metric(h_infinity, squared_differences)
        criterion_inset.plot(
            h_infinity, minimum_energy, "o", color="#c23b22", markersize=4.5, zorder=6
        )
        criterion_inset.annotate(
            f"minimum at $h={h_infinity:.4f}$",
            xy=(h_infinity, minimum_energy),
            xycoords="data",
            xytext=(0.44, 0.57),
            textcoords="axes fraction",
            ha="center",
            fontsize=8,
            color="#c23b22",
            arrowprops=dict(arrowstyle="->", color="#c23b22", linewidth=0.9),
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5),
        )
        criterion_inset.set_xscale("log")
        criterion_inset.set_xlim(h_min, h_max)
        criterion_inset.set_ylim(0, 1.08 * float(np.max(metric_values)))
        criterion_inset.set_xticks([h_min, 0.1, 1.0, h_max], labels=["0.01", "0.1", "1", "2"])
        criterion_inset.set_yticks([0, 0.02, 0.04, 0.06])
        criterion_inset.set_xlabel("bandwidth $h$ (log scale)", fontsize=8, labelpad=2)
        criterion_inset.set_ylabel("$\\widehat{Q}_0(h)$", fontsize=9, labelpad=2)
        criterion_inset.tick_params(axis="both", labelsize=7.5, pad=2)
        criterion_inset.grid(color="#E5E7EB", linewidth=0.5)
    plt.show()


def local_custom_beta_metric(bandwidth: float, squared_differences: np.ndarray, *, beta) -> float:
    """Coefficient of (Delta psi_beta)^2 for this sample."""
    h = float(bandwidth)
    return float(h ** (beta - 2.0) * local_beta2_log_metric(h, squared_differences))


def draw_sample(rng, n):
    return rng.normal(loc=0.0, scale=1.0, size=n)


def inverse_psi_beta(psi: float, beta: float) -> float:
    """
    Inverse of psi_beta.
    """
    psi = float(psi)
    if np.isclose(beta, 2.0):
        return float(np.exp(psi))
    power = 1.0 - beta / 2.0
    return float((power * psi) ** (1.0 / power))


def expected_q_n2_standard_gaussian(bandwidth, sample_size):
    return float(expected_gaussian_qhat_log_energy(np.log(bandwidth), sample_size)[0])


def expected_q_standard_gaussian(bandwidth, sample_size, beta):
    return float(
        expected_gaussian_qhat_log_energy(np.log(bandwidth), sample_size, r=1 - float(beta) / 2)[0]
    )


def asymptotic_c_beta(beta):
    beta = float(beta)
    if not -2.0 < beta < 3.0:
        raise ValueError("Need -2 < beta < 3.")
    return float(((3.0 - beta) / (beta + 2.0)) ** 0.2)


def exact_expected_q_minimizer(sample_size, beta, t_min, t_max, *, search):
    scale = int(sample_size) ** (-0.2)
    report = minimize_bandwidth_energy(
        lambda z: expected_gaussian_qhat_log_energy(z, int(sample_size), r=1 - float(beta) / 2),
        scale * t_min,
        scale * t_max,
        config=search,
    )
    return report["global_minimum"]["h"]


def richardson_local_metric_from_divergence(divergence, delta_psi):
    D = np.maximum(np.asarray(divergence, dtype=float), 0.0)
    number_of_scales = D.shape[0]
    if number_of_scales < 5:
        raise ValueError("Need at least five bandwidths.")
    q_local = np.full(number_of_scales, np.nan, dtype=float)
    i = np.arange(2, number_of_scales - 2)
    one_step = (D[i, i - 1] + D[i, i + 1]) / (2.0 * delta_psi**2)
    two_step = (D[i, i - 2] + D[i, i + 2]) / (8.0 * delta_psi**2)
    q_local[i] = (4.0 * one_step - two_step) / 3.0
    q_local[~np.isfinite(q_local) | (q_local <= 0.0)] = np.nan
    return q_local


def asymptotic_grid_size(sample_size, *, base_size, gamma, sample_sizes):
    n = int(sample_size)
    value = int(np.ceil(base_size * (n / sample_sizes[0]) ** gamma))
    return max(value, 25)


def psi_beta(h: float, beta: float) -> float:
    """
    psi_beta'(h) = h^{-beta/2}.
    """
    h = float(h)
    if np.isclose(beta, 2.0):
        return float(np.log(h))
    power = 1.0 - beta / 2.0
    return float(h**power / power)
