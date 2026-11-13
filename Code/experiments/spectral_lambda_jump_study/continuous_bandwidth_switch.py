"""Evaluate the Perron kernel between bandwidth nodes and refine peak crossings.

For the sampled Gaussian KDE family, evaluate the kernel itself at new h:
    phi_N(h) = sum_j exp(-lambda D(h,h_j)) v_j / rho(S).
At basis nodes this agrees with the matrix Perron vector. This is the
Nyström extension of the discrete eigenvector, not a fitted interpolation.
Its stationary points are found using analytic derivatives in log(h).

Endpoint-node and midpoint-node grids approximate the same uniform measure
in log(h) on [log(0.01), log(2)]. Convergence here is numerical evidence,
not an interval-certified theorem about the integral operator.
"""

from pathlib import Path
import json
import os
import sys

CODE_DIRECTORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_DIRECTORY))
os.environ.setdefault("MPLCONFIGDIR", str(CODE_DIRECTORY / ".matplotlib-cache"))

import numpy as np
from scipy.optimize import brentq
from scipy.signal import find_peaks
from threadpoolctl import threadpool_limits

from kde_tools.fourier import beta2_divergence_matrix_fft
from kde_tools.perron import perron_eigenpair_from_divergence
from kde_tools.similarities import beta2_divergence_matrix

OUTPUT_DIRECTORY = CODE_DIRECTORY / "outputs/data/spectral_lambda_jump_study/continuous_bandwidth_switch"


class KernelFamily:
    def __init__(self, sample, size, placement):
        self.sample = sample
        self.size = size
        self.placement = placement
        a, b = np.log(0.01), np.log(2.0)
        self.h_grid = (np.exp(np.linspace(a, b, size)) if placement == "endpoint"
                       else np.exp(a + (np.arange(size) + 0.5) * (b - a) / size))
        self.frequencies = 2 * np.pi * np.fft.rfftfreq(8192, d=24 / 8192)
        self.omega2 = self.frequencies ** 2
        self.omega4 = self.omega2 ** 2
        empirical_cf = np.exp(-1j * self.frequencies[:, None] * sample[None, :]).mean(axis=1)
        self.weights = 2 * np.abs(empirical_cf) ** 2 / 24
        self.weights[[0, -1]] *= 0.5
        self.basis = np.exp(-0.5 * self.h_grid[:, None] ** 2 * self.omega2[None, :])
        self.weighted_basis_transpose = (self.basis * self.weights).T
        gram = self.basis @ self.weighted_basis_transpose
        self.diagonal_gram = np.diag(gram).copy()
        self.divergence = np.maximum(0.5 * (self.diagonal_gram[:, None] + self.diagonal_gram[None, :]) - gram, 0)
        np.fill_diagonal(self.divergence, 0)
        self.cache = {}

    def eigenpair(self, lambda_value):
        key = float(lambda_value)
        if key not in self.cache:
            result = perron_eigenpair_from_divergence(self.divergence, key)
            # The shared eigenvalue belongs to exp(lambda*d_min)*(S-I).
            rho = 1 + np.exp(-key * result.minimum_divergence) * result.eigenvalue
            self.cache[key] = (result, float(rho))
        return self.cache[key]

    def evaluate(self, log_h, lambda_value, derivatives=False):
        h2 = float(np.exp(2 * log_h))
        smoother = np.exp(-0.5 * h2 * self.omega2)
        cross = smoother @ self.weighted_basis_transpose
        diagonal = float(np.sum(smoother * smoother * self.weights))
        divergence = np.maximum(0.5 * (diagonal + self.diagonal_gram) - cross, 0)
        kernel_row = np.exp(-lambda_value * divergence)
        result, rho = self.eigenpair(lambda_value)
        function_value = float(kernel_row @ result.vector / rho)
        if not derivatives:
            return function_value
        cross2 = (smoother * self.omega2) @ self.weighted_basis_transpose
        diagonal2 = float(np.sum(smoother * smoother * self.weights * self.omega2))
        cross4 = (smoother * self.omega4) @ self.weighted_basis_transpose
        diagonal4 = float(np.sum(smoother * smoother * self.weights * self.omega4))
        derivative_d = h2 * (cross2 - diagonal2)
        second_d = ((2 * h2 * h2 * diagonal4 - 2 * h2 * diagonal2)
                    - (h2 * h2 * cross4 - 2 * h2 * cross2))
        first = float((-lambda_value * derivative_d * kernel_row) @ result.vector / rho)
        second = float(((lambda_value ** 2 * derivative_d ** 2 - lambda_value * second_d)
                        * kernel_row) @ result.vector / rho)
        return function_value, first, second

    def local_peaks(self, lambda_value):
        peak_rows = []
        for low, high in ((0.095, 0.13), (0.18, 0.23)):
            root = brentq(lambda log_h: self.evaluate(log_h, lambda_value, True)[1],
                          np.log(low), np.log(high), xtol=1e-11, rtol=1e-12)
            value, derivative, curvature = self.evaluate(root, lambda_value, True)
            assert curvature < 0
            peak_rows.append({"h": float(np.exp(root)), "height": value,
                              "log_h_derivative": derivative,
                              "relative_log_h_curvature": curvature / value})
        return peak_rows

    def height_difference(self, lambda_value):
        peaks = self.local_peaks(lambda_value)
        return peaks[0]["height"] - peaks[1]["height"]

    def crossing(self):
        value = brentq(self.height_difference, 17.0, 19.5, xtol=1e-8, rtol=1e-11)
        peaks = self.local_peaks(value)
        valley_log_h = brentq(lambda log_h: self.evaluate(log_h, value, True)[1],
                              np.log(peaks[0]["h"] * 1.01), np.log(peaks[1]["h"] / 1.01), xtol=1e-11)
        valley_height, _, valley_curvature = self.evaluate(valley_log_h, value, True)
        assert valley_curvature > 0
        delta = 1e-3
        slope = (self.height_difference(value + delta) - self.height_difference(value - delta)) / (2 * delta)
        result, rho = self.eigenpair(value)
        node_errors = [abs(self.evaluate(np.log(self.h_grid[k]), value) - result.vector[k])
                       for k in (0, self.size // 2, self.size - 1)]
        return {"grid_size": self.size, "placement": self.placement, "lambda_crossing": value,
                "peaks": peaks, "peak_bandwidth_ratio": peaks[1]["h"] / peaks[0]["h"],
                "valley_h": float(np.exp(valley_log_h)),
                "relative_valley_depth": (min(row["height"] for row in peaks) - valley_height) / max(row["height"] for row in peaks),
                "relative_height_gap_slope_at_crossing": slope / max(row["height"] for row in peaks),
                "maximum_extension_discrepancy_at_checked_basis_nodes": max(node_errors),
                "relative_residual": result.relative_residual, "relative_spectral_gap": result.relative_gap,
                "height_gap_below": self.height_difference(value - 0.1),
                "height_gap_above": self.height_difference(value + 0.1)}

    def profile(self, h_values, lambda_value):
        result, rho = self.eigenpair(lambda_value)
        output = np.empty(len(h_values))
        for first in range(0, len(h_values), 64):
            current = h_values[first:first + 64]
            smoother = np.exp(-0.5 * current[:, None] ** 2 * self.omega2[None, :])
            cross = smoother @ self.weighted_basis_transpose
            diagonal = np.sum(smoother * smoother * self.weights, axis=1)
            divergence = np.maximum(0.5 * (diagonal[:, None] + self.diagonal_gram[None, :]) - cross, 0)
            output[first:first + len(current)] = np.exp(-lambda_value * divergence) @ result.vector / rho
        return output


def run_analysis():
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    sample = np.random.default_rng(20260823).normal(size=200)
    convergence_path = OUTPUT_DIRECTORY / "convergence.json"
    rows = json.loads(convergence_path.read_text()) if convergence_path.is_file() else []
    done = {(row["placement"], row["grid_size"]) for row in rows}
    with threadpool_limits(limits=1):
        for placement in ("endpoint", "midpoint"):
            for size in (201, 401, 801, 1601):
                if (placement, size) in done:
                    continue
                family = KernelFamily(sample, size, placement)
                row = family.crossing()
                if placement == "endpoint" and size == 201:
                    exact = beta2_divergence_matrix(sample, family.h_grid)
                    fft = beta2_divergence_matrix_fft(sample, family.h_grid)
                    row["maximum_divergence_discrepancy_from_analytic_gaussian"] = float(np.abs(family.divergence - exact).max())
                    row["maximum_divergence_discrepancy_from_existing_fft"] = float(np.abs(family.divergence - fft).max())
                rows.append(row)
                print(f"Continuous kernel peaks, {placement} N={size}: "
                      f"lambda={row['lambda_crossing']:.9f}; "
                      f"h={row['peaks'][0]['h']:.7f}, {row['peaks'][1]['h']:.7f}; "
                      f"relative valley={row['relative_valley_depth']:.3e}", flush=True)
                convergence_path.write_text(json.dumps(rows, indent=2) + "\n")
        family = KernelFamily(sample, 801, "midpoint")
        crossing = next(row["lambda_crossing"] for row in rows if row["grid_size"] == 801 and row["placement"] == "midpoint")
        h_values = np.geomspace(0.01, 2.0, 2401)
        phases = np.array([crossing - 0.1, crossing, crossing + 0.1])
        profiles = np.array([family.profile(h_values, value) for value in phases])
        phase_peaks = [family.local_peaks(value) for value in phases]
        gap_lambdas = np.linspace(crossing - 0.4, crossing + 0.4, 81)
        peaks_over_lambda = [family.local_peaks(value) for value in gap_lambdas]
        relative_gaps = np.array([(peaks[0]["height"] - peaks[1]["height"]) /
                                  max(row["height"] for row in peaks) for peaks in peaks_over_lambda])
        for profile in profiles:
            maxima = find_peaks(profile, prominence=1e-8 * profile.max())[0]
            assert len(maxima) == 2
            assert max(profile[0], profile[-1]) < profile[maxima].min()
        # Analytic derivatives checked independently by finite differences.
        checks = []
        for h_value in (0.105, 0.15, 0.215):
            t = np.log(h_value)
            first_step, second_step = 1e-5, 1e-4
            value, first, second = family.evaluate(t, crossing, True)
            first_left, first_right = family.evaluate(t - first_step, crossing), family.evaluate(t + first_step, crossing)
            second_left, second_right = family.evaluate(t - second_step, crossing), family.evaluate(t + second_step, crossing)
            numeric_first = (first_right - first_left) / (2 * first_step)
            numeric_second = (second_right - 2 * value + second_left) / second_step ** 2
            checks.append({"h": h_value, "analytic_first": first, "numeric_first": numeric_first,
                           "analytic_second": second, "numeric_second": numeric_second,
                           "first_difference_step": first_step, "second_difference_step": second_step})
            assert abs(first - numeric_first) < 1e-10 * value
            assert abs(second - numeric_second) < 5e-6 * value
    summary = {"sample_seed": 20260823, "sample_size": 200,
               "bandwidth_interval": [0.01, 2.0], "measure": "uniform in log(h)",
               "continuous_extension": "phi_N(h) = sum_j exp(-lambda D(h,h_j))*v_j/rho(S)",
               "definition_scope": "Kernel evaluation at new bandwidths; this is a Nyström approximation, not a fitted interpolation or a certified continuum theorem.",
               "convergence": rows, "derivative_checks": checks,
               "illustration_grid": {"size": 801, "placement": "midpoint", "lambda_crossing": crossing},
               "phase_lambda_values": phases.tolist(), "resolved_maxima_in_each_extended_phase": [2, 2, 2]}
    summary["phase_peaks"] = phase_peaks
    np.savez_compressed(OUTPUT_DIRECTORY / "continuous_profiles.npz", bandwidths=h_values,
                        lambda_phases=phases, profiles=profiles, gap_lambdas=gap_lambdas,
                        relative_height_gaps=relative_gaps)
    (OUTPUT_DIRECTORY / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved continuous-kernel diagnostics: {OUTPUT_DIRECTORY}", flush=True)
    return summary


def plot_analysis(summary=None):
    """Display the two competing kernel peaks and their continuous height gap."""
    import matplotlib.pyplot as plt
    from kde_tools.pdf_export import enable_automatic_pdf_saving

    if summary is None:
        summary = json.loads((OUTPUT_DIRECTORY / "summary.json").read_text())
    enable_automatic_pdf_saving("spectral_lambda_jump_study")
    image_directory = CODE_DIRECTORY / "outputs/images/spectral_lambda_jump_study/numerical_vs_sampling"
    image_directory.mkdir(parents=True, exist_ok=True)
    with np.load(OUTPUT_DIRECTORY / "continuous_profiles.npz", allow_pickle=False) as saved:
        h_values = saved["bandwidths"]
        profiles = saved["profiles"]
        phases = saved["lambda_phases"]
        gap_lambdas = saved["gap_lambdas"]
        relative_gaps = saved["relative_height_gaps"]

    crossing = summary["illustration_grid"]["lambda_crossing"]
    colors = ["#2563a6", "#737373", "#d96927"]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.35), constrained_layout=True)
    fig.suptitle("Figure 3.3 sample: two smooth peaks exchange dominance", fontsize=15)
    for index in (0, 2):
        peaks = summary["phase_peaks"][index]
        maximum = max(peak["height"] for peak in peaks)
        axes[0].plot(h_values, 1e4 * (profiles[index] / maximum - 1),
                     color=colors[index], lw=2, label=rf"$\lambda={phases[index]:.4f}$")
        winner = max(peaks, key=lambda peak: peak["height"])
        axes[0].plot(winner["h"], 0, marker="*", color=colors[index],
                     markersize=13, markeredgecolor="white", markeredgewidth=0.6)
        other = min(peaks, key=lambda peak: peak["height"])
        axes[0].plot(other["h"], 1e4 * (other["height"] / maximum - 1),
                     marker="o", color=colors[index], ms=5, mfc="white")
    axes[0].set(xlim=(0.102, 0.220), ylim=(-1.55, 0.18),
                xlabel="Bandwidth h", ylabel=r"$(\phi(h)/\max\phi-1)\times10^4$",
                title="Kernel evaluated between bandwidth nodes")
    axes[0].text(0.5, 0.045, "Stars mark the selected maximum; circles mark the other peak.",
                 transform=axes[0].transAxes, ha="center", fontsize=8.5,
                 bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9})
    axes[0].legend(loc="lower left", bbox_to_anchor=(0.01, 0.10), frameon=False, fontsize=9)

    axes[1].plot(gap_lambdas, 1e6 * relative_gaps, color="#3d536c", lw=2)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].axvline(crossing, color="#888888", lw=1, ls="--")
    axes[1].scatter(phases[[0, 2]],
                    [1e6 * (peaks[0]["height"] - peaks[1]["height"]) /
                     max(peak["height"] for peak in peaks)
                     for peaks in (summary["phase_peaks"][0], summary["phase_peaks"][2])],
                    color=[colors[0], colors[2]], s=40, zorder=3)
    axes[1].set(xlabel=r"Similarity parameter $\lambda$",
                ylabel=r"$(\phi_{\rm small}-\phi_{\rm large})/\max\phi\times10^6$",
                title="The peak-height difference crosses zero")
    axes[1].text(0.035, 0.90, "Smaller bandwidth wins", color=colors[0],
                 transform=axes[1].transAxes, fontsize=10)
    axes[1].text(0.965, 0.08, "Larger bandwidth wins", color=colors[2],
                 transform=axes[1].transAxes, ha="right", fontsize=10)
    axes[1].text(crossing + 0.018, 1.5, rf"$\lambda_c\approx{crossing:.4f}$", fontsize=9)
    for axis in axes:
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    image_path = image_directory / "continuous_peak_height_crossing.png"
    fig.savefig(image_path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return image_path


if __name__ == "__main__":
    run_analysis()
