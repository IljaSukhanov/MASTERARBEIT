"""Exploratory associations between sample geometry and Perron peak counts.

Reuse the saved 1,000 samples and the existing peak labels. Compare fixed
feature sets with five-fold out-of-fold logistic predictions. Associations
and predictive performance are descriptive and do not establish causation.

The KDE derivative norm is an original analytic calculation from the same
Gaussian inner products used in kde_tools.density.exact_gaussian_kde_l2_gram.
"""

from pathlib import Path
import json
import os
import sys

CODE_DIRECTORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE_DIRECTORY))
os.environ.setdefault("MPLCONFIGDIR", str(CODE_DIRECTORY / ".matplotlib-cache"))

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

SOURCE_DIRECTORY = CODE_DIRECTORY / "outputs/data/spectral_lambda_jump_study/2ce04db2a16e"
OUTPUT_DIRECTORY = SOURCE_DIRECTORY / "sample_geometry"
Q_BANDWIDTHS = np.geomspace(0.02, 0.8, 61)
PAIR_RADII = np.array([0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28])


def log_bandwidth_derivative_norm(samples, bandwidths):
    r"""Return ||d f_h / d log(h)||_2^2 using analytic Gaussian products.

    With u_rs=(X_r-X_s)^2/(4h^2), the expression is
    (2 sqrt(pi) h n^2)^(-1) sum_rs exp(-u_rs)(u_rs^2-3u_rs+3/4).
    Upper-triangle pairs count twice and n diagonal terms each equal 3/4.
    """
    n = len(samples)
    distances = np.abs(np.subtract.outer(samples, samples)[np.triu_indices(n, 1)])
    u = distances[None, :] ** 2 / (4 * bandwidths[:, None] ** 2)
    summands = np.exp(-u) * (u * u - 3 * u + 0.75)
    values = (0.75 * n + 2 * summands.sum(axis=1)) / (
        2 * np.sqrt(np.pi) * bandwidths * n * n
    )
    if np.any(values <= 0):
        raise ArithmeticError("Analytic derivative squared norms must be positive here.")
    return values


def basic_sample_features(samples):
    n = len(samples)
    scale = float(np.std(samples, ddof=1))
    standardized = (samples - samples.mean()) / scale
    sorted_values = np.sort(standardized)
    spacings = np.diff(sorted_values)
    central = spacings[np.abs(0.5 * (sorted_values[:-1] + sorted_values[1:])) <= 1.5]
    nearest = np.minimum(np.r_[np.inf, spacings], np.r_[spacings, np.inf])
    distances = np.abs(np.subtract.outer(standardized, standardized)[np.triu_indices(n, 1)])
    q25, q75 = np.quantile(standardized, [0.25, 0.75])
    row = {
        "sample_mean": float(samples.mean()), "sample_sd": scale,
        "abs_skewness": float(abs(skew(samples, bias=False))),
        "excess_kurtosis": float(kurtosis(samples, fisher=True, bias=False)),
        "iqr_over_sd": float(q75 - q25),
        "maximum_abs_standardized_value": float(np.abs(standardized).max()),
        "largest_standardized_spacing": float(spacings.max()),
        "central_spacing_cv": float(np.std(central) / np.mean(central)),
        "median_standardized_nearest_distance": float(np.median(nearest)),
    }
    row.update({f"pair_fraction_within_{radius:g}_sd": float(np.mean(distances < radius))
                for radius in PAIR_RADII})
    return row


def validate_derivative_norm():
    """Compare the analytic expression with direct integration of d f / d log(h)."""
    sample = np.random.default_rng(20260823).normal(size=200)
    bandwidths = np.array([0.04, 0.12, 0.3, 0.8])
    analytic = log_bandwidth_derivative_norm(sample, bandwidths)
    spatial_grid = np.linspace(-12.0, 12.0, 16384, endpoint=False)
    rows = []
    for bandwidth, value in zip(bandwidths, analytic):
        z = (spatial_grid[:, None] - sample[None, :]) / bandwidth
        derivative = ((z * z - 1) * np.exp(-0.5 * z * z)).mean(axis=1)
        derivative /= np.sqrt(2 * np.pi) * bandwidth
        direct = float(np.sum(derivative * derivative) * 24 / len(spatial_grid))
        relative_error = abs(float(value) - direct) / direct
        assert relative_error < 1e-10
        rows.append({"h": float(bandwidth), "analytic": float(value),
                     "direct_integral": direct, "relative_discrepancy": relative_error})
    return rows


def variance_controlled_interpolation(samples, example_ids, h_grid, lambda_value):
    """Redistribute sorted points while holding n, mean, and SD fixed.

    This is a deterministic constructed example, not another independent
    sampling experiment or a claim that every interpolation behaves this way.
    """
    from kde_tools.fourier import beta2_divergence_matrix_fft
    from kde_tools.perron import perron_eigenpair_from_divergence
    from scipy.signal import find_peaks

    endpoints = []
    for repetition in example_ids:
        sample = samples[repetition]
        endpoints.append(np.sort((sample - sample.mean()) / sample.std(ddof=1)))
    rows = []
    for fraction in np.linspace(0.0, 1.0, 21):
        sample = (1 - fraction) * endpoints[0] + fraction * endpoints[1]
        sample = (sample - sample.mean()) / sample.std(ddof=1)
        divergence = beta2_divergence_matrix_fft(sample, h_grid)
        result = perron_eigenpair_from_divergence(divergence, lambda_value)
        peaks = find_peaks(result.vector, prominence=1e-8 * result.vector.max())[0]
        rows.append({"interpolation_fraction": float(fraction),
                     "sample_mean": float(sample.mean()), "sample_sd": float(sample.std(ddof=1)),
                     "resolved_peak_count": len(peaks), "peak_h": h_grid[peaks].tolist()})
    return {"endpoint_sample_ids_zero_based": example_ids, "sample_size": len(samples[0]),
            "lambda": float(lambda_value), "common_sample_sd": 1.0,
            "construction": "Interpolate sorted standardized endpoints, then recenter and renormalize SD at each step.",
            "scope": "Original constructed example: geometry changes with fixed n, mean, and SD. Other shape characteristics may change.",
            "profiles": rows}


def run_analysis():
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with np.load(SOURCE_DIRECTORY / "checkpoint.npz", allow_pickle=False) as source:
        samples = source["samples"].copy()
        lambdas = source["lambda_values"].copy()
        h_grid = source["bandwidth_grid"].copy()
    outcomes = pd.read_csv(SOURCE_DIRECTORY / "peak_switches/sample_results.csv")
    records = json.loads((SOURCE_DIRECTORY / "peak_switches/checkpoint.json").read_text())["records"]
    outcome_ids = outcomes["repetition"].to_numpy(dtype=int)
    assert np.array_equal(outcome_ids, np.arange(len(samples)))
    target = outcomes["has_multiple_peaks_including_refined_crossings"].to_numpy(dtype=bool)
    rows = []
    q_profiles = np.empty((len(samples), len(Q_BANDWIDTHS)))
    with threadpool_limits(limits=1):
        for repetition, sample in enumerate(samples):
            row = {"repetition": repetition, "multiple_peaks_observed": bool(target[repetition]),
                   **basic_sample_features(sample)}
            rows.append(row)
            q_profiles[repetition] = log_bandwidth_derivative_norm(sample, Q_BANDWIDTHS)
            if (repetition + 1) % 250 == 0:
                print(f"Sample features: {repetition + 1}/{len(samples)}", flush=True)
        frame = pd.DataFrame(rows)
        ordinary = ["sample_sd", "abs_skewness", "excess_kurtosis", "iqr_over_sd",
                    "maximum_abs_standardized_value"]
        spacing = ["largest_standardized_spacing", "central_spacing_cv",
                   "median_standardized_nearest_distance"]
        pair_columns = [column for column in frame if column.startswith("pair_fraction_")]
        feature_sets = {
            "ordinary_distribution_summaries": (frame[ordinary].to_numpy(), ordinary),
            "summaries_and_spacing": (frame[ordinary + spacing + pair_columns].to_numpy(), ordinary + spacing + pair_columns),
            "multiscale_kde_change_profile": (np.log(q_profiles), [f"log_q_at_h_{h:g}" for h in Q_BANDWIDTHS]),
        }
        # Fixed feature sets, penalty, and folds; no tuning on held-out scores.
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=20261013)
        metrics = {}
        for name, (features, columns) in feature_sets.items():
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=4000))
            probability = cross_val_predict(model, features, target, cv=folds, method="predict_proba")[:, 1]
            frame[f"out_of_fold_probability_{name}"] = probability
            metrics[name] = {
                "feature_columns": columns, "out_of_fold_roc_auc": float(roc_auc_score(target, probability)),
                "out_of_fold_balanced_accuracy": float(balanced_accuracy_score(target, probability >= 0.5)),
            }
            print(name, metrics[name]["out_of_fold_roc_auc"], flush=True)
        group_statistics = []
        for column in ordinary + spacing + pair_columns:
            negative = frame.loc[~target, column].to_numpy()
            positive = frame.loc[target, column].to_numpy()
            pooled = np.sqrt(((len(negative) - 1) * negative.var(ddof=1) +
                              (len(positive) - 1) * positive.var(ddof=1)) / (len(samples) - 2))
            group_statistics.append({
                "feature": column, "one_peak_group_mean": float(negative.mean()),
                "multiple_peak_group_mean": float(positive.mean()),
                "standardized_mean_difference_multiple_minus_one": float((positive.mean() - negative.mean()) / pooled),
            })
        # Similar ordinary summaries, different peak counts at lambda=0.01.
        # Require the two-peak example to pass the stricter sensitivity filter.
        single_ids = np.flatnonzero(~target)
        multiple_ids = np.array([row["repetition"] for row in records if row["peak_counts"]["1e-06"][0] == 2])
        descriptors = frame[["sample_sd", "abs_skewness", "excess_kurtosis", "iqr_over_sd"]].to_numpy()
        descriptors = (descriptors - descriptors.mean(axis=0)) / descriptors.std(axis=0)
        distances = ((descriptors[single_ids, None, :] - descriptors[multiple_ids][None, :, :]) ** 2).sum(axis=2)
        left, right = np.unravel_index(np.argmin(distances), distances.shape)
        example_ids = [int(single_ids[left]), int(multiple_ids[right])]
        from kde_tools.fourier import beta2_divergence_matrix_fft
        from kde_tools.perron import perron_eigenpair_from_divergence
        from scipy.signal import find_peaks
        example_profiles = []
        for repetition in example_ids:
            divergence = beta2_divergence_matrix_fft(samples[repetition], h_grid)
            result = perron_eigenpair_from_divergence(divergence, lambdas[0])
            peaks = find_peaks(result.vector, prominence=1e-8 * result.vector.max())[0]
            minima = find_peaks(-divergence.sum(axis=1))[0]
            example_profiles.append({
                "repetition": repetition, "lambda": float(lambdas[0]),
                "ordinary_features": {column: float(frame.iloc[repetition][column]) for column in ordinary},
                "perron_peak_indices": peaks.tolist(), "perron_peak_h": h_grid[peaks].tolist(),
                "divergence_row_sum_local_minimum_h": h_grid[minima].tolist(),
                "perron_vector": result.vector.tolist(), "divergence_row_sums": divergence.sum(axis=1).tolist(),
            })
    summary = {
        "samples": len(samples), "one_peak_observed_samples": int(sum(~target)),
        "multiple_peaks_observed_samples": int(sum(target)),
        "outcome_definition": "Existing 201-point bandwidth grid, 101 lambdas, and refined candidate crossings; relative prominence 1e-8.",
        "method": "Original exploratory analysis; fixed five-fold out-of-fold standardized logistic models, C=1, no score-based tuning. No causal inference.",
        "models": metrics, "group_statistics": group_statistics,
        "matched_example_profiles": example_profiles,
        "matched_example_selection": "Minimum standardized distance on SD, absolute skewness, excess kurtosis, and IQR/SD; one-peak group versus two peaks at lambda=0.01, with the two-peak example also passing relative prominence 1e-6.",
        "geometry": "Analytic Gaussian KDE inner products depend on pairwise sample distances. Translation, reflection, and permutation preserve the exact divergence matrix.",
        "derivative_norm_formula": "q(h)=(2*sqrt(pi)*h*n^2)^(-1) sum_rs exp(-u_rs)*(u_rs^2-3*u_rs+3/4), u_rs=(X_r-X_s)^2/(4*h^2).",
        "derivative_norm_interpretation": "Squared L2 norm of d f_h / d log(h); an original derivation, not a universal peak-count criterion.",
        "derivative_norm_validation": validate_derivative_norm(),
        "variance_controlled_interpolation": variance_controlled_interpolation(
            samples, example_ids, h_grid, lambdas[0]
        ),
    }
    frame.to_csv(OUTPUT_DIRECTORY / "sample_features.csv", index=False)
    np.savez_compressed(OUTPUT_DIRECTORY / "kde_change_profiles.npz", bandwidths=Q_BANDWIDTHS,
                        q_profiles=q_profiles, multiple_peak_labels=target, bandwidth_grid=h_grid)
    (OUTPUT_DIRECTORY / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved analysis: {OUTPUT_DIRECTORY}", flush=True)
    print("Matched examples:", [(row["repetition"], row["perron_peak_h"]) for row in example_profiles], flush=True)
    return summary


def plot_matched_examples(summary):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    from kde_tools.density import gaussian_kde_values
    from kde_tools.pdf_export import enable_automatic_pdf_saving

    enable_automatic_pdf_saving("spectral_lambda_jump_study")
    image_directory = CODE_DIRECTORY / "outputs/images/spectral_lambda_jump_study/sample_geometry"
    image_directory.mkdir(parents=True, exist_ok=True)
    with np.load(SOURCE_DIRECTORY / "checkpoint.npz", allow_pickle=False) as source:
        samples = source["samples"].copy()
        h_grid = source["bandwidth_grid"].copy()
    examples = summary["matched_example_profiles"]
    colors = ["#2674a6", "#bd6225"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))
    for column, (example, color) in enumerate(zip(examples, colors)):
        sample = samples[example["repetition"]]
        sample = sample - sample.mean()
        x_grid = np.linspace(-3.5, 3.5, 1401)
        estimates = gaussian_kde_values(sample, np.array([0.1, 0.4]), x_grid)
        density_ax = axes[0, column]
        density_ax.plot(x_grid, estimates[0], color=color, lw=1.5, label="KDE, h = 0.10")
        density_ax.plot(x_grid, estimates[1], color="#535b63", lw=1.6, ls="--", label="KDE, h = 0.40")
        density_ax.plot(sample, np.full(len(sample), -0.02), "|", color=color, alpha=0.5, markersize=5)
        density_ax.set_xlim(-3.5, 3.5)
        density_ax.set_ylim(-0.04, 0.55)
        density_ax.set_xlabel("Sample value, centered")
        density_ax.set_ylabel("Estimated density")
        features = example["ordinary_features"]
        peak_label = "1 Perron peak" if len(example["perron_peak_h"]) == 1 else "2 Perron peaks"
        density_ax.set_title(
            f"Sample {example['repetition']} ({peak_label})\n"
            f"SD {features['sample_sd']:.4f}; |skew| {features['abs_skewness']:.4f}; "
            f"excess kurtosis {features['excess_kurtosis']:.4f}", fontsize=10.5
        )
        density_ax.legend(fontsize=8.5, loc="upper left")
        density_ax.grid(alpha=0.15)

        vector = np.array(example["perron_vector"])
        relative = (vector / vector.max() - 1) * 1e6
        profile_ax = axes[1, column]
        profile_ax.plot(h_grid, relative, color=color, lw=1.6)
        peak_indices = np.array(example["perron_peak_indices"], dtype=int)
        profile_ax.scatter(h_grid[peak_indices], relative[peak_indices], s=45, color=color, zorder=3)
        for index in peak_indices:
            profile_ax.annotate(f"h = {h_grid[index]:.3f}",
                                (h_grid[index], relative[index]),
                                xytext=(0, 12), textcoords="offset points", ha="center", fontsize=9)
        inside = (h_grid >= 0.04) & (h_grid <= 0.65)
        low = float(relative[inside].min())
        profile_ax.set_ylim(low * 1.12, -low * 0.22)
        profile_ax.set_xlim(0.04, 0.65)
        profile_ax.set_xscale("log")
        profile_ax.set_xticks([0.04, 0.08, 0.16, 0.32, 0.64])
        profile_ax.xaxis.set_major_formatter(ScalarFormatter())
        profile_ax.set_xticks([], minor=True)
        profile_ax.set_xlabel("Bandwidth h (logarithmic scale)")
        profile_ax.set_ylabel(r"$(v(h)/\max v - 1)\times 10^6$")
        profile_ax.set_title(r"Perron profile at the same $\lambda = 0.01$", fontsize=11)
        profile_ax.grid(alpha=0.15)
    fig.suptitle("Sample spacing and the number of Perron peaks", fontsize=14)
    fig.text(0.5, 0.015, "Examples matched on SD, absolute skewness, excess kurtosis, and IQR/SD. "
             "Sample IDs are zero based; profiles use the existing 201-point bandwidth grid.",
             ha="center", fontsize=8.5)
    fig.tight_layout(rect=[0, 0.045, 1, 0.96], h_pad=2.0)
    path = image_directory / "matched_samples_one_vs_two_peaks.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.show()  # Shared exporter automatically saves the displayed figure as PDF.
    plt.close(fig)
    return path


if __name__ == "__main__":
    run_analysis()
