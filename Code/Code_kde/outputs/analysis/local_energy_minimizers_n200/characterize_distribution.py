"""Describe and fit the conditional distributions of saved local minimizers.

These fits are finite-sample approximations, not exact or asymptotic laws.
No-local-minimum outcomes are counted separately. Model comparisons use
AIC and ten-fold predictive log likelihood. The KS distance is descriptive;
no unadjusted KS p-values are reported after parameter fitting.
"""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

KDE_DIR = Path(__file__).resolve().parents[3]
from kde_tools.energy_minimization import unified_energy_run_directory, energy_search_settings, energy_search_id

DATA_DIR = unified_energy_run_directory(KDE_DIR / "outputs/data/local_energy_minimizers_n200")
MODELS = {
    "normal": (stats.norm, {}, 2),
    "lognormal": (stats.lognorm, {"floc": 0}, 2),
    "gamma": (stats.gamma, {"floc": 0}, 2),
    "skew_normal": (stats.skewnorm, {}, 3),
}


def main():
    data = pd.read_csv(DATA_DIR/"results.csv")
    descriptions, fits, predictions = [], [], []
    for method in ("r0", "r13over14", "r1"):
        part = data[data.method == method]
        observed = part.dropna(subset=["selected_h"])
        x = observed.selected_h.to_numpy()
        sample_ids = observed.sample_id.to_numpy(dtype=int)
        descriptions.append({
            "method": method, "observed_minima": len(x), "samples": len(part),
            "no_minimum_fraction": float(part.selected_h.isna().mean()),
            "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
            "skewness": float(stats.skew(x, bias=False)),
            "excess_kurtosis": float(stats.kurtosis(x, bias=False)),
            "q05": float(np.quantile(x, .05)), "median": float(np.median(x)),
            "q95": float(np.quantile(x, .95)),
        })
        for name, (distribution, options, parameters) in MODELS.items():
            fit = tuple(float(v) for v in distribution.fit(x, **options))
            log_likelihood = float(distribution.logpdf(x, *fit).sum())
            out_of_sample = np.empty(len(x))
            for fold in range(10):
                test = sample_ids % 10 == fold
                fitted = distribution.fit(x[~test], **options)
                out_of_sample[test] = distribution.logpdf(x[test], *fitted)
            if not np.all(np.isfinite(out_of_sample)):
                raise ArithmeticError("Nonfinite cross-validated log likelihood.")
            fits.append({
                "method": method, "model": name, "parameters_json": json.dumps(fit),
                "number_of_free_parameters": parameters,
                "log_likelihood": log_likelihood, "AIC": 2*parameters-2*log_likelihood,
                "KS_distance_descriptive": float(stats.kstest(x, distribution.cdf, args=fit).statistic),
                "ten_fold_mean_predictive_log_likelihood": float(out_of_sample.mean()),
            })
            for sample_id, log_score in zip(sample_ids, out_of_sample):
                predictions.append({"sample_id": int(sample_id), "method": method,
                                    "model": name, "held_out_log_density": float(log_score)})
    descriptive = pd.DataFrame(descriptions)
    fitted = pd.DataFrame(fits)
    fitted["delta_AIC"] = fitted.AIC-fitted.groupby("method").AIC.transform("min")
    descriptive.to_csv(DATA_DIR/"distribution_descriptive.csv", index=False)
    fitted.to_csv(DATA_DIR/"distribution_fits.csv", index=False)
    pd.DataFrame(predictions).to_csv(DATA_DIR/"distribution_predictive_scores.csv", index=False)
    print(descriptive.to_string(index=False))
    print(fitted.to_string(index=False))
    return descriptive, fitted


def plot_fits(descriptive, fitted):
    import sys
    if str(KDE_DIR) not in sys.path:
        sys.path.insert(0, str(KDE_DIR))
    import matplotlib.pyplot as plt

    data = pd.read_csv(DATA_DIR/"results.csv")
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.1), sharex=True, sharey=True)
    bins = np.linspace(0, .85, 41)
    grid = np.linspace(0, .85, 700)
    for ax, method, label, color in zip(
        axes, ("r0", "r13over14", "r1"), ("0", "13/14", "1"),
        ("#0072B2", "#009E73", "#D55E00"),
    ):
        x = data.loc[data.method == method, "selected_h"].dropna().to_numpy()
        info = descriptive[descriptive.method == method].iloc[0]
        models = fitted[fitted.method == method]
        best = models.loc[models.AIC.idxmin()]
        normal = models[models.model == "normal"].iloc[0]
        ax.hist(x, bins=bins, density=True, color=color, alpha=.55,
                edgecolor="white", linewidth=.5)
        if best.model != "normal":
            params = json.loads(normal.parameters_json)
            ax.plot(grid, stats.norm.pdf(grid, *params), color="#777777",
                    linestyle="--", linewidth=1.5, label="normal fit")
        distribution = MODELS[best.model][0]
        params = json.loads(best.parameters_json)
        ax.plot(grid, distribution.pdf(grid, *params), color="#222222", linewidth=2.1,
                label=best.model.replace("_", "-")+" fit")
        ax.set(title=rf"$r={label}$", xlabel=r"local minimizer $h$", xlim=(0, .85))
        ax.legend(loc="upper right", frameon=True, framealpha=1, fontsize=9)
        ax.text(.97, .75,
                f"observed no-minimum rate: {info.no_minimum_fraction:.1%}\n"
                f"sample skewness: {info.skewness:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.7,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .95})
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("density conditional on a local minimum")
    fig.suptitle(r"Conditional distributions of local minimizers: empirical fits ($n=200$)", fontsize=16)
    fig.text(.5, .018,
             "Same 1,000 paired standard-normal samples; lowest-energy interior minimum when several exist.\n"
             "Fits are descriptive approximations. Samples without a minimum are accounted for separately.",
             ha="center", fontsize=9.5, color="#444444")
    fig.tight_layout(rect=(0, .105, 1, .94), w_pad=2)
    directory = KDE_DIR/"outputs/images/local_energy_minimizers_n200"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/"distribution_fits.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print("PNG:", path)
    return path


def plot_histogram_overlay():
    """Overlay only the histogram bars, using common conditional-density bins."""
    import sys
    if str(KDE_DIR) not in sys.path:
        sys.path.insert(0, str(KDE_DIR))
    import matplotlib.pyplot as plt

    data = pd.read_csv(DATA_DIR/"results.csv")
    bins = np.linspace(0, .85, 41)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5.7))
    records = []
    for method, label, color in zip(
        ("r0", "r13over14", "r1"), ("0", "13/14", "1"),
        ("#0072B2", "#009E73", "#D55E00"),
    ):
        x = data.loc[data.method == method, "selected_h"].dropna().to_numpy()
        counts, _ = np.histogram(x, bins=bins)
        assert counts.sum() == len(x)
        density = counts/(len(x)*np.diff(bins))
        assert np.isclose(np.sum(density*np.diff(bins)), 1.0)
        ax.hist(x, bins=bins, density=True, color=color, alpha=.42,
                edgecolor=color, linewidth=.55,
                label=rf"$r={label}$ ({len(x)} minima)")
        for left, right, count, height in zip(bins[:-1], bins[1:], counts, density):
            records.append({"method": method, "bin_left": left, "bin_right": right,
                            "count": int(count), "conditional_density": height})
    ax.set(xlim=(0, .85), xlabel=r"interior local minimizer $h$",
           ylabel="density conditional on a local minimum",
           title=r"Overlaid histograms of interior local minimizers ($n=200$)")
    ax.legend(loc="upper right", frameon=True, framealpha=1, fontsize=10)
    ax.grid(axis="x", visible=False)
    fig.text(.5, .017,
             "1,000 paired standard-normal samples; lowest-energy interior minimum when several exist.",
             ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, .045, 1, 1))
    directory = KDE_DIR/"outputs/images/local_energy_minimizers_n200"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/"histogram_overlay.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.show()  # Shared automatic PDF export.
    plt.close(fig)
    pd.DataFrame(records).to_csv(DATA_DIR/"histogram_overlay.csv", index=False)
    print("PNG:", path)
    return path


def plot_figure39_style():
    """Use the outlined, geometric-bin, percentage style of current Figure 3.9.

    Frequencies divide by all 1,000 samples, as in that reference figure.
    Original samples and selected local minima are unchanged.
    """
    import sys
    if str(KDE_DIR) not in sys.path:
        sys.path.insert(0, str(KDE_DIR))
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    data = pd.read_csv(DATA_DIR/"results.csv")
    # Same bandwidth-grid centres and index-bin convention as Figure 3.9's
    # continuous_ise_qhat_distributions_vs_n notebook.
    bandwidths = np.geomspace(.01, 2.0, 201)
    positions = np.arange(len(bandwidths))
    edges = np.arange(len(bandwidths)+1)-.5
    methods = ("r0", "r13over14", "r1")
    labels = (r"$\widehat Q_0$", r"$\widehat Q_{13/14}$", r"$\widehat Q_1$")
    colors = ("#0072B2", "#009E73", "#D55E00")
    styles = ("-", "-.", "--")
    frequencies, records = [], []
    for method in methods:
        part = data[data.method == method]
        x = part.selected_h.dropna().to_numpy()
        if np.any(x < bandwidths[0]) or np.any(x > bandwidths[-1]):
            raise ValueError("The reference bandwidth bins do not cover the selected minima.")
        locations = np.interp(np.log(x), np.log(bandwidths), positions)
        counts, _ = np.histogram(locations, bins=edges)
        assert counts.sum() == len(x)
        frequency = counts/len(part)
        missing = float(part.selected_h.isna().mean())
        assert np.isclose(frequency.sum()+missing, 1.0)
        frequencies.append(frequency)
        for index, (count, value) in enumerate(zip(counts, frequency)):
            records.append({"method": method, "grid_index": index,
                            "bandwidth_center": bandwidths[index], "count": int(count),
                            "relative_frequency_all_samples": value,
                            "denominator": len(part)})
    occupied = np.flatnonzero(np.any(np.stack(frequencies) > 0, axis=0))
    last = min(len(bandwidths)-1, int(occupied[-1])+3)
    ticks = np.unique(np.linspace(0, last, 7, dtype=int))
    ymax = 1.14*max(float(x.max()) for x in frequencies)

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for values, color, style in zip(frequencies, colors, styles):
        ax.stairs(values, edges, fill=True, color=color, alpha=.07, linewidth=0)
        ax.stairs(values, edges, color=color, linestyle=style, linewidth=1.5)
    ax.set_title(r"$n=200$", fontsize=13, pad=10)
    ax.set_xticks(ticks, [f"{bandwidths[i]:.3g}" for i in ticks])
    ax.set_xlim(-.7, last+.5)
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("continuously selected bandwidth h", fontsize=10)
    ax.set_ylabel("relative frequency", fontsize=10)
    ax.tick_params(labelsize=9, colors="#304967", labelbottom=True, labelleft=True)
    ax.grid(axis="y", color="#DDE5EC", linewidth=.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#B8C5D1")

    fig.suptitle("Interior local bandwidth minimizers", fontsize=16, y=.985)
    fig.text(.5, .925, "Gaussian samples; M = 1000", ha="center", fontsize=10)
    handles = [Line2D([0], [0], color=color, linestyle=style, linewidth=2, label=label)
               for color, style, label in zip(colors, styles, labels)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .90),
               ncol=3, frameon=False, fontsize=12)
    fig.text(.5, .025,
             "Bin frequencies divide by all 1,000 samples.\n"
             "Lowest-energy interior minimum selected when several exist.",
             ha="center", fontsize=8.5, color="#444444")
    fig.subplots_adjust(left=.11, right=.98, bottom=.17, top=.75)
    directory = KDE_DIR/"outputs/images/local_energy_minimizers_n200"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/"histogram_overlay_figure39_style.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.show()  # Shared automatic PDF export under the notebook name.
    plt.close(fig)
    pd.DataFrame(records).to_csv(DATA_DIR/"histogram_overlay_figure39_style.csv", index=False)
    print("PNG:", path)
    return path


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(KDE_DIR))
    from kde_tools.pdf_export import enable_automatic_pdf_saving
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--histogram-overlay", action="store_true")
    parser.add_argument("--figure-39-style", action="store_true")
    args = parser.parse_args()
    enable_automatic_pdf_saving("local_energy_minimizers_n200")
    if args.figure_39_style:
        plot_figure39_style()
    elif args.histogram_overlay:
        plot_histogram_overlay()
    else:
        if args.plot_only:
            descriptive = pd.read_csv(DATA_DIR/"distribution_descriptive.csv")
            fitted = pd.read_csv(DATA_DIR/"distribution_fits.csv")
        else:
            descriptive, fitted = main()
        plot_fits(descriptive, fitted)
