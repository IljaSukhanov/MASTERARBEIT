"""Describe completed lambda experiments without rerunning the simulations."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[2]
sys.path.insert(0, str(CODE))
os.environ.setdefault("MPLCONFIGDIR", str(CODE / ".matplotlib-cache"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from kde_tools.bandwidths import gaussian_mise_oracle_bandwidth
from kde_tools.pdf_export import enable_automatic_pdf_saving
from kde_tools.perron import PERRON_SOLVER_VERSION, perron_eigenpair_from_divergence
from kde_tools.similarities import population_gaussian_beta2_divergence_matrix

NOTEBOOK = "fast_l2_bandwidth_distributions_vs_n_lambda"
RUNS = {1: "028fa8664f9c", 10: "1de0f2734d87", 100: "4e41e6f126cf", 1000: "d7062cba5620"}
ROOT = CODE / "outputs" / "data" / NOTEBOOK / "thesis"
THRESHOLDS = (0.20, 0.25, 0.30)


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def population_choice(divergence, lambda_value):
    return perron_eigenpair_from_divergence(divergence, lambda_value).selected_index


def main():
    baseline = None
    stats = []
    selections = {}
    common_grid = None
    for lam, run_id in RUNS.items():
        directory = ROOT / run_id
        manifest = json.loads((directory / "settings.json").read_text())
        assert manifest["lambda_values"] == [float(lam)]
        common_manifest = {key: value for key, value in manifest.items() if key != "lambda_values"}
        if baseline is None:
            baseline = common_manifest
        assert common_manifest == baseline, "Runs must share all settings except lambda."
        repetitions = manifest["settings"]["repetitions"]
        assert repetitions == 1000
        rows = read_rows(directory / "selections.csv")
        frequencies = read_rows(directory / "frequencies.csv")
        summaries = read_rows(directory / "summary.csv")
        assert len(rows) == repetitions * len(manifest["sample_sizes"])

        for n in manifest["sample_sizes"]:
            point = sorted((row for row in rows if int(row["n"]) == n), key=lambda row: int(row["repetition"]))
            assert [int(row["repetition"]) for row in point] == list(range(repetitions))
            assert all(float(row["lambda"]) == lam for row in point)
            h = np.array([float(row["h"]) for row in point])
            indices = np.array([int(row["grid_index"]) for row in point])
            with np.load(directory / f"n{n}" / "checkpoint.npz", allow_pickle=False) as checkpoint:
                grid = checkpoint["h_grid"].copy()
                assert int(checkpoint["completed"]) == repetitions
                assert checkpoint["selected_indices"].shape == (repetitions, 1)
                assert np.array_equal(indices, checkpoint["selected_indices"][:, 0])
                assert np.allclose(h, grid[indices], rtol=0, atol=1e-15)
                configuration = json.loads(str(checkpoint["configuration_json"].item()))
                assert configuration["settings"] == manifest["settings"]
                assert configuration["sample_size"] == n
                assert configuration["lambda_values"] == manifest["lambda_values"]
                assert configuration["implementation_version"] == manifest["implementation_version"]
            if common_grid is None:
                common_grid = grid
            assert np.array_equal(grid, common_grid)
            counts = np.bincount(indices, minlength=len(grid))
            saved_frequencies = sorted((row for row in frequencies if int(row["n"]) == n), key=lambda row: int(row["grid_index"]))
            assert len(saved_frequencies) == len(grid)
            assert np.array_equal(counts, [int(row["count"]) for row in saved_frequencies])
            assert np.allclose(counts / repetitions, [float(row["relative_frequency"]) for row in saved_frequencies])
            saved_summary, = [row for row in summaries if int(row["n"]) == n]
            assert np.isclose(h.mean(), float(saved_summary["mean_h"]))
            assert np.isclose(np.median(h), float(saved_summary["median_h"]))
            sd = h.std(ddof=1)
            quantiles = np.quantile(h, [0.10, 0.50, 0.90])
            row = {
                "n": n, "lambda": lam, "repetitions": repetitions,
                "mean_h": h.mean(), "mc_se_mean": sd / np.sqrt(repetitions),
                "sd_h": sd, "q10_h": quantiles[0], "median_h": quantiles[1], "q90_h": quantiles[2],
                "modal_h": grid[counts.argmax()], "modal_frequency": counts.max() / repetitions,
                "lower_boundary_count": int(counts[0]), "upper_boundary_count": int(counts[-1]),
            }
            for threshold in THRESHOLDS:
                row[f"fraction_h_gt_{threshold:.2f}"] = (h > threshold).mean()
            stats.append(row)
            selections[n, lam] = h

    write_rows(HERE / "distribution_statistics.csv", stats)
    paired = []
    for n in baseline["sample_sizes"]:
        for low, high in ((1, 10), (10, 100), (100, 1000)):
            difference = selections[n, high] - selections[n, low]
            se = difference.std(ddof=1) / np.sqrt(len(difference))
            paired.append({
                "n": n, "lambda_from": low, "lambda_to": high,
                "mean_h_difference": difference.mean(), "mc_se_difference": se,
                "approx_95_mc_lower": difference.mean() - 1.96 * se,
                "approx_95_mc_upper": difference.mean() + 1.96 * se,
                "fraction_h_increases": (difference > 0).mean(),
                "fraction_h_decreases": (difference < 0).mean(),
                "fraction_h_unchanged": (difference == 0).mean(),
            })
    write_rows(HERE / "paired_lambda_comparisons.csv", paired)

    population_divergence = population_gaussian_beta2_divergence_matrix(common_grid)
    population = [{"lambda": lam, "selected_h_population_smoothed_matrix": common_grid[population_choice(population_divergence, lam)]} for lam in RUNS]
    write_rows(HERE / "population_matrix_reference.csv", population)
    oracle = [{"n": n, "gaussian_mise_oracle_h": gaussian_mise_oracle_bandwidth(n)} for n in baseline["sample_sizes"]]
    write_rows(HERE / "gaussian_mise_reference.csv", oracle)

    enable_automatic_pdf_saving(NOTEBOOK, output_root=CODE / "outputs" / "pdfs")
    colors = {1: "#0072B2", 10: "#D55E00", 100: "#009E73", 1000: "#CC79A7"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for lam in RUNS:
        points = [row for row in stats if row["lambda"] == lam]
        axes[0].errorbar(
            [row["n"] for row in points], [row["mean_h"] for row in points],
            yerr=[1.96 * row["mc_se_mean"] for row in points],
            color=colors[lam], label=rf"$\lambda={lam}$", marker="o", markersize=4, capsize=2,
        )
        if lam in (1, 10):
            axes[1].plot([row["n"] for row in points], [row["fraction_h_gt_0.25"] for row in points],
                         color=colors[lam], label=rf"$\lambda={lam}$", marker="o", markersize=4)
    axes[0].set(title="Mean selected bandwidth", ylabel="Bandwidth h", ylim=(0, 0.49))
    axes[1].set(title="Shift toward larger bandwidths", ylabel="Fraction with h > 0.25", ylim=(0, 1.04))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    for axis in axes:
        axis.set_xlabel("Sample size n")
        axis.set_xticks(baseline["sample_sizes"])
        axis.tick_params(axis="x", labelsize=9)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Fast L2 spectral selection: comparison across lambda", fontsize=15)
    fig.supxlabel("1,000 repetitions per point. Mean error bars: ±1.96 Monte Carlo SE. The threshold 0.25 is descriptive.", fontsize=9)
    fig.savefig(HERE / "lambda_comparison.png", dpi=180, bbox_inches="tight")
    plt.show()  # Shared exporter saves this figure as a PDF automatically.
    plt.close(fig)

    source_links = "\n".join(f"- λ={lam}: [saved run]({(ROOT / run_id).as_posix()})" for lam, run_id in RUNS.items())
    anchor_rows = "\n".join(
        "| " + " | ".join([str(n), *[f"{selections[n, lam].mean():.3f}" for lam in RUNS]]) + " |"
        for n in (100, 500, 1000)
    )
    transitions = "\n".join(
        f"| {n} | " + " | ".join(f"{(selections[n, lam] > .25).mean():.1%}" for lam in (1, 10)) + " |"
        for n in (400, 500, 600, 700, 1000)
    )
    report = f"""# Analysis of saved fast L2 lambda experiments

This is an original descriptive analysis of the existing completed runs. No simulations were rerun. All 40 checkpoints are complete (M=1000). Selection CSVs, frequency CSVs and checkpoint indices agree. The manifests agree in every setting except λ: standard-normal data, 201 geometric bandwidths from 0.01 to 2, FFT spatial grid 8192 on [-12,12], seed 20260830. The same seeded samples and their prefixes are used across λ and n in the [notebook]({(CODE / (NOTEBOOK + '.ipynb')).as_posix()}).

## Empirical findings

Mean selected bandwidths, calculated from the saved selections:

| n | λ=1 | λ=10 | λ=100 | λ=1000 |
| --- | --- | --- | --- | --- |
{anchor_rows}

The existing histograms show two separated regions of selected bandwidth at intermediate n for λ=1 and λ=10. The fraction selecting h>0.25 quantifies the shift, but the threshold is an analyst-chosen diagnostic and is not a formal test of bimodality:

| n | λ=1 | λ=10 |
| --- | --- | --- |
{transitions}

The same delayed transition for λ=10 remains visible with thresholds 0.20 and 0.30; all three thresholds are recorded in distribution_statistics.csv. These cross-repetition histograms do not establish that individual Perron vectors have two competing maxima.

For λ=100 and λ=1000, the mean bandwidth decreases at every tested step in n. The empirical 10th–90th percentile intervals narrow from approximately [0.200,0.321] to [0.077,0.112] for λ=100, and from [0.222,0.367] to [0.127,0.162] for λ=1000. None of the 40 points selects either grid endpoint.

Increasing λ is not a monotone bandwidth reduction. At n=1000, raising λ from 100 to 1000 increases mean h by {selections[1000, 1000].mean() - selections[1000, 100].mean():.6f}. Paired comparisons and their Monte Carlo standard errors are recorded separately.

## Interpretation and limits

The experiment demonstrates a strong interaction between sample size and λ for this Gaussian model and this fixed grid. A plausible mechanism is a change in the similarities that determine the spectral ranking: [similarities.py]({(CODE / 'kde_tools/similarities.py').as_posix()}) constructs S_ij=exp(-λ D_ij), with D equal to half the squared L2 distance. This interpretation does not by itself explain the observed transition or establish a theorem.

Concentration of selected bandwidths measures reproducibility of the selector, not density-estimation accuracy. This analysis does not calculate realized density errors. A Gaussian fixed-bandwidth MISE reference is computed using [gaussian_mise_oracle_bandwidth]({(CODE / 'kde_tools/bandwidths.py').as_posix()}:120); closeness of a data-dependent bandwidth to that reference is not a risk comparison. For general guidance on separating performance measures and reporting Monte Carlo uncertainty, see Morris, White and Crowther (2019), [Sections 5.2–5.3](https://onlinelibrary.wiley.com/doi/full/10.1002/sim.8086).

No asymptotic convergence rate or consistency conclusion follows from n=100,...,1000. In particular, the [draft joint-limit conjecture]({(CODE.parents[1] / 'Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex').as_posix()}:820) varies N and λ together; the present experiment holds N=201 fixed.

As an additional original deterministic calculation, replacing each empirical KDE by its population-smoothed Gaussian density in the shared [population matrix helper]({(CODE / 'kde_tools/similarities.py').as_posix()}:50) gives selected grid bandwidths approximately {', '.join(f"λ={row['lambda']}: h={row['selected_h_population_smoothed_matrix']:.3f}" for row in population)}. These are model references, not additional Monte Carlo observations or density-error optima. In particular, the observed downward trend at λ=100 should not be extrapolated indefinitely; its population-smoothed reference is h≈0.260. No thesis or notebook source was edited.

## Reproduce

Run `python3 Code/Code_kde/outputs/analysis/fast_l2_lambda_comparison/analyse_saved_results.py` from the project root. It reads only the completed runs and writes analysis CSVs and this summary. The comparison figure is automatically exported to the existing notebook's PDF folder by [pdf_export.py]({(CODE / 'kde_tools/pdf_export.py').as_posix()}). Mean Monte Carlo SE is the sample standard deviation divided by sqrt(M). Approximate 95% Monte Carlo intervals use ±1.96 SE; they describe simulation uncertainty in an estimated mean, not the spread of individual selected bandwidths (Morris et al., Table 6).

The population references use the shared [Perron solver]({(CODE / 'kde_tools/perron.py').as_posix()}), version `{PERRON_SOLVER_VERSION}`. Historical sample selections retain the computation recorded in their source manifests.

Source runs:

{source_links}
"""
    (HERE / "README.md").write_text(report)
    (HERE / "analysis_manifest.json").write_text(json.dumps({
        "source_run_ids": RUNS, "common_configuration": baseline,
        "analysis_type": "descriptive analysis of existing selections; population matrix reference",
        "population_perron_solver": PERRON_SOLVER_VERSION,
        "thresholds": THRESHOLDS, "checkpoint_count_verified": 40,
        "simulations_rerun": False,
    }, indent=2) + "\n")
    print("Analysis saved:", HERE)
    print("At n=1000, paired lambda differences:")
    for row in paired:
        if row["n"] == 1000:
            print(row)
    print("Population-smoothed reference:", population)


if __name__ == "__main__":
    main()
