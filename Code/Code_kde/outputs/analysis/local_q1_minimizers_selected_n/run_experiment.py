"""One interior local Q_1 minimum per paired sample, restricted to (0.01, 2).

Fourier gradients and curvatures locate brackets only. Exact Gaussian-pair
derivatives refine curvature extrema and every accepted local minimum. Both
endpoints are excluded. Multiple minima: choose the smallest energy; no
minimum: retain a missing outcome, with no endpoint substitution.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

KDE_DIR = Path(__file__).resolve().parents[3]
ROOT = KDE_DIR.parents[1]
sys.path.insert(0, str(KDE_DIR))
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/kde-hmax10-mpl-cache")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
from kde_tools.experiment_inputs import paired_gaussian_sample_bank, sampling_inputs
from threadpoolctl import threadpool_limits

from kde_tools.qhat import minimize_qhat
from kde_tools.energy_minimization import energy_search_id, energy_search_settings, search_config, DEFAULT_ENERGY_SEARCH
from kde_tools.pdf_export import enable_automatic_pdf_saving

NAME = "local_q1_minimizers_selected_n"
from kde_tools.energy_minimization import unified_energy_run_directory, energy_search_settings, energy_search_id

DATA_DIR = unified_energy_run_directory(KDE_DIR/"outputs/data"/NAME)
IMAGE_DIR = KDE_DIR/"outputs/images"/NAME
DELIVERY_DIR = ROOT/"output/pdf"/NAME
SAMPLE_SIZES, M, SEED = (100, 1000, 1500, 2000), 1000, 20260830
H_MIN, H_MAX = 0.01, 2.0
HALF_WIDTH, SPATIAL_POINTS, SCAN_POINTS = (DEFAULT_ENERGY_SEARCH.half_width,
    DEFAULT_ENERGY_SEARCH.spatial_points, DEFAULT_ENERGY_SEARCH.scan_points)
VERSION = "unified_q1_local_interval_v2_" + energy_search_id()


def save_csv(path, rows, fields=None):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


class LocalSearch:
    def __init__(self, points=SCAN_POINTS, spatial_points=SPATIAL_POINTS):
        self.config = search_config(scan_points=points, spatial_points=spatial_points,
                                    half_width=HALF_WIDTH)

    def analyze(self, samples):
        return minimize_qhat(samples, H_MIN, H_MAX, r=1., config=self.config)

    def minima(self, samples):
        report = self.analyze(samples)
        return report["local_minima"], bool(report["diagnostics"]["resolution_stable"])


def same_minima(first, second):
    return len(first) == len(second) and all(abs(np.log(float(a["h"])/float(b["h"]))) < 1e-7 for a, b in zip(first, second))


def independent_dense_minima(sample, points=1025):
    # Exact objective scans cross-check Fourier-assisted candidate discovery.
    return minimize_qhat(sample, H_MIN, H_MAX, r=1.,
        config=search_config(scan_backend="exact", scan_points=points))["local_minima"]


def main():
    for directory in (DATA_DIR, IMAGE_DIR, DELIVERY_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    samples = paired_gaussian_sample_bank(M, 2000)
    sample_hash = hashlib.sha256(samples.tobytes()).hexdigest()
    assert sample_hash == sampling_inputs()["paired_sample_bank"]["sample_bank_sha256"]
    settings = dict(version=VERSION, energy_search=energy_search_settings(), sample_sizes=list(SAMPLE_SIZES), repetitions=M, seed=SEED,
                    sample_bank_sha256=sample_hash, r=1.0, h_min=H_MIN, h_max=H_MAX,
                    endpoint_candidates=False, selection="lowest energy among detected interior local minima; no endpoint fallback",
                    scan_points=SCAN_POINTS, spatial_points=SPATIAL_POINTS,
                    normalization="bin count divided by all 1000 repetitions")
    settings_path = DATA_DIR/"settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise RuntimeError("Saved experiment settings differ.")
    settings_path.write_text(json.dumps(settings, indent=2)+"\n")
    selection_path, candidate_path = DATA_DIR/"selections.csv", DATA_DIR/"candidates.csv"
    rows = list(csv.DictReader(selection_path.open())) if selection_path.exists() else []
    candidates = list(csv.DictReader(candidate_path.open())) if candidate_path.exists() else []
    done = {(int(row["n"]), int(row["repetition"])) for row in rows}
    with threadpool_limits(limits=1):
        search = LocalSearch()
        refined_search = LocalSearch(2*SCAN_POINTS-1, 2*SPATIAL_POINTS)

        def select(n, repetition):
            sample = samples[repetition, :n]
            analysis = search.analyze(sample)
            if analysis["diagnostics"]["resolution_stable"] is False:
                analysis = minimize_qhat(sample, H_MIN, H_MAX, r=1.,
                    config=search_config(search.config, scan_backend="exact"))
            found = analysis["local_minima"]
            signature_agrees = bool(analysis["diagnostics"]["resolution_stable"])
            best = min(range(len(found)), key=lambda i: (found[i]["energy"], found[i]["h"])) if found else None
            row = dict(n=n, repetition=repetition, local_minimum_count=len(found),
                       selected_h=found[best]["h"] if best is not None else np.nan,
                       selected_energy=found[best]["energy"] if best is not None else np.nan,
                       scan_signature_agrees=signature_agrees,
                       interval_minimum_h=analysis["global_minimum"]["h"],
                       interval_minimum_energy=analysis["global_minimum"]["energy"],
                       interval_minimum_boundary=analysis["global_minimum"]["boundary"],
                       analysis_json=json.dumps(analysis, sort_keys=True))
            points = [dict(n=n, repetition=repetition, candidate=i, selected=i == best, **point) for i, point in enumerate(found)]
            return row, points

        if not done:
            row, points = select(SAMPLE_SIZES[0], 0)
            rows.append(row)
            candidates.extend(points)
            done.add((SAMPLE_SIZES[0], 0))
        jobs = [(n, i) for n in SAMPLE_SIZES for i in range(M) if (n, i) not in done]
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(select, n, i) for n, i in jobs]
            for future in as_completed(futures):
                row, points = future.result()
                rows.append(row)
                candidates.extend(points)
                if len(rows) % 100 == 0:
                    save_csv(candidate_path, candidates)
                    save_csv(selection_path, rows)
                    print(f"Completed {len(rows)}/{M*len(SAMPLE_SIZES)} samples; elapsed {time.monotonic()-started:.0f}s", flush=True)
        rows.sort(key=lambda row: (int(row["n"]), int(row["repetition"])))
        candidates.sort(key=lambda row: (int(row["n"]), int(row["repetition"]), float(row["h"])))
        save_csv(candidate_path, candidates)
        save_csv(selection_path, rows)
        assert len(rows) == M*len(SAMPLE_SIZES)

        checks = []
        for n in SAMPLE_SIZES:
            for repetition in (0, M//2, M-1):
                original = [point for point in candidates if int(point["n"]) == n and int(point["repetition"]) == repetition]
                refined, _ = refined_search.minima(samples[repetition, :n])
                assert same_minima(original, refined)
                checks.append(dict(n=n, repetition=repetition, kind="doubled Fourier scan and spatial resolution",
                                   minima=len(original), maximum_h_difference=max([abs(float(a["h"])-b["h"]) for a, b in zip(original, refined)] or [0.])))
        # One fully exact dense scan per n checks the bracket-finding method independently.
        for n in SAMPLE_SIZES:
            original = [point for point in candidates if int(point["n"]) == n and int(point["repetition"]) == 0]
            exact = independent_dense_minima(samples[0, :n])
            assert same_minima(original, exact)
            checks.append(dict(n=n, repetition=0, kind="independent exact 1025-node gradient/curvature scan",
                               minima=len(original), maximum_h_difference=max([abs(float(a["h"])-b["h"]) for a, b in zip(original, exact)] or [0.])))
        # Compare n=200 samples against fresh exact scans, independent of saved results.
        for repetition in range(0, M, 10):
            expected = independent_dense_minima(samples[repetition, :200])
            actual, _ = search.minima(samples[repetition, :200])
            assert same_minima(expected, actual)
        monotone, _ = search.minima(np.zeros(200))
        assert not monotone

    summary = []
    for n in SAMPLE_SIZES:
        part = [row for row in rows if int(row["n"]) == n]
        selected = np.array([float(row["selected_h"]) for row in part])
        valid = selected[np.isfinite(selected)]
        counts = [int(row["local_minimum_count"]) for row in part]
        assert len(part) == M and np.all((valid > H_MIN) & (valid < H_MAX))
        summary.append(dict(n=n, samples=M, with_local_minimum=len(valid), without_local_minimum=M-len(valid),
                            multiple_local_minima=sum(count > 1 for count in counts), total_local_minima=sum(counts),
                            median_h=float(np.median(valid)), q05_h=float(np.quantile(valid, .05)), q95_h=float(np.quantile(valid, .95))))
    save_csv(DATA_DIR/"summary.csv", summary)
    validation = dict(sample_bank_matches_canonical_input=True, endpoint_substitutions=0,
                      all_candidates_strictly_inside_interval=True,
                      all_candidates_have_positive_curvature=all(float(point["log_curvature"]) > 0 for point in candidates),
                      maximum_stationarity_correction=max(float(point["newton_correction"]) for point in candidates),
                      fresh_exact_n200_samples_checked=len(range(0, M, 10)), monotone_gaussian_has_no_local_minimum=True,
                      checks=checks, numerical_search_is_not_an_exhaustiveness_certificate=True)
    (DATA_DIR/"validation.json").write_text(json.dumps(validation, indent=2)+"\n")
    plot_results(rows, summary)


def plot_results(rows, summary):
    pdf_dir = enable_automatic_pdf_saving(NAME)
    bandwidths = np.geomspace(H_MIN, H_MAX, 201)
    positions, edges = np.arange(201), np.arange(202)-.5
    frequencies, histogram_rows = {}, []
    for n in SAMPLE_SIZES:
        h = np.array([float(row["selected_h"]) for row in rows if int(row["n"]) == n])
        h = h[np.isfinite(h)]
        count, _ = np.histogram(np.interp(np.log(h), np.log(bandwidths), positions), bins=edges)
        assert count.sum() == len(h)
        frequencies[n] = count/M
        histogram_rows.extend(dict(n=n, bin_index=i, bandwidth_center=float(bandwidths[i]), count=int(count[i]), relative_frequency_all_samples=float(count[i]/M)) for i in range(201))
    save_csv(DATA_DIR/"histogram.csv", histogram_rows)
    plt.style.use("default")
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.1), sharex=True, sharey=True)
    ticks = np.unique(np.linspace(0, 200, 7, dtype=int))
    ymax = 1.13*max(float(values.max()) for values in frequencies.values())
    for ax, n, info in zip(axes.flat, SAMPLE_SIZES, summary):
        ax.stairs(frequencies[n], edges, fill=True, color="#D55E00", alpha=.10, linewidth=0)
        ax.stairs(frequencies[n], edges, color="#D55E00", linestyle="--", linewidth=1.8)
        ax.set_title(rf"$n={n}$", fontsize=13, pad=10)
        ax.set_xticks(ticks, [f"{bandwidths[i]:.3g}" for i in ticks])
        ax.set(xlim=(-.7, 200.7), ylim=(0, ymax), xlabel="local minimizing bandwidth h", ylabel="relative frequency")
        ax.text(.025, .94, f"Local minimum found: {info['with_local_minimum']}/{M}", transform=ax.transAxes, va="top", fontsize=9, color="#304967")
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.tick_params(colors="#304967", labelsize=9, labelbottom=True, labelleft=True)
        ax.grid(axis="y", color="#DDE5EC", linewidth=.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#B8C5D1")
    fig.suptitle(r"Interior local minimizers of $\widehat Q_1$", fontsize=18, y=.985)
    fig.text(.5, .928, "Gaussian samples; M = 1000 per n; local minima restricted to (0.01, 2)", ha="center", fontsize=11)
    fig.legend(handles=[Line2D([0], [0], color="#D55E00", linestyle="--", linewidth=2, label=r"$\widehat Q_1$")], loc="upper center", bbox_to_anchor=(.5, .91), frameon=False, fontsize=12)
    fig.text(.5, .025, "One entry per sample: lowest-energy local minimum when several exist.\nBin frequencies divide by all 1,000 samples; samples with no local minimum contribute no entry.", ha="center", fontsize=9)
    fig.subplots_adjust(left=.075, right=.985, bottom=.12, top=.80, hspace=.42, wspace=.20)
    fig.savefig(IMAGE_DIR/"local_q1_minimizers_selected_n.png", dpi=180)
    plt.show()
    plt.close(fig)
    produced = pdf_dir/"001_interior_local_minimizers_of_widehat_q_1.pdf"
    if not produced.is_file():
        produced, = pdf_dir.glob("001_*.pdf")
    shutil.copyfile(produced, DELIVERY_DIR/"local_q1_minimizers_selected_n.pdf")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot_results(list(csv.DictReader((DATA_DIR/"selections.csv").open())),
                     list(csv.DictReader((DATA_DIR/"summary.csv").open())))
    else:
        main()
