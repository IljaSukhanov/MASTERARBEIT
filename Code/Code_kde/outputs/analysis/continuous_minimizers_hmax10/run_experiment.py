"""Repeat the selected Figure 3.9 panels on [0.01, 10], using paired samples.

The original thesis assets are retained for comparison. Each displayed figure
is exported by kde_tools.pdf_export, then copied to output/pdf for delivery.
The finite basin scan is a numerical search, not a global-optimality proof.
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
from threadpoolctl import threadpool_limits

from kde_tools.continuous_minimizers import METHODS, continuous_ise_qr_minima
from kde_tools.pdf_export import enable_automatic_pdf_saving

NAME = "continuous_minimizers_hmax10"
DATA_DIR = KDE_DIR/"outputs/data"/NAME
IMAGE_DIR = KDE_DIR/"outputs/images"/NAME
DELIVERY_DIR = ROOT/"output/pdf/continuous_minimizers_selected_n_hmax10"
SAMPLE_SIZES, M, SEED = (100, 1000, 1500, 2000), 1000, 20260830
SEARCH = dict(h_min=0.01, h_max=10.0, scan_points=513,
              spatial_points=65536, half_width=72.0, sigma2=1.0)
COLORS = ("#7B4AB2", "#0072B2", "#D55E00", "#009E73")
STYLES = ("-", "-", "--", "-.")
LABELS = ("ISE", r"$\widehat Q_0$", r"$\widehat Q_1$", r"$\widehat Q_{13/14}$")


def save_csv(path, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    for directory in (DATA_DIR, IMAGE_DIR, DELIVERY_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    samples = np.concatenate((
        np.random.default_rng(SEED).normal(size=(M, 1000)),
        np.random.default_rng(np.random.SeedSequence([SEED, 1])).normal(size=(M, 1000)),
    ), axis=1)
    original_settings = KDE_DIR/"outputs/data/continuous_ise_qhat_distributions_vs_n/thesis/5b5fa0b0a92f/settings.json"
    original = json.loads(original_settings.read_text())
    sample_hash = hashlib.sha256(samples.tobytes()).hexdigest()
    assert sample_hash == original["sample_bank_sha256"]
    settings = dict(sample_sizes=SAMPLE_SIZES, repetitions=M, seed=SEED,
                    sample_bank_sha256=sample_hash, search=SEARCH,
                    selection="lowest exact objective among refined detected basins and both endpoints")
    (DATA_DIR/"settings.json").write_text(json.dumps(settings, indent=2)+"\n")
    path = DATA_DIR/"selections.csv"
    rows = list(csv.DictReader(path.open())) if path.exists() else []
    done = {(int(row["n"]), int(row["repetition"])) for row in rows}

    def select(n, repetition):
        found = continuous_ise_qr_minima(samples[repetition, :n], **SEARCH)
        return [dict(n=n, repetition=repetition, method=method, **found[method])
                for method in METHODS]

    started = time.monotonic()
    with threadpool_limits(limits=1):
        # Warm shared matrix and JIT caches before launching workers.
        if not done:
            rows.extend(select(SAMPLE_SIZES[0], 0))
            done.add((SAMPLE_SIZES[0], 0))
        jobs = [(n, i) for n in SAMPLE_SIZES for i in range(M) if (n, i) not in done]
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(select, n, repetition) for n, repetition in jobs]
            for future in as_completed(futures):
                rows.extend(future.result())
                completed = len(rows)//len(METHODS)
                if completed % 100 == 0:
                    save_csv(path, rows)
                    print(f"Completed {completed}/{M*len(SAMPLE_SIZES)} paired samples; elapsed {time.monotonic()-started:.0f}s", flush=True)
    rows.sort(key=lambda row: (int(row["n"]), int(row["repetition"]), METHODS.index(row["method"])))
    save_csv(path, rows)
    assert len(rows) == M*len(SAMPLE_SIZES)*len(METHODS)

    checks = []
    with threadpool_limits(limits=1):
        for n in SAMPLE_SIZES:
            for repetition in (0, M//2, M-1):
                doubled = continuous_ise_qr_minima(samples[repetition, :n], **{**SEARCH, "scan_points": 1025})
                for method in METHODS:
                    stored, = [row for row in rows if int(row["n"]) == n and int(row["repetition"]) == repetition and row["method"] == method]
                    difference = abs(float(stored["h"])-doubled[method]["h"])
                    assert difference < 1e-7
                    checks.append(dict(n=n, repetition=repetition, method=method, difference=difference))
    summary = []
    for n in SAMPLE_SIZES:
        for method in METHODS:
            part = [row for row in rows if int(row["n"]) == n and row["method"] == method]
            assert len(part) == M
            h = np.array([float(row["h"]) for row in part])
            boundary = np.array([int(row["boundary"]) for row in part])
            assert np.all((h >= SEARCH["h_min"]) & (h <= SEARCH["h_max"]))
            summary.append(dict(n=n, method=method, samples=M,
                                lower_boundary_count=int(np.sum(boundary == -1)),
                                upper_boundary_count=int(np.sum(boundary == 1)),
                                median_h=float(np.median(h))))
    save_csv(DATA_DIR/"summary.csv", summary)
    validation = dict(sample_bank_matches_original=True,
                      completed_samples=M*len(SAMPLE_SIZES), selectors=len(rows),
                      refinement_checks=checks,
                      maximum_refinement_discrepancy=max(row["difference"] for row in checks))
    (DATA_DIR/"validation.json").write_text(json.dumps(validation, indent=2)+"\n")

    pdf_dir = enable_automatic_pdf_saving(NAME)
    bandwidths = np.geomspace(SEARCH["h_min"], SEARCH["h_max"], 201)
    positions = np.arange(201)
    edges = np.arange(202)-0.5
    ticks = np.unique(np.linspace(0, 200, 7, dtype=int))
    histogram_rows = []
    plt.style.use("default")
    for panel_index, n in enumerate(SAMPLE_SIZES, start=1):
        fig, ax = plt.subplots(figsize=(7.3, 4.2))
        for method, color, style in zip(METHODS, COLORS, STYLES):
            h = np.array([float(row["h"]) for row in rows if int(row["n"]) == n and row["method"] == method])
            count, _ = np.histogram(np.interp(np.log(h), np.log(bandwidths), positions), bins=edges)
            assert count.sum() == M
            frequency = count/M
            ax.stairs(frequency, edges, fill=True, color=color, alpha=.07, linewidth=0)
            ax.stairs(frequency, edges, color=color, linestyle=style, linewidth=1.8 if method == "ISE" else 1.5)
            histogram_rows.extend(dict(n=n, method=method, bin_index=i, bandwidth_center=float(bandwidths[i]), count=int(count[i]), relative_frequency=float(frequency[i])) for i in range(201))
        ax.set_title(rf"$n={n}$", fontsize=13, pad=10)
        ax.set_xticks(ticks, [f"{bandwidths[i]:.3g}" for i in ticks])
        ax.set(xlim=(-.7, 200.7), ylim=(0, 1.06),
               xlabel="continuously selected bandwidth h", ylabel="relative frequency")
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.tick_params(colors="#304967", labelsize=9)
        ax.grid(axis="y", color="#DDE5EC", linewidth=.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#B8C5D1")
        fig.suptitle("Global bandwidth minimizers", fontsize=16, y=.985)
        fig.text(.5, .90, "Gaussian samples; M = 1000; full interval [0.01, 10]", ha="center", fontsize=10)
        handles = [Line2D([0], [0], color=c, linestyle=s, linewidth=2, label=l) for c, s, l in zip(COLORS, STYLES, LABELS)]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .86), ncol=4, frameon=False, fontsize=11)
        fig.subplots_adjust(left=.12, right=.98, bottom=.16, top=.66)
        fig.savefig(IMAGE_DIR/f"continuous_minimizers_n{n}.png", dpi=190)
        plt.show()
        plt.close(fig)
        produced = pdf_dir/f"{panel_index:03d}_global_bandwidth_minimizers.pdf"
        assert produced.is_file()
        shutil.copyfile(produced, DELIVERY_DIR/f"continuous_minimizers_n{n}.pdf")
    save_csv(DATA_DIR/"histogram.csv", histogram_rows)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
