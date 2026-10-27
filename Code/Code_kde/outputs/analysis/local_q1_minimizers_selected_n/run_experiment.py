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
from scipy.optimize import brentq
from scipy.spatial.distance import pdist
from threadpoolctl import threadpool_limits

from kde_tools.continuous_minimizers import qr_log_energy
from kde_tools.pdf_export import enable_automatic_pdf_saving
from kde_tools.qhat import _energy, _scan_matrix

NAME = "local_q1_minimizers_selected_n"
DATA_DIR = KDE_DIR/"outputs/data"/NAME
IMAGE_DIR = KDE_DIR/"outputs/images"/NAME
DELIVERY_DIR = ROOT/"output/pdf"/NAME
SAMPLE_SIZES, M, SEED = (100, 1000, 1500, 2000), 1000, 20260830
H_MIN, H_MAX = 0.01, 2.0
HALF_WIDTH, SPATIAL_POINTS, SCAN_POINTS = 16.0, 65536, 1025
VERSION = "q1_local_interval_curvature_augmented_v1"


def save_csv(path, rows, fields=None):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


class LocalSearch:
    def __init__(self, points=SCAN_POINTS, spatial_points=SPATIAL_POINTS):
        self.spatial_points = spatial_points
        self.z, self.omega, _, q0 = _scan_matrix(H_MIN, H_MAX, points, HALF_WIDTH, spatial_points)
        u = (np.exp(self.z[:, None])*self.omega[None, :])**2
        q1 = q0*np.exp(-2*self.z[:, None])
        a = 2-2*u
        self.matrices = np.stack((q1, q1*a, q1*(a*a-4*u)))

    def approximate(self, samples):
        position = (samples+HALF_WIDTH)/(2*HALF_WIDTH/self.spatial_points)
        left = np.floor(position).astype(int)
        fraction = position-left
        mass = (np.bincount(left % self.spatial_points, weights=1-fraction, minlength=self.spatial_points)
                + np.bincount((left+1) % self.spatial_points, weights=fraction, minlength=self.spatial_points))
        power = np.abs(np.fft.rfft(mass/len(samples))[:len(self.omega)])**2
        return self.matrices @ power

    def minima(self, samples):
        values = self.approximate(samples)
        d2 = pdist(samples[:, None], metric="sqeuclidean")
        cache = {}

        def exact(z):
            z = float(z)
            if z not in cache:
                cache[z] = qr_log_energy(d2, len(samples), z, 1.0)
            return cache[z]

        nodes = {float(self.z[0]), float(self.z[-1])}
        # Exact curvature extrema expose shallow minimum/maximum pairs.
        for index in np.flatnonzero(values[2, :-1]*values[2, 1:] < 0):
            lo, hi = int(index), int(index+1)
            for _ in range(9):
                if exact(self.z[lo])[2]*exact(self.z[hi])[2] < 0:
                    break
                lo, hi = max(0, lo-1), min(len(self.z)-1, hi+1)
            else:
                raise ArithmeticError("Fourier curvature bracket did not bracket an exact extremum.")
            nodes.add(float(brentq(lambda z: exact(z)[2], self.z[lo], self.z[hi], xtol=2e-13)))
        # Also retain every bracket indicated directly by the scanned gradient.
        for index in np.flatnonzero(values[1, :-1]*values[1, 1:] < 0):
            nodes.update((float(self.z[index]), float(self.z[index+1])))
        nodes = sorted(nodes)
        result = []
        for lo, hi in zip(nodes[:-1], nodes[1:]):
            if exact(lo)[1] < 0 < exact(hi)[1]:
                z = float(brentq(lambda t: exact(t)[1], lo, hi, xtol=2e-13))
                value, gradient, curvature = exact(z)
                h = float(np.exp(z))
                if not H_MIN < h < H_MAX or value <= 0 or curvature <= 0 or abs(gradient/curvature) > 1e-8:
                    raise ArithmeticError("Local-minimum interval, stationarity, or curvature check failed.")
                if result and abs(np.log(result[-1]["h"])-z) < 1e-8:
                    continue
                result.append(dict(h=h, energy=float(value), log_derivative=float(gradient),
                                   log_curvature=float(curvature), newton_correction=float(abs(gradient/curvature))))
        def signature(array):
            return (int(np.sum((array[1, :-1] < 0) & (array[1, 1:] > 0))),
                    int(np.sum(array[2, :-1]*array[2, 1:] < 0)))
        return result, signature(values) == signature(values[:, ::2])


def same_minima(first, second):
    return len(first) == len(second) and all(abs(np.log(float(a["h"])/float(b["h"]))) < 1e-7 for a, b in zip(first, second))


def independent_dense_minima(sample, points=1025):
    # Reuse the earlier exact derivative scan, without any Fourier approximation.
    import runpy
    old = runpy.run_path(str(KDE_DIR/"outputs/analysis/local_energy_minimizers_n200/run_experiment.py"))
    d2 = pdist(sample[:, None], metric="sqeuclidean")
    z = np.linspace(np.log(H_MIN), np.log(H_MAX), points)
    base = np.array([_energy(d2, len(sample), t) for t in z])
    return old["roots_from_scan"](d2, len(sample), z, base, 1.0)


def main():
    for directory in (DATA_DIR, IMAGE_DIR, DELIVERY_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    samples = np.concatenate((np.random.default_rng(SEED).normal(size=(M, 1000)),
        np.random.default_rng(np.random.SeedSequence([SEED, 1])).normal(size=(M, 1000))), axis=1)
    sample_hash = hashlib.sha256(samples.tobytes()).hexdigest()
    original_path = KDE_DIR/"outputs/data/continuous_ise_qhat_distributions_vs_n/thesis/5b5fa0b0a92f/settings.json"
    assert sample_hash == json.loads(original_path.read_text())["sample_bank_sha256"]
    settings = dict(version=VERSION, sample_sizes=list(SAMPLE_SIZES), repetitions=M, seed=SEED,
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
        refined_search = LocalSearch(2049, 131072)

        def select(n, repetition):
            sample = samples[repetition, :n]
            found, signature_agrees = search.minima(sample)
            if not signature_agrees:
                refined, _ = refined_search.minima(sample)
                if not same_minima(found, refined):
                    dense = independent_dense_minima(sample, 2049)
                    if not same_minima(refined, dense):
                        raise ArithmeticError("Local minima did not stabilize under refinement.")
                found = refined
            best = min(range(len(found)), key=lambda i: (found[i]["energy"], found[i]["h"])) if found else None
            row = dict(n=n, repetition=repetition, local_minimum_count=len(found),
                       selected_h=found[best]["h"] if best is not None else np.nan,
                       selected_energy=found[best]["energy"] if best is not None else np.nan,
                       scan_signature_agrees=signature_agrees)
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
        # Check against 100 saved samples from the earlier n=200 exact experiment.
        baseline_path = KDE_DIR/"outputs/data/local_energy_minimizers_n200/candidates.csv"
        baseline = list(csv.DictReader(baseline_path.open()))
        for repetition in range(0, M, 10):
            expected = [dict(h=float(point["h"])) for point in baseline if int(point["sample_id"]) == repetition and point["method"] == "r1" and H_MIN < float(point["h"]) < H_MAX]
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
    validation = dict(sample_bank_matches_figure39=True, endpoint_substitutions=0,
                      all_candidates_strictly_inside_interval=True,
                      all_candidates_have_positive_curvature=all(float(point["log_curvature"]) > 0 for point in candidates),
                      maximum_stationarity_correction=max(float(point["newton_correction"]) for point in candidates),
                      baseline_n200_samples_checked=100, monotone_gaussian_has_no_local_minimum=True,
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
