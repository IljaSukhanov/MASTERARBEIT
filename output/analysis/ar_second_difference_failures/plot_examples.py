"""Inspect saved AR(3) curves without rerunning or changing the experiment."""

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "Code/Code_AR_processes/outputs/csv/ar_order_minimum_width/width_curves.csv"
SELECTIONS = ROOT / "Code/Code_AR_processes/outputs/csv/ar3_implied_support_width/derivative_selector_results.csv"

with SOURCE.open() as handle:
    source_rows = [r for r in csv.DictReader(handle) if r["true_order"] == "3"]
with SELECTIONS.open() as handle:
    saved_orders = {
        int(r["replication"]): int(r["second_difference_order"])
        for r in csv.DictReader(handle)
    }

curves = {}
for row in source_rows:
    curves.setdefault(int(row["replication"]), []).append(row)
for replication, rows in curves.items():
    rows.sort(key=lambda row: int(row["k"]))
    curves[replication] = np.array([float(row["minimum_width"]) for row in rows])

counts = Counter()
for replication, widths in curves.items():
    selected_order = int(np.argmax(np.diff(widths, n=2)) + 1)
    assert selected_order == saved_orders[replication]
    counts[selected_order] += 1

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BLUE, ORANGE, GREEN = "#0072B2", "#D55E00", "#009E73"
records = []

for replication in [2, 5, 1]:
    widths = curves[replication]
    first = np.diff(widths)
    second = np.diff(widths, n=2)
    centers = np.arange(2, len(widths))
    peak = int(np.argmax(second))
    selected_k = int(centers[peak])
    selected_p = selected_k - 1

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.23, top=0.76, wspace=0.31)
    fig.suptitle(
        f"Replication {replication}: true AR order 3; selected order {selected_p}",
        fontsize=17, y=0.97,
    )
    ax = axes[0]
    ax.plot(np.arange(1, len(widths) + 1), widths, "o-", color=BLUE, lw=1.8, ms=5)
    ax.set(title="Saved minimum-width estimates", xlabel="Embedding dimension k", ylabel="Width")
    ax.set_xticks(np.arange(1, 11))
    ax.set_xlim(0.7, 10.3)
    ax.set_ylim(bottom=0)

    ax = axes[1]
    colors = [ORANGE if i == peak else BLUE for i in range(len(second))]
    ax.bar(centers, second, width=0.62, color=colors, alpha=0.87)
    ax.axhline(0, color="0.45", lw=0.8)
    ax.set(title="Second differences, centered at k", xlabel="Embedding dimension k", ylabel="Change in slope")
    ax.set_xticks(centers)
    ax.set_xlim(1.5, 9.5)
    ymin, ymax = float(min(0, second.min())), float(second.max())
    span = ymax - ymin
    ax.set_ylim(ymin - 0.12 * span, ymax + 0.32 * span)
    ax.text(0.97, 0.94, f"Max: {second[peak]:.3f} at k = {selected_k}",
            transform=ax.transAxes, ha="right", va="top", fontsize=12)

    for ax in axes:
        ax.axvline(4, color=GREEN, ls="--", lw=1.4,
                   label="True-order reference: k = 4")
        ax.axvline(selected_k, color=ORANGE, ls=":", lw=1.7,
                   label=f"Selected peak: k = {selected_k}")
        ax.grid(axis="y", alpha=0.16)
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.005), fontsize=11)
    fig.savefig(OUT / f"replication_{replication:02d}_failure.png", dpi=180)
    # Preserve a vector copy of every displayed figure as well.
    fig.savefig(OUT / f"replication_{replication:02d}_failure.pdf")
    plt.close(fig)

    records.append({
        "replication": replication,
        "true_order": 3,
        "selected_order": selected_p,
        "peak_dimension": selected_k,
        "maximum_second_difference": float(second[peak]),
        "second_difference_at_k4": float(second[2]),
        "widths": widths.tolist(),
        "first_differences": first.tolist(),
        "second_differences": second.tolist(),
    })

(OUT / "examples.json").write_text(json.dumps({
    "source": str(SOURCE),
    "saved_selections": str(SELECTIONS),
    "counts_by_selected_order": dict(sorted(counts.items())),
    "examples": records,
}, indent=2) + "\n")

print(json.dumps({"counts": dict(sorted(counts.items())), "examples": [
    {k: v for k, v in r.items() if k not in {"widths", "first_differences", "second_differences"}}
    for r in records
]}, indent=2))
