"""Export one order-versus-error PDF per Beta shape in the thesis table.

Run from any directory with Python and Matplotlib. The thesis is read only.
The figures reproduce the table's rounded mean coefficient MAE values;
they do not recompute experiments or infer unavailable uncertainty intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


AR_DIRECTORY = Path(__file__).resolve().parents[1]
PROJECT_ROOT = AR_DIRECTORY.parents[1]
SOURCE = PROJECT_ROOT / "Abschlussarbeit_2021/LaTeX/chapters/3_application_to_time_series.tex"
DEFAULT_OUTPUT = AR_DIRECTORY / "outputs/pdfs/ar_coefficient_estimation_by_shape"

# Match the colors in the existing combined coefficient-accuracy figure.
STYLES = (
    ("OLS", r"OLS ($L_2$)", "#D55E00", "s", "--"),
    ("Yule-Walker", "Yule-Walker", "#009E73", "^", "-."),
    ("Horvath L-infinity", r"Horváth ($L_\infty$)", "#CC79A7", "D", ":"),
    ("Minimum width", "Minimum width", "#0072B2", "o", "-"),
)


def read_table(source: Path) -> tuple[str, list[dict]]:
    text = source.read_text()
    label = text.index(r"\label{tab:coefficient-estimation-all}")
    start = text.rfind(r"\begin{table}", 0, label)
    end = text.index(r"\end{table}", label) + len(r"\end{table}")
    table = text[start:end]
    blocks = re.split(
        r"\\multicolumn\{6\}\{l\}\{\$\\operatorname\{Beta\}\(([^)]+)\)\$\}",
        table,
    )
    panels = []
    for shape, block in zip(blocks[1::2], blocks[2::2]):
        alpha, beta = (float(value) for value in shape.split(","))
        rows = re.findall(
            r"(OLS \(\$L_2\$\)|Yule--Walker|Horv\\'ath \(\$L_\\infty\$\)|Minimum width)\s*&(.+?)\\\\",
            block,
            flags=re.DOTALL,
        )
        values = {}
        for name, entries in rows:
            method = (
                "OLS" if name.startswith("OLS") else
                "Yule-Walker" if name == "Yule--Walker" else
                "Horvath L-infinity" if name.startswith("Horv") else
                "Minimum width"
            )
            errors = [float(number) for number in re.findall(r"\d+\.\d+", entries)]
            if len(errors) != 5 or method in values:
                raise ValueError(f"Unexpected table row: {shape}, {name}")
            values[method] = errors
        if set(values) != {style[0] for style in STYLES}:
            raise ValueError(f"Missing or unexpected methods for Beta({shape})")
        panels.append({"alpha": alpha, "beta": beta, "mean_mae": values})
    if len(panels) != 5:
        raise ValueError("Expected five Beta distributions in the thesis table")
    return table, panels


def export_figures(source: Path, directory: Path) -> list[Path]:
    table, panels = read_table(source)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    with plt.rc_context({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 11,
        "mathtext.fontset": "dejavuserif",
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "axes.linewidth": 0.8,
        "legend.fontsize": 9.5,
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
    }):
        for index, panel in enumerate(panels, start=1):
            figure, axis = plt.subplots(figsize=(6.4, 4.5))
            figure.subplots_adjust(left=0.15, right=0.97, bottom=0.16, top=0.83)
            for method, label, color, marker, linestyle in STYLES:
                axis.plot(
                    range(1, 6), panel["mean_mae"][method],
                    label=label, color=color, marker=marker,
                    linestyle=linestyle, markeredgewidth=0.7,
                    markeredgecolor="white", zorder=3,
                )
            alpha, beta = panel["alpha"], panel["beta"]
            figure.suptitle(rf"$\mathrm{{Beta}}({alpha:g},\,{beta:g})$ innovations", y=0.965)
            figure.text(0.5, 0.875, r"100 processes per order, $T=1000$", ha="center", fontsize=10, color="0.35")
            axis.set(
                xlabel=r"AR order $p$", ylabel="Mean coefficient MAE",
                xticks=range(1, 6), xlim=(0.8, 5.2), ylim=(-0.005, 0.20),
                yticks=[0, 0.05, 0.10, 0.15, 0.20],
            )
            axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            axis.grid(axis="y", color="0.90", linewidth=0.7, zorder=0)
            axis.legend(loc="upper left", frameon=False)
            token = f"beta_{alpha:g}_{beta:g}".replace(".", "p")
            path = directory / f"{index:02d}_{token}_mean_mae_vs_order.pdf"
            figure.savefig(path, metadata={
                "Title": f"Beta({alpha:g}, {beta:g}): mean coefficient MAE versus AR order",
                "Subject": "Thesis table tab:coefficient-estimation-all; rounded four-decimal values; common linear y-axis.",
                "Keywords": "AR order, coefficient estimation, MAE, Beta innovations",
            })
            plt.close(figure)
            paths.append(path)
    provenance = {
        "source": str(source.relative_to(PROJECT_ROOT)),
        "table_label": "tab:coefficient-estimation-all",
        "table_sha256": hashlib.sha256(table.encode()).hexdigest(),
        "values": "Rounded mean coefficient MAE as printed in the thesis table (four decimals).",
        "uncertainty": "Not included: the source table does not provide uncertainty intervals.",
        "orders": [1, 2, 3, 4, 5],
        "processes_per_order_and_shape": 100,
        "sample_size": 1000,
        "common_y_limits": [-0.005, 0.20],
        "figures": [dict(panel, pdf=path.name) for panel, path in zip(panels, paths)],
    }
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    for exported_path in export_figures(SOURCE, arguments.output_directory.resolve()):
        print(exported_path)
