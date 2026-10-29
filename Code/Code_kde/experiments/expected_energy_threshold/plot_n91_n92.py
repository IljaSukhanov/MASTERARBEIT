"""Plot the exact expected Gaussian KDE energy across the n=91/92 threshold.

Original derivation and assumptions:
output/research/gaussian_qhat_properties_2026-10-29/report.md, Section 5.
Every displayed figure is exported automatically to PDF by the shared helper.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qhat-matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from matplotlib.ticker import FormatStrFormatter

KDE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KDE_ROOT))

from kde_tools.pdf_export import enable_automatic_pdf_saving
from kde_tools.qhat import expected_gaussian_qhat_log_energy


def main():
    r = 13/14
    sigma2 = 1.0
    a, b = 1+2*r, 4-2*r
    output = KDE_ROOT / "outputs" / "expected_energy_threshold"
    output.mkdir(parents=True, exist_ok=True)
    pdf_directory = enable_automatic_pdf_saving("expected_qhat_n91_n92")

    def energies(h, n):
        exact = np.array([expected_gaussian_qhat_log_energy(np.log(x), n, r, sigma2)
                          for x in np.asarray(h)])
        # Independent check of the expectation's closed form.
        closed = 3/(16*np.sqrt(np.pi)*n)*(h**(-a)+(n-1)*h**b/(sigma2+h*h)**2.5)
        np.testing.assert_allclose(exact[:, 0], closed, rtol=2e-14, atol=1e-16)
        return exact

    extrema = {}
    s_peak = b/7
    for n in (91, 92):
        sign = lambda s: -a+(n-1)*s**2.5*(b-5*s)
        extrema[n] = []
        if sign(s_peak) > 0:
            for kind, (lo, hi) in zip(("minimum", "maximum"),
                                      ((1e-12, s_peak), (s_peak, b/5))):
                s = brentq(sign, lo, hi, xtol=1e-15)
                h = float(np.sqrt(sigma2*s/(1-s)))
                value, first, second = expected_gaussian_qhat_log_energy(np.log(h), n, r, sigma2)
                assert abs(first) < 1e-14
                assert (second > 0) == (kind == "minimum")
                extrema[n].append(dict(kind=kind, h=h, expected_energy=float(value)))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "semibold", "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.3), layout="constrained",
                             gridspec_kw={"height_ratios": [1, 1.25]})
    fig.suptitle(r"Expected Gaussian KDE scale energy: $r=13/14$, $X_i\sim\mathcal{N}(0,1)$",
                 fontsize=13)
    overview = np.linspace(0.12, 2, 1800)
    zoom = np.linspace(0.59, 0.72, 1800)
    colors = {91: "#235789", 92: "#c26915"}

    for col, n in enumerate((91, 92)):
        color = colors[n]
        full_values = energies(overview, n)
        zoom_values = energies(zoom, n)
        ax = axes[0, col]
        ax.plot(overview, full_values[:, 0], color=color, linewidth=2.2)
        ax.axvspan(zoom[0], zoom[-1], color=color, alpha=0.10)
        ax.set(title=f"n = {n} · overview", xlim=(overview[0], overview[-1]),
               yscale="log", ylim=(0.004, 0.6), xlabel="Bandwidth h",
               ylabel=r"$\mathbb{E}[\widehat{Q}_{13/14}(h)]$")
        ax.grid(axis="y", which="major", alpha=0.18)

        ax = axes[1, col]
        ax.plot(zoom, zoom_values[:, 0], color=color, linewidth=2.4)
        ax.set(title=f"n = {n} · close-up", xlim=(zoom[0], zoom[-1]),
               xlabel="Bandwidth h", ylabel=r"$\mathbb{E}[\widehat{Q}_{13/14}(h)]$")
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.6f"))
        ax.grid(alpha=0.17)
        if n == 91:
            # The strict derivative sign is verified by its analytic maximum.
            assert -a+(n-1)*s_peak**2.5*(b-5*s_peak) < 0
            ax.text(0.055, 0.09, "Strictly decreasing\nNo stationary points",
                    transform=ax.transAxes, color=color, va="bottom")
            ax.margins(y=0.16)
        else:
            for point in extrema[n]:
                h, q = point["h"], point["expected_energy"]
                is_min = point["kind"] == "minimum"
                ax.scatter(h, q, color=color, edgecolors="white", linewidths=0.7,
                           marker="v" if is_min else "^", s=65, zorder=5)
                ax.annotate(("Local minimum" if is_min else "Local maximum")
                            +f"\nh = {h:.5f}", (h, q),
                            xytext=(-4, -34) if is_min else (0, 31),
                            textcoords="offset points", ha="center",
                            va="top" if is_min else "bottom", fontsize=9,
                            arrowprops={"arrowstyle": "-", "color": color,
                                        "linewidth": 0.8})
            ax.set_ylim(0.021146, 0.021202)

    png_path = output / "expected_qhat_n91_n92.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.show()  # Shared exporter automatically saves the same figure as PDF.
    plt.close(fig)
    result = dict(r=r, sampling_distribution="iid N(0,1)", sigma2=sigma2,
                  expectation="3/(16 sqrt(pi) n) [h^(-1-2r) + (n-1) h^(4-2r)/(1+h^2)^(5/2)]",
                  critical_sample_size=float(1+a/(2*(b/7)**3.5)),
                  stationary_points=extrema, png=str(png_path),
                  pdfs=[str(p) for p in sorted(pdf_directory.glob("*.pdf"))])
    (output / "expected_qhat_n91_n92.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
