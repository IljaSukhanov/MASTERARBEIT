"""Plot the empirical Q_1 on [0.01,20] for the preceding fixed n=200 sample.

Q_1(h)=(1/2)||partial_h f_hat_h||_2^2, using thesis prop:main_thm.
The shared exact Gaussian-pair objective supplies values and derivatives.
The separate spatial integral below checks the definition directly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

KDE_DIR = Path(__file__).resolve().parents[3]
if str(KDE_DIR) not in sys.path:
    sys.path.insert(0, str(KDE_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.spatial.distance import pdist

from kde_tools.continuous_minimizers import qr_log_energy
from kde_tools.pdf_export import enable_automatic_pdf_saving

NAME = "energy_r1_wide_domain"
DATA_DIR = KDE_DIR / "outputs" / "data" / NAME
IMAGE_DIR = KDE_DIR / "outputs" / "images" / NAME
SAMPLE_DIR = KDE_DIR / "outputs" / "data" / "linear_grid_joint_n_lambda_limit"
H_MIN, H_MAX = 0.01, 20.0


def main() -> None:
    enable_automatic_pdf_saving(NAME)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    samples = np.loadtxt(SAMPLE_DIR / "sample.csv", skiprows=1)
    previous = json.loads((SAMPLE_DIR / "settings.json").read_text())
    sample_hash = hashlib.sha256(samples.tobytes()).hexdigest()
    if sample_hash != previous["sample_sha256"]:
        raise RuntimeError("The stored sample differs from the preceding experiment.")
    d2 = pdist(samples[:, None], metric="sqeuclidean")

    def energy(h: float) -> np.ndarray:
        return qr_log_energy(d2, len(samples), np.log(h), r=1.0)

    h = np.geomspace(H_MIN, H_MAX, 3001)
    values = np.array([energy(bandwidth) for bandwidth in h])
    if not np.all(np.isfinite(values)) or np.any(values[:, 0] <= 0):
        raise RuntimeError("Nonpositive or nonfinite energy values.")
    critical = []
    sign_changes = np.flatnonzero(values[:-1, 1] * values[1:, 1] < 0)
    for i in sign_changes:
        log_h = brentq(
            lambda z: qr_log_energy(d2, len(samples), z, 1.0)[1],
            np.log(h[i]), np.log(h[i+1]), xtol=1e-13,
        )
        point_h = float(np.exp(log_h))
        point = energy(point_h)
        critical.append({
            "h": point_h, "q": float(point[0]),
            "kind": "minimum" if point[2] > 0 else "maximum",
        })
    minimum, = [point for point in critical if point["kind"] == "minimum"]
    maximum, = [point for point in critical if point["kind"] == "maximum"]
    crossing = brentq(lambda t: energy(t)[0] - minimum["q"], maximum["h"], 2.0)

    # Integrate the squared analytic Gaussian-KDE bandwidth derivative in x.
    # This independently checks the pair-sum objective and its factor 1/2.
    checks = []
    for bandwidth in [minimum["h"], 2.0, H_MAX]:
        x = np.linspace(samples.min()-9*bandwidth, samples.max()+9*bandwidth, 20001)
        u = (x[:, None] - samples[None, :]) / bandwidth
        phi = np.exp(-0.5*u*u) / (np.sqrt(2*np.pi)*bandwidth)
        derivative = np.mean(phi*(u*u-1.0)/bandwidth, axis=1)
        direct = float(0.5*np.trapz(derivative**2, x))
        analytic = float(energy(bandwidth)[0])
        relative_error = abs(direct-analytic)/analytic
        if relative_error > 1e-8:
            raise RuntimeError("Direct spatial integration disagrees with the energy formula.")
        checks.append({"h": bandwidth, "analytic": analytic,
                       "direct_spatial_integral": direct, "relative_error": relative_error})

    summary = {
        "r": 1.0, "sample_size": len(samples), "seed": previous["seed"],
        "sample_sha256": sample_hash, "h_min": H_MIN, "h_max": H_MAX,
        "scan_points": len(h), "critical_points": critical,
        "q_at_previous_h_max_2": float(energy(2.0)[0]),
        "q_at_new_h_max_20": float(energy(H_MAX)[0]),
        "descending_branch_crosses_local_minimum_level_at_h": float(crossing),
        "direct_integration_checks": checks,
    }
    pd.DataFrame({"h": h, "q_r1": values[:, 0],
                  "log_h_derivative": values[:, 1], "log_h_curvature": values[:, 2]}).to_csv(
        DATA_DIR / "energy.csv", index=False,
    )
    (DATA_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4))
    color, min_color, max_color = "#0072B2", "#D55E00", "#8C5DA8"
    ax = axes[0]
    ax.loglog(h, values[:, 0], color=color, linewidth=2.2, label=r"$\widehat Q_1(h)$")
    ax.scatter(minimum["h"], minimum["q"], color=min_color, s=35, zorder=4,
               label=rf"local minimum: $h={minimum['h']:.4f}$")
    ax.axvline(2.0, color="#777777", linestyle="--", linewidth=1.3,
               label=r"previous upper limit $h=2$")
    ax.scatter(H_MAX, energy(H_MAX)[0], color=color, s=30, zorder=4)
    mantissa, exponent = f"{energy(H_MAX)[0]:.2e}".split("e")
    ax.annotate(
        rf"$h=20$: $\widehat Q_1={mantissa}\times10^{{{int(exponent)}}}$",
        xy=(H_MAX, energy(H_MAX)[0]), xytext=(0.7, 9e-5),
        arrowprops={"arrowstyle": "->", "color": color}, fontsize=9.5,
    )
    ax.set(xlim=(H_MIN, H_MAX*1.12), xlabel=r"bandwidth $h$ (log scale)",
           ylabel=r"energy $\widehat Q_1(h)$ (log scale)",
           title=r"Full domain: $0.01\leq h\leq20$")
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    ax.grid(which="major", color="#D5DEE5", linewidth=0.7)

    ax = axes[1]
    detail = (h >= 0.45) & (h <= 1.1)
    ax.plot(h[detail], values[detail, 0], color=color, linewidth=2.4)
    ax.axhline(minimum["q"], color="#999999", linestyle=":", linewidth=1)
    ax.scatter(minimum["h"], minimum["q"], color=min_color, s=40, zorder=4)
    ax.scatter(maximum["h"], maximum["q"], color=max_color, s=40, zorder=4)
    ax.annotate(
        f"local minimum\nh = {minimum['h']:.4f}",
        xy=(minimum["h"], minimum["q"]), xytext=(0.48, 0.0191),
        color=min_color, fontsize=10,
        arrowprops={"arrowstyle": "->", "color": min_color},
    )
    ax.annotate(
        f"local maximum\nh = {maximum['h']:.4f}",
        xy=(maximum["h"], maximum["q"]), xytext=(0.82, 0.0231),
        color=max_color, fontsize=10,
        arrowprops={"arrowstyle": "->", "color": max_color},
    )
    ax.set(xlim=(0.45, 1.1), ylim=(0.018, 0.0245),
           xlabel=r"bandwidth $h$", ylabel=r"energy $\widehat Q_1(h)$",
           title="Detail of the shallow interior minimum (linear axes)")
    fig.suptitle(r"Energy functional $\widehat Q_1(h)=\frac{1}{2}\|\partial_h\widehat f_h\|_2^2$", fontsize=16)
    fig.text(0.5, 0.015, r"Same fixed $n=200$ standard-normal sample; Gaussian kernel; seed 20260823",
             ha="center", fontsize=10, color="#444444")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94), w_pad=2.5)
    image_path = IMAGE_DIR / "energy_r1_wide_domain.png"
    fig.savefig(image_path, dpi=180, bbox_inches="tight")
    plt.show()  # Shared automatic exporter saves the corresponding PDF.
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    print("PNG:", image_path)


if __name__ == "__main__":
    main()
