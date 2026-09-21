"""Reproducible illustration for the width-harmonics explanatory note.

All Q quantities refer to half-width, with uniform probability measure on
the sphere. No thesis sources or existing experiment code are modified.
"""
from pathlib import Path
import json
import sys
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[3] / "tmp/pdfs/width_harmonics/mpl-cache"))

import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from Code.Code_AR_processes.ar_tools.functions import (
    simulate_uniform_ar,
    sliding_window_point_cloud,
)

OUT = Path(__file__).resolve().parent
PDF = ROOT / "output/pdf/mean_width_and_q2_example.pdf"
PDF.parent.mkdir(parents=True, exist_ok=True)


def ar1_population(phi, k):
    """Exact AR(1) sums: first k-1 generators plus a geometric tail."""
    generators = np.array([
        [phi ** (j - ell) if j >= ell else 0.0 for ell in range(k)]
        for j in range(k)
    ])
    lengths = np.linalg.norm(generators, axis=1)
    weights = np.ones(k)
    weights[-1] = 1 / (1 - abs(phi))
    R = weights @ lengths
    S = np.einsum("j,ji,jl->il", weights / lengths, generators, generators)
    c = np.exp(gammaln(k / 2) - .5 * np.log(np.pi) - gammaln((k + 1) / 2))
    A = c * k * (k + 2) / (2 * (k + 1)) * (S - R * np.eye(k) / k)
    return c * R, A


phi = 0.6
T = 2000
seed = 20260909
series = simulate_uniform_ar([phi], T, burnin=1000, rng=seed)
cloud = sliding_window_point_cloud(series, 2)

# An equally spaced angular rule on S^1; no random-direction integration error.
M = 32768
theta = np.arange(M) * (2 * np.pi / M)
directions = np.column_stack((np.cos(theta), np.sin(theta)))
emp_half_width = np.empty(M)
for start in range(0, M, 512):
    projections = cloud @ directions[start:start + 512].T
    emp_half_width[start:start + 512] = np.ptp(projections, axis=0) / 2
q0, A = ar1_population(phi, 2)
q0_hat = emp_half_width.mean()
A_hat = 4 * np.einsum("j,ji,jl->il", emp_half_width, directions, directions) / M
A_hat -= 2 * q0_hat * np.eye(2)
q2 = np.einsum("ji,il,jl->j", directions, A, directions)
q2_hat = np.einsum("ji,il,jl->j", directions, A_hat, directions)
pop_half_width = np.abs(directions[:, 0]) + np.abs(phi * directions[:, 0] + directions[:, 1]) / (1 - phi)

# Independent integral checks, beyond re-evaluating the generator formula.
A_integral = 4 * np.einsum("j,ji,jl->il", pop_half_width, directions, directions) / M
A_integral -= 2 * pop_half_width.mean() * np.eye(2)
assert abs(q0 - pop_half_width.mean()) < 2e-7
assert np.max(np.abs(A_integral - A)) < 2e-7
assert abs(np.trace(A)) < 1e-12
assert abs(np.trace(A_hat)) < 1e-12
lambda_errors = []
for k in (2, 3, 4, 8):
    c = np.exp(gammaln(k / 2) - .5 * np.log(np.pi) - gammaln((k + 1) / 2))
    normalizer = np.exp(gammaln(k / 2) - .5 * np.log(np.pi) - gammaln((k - 1) / 2))
    integral = 2 * quad(lambda t: t * (k * t*t - 1) / (k - 1) * normalizer * (1-t*t) ** ((k-3)/2), 0, 1, epsabs=1e-11)[0]
    lambda_errors.append(abs(integral - c / (k + 1)))
assert max(lambda_errors) < 1e-10

evals, evecs = np.linalg.eigh(A)
evals_hat, evecs_hat = np.linalg.eigh(A_hat)
u = evecs[:, 0]
u_hat = evecs_hat[:, 0]
minimum = 2 / np.sqrt(1 + phi**2)
rows = []
for k in range(1, 9):
    q0_k, A_k = ar1_population(phi, k)
    rows.append({"k": k, "mean_width": float(2*q0_k),
                 "min_q2": None if k == 1 else float(np.linalg.eigvalsh(A_k)[0]),
                 "minimum_width": 2/(1-phi) if k == 1 else minimum,
                 "quadratic_min_width": None if k == 1 else float(2*(q0_k+np.linalg.eigvalsh(A_k)[0]))})
summary = {
    "normalization": "Q_l is a harmonic component of half-width; sphere measure has total mass 1",
    "model": {"phi": phi, "innovation_support": [-1, 1], "T": T, "seed": seed, "angular_directions": M},
    "population": {"mean_width": 2*q0, "min_q2": float(evals[0]), "quadratic_min_width": float(2*(q0+evals[0])), "minimum_width": minimum},
    "empirical": {"mean_width": 2*q0_hat, "min_q2": float(evals_hat[0]), "quadratic_min_width": float(2*(q0_hat+evals_hat[0]))},
    "cross_evaluations": {"Q2_at_empirical_minimizer": float(u_hat @ A @ u_hat), "empirical_Q2_at_population_minimizer": float(u @ A_hat @ u)},
    "dimension_comparison": rows,
    "checks": {"mean_integral_error": float(abs(q0-pop_half_width.mean())), "A_integral_max_error": float(np.max(np.abs(A_integral-A))), "lambda2_integral_max_error": float(max(lambda_errors))},
}
(OUT / "example_values.json").write_text(json.dumps(summary, indent=2) + "\n")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "medium", "pdf.fonttype": 42})
blue, orange, green, grey = "#176B9B", "#B34C16", "#23714D", "#666666"
degrees = np.degrees(theta[:M//2 + 1])
s = slice(0, M//2 + 1, 16)

with PdfPages(PDF) as pdf:
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 8.5), sharex=True)
    fig.subplots_adjust(left=.105, right=.975, top=.865, bottom=.175, hspace=.38)
    fig.suptitle("From mean width to the degree-two component", x=.105, ha="left", y=.975, fontsize=18)
    fig.text(.105, .925, "AR(1): coefficient 0.6  |  Uniform[-1, 1] innovations  |  embedding dimension k = 2", fontsize=11)
    ax = axes[0]
    ax.plot(degrees[::16], 2*pop_half_width[s], color=blue, lw=2.2, label="Population width")
    ax.plot(degrees[::16], 2*emp_half_width[s], color=orange, ls=":", lw=2.0, label="Empirical width (T = 2,000)")
    ax.plot(degrees[::16], 2*(q0+q2[s]), color=green, ls="--", lw=2, label=r"Approximation $2(Q_0+Q_2)$")
    ax.axhline(2*q0, color=grey, ls="-.", lw=1.4, label=r"Mean width $2Q_0$")
    ax.set(title="Full width and its approximation", ylabel="Width (data units)")
    ax.set_xlim(0, 180)
    ax.grid(axis="y", alpha=.18)
    ax.legend(loc="upper right", ncol=1, frameon=False, fontsize=9.5)
    ax = axes[1]
    ax.plot(degrees[::16], q2[s], color=blue, lw=2.2, label=r"Population $Q_2$")
    ax.plot(degrees[::16], q2_hat[s], color=orange, ls=":", lw=2, label=r"Empirical $\widehat Q_2$")
    ax.axhline(0, color=grey, lw=.9)
    angle = float(np.degrees(np.arctan2(u[1], u[0])) % 180)
    ax.scatter([angle], [evals[0]], s=45, color=blue, zorder=4)
    ax.annotate(f"min Q2 = {evals[0]:.3f}", (angle, evals[0]), xytext=(-100, 20), textcoords="offset points", arrowprops={"arrowstyle":"-", "color":grey}, fontsize=10)
    ax.set(title="The signed degree-two correction", xlabel=r"Direction angle $\theta$ (degrees)", ylabel="Half-width correction (data units)")
    ax.set_xticks(np.arange(0, 181, 30))
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=.18)
    fig.text(.105, .035, f"True minimum width: {minimum:.3f}     Minimum of approximation: {2*(q0+evals[0]):.3f}\nThe degree-two component has angular average zero. Higher components contain the missing detail.", fontsize=10, linespacing=1.7)
    pdf.savefig(fig)
    fig.savefig(OUT / "width_decomposition.png", dpi=165)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 9.5), sharex=True)
    fig.subplots_adjust(left=.14, right=.97, top=.86, bottom=.17, hspace=.36)
    fig.suptitle("The minimum-width plateau does not transfer to Q2", x=.14, ha="left", y=.97, fontsize=17)
    fig.text(.14, .925, "Same AR(1) model  |  exact geometric-tail formulas for the population quantities", fontsize=11)
    ks = np.arange(1, 9)
    axes[0].plot(ks, [r["mean_width"] for r in rows], "o-", color=blue, lw=2)
    axes[0].set(ylabel="Mean width", title=r"Mean width $2Q_{0,k}$")
    axes[1].plot(ks[1:], [r["min_q2"] for r in rows[1:]], "s-", color=orange, lw=2)
    axes[1].set(ylabel="Min Q2", title=r"Degree-two minimum $\min_u Q_{2,k}(u)$")
    axes[2].plot(ks, [r["minimum_width"] for r in rows], "D-", color=green, lw=2)
    axes[2].set(ylabel="Minimum width", title=r"Full minimum width $m_k$", xlabel="Embedding dimension k")
    for ax in axes:
        ax.set_xticks(ks)
        ax.axvline(2, color=grey, ls=":", lw=1)
        ax.grid(axis="y", alpha=.18)
    fig.text(.14, .035, "The dotted line marks k = p + 1 = 2. Only the full minimum width is constant from there.\nQ2 is used for k >= 2; its value is a signed correction, with the same physical units as width.", fontsize=10, linespacing=1.7)
    pdf.savefig(fig)
    fig.savefig(OUT / "dimension_comparison.png", dpi=165)
    plt.close(fig)

print(json.dumps(summary, indent=2))
print(f"PDF: {PDF}")
