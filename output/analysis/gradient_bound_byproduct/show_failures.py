"""Export calculations and plots for the 16 additional underselections.

Uses saved gradient fits only. Minimax programs explain the already observed
classifications; they do not alter the fitted directions or bound estimates.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from check import INPUT, MAXIMUM_ORDER, OUT, canonical_direction
from diagnose_failures import minimax_radius


IMAGE_DIR = OUT / "failed_case_figures"
TEAL, RED, PURPLE, GRAY = "#16766b", "#c44736", "#62509a", "#657080"


def main():
    IMAGE_DIR.mkdir(exist_ok=True)
    diagnostics = pd.read_csv(OUT / "failure_diagnostics.csv")
    failed = diagnostics[(diagnostics.selected_order < 3) &
                         (diagnostics.known_radius_selected_order == 3)].copy()
    assert len(failed) == 16
    estimates = pd.read_csv(OUT / "selections.csv")
    calculations, radii_rows, residual_rows = [], [], []
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .16, "grid.linewidth": .6})
    with threadpool_limits(limits=1):
        for _, diagnostic in failed.iterrows():
            replication = int(diagnostic.replication)
            series = pd.read_csv(INPUT / "processes" / f"process_{replication:03d}.csv").X_t.to_numpy()
            fits = pd.read_csv(INPUT / "replications" / f"replication_{replication:03d}.csv")
            curve = fits[fits.variant == "Padded warm start"].sort_values("k")
            fit = curve[curve.k == MAXIMUM_ORDER + 1].iloc[0]
            u = fit[[f"u_{j}" for j in range(MAXIMUM_ORDER + 1)]].to_numpy(dtype=float)
            v = canonical_direction(u)
            cloud = np.column_stack([
                series[MAXIMUM_ORDER-j:len(series)-j]
                for j in range(MAXIMUM_ORDER + 1)
            ])
            canonical_projection = cloud @ v
            residual = canonical_projection / v[0]
            numerator = float(np.max(np.abs(canonical_projection)))
            radius = numerator / v[0]
            assert abs(radius - diagnostic.estimated_radius) < 1e-9
            sample_max = int(np.argmax(np.abs(residual)))
            response, lags = cloud[:, 0], cloud[:, 1:]
            radii = [minimax_radius(response, lags, q) for q in range(MAXIMUM_ORDER + 1)]
            selected = next(q for q, value in enumerate(radii) if value <= radius + 1e-8)
            assert selected == int(diagnostic.selected_order)
            assert radii[3] <= 1 < radii[2]
            assert radii[selected] <= radius + 1e-8
            if selected > 0:
                assert radii[selected-1] > radius + 1e-8
            phi = np.array(json.loads(fit.coefficients))
            original_width = float(np.ptp(cloud @ u))
            canonical_width = float(np.ptp(canonical_projection))
            calculations.append(dict(
                replication=replication, true_order=3, pilot_embedding_dimension=9,
                achieved_gradient_width=original_width,
                canonical_projection_width=canonical_width,
                maximum_absolute_canonical_projection=numerator,
                canonical_first_coordinate=float(v[0]), estimated_radius=float(radius),
                maximizing_observation_time=sample_max + MAXIMUM_ORDER + 1,
                order_2_minimax_radius=radii[2], order_3_minimax_radius=radii[3],
                selected_order=selected, known_radius_selected_order=3,
                canonical_direction=json.dumps(v.tolist()),
                original_direction=json.dumps(u.tolist()), coefficients=json.dumps(phi.tolist()),
            ))
            radii_rows.extend(dict(replication=replication, candidate_order=q, radius=value)
                              for q, value in enumerate(radii))
            residual_rows.append(pd.DataFrame(dict(
                replication=replication,
                t=np.arange(MAXIMUM_ORDER+1, len(series)+1),
                canonical_projection=canonical_projection, fitted_residual=residual,
            )))
            fig, (left, right) = plt.subplots(1, 2, figsize=(11.2, 4.7))
            fig.subplots_adjust(left=.075, right=.98, bottom=.20, top=.76, wspace=.25)
            fig.suptitle(f"Run {replication:02d} | True order 3 | Selected order {selected}",
                         x=.075, y=.98, ha="left", fontsize=15, fontweight="bold")
            fig.text(.075, .89,
                     f"Bound calculation: {numerator:.9f} / {v[0]:.9f} = {radius:.9f}"
                     "    (maximum absolute corrected projection / first coordinate)", fontsize=10)
            left.plot(curve.k, curve.minimum_width, "o-", color=PURPLE, markersize=4,
                      label="Saved gradient width")
            plateau = 2 / np.sqrt(1 + phi @ phi)
            left.hlines(plateau, 4, 10, color=TEAL, linestyle="--", linewidth=1.5,
                        label="Population minimum, k >= 4")
            left.axvline(4, color=TEAL, alpha=.4, linewidth=1)
            left.scatter([9], [original_width], s=75, facecolors="none", edgecolors=RED,
                         linewidths=1.8, zorder=5, label="Pilot fit at k = 9")
            left.set(title="Width optimization output", xlabel="Embedding dimension k",
                     ylabel="Geometric width", xticks=np.arange(1, 11), xlim=(.7, 10.3))
            left.legend(fontsize=8, frameon=False, loc="upper right")
            right.axhspan(radii[3], radii[2], color=TEAL, alpha=.12)
            right.plot(range(9), radii, "o-", color=GRAY, markersize=4,
                       label="Best achievable residual radius")
            right.axhline(radius, color=RED, linewidth=1.7, label=f"Estimated bound: {radius:.4f}")
            right.axhline(1, color=TEAL, linewidth=1.4, linestyle="--", label="True bound: 1")
            right.scatter([selected], [radii[selected]], color=RED, s=55, zorder=6)
            right.axvline(selected, color=RED, alpha=.25, linewidth=1)
            right.set(title="First order feasible under the estimated bound",
                      xlabel="Candidate AR order q", ylabel="Residual radius (log scale)",
                      xticks=np.arange(9), xlim=(-.3, 8.3), yscale="log")
            right.legend(fontsize=8, frameon=False, loc="upper right")
            fig.text(.075, .075,
                     f"Order 3 requires {radii[3]:.9f} <= bound < {radii[2]:.9f}."
                     f"  Estimated bound {radius:.9f} admits order {selected}.", fontsize=10)
            fig.text(.075, .025,
                     "Left: saved per-dimension observation windows. Right: common t = 9,...,2000."
                     " Shading marks the interval selecting order 3.", fontsize=8, color=GRAY)
            fig.savefig(IMAGE_DIR / f"run_{replication:02d}.png", dpi=180)
            plt.close(fig)
    calculations = pd.DataFrame(calculations)
    calculations.to_csv(OUT / "sixteen_failed_calculations.csv", index=False)
    pd.DataFrame(radii_rows).to_csv(OUT / "sixteen_failed_order_radii.csv", index=False)
    pd.concat(residual_rows, ignore_index=True).to_csv(OUT / "sixteen_failed_residuals.csv", index=False)
    fig, ax = plt.subplots(figsize=(10.2, 8.1))
    fig.subplots_adjust(left=.21, right=.96, bottom=.12, top=.87)
    for position, row in calculations.iterrows():
        ax.plot([row.order_3_minimax_radius, row.order_2_minimax_radius],
                [position, position], color=TEAL, linewidth=6, solid_capstyle="butt", alpha=.8)
        ax.scatter([row.estimated_radius], [position], color=RED, s=40, zorder=5)
        ax.text(row.estimated_radius+.075, position, f"{row.estimated_radius:.4f}",
                va="center", fontsize=9, color=RED)
    ax.axvline(1, color=GRAY, linestyle="--", linewidth=1, zorder=0)
    ax.set(yticks=np.arange(16),
           yticklabels=[f"Run {int(row.replication):02d}   -> order {int(row.selected_order)}"
                        for _, row in calculations.iterrows()],
           xlim=(.9, float(calculations.estimated_radius.max())+.65), ylim=(15.7, -.7),
           xlabel="Innovation bound / residual radius")
    ax.grid(axis="y", visible=False)
    fig.suptitle("All 16 additional underselections", x=.07, y=.965,
                 ha="left", fontsize=18, fontweight="bold")
    fig.text(.07, .915,
             "True order = 3; true bound = 1. Each red estimate exceeds the interval selecting order 3.",
             fontsize=10)
    ax.legend(handles=[Line2D([0], [0], color=TEAL, linewidth=6,
                              label="Bound interval selecting order 3"),
                       Line2D([0], [0], color=RED, marker="o", linestyle="none",
                              label="Gradient-derived bound"),
                       Line2D([0], [0], color=GRAY, linestyle="--", label="True bound = 1")],
              loc="lower right", frameon=False, fontsize=9)
    fig.text(.07, .035,
             "Saved padded-start fits at embedding k = 9 (8 lags), n = 1992."
             " Known-bound selection returns order 3 in every case.\n"
             "Original diagnostic calculation; all 16 cases are included. No gradient fits were rerun.",
             fontsize=9, color=GRAY)
    fig.savefig(OUT / "sixteen_failed_overview.png", dpi=180)
    plt.close(fig)
    print(calculations[["replication", "maximum_absolute_canonical_projection",
                        "canonical_first_coordinate", "estimated_radius",
                        "order_2_minimax_radius", "selected_order"]].to_string(index=False))


if __name__ == "__main__":
    main()
