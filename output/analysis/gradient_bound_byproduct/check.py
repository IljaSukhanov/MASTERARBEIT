"""Gradient-only support estimates on the 50 saved padded-start AR(3) runs.

Research diagnostic, not an addition to the active selector. No gradient fit is
rerun or changed. Truth is used only for checks and reporting. Root reflection
is standard minimum-phase conversion; its application here is exploratory.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / "Code/Code_AR_processes/outputs/csv/ar3_order_selection_warm_starts"
OUT = Path(__file__).resolve().parent
MAXIMUM_ORDER = 8


def canonical_direction(direction):
    """Preserve |U(exp(i omega))| and put all lag-polynomial roots outside.

    For a lag-variable root a with |a|<1, replace (B-a) by (1-conj(a)*B).
    Their magnitudes agree on |B|=1. In particular, a=0 removes a delay.
    Only exactly zero trailing coefficients are trimmed; no statistical
    coefficient threshold is used. Unit-circle roots remain on the boundary.
    """
    u = np.asarray(direction, dtype=float)
    if u.ndim != 1 or not np.isfinite(u).all() or np.linalg.norm(u) == 0:
        raise ValueError("Expected a finite, nonzero vector")
    u = u / np.linalg.norm(u)
    coefficients = np.trim_zeros(u, "b")
    roots = np.polynomial.polynomial.polyroots(coefficients)
    v = np.array([complex(coefficients[-1])])
    for root in roots:
        factor = [1.0, -np.conj(root)] if abs(root) < 1 else [-root, 1.0]
        v = np.convolve(v, factor)
    if np.max(np.abs(v.imag)) > 1e-8 * max(1.0, np.linalg.norm(v.real)):
        raise ArithmeticError("Root reconstruction has a material imaginary part")
    v = v.real
    v = np.pad(v, (0, len(u) - len(v)))
    v *= np.sign(v[0])
    v /= np.linalg.norm(v)
    return v


def minimum_feasible_order(response, lags, radius):
    for q in range(lags.shape[1] + 1):
        if q == 0:
            feasible = np.max(np.abs(response)) <= radius
        else:
            design = lags[:, :q]
            fit = linprog(
                np.zeros(q), A_ub=np.vstack([-design, design]),
                b_ub=np.r_[radius - response, radius + response],
                bounds=[(None, None)] * q, method="highs",
                options={"primal_feasibility_tolerance": 1e-9},
            )
            if fit.status not in (0, 2):
                raise RuntimeError(fit.message)
            feasible = fit.success
            if feasible and np.max(np.abs(response - design @ fit.x)) > radius + 1e-8:
                raise ArithmeticError("Feasibility solver residual check failed")
        if feasible:
            return q
    return None


def main():
    rows, source_hashes = [], {}
    spectral_error = 0.0
    exact_recovery_error = 0.0
    frequency_grid = np.exp(2j * np.pi * np.arange(1024) / 1024)
    with threadpool_limits(limits=1):
        for replication in range(1, 51):
            process_path = INPUT / "processes" / f"process_{replication:03d}.csv"
            fit_path = INPUT / "replications" / f"replication_{replication:03d}.csv"
            for path in (process_path, fit_path):
                source_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            series = pd.read_csv(process_path).X_t.to_numpy()
            fits = pd.read_csv(fit_path)
            fit = fits[(fits.variant == "Padded warm start") & (fits.k == MAXIMUM_ORDER + 1)].iloc[0]
            u = fit[[f"u_{j}" for j in range(MAXIMUM_ORDER + 1)]].to_numpy(dtype=float)
            v = canonical_direction(u)
            cloud = np.column_stack([
                series[MAXIMUM_ORDER-j:len(series)-j]
                for j in range(MAXIMUM_ORDER + 1)
            ])
            projections = cloud @ u
            residuals = cloud @ v / v[0]
            response, lags = cloud[:, 0], cloud[:, 1:]
            # Separate regression below is a labelled comparison only. The
            # four gradient estimates above/below do not use its coefficients.
            ols = np.linalg.lstsq(lags, response, rcond=None)[0]
            estimates = {
                "raw_maxabs": np.max(np.abs(projections)) / abs(u[0]),
                "canonical_maxabs": np.max(np.abs(residuals)),
                "canonical_moment": np.sqrt(3 * np.mean(residuals**2)),
                "gain_corrected_maxabs": np.max(np.abs(projections)) / v[0],
                "known_radius_benchmark": 1.0,
                "ols_maxabs_comparison": np.max(np.abs(response - lags @ ols)),
            }
            spectral_error = max(spectral_error, float(np.max(np.abs(
                np.abs(np.polynomial.polynomial.polyval(frequency_grid, u))
                - np.abs(np.polynomial.polynomial.polyval(frequency_grid, v))
            ))))
            # All exact shifted population minimizers must return the same AR
            # filter. These checks use truth; the four estimates never do.
            phi = np.array(json.loads(fit.coefficients))
            truth = np.r_[1.0, -phi]
            truth /= np.linalg.norm(truth)
            canonical_truth = np.pad(truth, (0, MAXIMUM_ORDER + 1 - len(truth)))
            for shift in range(MAXIMUM_ORDER + 2 - len(truth)):
                shifted = np.pad(truth, (shift, MAXIMUM_ORDER + 1 - len(truth) - shift))
                exact_recovery_error = max(exact_recovery_error, float(np.max(np.abs(
                    canonical_direction(shifted) - canonical_truth
                ))))
            for method, radius in estimates.items():
                selected = minimum_feasible_order(response, lags, float(radius))
                rows.append(dict(
                    replication=replication, true_order=3, method=method,
                    estimated_radius=float(radius), selected_order=selected,
                    feasible=selected is not None, raw_first_coordinate=float(u[0]),
                    canonical_first_coordinate=float(v[0]),
                ))
            if replication % 10 == 0:
                print(f"Checked {replication}/50 saved runs", flush=True)
    assert spectral_error < 1e-8
    assert exact_recovery_error < 1e-8
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "selections.csv", index=False)
    summary = []
    for method, group in results.groupby("method", sort=False):
        summary.append(dict(
            method=method, correct=int((group.selected_order == 3).sum()),
            under=int((group.selected_order < 3).sum()),
            over=int((group.selected_order > 3).sum()),
            infeasible=int((~group.feasible).sum()), total=len(group),
            radius_min=float(group.estimated_radius.min()),
            radius_median=float(group.estimated_radius.median()),
            radius_max=float(group.estimated_radius.max()),
            radius_below_truth=int((group.estimated_radius < 1).sum()),
        ))
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "summary.csv", index=False)
    (OUT / "verification.json").write_text(json.dumps(dict(
        dataset="50 existing padded-start AR(3) runs; no new simulations",
        maximum_order=MAXIMUM_ORDER, embedding_dimension=MAXIMUM_ORDER + 1,
        observation_times="t=9,...,2000 (one-based), common to every order",
        variant="Padded warm start", changes_to_gradient="none",
        known_radius="1.0, used only for diagnostics and the labelled benchmark",
        maximum_frequency_magnitude_error=spectral_error,
        maximum_exact_shifted_population_recovery_error=exact_recovery_error,
        source_sha256=source_hashes,
    ), indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    print(f"Maximum frequency-magnitude error: {spectral_error:.3g}")
    print(f"Maximum exact shifted-filter recovery error: {exact_recovery_error:.3g}")


if __name__ == "__main__":
    main()
