#!/usr/bin/env python3
"""Monte Carlo experiment for Toeplitz-corrected spectral KDE bandwidths.

The KDEs are evaluated by Fourier inversion on a fine periodic spatial grid.
For each Rényi order, the divergence is symmetrized before constructing
S = exp(-D).  The selected bandwidth maximizes v / v0, where v is the Perron
vector of S and v0 is the Perron vector of the symmetric Toeplitz projection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from kde_tools.fft import kde_family_fft
from kde_tools.experiments import summarize
from kde_tools.similarities import (
    symmetrized_renyi_similarity,
    toeplitz_corrected_choice,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--spatial-points", type=int, default=8192)
    parser.add_argument("--half-width", type=float, default=12.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    h_min = 0.003509098944391056
    h_max = 2.0
    grid_size = 191
    alphas = (0.25, 0.50, 0.75)
    bandwidths = np.geomspace(h_min, h_max, grid_size)
    rng = np.random.default_rng(args.seed)
    selected = {str(alpha): [] for alpha in alphas}
    selected_indices = {str(alpha): [] for alpha in alphas}

    start = perf_counter()
    for run in range(args.runs):
        sample = rng.normal(size=args.sample_size)
        densities, dx = kde_family_fft(sample, bandwidths, args.half_width, args.spatial_points)

        # Orders .25 and .75 share the same pair of directional overlaps;
        # constructing them separately keeps the implementation transparent.
        for alpha in alphas:
            similarity = symmetrized_renyi_similarity(densities, dx, alpha)
            choice, index, _ = toeplitz_corrected_choice(similarity, bandwidths)
            selected[str(alpha)].append(choice)
            selected_indices[str(alpha)].append(index)

        if (run + 1) % 10 == 0 or run + 1 == args.runs:
            elapsed = perf_counter() - start
            print(f"completed {run + 1}/{args.runs} runs in {elapsed:.1f}s", flush=True)

    results = {
        "settings": {
            "runs": args.runs,
            "sample_size": args.sample_size,
            "distribution": "N(0,1)",
            "grid_size": grid_size,
            "h_min": h_min,
            "h_max": h_max,
            "grid": "geometric",
            "alphas": list(alphas),
            "divergence": "symmetrized Renyi",
            "similarity": "exp(-D)",
            "selection": "argmax(v/v0)",
            "seed": args.seed,
            "spatial_points": args.spatial_points,
            "spatial_domain": [-args.half_width, args.half_width],
            "oracle_bandwidth": 0.383039449,
        },
        "summary": {alpha: summarize(np.asarray(values)) for alpha, values in selected.items()},
        "selected": selected,
        "selected_indices": selected_indices,
        "elapsed_seconds": perf_counter() - start,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main()
