#!/usr/bin/env python3
"""KL versus reverse-KL experiment using ordinary KDE quadrature."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from kde_tools.bandwidths import gaussian_mise_oracle_bandwidth
from kde_tools.density import gaussian_kde_values, quadrature_grid_for_kdes
from kde_tools.experiments import summarize
from kde_tools.similarities import directional_kl_similarity, toeplitz_corrected_choice


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--quadrature-points", type=int, default=4001)
    parser.add_argument("--padding", type=float, default=8.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    h_min = 0.003509098944391056
    h_max = 2.0
    grid_size = 191
    bandwidths = np.geomspace(h_min, h_max, grid_size)
    rng = np.random.default_rng(args.seed)
    selected = {"KL": [], "reverse_KL": []}
    selected_indices = {"KL": [], "reverse_KL": []}
    kl_scores: list[list[float]] = []

    start = perf_counter()
    for run in range(args.runs):
        sample = rng.normal(size=args.sample_size)
        x_grid = quadrature_grid_for_kdes(
            sample,
            bandwidths,
            m_x=args.quadrature_points,
            padding=args.padding,
        )
        densities = gaussian_kde_values(sample, bandwidths, x_grid)
        forward_similarity = directional_kl_similarity(densities, x_grid)
        similarities = {"KL": forward_similarity, "reverse_KL": forward_similarity.T}
        for name, similarity in similarities.items():
            choice, index, score = toeplitz_corrected_choice(
                similarity, bandwidths, directed=True
            )
            selected[name].append(choice)
            selected_indices[name].append(index)
            if name == "KL":
                kl_scores.append(score.tolist())
        if (run + 1) % 10 == 0 or run + 1 == args.runs:
            print(f"completed {run + 1}/{args.runs}", flush=True)

    results = {
        "settings": {
            "runs": args.runs,
            "sample_size": args.sample_size,
            "distribution": "N(0,1)",
            "grid_size": grid_size,
            "h_min": h_min,
            "h_max": h_max,
            "grid": "geometric",
            "divergences": ["KL(p_i || p_j)", "KL(p_j || p_i)"],
            "similarity": "exp(-D)",
            "centrality": "Doob stationary mass pi_i proportional to l_i r_i",
            "null": "signed-lag Toeplitz projection",
            "selection": "argmax(pi/pi0)",
            "seed": args.seed,
            "density_evaluation": "direct Gaussian KDE",
            "integration": "composite trapezoidal quadrature",
            "quadrature_points": args.quadrature_points,
            "quadrature_padding": args.padding,
            "oracle_bandwidth": gaussian_mise_oracle_bandwidth(args.sample_size),
        },
        "summary": {name: summarize(np.asarray(values)) for name, values in selected.items()},
        "selected": selected,
        "selected_indices": selected_indices,
        "bandwidth_grid": bandwidths.tolist(),
        "kl_scores": kl_scores,
        "identical_selections": selected_indices["KL"] == selected_indices["reverse_KL"],
        "elapsed_seconds": perf_counter() - start,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results["summary"], indent=2))
    print(f"identical selections: {results['identical_selections']}")


if __name__ == "__main__":
    main()
