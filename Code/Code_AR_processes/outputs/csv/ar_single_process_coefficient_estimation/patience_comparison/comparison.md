# Patience stopping comparison for one AR(3) process

This is an empirical comparison on the single process configured in `ar_single_process_coefficient_estimation.ipynb`: p=3, T=2000, burn-in=1000, coefficient seed 20260910, process seed 20260911, optimizer seed 20260912. All variants receive the same 12 starting directions, step scale 0.2, geometric decay 0.997, and improvement tolerance 1e-10.

| Variant | Total evaluated iterations | Starts stopped by patience | Winning start | Winning width | Largest change in winning coefficients versus fixed 5,000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| limit_5000 | 60,000 | 0 | 3 | 1.305535216612 | 0 |
| patience_200_cap_5000 | 50,452 | 2 | 3 | 1.305535216612 | 0 |
| patience_200_cap_10000 | 72,498 | 12 | 3 | 1.305535204532 | 1.0780888e-07 |

With the same 5,000-iteration cap, patience 200 saved 15.91% of evaluated iterations and returned an identical winning coefficient vector. Starts 5 and 8 stopped after 201 and 251 evaluations, respectively, and had worse widths than their corresponding full runs. The final winner was unchanged because start 3 won both comparisons.

These are observations for one process, not an equivalence or convergence guarantee. Timing in `summary.csv` is one measurement per variant and excludes simulation, diameter calculation, and plotting; use iteration counts to compare work without overinterpreting timing noise.

The configuration and full-precision per-start estimates are in `configuration.json` and `per_start.csv`. The implementation is `ar_tools/descent.py`, function `projected_subgradient_empirical_width` (patience counter, improvement tolerance, and tangent-subgradient check).
