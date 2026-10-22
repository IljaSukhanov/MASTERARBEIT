# Width co-winner counts versus mean coefficient MAE

Analysis of the existing 100-model, one-path-per-model tuning results, T=500, delta=1e-8. No optimizer or notebook settings were changed. Both compared configurations start at e1 and use diameter-normalized initial steps gamma0=s/L. Width co-winner counts use the common reference across all 288 configurations with tolerance 1e-8.

| Configuration | s | H | Width co-winner count | Mean coefficient MAE | Median coefficient MAE | Maximum coefficient MAE |
| --- | --- | --- | --- | --- | --- | --- |
| Highest width count | 0.2 | 50 | 54 | 0.0472194007082 | 0.0102494022509 | 0.783999748217 |
| Lowest mean MAE | 0.1 | 200 | 1 | 0.0254716199286 | 0.0109823168919 | 0.264319305283 |

The highest-width-count configuration has lower width by more than 1e-8 on 79 of the 100 paired paths; the lowest-mean-MAE configuration has lower width on 21. There are no width ties within 1e-8 for this pair.
The highest-width-count configuration has smaller coefficient MAE on 52 paths, versus 48 for the other configuration. The magnitudes of errors, not just their frequency, determine the mean.
On 31 paths, the highest-width-count configuration has both a lower achieved width and a larger coefficient MAE than the lowest-mean-MAE configuration.

Example, model 84: the width winner reaches width 1.07453846165 with MAE 0.0631724851929; the mean-MAE winner reaches width 1.07622258194 with MAE 0.0331257531278. This is a direct observed counterexample to monotone finite-sample ordering of coefficient MAE by achieved empirical width.

Interpretation: the comparison combines two distinct targets (empirical width and coefficient recovery) and two distinct summaries (frequency of near-best widths and arithmetic mean errors). The observations do not isolate finite-sample error from optimization error, and neither configuration is certified to attain the empirical global minimum. The width-count winner has a slightly better median coefficient MAE but a worse mean and larger maximum; its larger errors outweigh its modest advantage in the number of paths with smaller MAE.

Sources: [current tuning notebook](../../Code/Code_AR_processes/xx_notebooks_tuning/ar_tuning_experiments.ipynb), [co-winner implementation](../../Code/Code_AR_processes/ar_tools/tuning.py), [MAE convention](../../Code/Code_AR_processes/ar_tools/coefficient_errors.py), and the existing recorded best directions for all 28,800 fits.
