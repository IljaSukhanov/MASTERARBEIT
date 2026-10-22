# Why width winners can use the full iteration budget

A targeted replay of models 1–3 from the current tuning notebook, using its original coefficient/innovation seeds, T=500, burn-in 2000, one e1 start, normalized scale s=0.2, 2000 allowed iterations, patience 200 and improvement_tolerance=0. No tuning source or settings were changed for this replay.

| Model | Half-life H | Evaluations | Best achieved width | First evaluation within 1e-8 of its eventual width | Last strict improvement at evaluation | Size of last strict improvement |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 301 | 4.154101897917183 | 53 | 101 | 8.88e-16 |
| 1 | 50 | 2000 | 0.9942104436394108 | 953 | 1978 | 2.66e-15 |
| 2 | 2 | 304 | 4.047162159672572 | 57 | 104 | 8.88e-16 |
| 2 | 50 | 2000 | 1.3147412080775065 | 1104 | 1997 | 4.00e-15 |
| 3 | 2 | 302 | 3.083687480296229 | 53 | 102 | 4.44e-16 |
| 3 | 50 | 2000 | 1.228150515029825 | 1033 | 1996 | 1.33e-15 |

The proximity calculation uses each run's own final best-so-far width. It does not establish proximity to the global minimum, nor the first time it became a co-winner against all 288 configurations.

Interpretation of the observed histories: these H=50 runs attained a width within 1e-8 of their eventual result much earlier than they stopped. Later improvements far smaller than the 1e-8 win tolerance still reset the zero-tolerance patience counter. H=2 stopped earlier but at substantially larger widths on these same three clouds. These observations describe the replayed runs, not a guarantee for all configurations or models.

For a future speed comparison, first attainment of a shared per-path target (best width found by all candidates plus a declared tolerance) would answer a different question from whether the strict-improvement patience counter stopped before the cap. This is a suggested evaluation design, not a change to the current notebook.

Implementation sources: [descent stopping and record updates](../../Code/Code_AR_processes/ar_tools/descent.py), [paired model generation and schedules](../../Code/Code_AR_processes/ar_tools/tuning.py), [current notebook settings](../../Code/Code_AR_processes/xx_notebooks_tuning/ar_tuning_experiments.ipynb).
