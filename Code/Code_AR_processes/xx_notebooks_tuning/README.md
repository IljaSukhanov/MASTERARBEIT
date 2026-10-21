# AR tuning notebooks

This folder contains the tuning notebook. It uses the current shared
geometric optimizer, absolute initial steps, achieved-width selection,
and explicit fixed seeds. It displays tables in the notebook and exports only
PDF/PNG figures; it writes no JSON/CSV data and computes no empirical global optimum.

| Notebook | Purpose |
|---|---|
| [ar_tuning_experiments.ipynb](ar_tuning_experiments.ipynb) | Thesis experiments 2 and 3: independent paths with fixed coefficients, and paths with Jones-sampled coefficient vectors (both default 100). Both use one common fixed starting direction and absolute steps. Show width and coefficient-error win counts for experiment 2, plus model diagnostics for experiment 3. |

Experiment 2 uses phi=(0.6,-0.3,0.2).
Experiment 3 samples a fresh causal AR(3) coefficient vector for each path
using the existing `jones_ar_coefficients` sampler. All use uniform innovations on [-1,1].
The notebook uses one fixed starting direction, scales 0.1,0.2,...,0.9,
half-lives H in {2,5,10,20,50,100,200,1000}, beta=2^(-1/H), and a best-value
tracking rule that accepts every strictly smaller width (tolerance zero).
The notebook uses a 2,000-iteration cap and patience of 200.
The notebook awards a full +1 to every configuration within an
absolute tolerance epsilon=1e-8 of each path's best achieved minimum width.
Each grid value is used directly as the initial step gamma_0. Set
`number_of_realizations` in the notebook’s settings cell to choose its
replication count; the default is 100. The tables and plots follow this value.

Shared computations live in [tuning.py](../ar_tools/tuning.py) and
[descent.py](../ar_tools/descent.py). No resume/cache workflow is used.
The shared setup creates the common starting direction without simulating
a preliminary path. Both experiments obtain their achieved widths and
coefficient diagnostics directly from the repeated-path grid results.

The notebook also compares win frequencies using the same achieved
minimum empirical widths. On each realized path it awards +1 to every
configuration whose achieved width is at most the path's best achieved width
plus epsilon. Set `epsilon_win_tolerance` in the settings cell; its default
is 1e-8. Scores use full precision. Points are not split, and evaluation
counts do not exclude qualifying configurations. Each
configuration can win at most once per path, but the total credits across
configurations can exceed the number of realizations. All configurations
tied for the highest final count are displayed and highlighted. Count matrices
are exported as PDF/PNG. Experiment 2 also gives a separate +1 for coefficient
MAE within `coefficient_epsilon_win_tolerance` (default 1e-2) of the
smallest finite coefficient MAE on that path, across the absolute-step grid.
It reuses each run's recorded minimum-width direction and the known simulation
coefficients. Undefined coefficient estimates receive no coefficient credit.
Each matrix cell shows width counts above coefficient-MAE counts; red boxes
mark the highest width counts, and green boxes mark the highest
coefficient-MAE counts, including all ties. Width counts determine the colors.
The shared `summarize_configuration_wins` helper
includes zero-win configurations and checks that each paired case is complete.
The optimizer and absolute-gap comparisons continue to use full precision.
Equal win counts use parameter labels for display order; all tied leaders remain.
The count section reuses the 2,000-iteration/patience-200 descent results;
it performs no extra simulation or parameter search.
Shared matrix plotting routines live in `ar_tools/tuning_plots.py`.
Experiment 2 progress lines and the per-path table show the coefficient estimates
and their MAE for each path's lowest-width configuration.

Experiment 3 appears below the fixed-coefficient results. Set
`number_of_coefficient_realizations` (default 100), `jones_coefficient_seed`
(20261023) and `jones_process_seed` (20261024) in its own settings cell.
It uses T=500, burn-in 2000, the same 2000/200 stopping settings and the exact
fixed `initial_direction` from the shared setup. All 72 configurations
are evaluated on every sampled model. Its primary selection uses the same
epsilon=1e-8 co-winner counts. The repeated-path runners return raw grid results
without aggregate width ranking or selection (`summarize=False`).
The coefficient and innovation streams are independent, and no spectral-radius
cutoff filters the Jones draws. The tables retain every model's coefficients,
partial autocorrelations, spectral radius and both seeds. Shared runners in
`tuning.py` perform the fixed- and random-coefficient experiments without
duplicating the optimizer. Only win-count matrices are exported,
under a separate `jones_<run-id>` directory; iteration plots and JSON/CSV data
exports are absent. Sampling reference: [Jones (1987), pp. 134–138](https://doi.org/10.2307/2347544).
Its final post-fit diagnostic reports nonstationary recovered estimates,
undefined coefficients and coefficient MAE above 1 across the full grid,
the leading shared configurations and each model's lowest-width grid result.
These diagnostics do not filter results or alter tuning decisions.
Each progress line also reports the theoretical population minimum
2/sqrt(1+||phi||^2) and the signed attained-empirical-minus-population gap.
The model table and final width diagnostic retain this reference and the
empirical width in the true-coefficient direction. The latter is a feasible
direction on the same cloud: a larger attained width demonstrates a missed
available improvement. Its positive gap gives a lower bound on empirical
suboptimality. The population discrepancy can be negative and is not an
empirical optimization gap. Neither benchmark changes the tuning score.

Figures are saved under `Code/Code_AR_processes/outputs/pdfs/<experiment>/<run-id>/`
and the corresponding `outputs/images/` folder. The experiment names match the
notebook filenames. Run IDs include settings and shared implementation hashes.

The four-coordinate AR(3) embedding uses S^3; the thesis's tuning
paragraph currently writes S^2. No thesis sources are edited.

Specification: [AR thesis chapter, Tuning experiments](../../../Abschlussarbeit_2021/LaTeX/chapters/3_application_to_time_series.tex).

Coefficient MAE is the mean absolute error across the three coefficients,
computed by [`coefficient_mae`](../ar_tools/coefficient_errors.py).
The coefficient tolerance is measured in these per-coefficient MAE units.
