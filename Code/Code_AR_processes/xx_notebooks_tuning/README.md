# AR tuning notebooks

This folder contains two complementary notebooks. Both use the current shared
geometric optimizer, actual Euclidean cloud diameter, achieved-width selection,
and explicit fixed seeds. They display tables in the notebook and export only
PDF/PNG figures; neither writes JSON/CSV data or computes an empirical global optimum.

| Notebook | Purpose |
|---|---|
| [ar_unified_tuning_experiments.ipynb](ar_unified_tuning_experiments.ipynb) | Thesis experiments 1, 2 and 3: one fixed AR(3) path, fresh paths with fixed coefficients, and fresh paths with Jones-sampled coefficient vectors (both default 100). All use one common fixed starting direction. Compare absolute and diameter-normalized steps, show achieved-width and win-count matrices, and retain the fixed-path recovered-coefficient table. |
| [ar_tuning_sample_size_and_validation.ipynb](ar_tuning_sample_size_and_validation.ipynb) | Additional sample-size and validation study: tune jointly over T=500 and T=2000 on six paths, freeze the selected setting, then compare it with a fixed baseline on six separate validation paths. Each smaller sample is a prefix of its paired longer path. |

Experiments 1 and 2 and the extension use phi=(0.6,-0.3,0.2).
Experiment 3 samples a fresh causal AR(3) coefficient vector for each path
using the existing `jones_ar_coefficients` sampler. All use uniform innovations on [-1,1].
Both notebooks use one fixed starting direction, scales 0.1,0.2,...,0.9,
half-lives H in {2,5,10,20,50,100,200,1000}, beta=2^(-1/H), and a best-value
tracking rule that accepts every strictly smaller width (tolerance zero).
Both notebooks use a 2,000-iteration cap and patience of 200.
The unified notebook awards a full +1 to every configuration within an
absolute tolerance epsilon=1e-8 of each path's best achieved minimum width.
The normalized initial step is c/L for each realized cloud. Set
`number_of_realizations` in the unified notebook’s settings cell to choose its
replication count; the default is 100. The tables and plots follow this value.

In the extension, the baseline remains frozen at the unified experiment's earlier
20-path pilot winner: absolute gamma_0=0.7, H=50. The extension uses process-seed
roots 20261021 for tuning and 20261022 for validation. Its validation results
do not revise the selected configuration. Joint tuning gives each
replication/sample-size case equal weight; no relative-gap ranking is used.

Shared computations live in [tuning.py](../ar_tools/tuning.py) and
[descent.py](../ar_tools/descent.py). The one-size and multiple-size tuning
experiments use the same runner. Frozen validation evaluates only the supplied
settings and performs no parameter search. No resume/cache workflow is used.
The unified notebook obtains its recovered-coefficient table and achieved
widths directly from the grid results.

The unified notebook also compares win frequencies using the same achieved
minimum empirical widths. On each realized path it awards +1 to every
configuration whose achieved width is at most the path's best achieved width
plus epsilon. Set `epsilon_win_tolerance` in the settings cell; its default
is 1e-8. Scores use full precision. Points are not split, and evaluation
counts do not exclude qualifying configurations. Each
configuration can win at most once per path, but the total credits across
configurations can exceed the number of realizations. All configurations
tied for the highest final count are displayed and highlighted. Count matrices
are exported as PDF/PNG. The shared `summarize_configuration_wins` helper
includes zero-win configurations and checks that each paired case is complete.
The optimizer and mean-width ranking continue to use full precision.
The count section reuses the 2,000-iteration/patience-200 descent results;
it performs no extra simulation or parameter search.
Shared matrix plotting routines live in `ar_tools/tuning_plots.py`.

Experiment 3 appears below the fixed-coefficient results. Set
`number_of_coefficient_realizations` (default 100), `jones_coefficient_seed`
(20261023) and `jones_process_seed` (20261024) in its own settings cell.
It uses T=500, burn-in 2000, the same 2000/200 stopping settings and the exact
fixed `initial_direction` from experiment 1. All 144 labeled configurations
are evaluated on every sampled model. Its primary selection uses the same
epsilon=1e-8 co-winner counts; mean achieved widths remain a separate diagnostic.
The coefficient and innovation streams are independent, and no spectral-radius
cutoff filters the Jones draws. The tables retain every model's coefficients,
partial autocorrelations, spectral radius and both seeds. Shared runners in
`tuning.py` perform the fixed- and random-coefficient experiments without
duplicating the optimizer. Only mean-width and win-count matrices are exported,
under a separate `jones_<run-id>` directory; iteration plots and JSON/CSV data
exports are absent. Sampling reference: [Jones (1987), pp. 134–138](https://doi.org/10.2307/2347544).
Its final post-fit diagnostic reports nonstationary recovered estimates,
undefined coefficients and coefficient MSE above 1 across the full grid,
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
