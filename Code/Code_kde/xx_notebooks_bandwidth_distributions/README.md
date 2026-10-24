# Bandwidth distributions

Compare the sampling distributions of spectral and benchmark bandwidth choices.

| Notebook | Purpose |
| --- | --- |
| [fast_l2_bandwidth_distributions_vs_n_lambda.ipynb](fast_l2_bandwidth_distributions_vs_n_lambda.ipynb) | Spectral bandwidth distributions across sample sizes and lambda values, plus saved-sample Q-hat comparisons. |

The exact Gaussian L²/ISE benchmark is now the `exact_gaussian` profile in
[section 2 of the main notebook](../xx_notebooks_overview/spectral_scale_selection_kde.ipynb),
with its original preview/thesis settings, independent seeds, ISE exports,
and overlaid/individual plots. Its historical saved results are retained.

The localized N = 1000, lambda = N^(3/2) comparison is now the
`localized` profile in [section 2 of the main notebook](../xx_notebooks_overview/spectral_scale_selection_kde.ipynb).

The main sample-size sweep and the fast L² notebook share their spectral
runner and `../outputs/data/shared_spectral_sweeps/` cache. Their additional
comparison methods and plots retain their existing roles.

Run cells in order. Setup locates the shared helpers in `../kde_tools/` and
enables automatic PDF exports under `../outputs/pdfs/<notebook-name>/`.
Saved data, images, and checkpoints keep their existing locations in
`../outputs/`.

[All KDE notebooks and code guide](../README.md)

The continuous Q-hat overlay shares the Q₀ runner and compatible checkpoints
with the continuous-minimizer notebook under `../outputs/data/shared_continuous_minimizers/`.
