# Asymptotics

Study bandwidth-grid size, localization scaling, and similarity eigenvalues.

| Notebook | Purpose |
| --- | --- |
| [experiment_1_joint_N_lambda_limit.ipynb](experiment_1_joint_N_lambda_limit.ipynb) | Joint N and lambda limits, mesh diagnostics, and fixed-sample scaling experiments. |
| [similarity_max_eigenvalue_vs_n.ipynb](similarity_max_eigenvalue_vs_n.ipynb) | Largest similarity eigenvalue versus bandwidth-grid size, with numerical reference checks. |

The joint-limit notebook's repeated-sample and sample-size sections use the
shared spectral runner and `../outputs/data/shared_spectral_sweeps/` cache.
They retain their original seeds, eigenpair diagnostics, summaries, and figures.

Run cells in order. Setup locates the shared helpers in `../kde_tools/` and
enables automatic PDF exports under `../outputs/pdfs/<notebook-name>/`.
Saved data, images, and checkpoints keep their existing locations in
`../outputs/`.

[All KDE notebooks and code guide](../README.md)
