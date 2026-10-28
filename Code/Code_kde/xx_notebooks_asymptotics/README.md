# Asymptotics

Study bandwidth-grid size, localization scaling, and similarity eigenvalues.

| Notebook | Purpose |
| --- | --- |
| [experiment_1_joint_N_lambda_limit.ipynb](experiment_1_joint_N_lambda_limit.ipynb) | Joint N and lambda limits, mesh diagnostics, and fixed-sample scaling experiments. |
| [linear_grid_joint_N_lambda_limit.ipynb](linear_grid_joint_N_lambda_limit.ipynb) | Figure 3.6 experiment with `GRID_R = "1"` (linear) or `"13/14"`, with separate saved results, PDFs, and numerical checks. |
| [similarity_max_eigenvalue_vs_n.ipynb](similarity_max_eigenvalue_vs_n.ipynb) | Largest similarity eigenvalue versus bandwidth-grid size, with numerical reference checks. |

The joint-limit notebook's repeated-sample and sample-size sections use the
shared spectral runner and `../outputs/data/shared_spectral_sweeps/` cache.
They retain their original seeds, eigenpair diagnostics, summaries, and figures.

[Setup, automatic exports, and output locations](../README.md#running-notebooks-and-exports).

The two grid launchers are now one notebook. Choose `GRID_R` in setup and rerun
all cells. The r=13/14 experiment retains its original data, image, and PDF
directory names; its separate notebook has been deleted.

[All KDE notebooks and code guide](../README.md)
