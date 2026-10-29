# KDE thesis work

Start in **[notebooks/](notebooks/)**. These five notebooks correspond to material
included in the KDE chapter or explicitly requested by its TODOs.

| Notebook | Thesis purpose |
| --- | --- |
| [kde_numerics.ipynb](notebooks/kde_numerics.ipynb) | Classical bandwidth illustration, Fourier-error figure, and exact/Fourier runtime table. |
| [spectral_scale_selection_kde.ipynb](notebooks/spectral_scale_selection_kde.ipynb) | Perron profiles, grid fineness, lambda path, repeated-sampling histograms, and sample-size comparisons already presented. |
| [experiment_1_joint_N_lambda_limit.ipynb](notebooks/experiment_1_joint_N_lambda_limit.ipynb) | Joint-limit figure and planned changes of r, repeated sampling, mesh checks, and high-exponent regimes. |
| [continuous_ise_qhat_distributions_vs_n.ipynb](notebooks/continuous_ise_qhat_distributions_vs_n.ipynb) | The four presented ISE/Q panels at n=100,1000,1500,2000, plus a fixed-sample control for the local-minimum discussion. |
| [spectral_lambda_jump_study.ipynb](notebooks/spectral_lambda_jump_study.ipynb) | Planned explanation of the sharp jump, competing peaks, repeated-sampling switch locations, and numerical-refinement checks. |

Scope: [the KDE chapter](../../Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex),
its included figures/table, and its explicit TODOs. Broader exploratory variants
and historical output folders have been removed. The retained dependencies and
checks are recorded in [thesis_manifest.json](outputs/thesis_manifest.json).

## Run

Use the Python 3.12 environment in [requirements.txt](requirements.txt) and run
cells in order. Setup finds the shared package and enables automatic PDF exports
under `outputs/pdfs/<notebook-name>/`. Restart the kernel after changing helpers.
Full sampling runs can be expensive; settings, checkpoints, and checks remain.

For the two presented histogram profiles, run the spectral notebook's histogram
section with `beta2_profile = "figure_3_5"` and then `"localized"`. The joint
notebook's final section exposes `GRID_R = "1"` or `"13/14"`. The energy notebook
shows local minima and interval endpoints together in its final control section.
See [the shared search](kde_tools/energy_minimization.py) for its recorded controls.

## Folders

- `notebooks/`: the five entry points above.
- `kde_tools/`: shared calculations, checkpoint handling, plotting, and PDF export.
- `experiments/`: seven helpers for the joint-limit and planned jump/peak sections.
- `inputs/`: [raw inputs and sampling identities](inputs/README.md).
- `outputs/`: data, included thesis assets, and PDFs for presented/planned results.
- `tests/`: numerical checks and small isolated notebook runs.

Executable scripts no longer live inside generated outputs. Existing included
figures retain their exact paths and contents. The numerical notebook writes
copies at the classical/error figure paths; the energy notebook exports the
four selected panels directly. Five referenced PDFs were already missing before
cleanup: the joint-limit PDF and those four energy panels. Their producing code
remains. This cleanup does not rerun full experiments or edit thesis sources.

## Check

From this directory, use the configured interpreter:

```bash
MPLBACKEND=Agg /opt/anaconda3/bin/python3 -m unittest discover -s tests
```
