# KDE thesis work

Start in **[notebooks/](notebooks/)**. These five notebooks correspond to material
included in the KDE chapter or explicitly requested by its TODOs.
Lambda-jump studies are deferred to a possible outlook discussion.

| Notebook | Thesis purpose |
| --- | --- |
| [01_classical_bandwidth_kde_illustration.ipynb](notebooks/01_classical_bandwidth_kde_illustration.ipynb) | Classical bandwidth illustration. |
| [02_kde_numerics.ipynb](notebooks/02_kde_numerics.ipynb) | Fourier-error figure and exact/Fourier runtime table. |
| [spectral_scale_selection_kde.ipynb](notebooks/spectral_scale_selection_kde.ipynb) | Perron profiles, grid fineness, lambda path, repeated-sampling histograms, and sample-size comparisons already presented. |
| [experiment_1_joint_N_lambda_limit.ipynb](notebooks/experiment_1_joint_N_lambda_limit.ipynb) | Joint-limit figure and planned changes of r, repeated sampling, mesh checks, and high-exponent regimes. |
| [continuous_ise_qhat_distributions_vs_n.ipynb](notebooks/continuous_ise_qhat_distributions_vs_n.ipynb) | The four presented ISE/Q panels at n=100,1000,1500,2000, plus a fixed-sample control for the local-minimum discussion. |

Scope: [the KDE chapter](../../Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex),
its included figures/table, and its explicit TODOs. Broader exploratory variants
and historical output folders have been removed. The retained dependencies and
checks are recorded in [thesis_manifest.json](outputs/thesis_manifest.json).

## Run

Use the Python 3.12 environment in [requirements.txt](requirements.txt) and run
each notebook from its `notebooks/` folder, with cells in order. Its setup imports
only the functions it needs and calls `setup_notebook(...)` once. Every displayed
figure is saved automatically under `outputs/pdfs/<notebook-name>/`, including
figures displayed by the notebook's inline backend. The shared setup uses a
persistent font cache. Restart the kernel after changing Python helpers.
Full sampling runs can be expensive; settings, checkpoints, and checks remain.

## Simple notebook structure

Notebooks contain imports, editable experiment settings, calculations, and plots.
Reusable function definitions live in Python files:

| Experiment | Imported functions |
| --- | --- |
| Classical bandwidth comparison | [classical_experiments.py](kde_tools/classical_experiments.py) |
| Fourier error and runtime comparison | [numerical_experiments.py](kde_tools/numerical_experiments.py) |
| Spectral bandwidth experiments | [spectral_experiments.py](kde_tools/spectral_experiments.py) |
| Joint grid/scale limits | [joint_experiments.py](kde_tools/joint_experiments.py) |
| Continuous minimizer distributions | [continuous_experiments.py](kde_tools/continuous_experiments.py) |

[notebook_setup.py](kde_tools/notebook_setup.py) handles paths and PDF saving;
the small local [_setup.py](notebooks/_setup.py) makes the package importable.
The existing mathematical modules remain the common calculation layer. Expensive
sampling runs retain their compatible checkpoints; repeated path searches,
wildcard imports, helper reloading, and local function definitions are removed
from the notebooks.

For the two presented histogram profiles, run the spectral notebook's histogram
section with `beta2_profile = "figure_3_5"` and then `"localized"`. The joint
notebook's final section exposes `GRID_R = "1"` or `"13/14"`. The energy notebook
shows local minima and interval endpoints together in its final control section.
See [the shared search](kde_tools/energy_minimization.py) for its recorded controls.

## Folders

- `notebooks/`: the five entry points above.
- `kde_tools/`: shared calculations, checkpoint handling, plotting, and PDF export.
- `experiments/`: joint-limit helpers and optional diagnostic scripts.
- `inputs/`: [raw inputs and sampling identities](inputs/README.md).
- `outputs/`: data, included thesis assets, and PDFs for presented/planned results.
- `tests/`: numerical checks and small isolated notebook runs.

Executable scripts no longer live inside generated outputs. Existing included
figures retain their exact paths and contents. The classical and numerical notebooks
write to the classical/error figure paths; the energy notebook exports the
four selected panels directly, with the endpoint-frequency insets described by
the thesis caption. The joint-limit PDF and four energy panels that were missing
before cleanup were regenerated on 2026-10-29 using the retained settings. All
thesis image references now resolve; see the manifest for the compilation checks.

## Check

From this directory, use the configured interpreter:

```bash
MPLBACKEND=Agg /opt/anaconda3/bin/python3 -m unittest discover -s tests
```
