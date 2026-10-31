# KDE thesis work

Start in **[notebooks/](notebooks/)**. These five notebooks correspond to material
included in the KDE chapter or explicitly requested by its TODOs.
Lambda-jump studies are deferred to a possible outlook discussion.

| Notebook | Thesis purpose |
| --- | --- |
| [02_kde_numerics.ipynb](notebooks/02_kde_numerics.ipynb) | Fourier-error figure and exact/Fourier runtime table. |
| [03_fixed_sample_experiments.ipynb](notebooks/03_fixed_sample_experiments.ipynb) | Fixed-sample Perron profiles, grid-fineness and lambda paths, joint grid/scale limits, and alternative grid parameters. |
| [04_repeated_sampling_experiments.ipynb](notebooks/04_repeated_sampling_experiments.ipynb) | Fixed-sample-size distributions: the two thesis histogram profiles, calibrated-selector sampling, and custom-beta Monte Carlo. |
| [05_increasing_sample_size_experiments.ipynb](notebooks/05_increasing_sample_size_experiments.ipynb) | Spectral sample-size sweeps, continuous ISE/energy-minimizer distributions, and simultaneous sample/grid/scale limits. |
| [06_additional_experiments.ipynb](notebooks/06_additional_experiments.ipynb) | Mesh sensitivity, local Gaussian and row-sum mechanisms, convergence diagnostics, and local-minimum/endpoint controls. |

Scope: [the KDE chapter](../../Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex),
its included figures/table, and its explicit TODOs. Broader exploratory variants
and historical output folders have been removed. The earlier cleanup and compilation checks are recorded in
[thesis_manifest.json](outputs/thesis_manifest.json); its notebook inventory is historical.

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
| Fourier error and runtime comparison | [numerical_experiments.py](kde_tools/numerical_experiments.py) |
| Spectral bandwidth experiments | [spectral_experiments.py](kde_tools/spectral_experiments.py) |
| Joint grid/scale limits | [joint_experiments.py](kde_tools/joint_experiments.py) |
| Continuous minimizer distributions | [continuous_experiments.py](kde_tools/continuous_experiments.py) |

[notebook_setup.py](kde_tools/notebook_setup.py) handles paths and PDF saving;
the small local [_setup.py](notebooks/_setup.py) makes the package importable.
The existing mathematical modules remain the common calculation layer. Expensive
sampling runs retain their compatible checkpoints; repeated path searches,
wildcard imports, helper reloading, and local function definitions are removed
from the notebooks. Active notebook histograms use Matplotlib for direct PDF export.

The sequence 03–05 follows the three numerical-experiment subsections in the
thesis. Each notebook runs independently; no notebook executes another notebook.
Saved dataset names and compatible checkpoints retain their original paths.

For the two presented histogram profiles, run notebook 04's histogram section with `beta2_profile = "figure_3_5"` and then `"localized"`. Notebook 03's final section exposes `GRID_R = "1"` or `"13/14"`. Notebook 06 shows local minima and interval endpoints together in its final control section.
See [the shared search](kde_tools/energy_minimization.py) for its recorded controls.

## Folders

- `notebooks/`: the five entry points above.
- `kde_tools/`: shared calculations, checkpoint handling, plotting, and PDF export.
- `experiments/`: joint-limit helpers and optional diagnostic scripts.
- `inputs/`: [raw inputs and sampling identities](inputs/README.md).
- `outputs/`: data, included thesis assets, and PDFs for presented/planned results.
- `tests/`: numerical checks and small isolated notebook runs.

Executable scripts no longer live inside generated outputs. The numerical notebook
writes to the Fourier-error figure path; notebook 05 exports the
four selected panels directly, with the endpoint-frequency insets described by
the thesis caption. The joint-limit PDF and four energy panels that were missing
before cleanup were regenerated on 2026-10-29 using the retained settings. The
manifest records the compilation checks performed during that cleanup.

## Check

From this directory, use the configured interpreter:

```bash
MPLBACKEND=Agg /opt/anaconda3/bin/python3 -m unittest discover -s tests
```
