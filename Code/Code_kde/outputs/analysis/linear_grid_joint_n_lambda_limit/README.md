# Linear-grid version of current Figure 3.6

This run repeats the initial experiment in
[`experiment_1_joint_N_lambda_limit.ipynb`](../../../xx_notebooks_asymptotics/experiment_1_joint_N_lambda_limit.ipynb),
changing only the bandwidth grid to the thesis's linear case, r=1.
The shared grid helper calls this `beta=0`; this parameter controls the
grid coordinate, while the divergence remains half the squared L2 distance.

- One fixed standard-normal sample of size 200, seed 20260823.
- Bandwidths equally spaced between 0.01 and 2, with both endpoints included.
- N = 100, 200, ..., 3000.
- Lambda = 1, N^(1/2), N, N^(3/2), N^2, N^3.
- Same shared Fourier approximation and stabilized Perron solver as the original notebook.

The [combined grid notebook](../../../xx_notebooks_asymptotics/linear_grid_joint_N_lambda_limit.ipynb)
enables automatic PDF export in its setup cell. The standalone workflow is
[`run_experiment.py`](run_experiment.py). It saves a checkpoint after every
completed grid size and reuses those checkpoints on reruns. To run from the
project root:

```sh
MPLBACKEND=Agg python Code/Code_kde/outputs/analysis/linear_grid_joint_n_lambda_limit/run_experiment.py
```

The same runner also accepts `--r 13/14`, which saves that grid's results in
separate `r13over14_joint_n_lambda_limit` directories. In the combined notebook,
set `GRID_R = "13/14"` in setup; set it to `"1"` for the linear case. Both
choices keep their original result and automatic PDF directories.

Data and diagnostics are saved under
`outputs/data/linear_grid_joint_n_lambda_limit/unified_<settings-id>/`.
The reference calculation scans the complete interval, refines every detected
local minimum, and compares both endpoints. It uses the thesis's coefficient
Q_1(h) = (1/2) ||partial_h f_hat_h||_2^2; see
[`prop:main_thm`](../../../../../Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex).
The exact Gaussian divergence validates a 25-point grid, and doubled frequency
resolution must preserve all six selected bandwidths at N=3000. Residuals
and spectral gaps are stored for every eigenpair.

The PNG is saved under `outputs/images/linear_grid_joint_n_lambda_limit/`.
Automatic PDFs are saved under `outputs/pdfs/linear_grid_joint_n_lambda_limit/`.
No thesis files are edited by this workflow.
