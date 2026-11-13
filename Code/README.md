# KDE thesis code

This directory contains the code, inputs and saved results associated with the
numerical experiments in the thesis. Run commands from this directory using the
Python environment specified in `requirements.txt`.

## Thesis workflows

| Result | Entry point |
| --- | --- |
| Fourier approximation error and runtime comparison | `notebooks/02_kde_numerics.ipynb` |
| Fixed-sample Perron profiles, bandwidth paths and joint-limit figure | `notebooks/03_fixed_sample_experiments.ipynb` |
| Repeated-sample distributions with n=200, N=400 and 1,000 repetitions | `experiments/repeated_alpha_evolution/run_experiment.py` and `plot_results.py` |
| Increasing-sample-size figures in the outlook | `experiments/outlook_four_lambdas.py` |
| Refinement plot and Brent versus spectral timing table | `experiments/qhat0_interval_example/` |
| Additional checks for the fixed-sample experiments | `notebooks/06_additional_experiments.ipynb` |

Run notebook cells in order. The joint-limit helper imported by notebook 03 is
`experiments/joint_limits/run_experiment.py`. The repeated-sample plotting script
reads the saved N=400 checkpoint from `outputs/data/repeated_alpha_evolution/n400/`.

To regenerate the repeated-sample plots from that checkpoint:

```bash
python experiments/repeated_alpha_evolution/plot_results.py
```

`outputs/images/thesis/` and `outputs/pdfs/` contain PDFs directly included by
the thesis. Saved data and checkpoints in `outputs/` also preserve reported
measurements and inputs read by later scripts. In particular, do not discard the
Brent checkpoints and prior results without first updating the corresponding
scripts. Some full numerical runs are computationally expensive, and execution
times depend on the machine and runtime environment.

The saved grid comparison contains 57 of the 100 reference checkpoints. The
Brent refinement script recomputes missing references using the same exact
search and checks their resolution. The peak-switch analysis's default
repeated-sample checkpoint is missing, but its fixed-sample thesis value can be
reproduced independently:

```bash
python experiments/spectral_lambda_jump_study/repeated_peak_switch_analysis.py --reference-only
```

The full peak-switch study still requires its original checkpoint or a
compatible replacement, supplied with `--source`.

## Tests

```bash
MPLBACKEND=Agg python -m unittest discover -s tests
```

From the project root, use `MPLBACKEND=Agg python -m unittest discover -s Code/tests`.

The tests run reduced examples and use isolated temporary output directories.
