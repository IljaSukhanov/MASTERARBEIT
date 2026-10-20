# AR tuning notebooks

These notebooks search initial step sizes and geometric decay settings for the
AR minimum-width optimizer. Run each notebook from top to bottom.

| Notebook | Tuning task |
|---|---|
| [ar_projected_subgradient_eta0_beta_tuning.ipynb](ar_projected_subgradient_eta0_beta_tuning.ipynb) | Compare schedule families, search initial-step/decay pairs, and compare the selected schedule on fresh samples. |
| [ar3_single_path_geometric_descent.ipynb](ar3_single_path_geometric_descent.ipynb) | Search a step-size/decay grid on one fixed AR(3) path and inspect the selected trajectory. |
| [ar_single_process_coefficient_estimation.ipynb](ar_single_process_coefficient_estimation.ipynb) | Fit one configurable AR process; section 5 adds a step-size/decay tuning grid. |

The notebooks locate the existing shared utilities under
`Code/Code_AR_processes/ar_tools/`. Output files continue to be saved under
`Code/Code_AR_processes/outputs/`, using their existing experiment names. The
single-path notebook uses the experiment name `ar3_geo`.
