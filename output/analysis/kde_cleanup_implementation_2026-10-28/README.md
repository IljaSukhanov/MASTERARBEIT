# KDE cleanup completed — 28 October 2026

The active notebook collection was reduced from **20 to 16**. Four notebooks
and one duplicate runner were **deleted**, with no archive copies added to
the project. Existing thesis files, input data, numerical results, and exported
figures were preserved. This implements the four notebook reductions from the
[pre-cleanup inventory](../kde_cleanup_check_2026-10-28/inventory.json).

## Deleted files and retained views

| Deleted file | Retained behavior |
| --- | --- |
| `xx_notebooks_overview/g_rs_error_vs_k_sample_sizes.ipynb` | Historical experiment removed as requested; its existing numerical data and exported figures remain. |
| `xx_notebooks_overview/g_rs_error_vs_k_single_sample.ipynb` | The [repeated-sample notebook](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/xx_notebooks_overview/g_rs_error_vs_k_repeated_samples.ipynb) now includes the realization-zero plot, signed-error CSV, settings, exact Gram matrix, and report. |
| `xx_notebooks_asymptotics/r13over14_joint_N_lambda_limit.ipynb` | The [combined grid notebook](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/xx_notebooks_asymptotics/linear_grid_joint_N_lambda_limit.ipynb) supports both `GRID_R = "1"` and `"13/14"`, retaining each experiment's data, image, and PDF namespace. |
| `xx_notebooks_continuous_minimizers/q1_local_minimizers_selected_n.ipynb` | The [local-Q-r notebook](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/xx_notebooks_continuous_minimizers/local_qr_minimizers_vs_n.ipynb) includes the Q₁-only selected-size histogram and CSV/diagnostic exports. |
| `outputs/analysis/local_q1_minimizers_selected_n/run_experiment.py` | Its useful view is consolidated in the [shared local-Q-r runner](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/outputs/analysis/local_qr_minimizers_vs_n/run_experiment.py:273), using the shared selections instead of a second Monte Carlo loop. |

Paths in the first column are relative to `Code/Code_kde/`. The first deletion
intentionally removes the five-size experiment's executable notebook source;
the other deletions remove entry points whose useful views are retained.

## Behavior of the merged views

The single-sample view reuses realization zero's verified Gram matrix, errors,
and worst-pair indices. It evaluates the Fourier approximation at those pairs
to recover signed errors. New view exports go under `single_sample/` within
the repeated experiment's run directory; its plots use the repeated notebook's
automatic PDF exporter. Its [regression check](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/tests/test_notebook_cleanup.py:52)
also verifies that resuming the repeated calculation does not rewrite the
numerical cache files.

The grid notebook explicitly configures the chosen r before running and enables
the PDF exporter for the returned experiment name. Switching back to r=1 resets
the configuration and its output namespace. Stale hardcoded result tables and
links to absent pre-unification files were removed from the combined notebook;
its displayed table and plot now use the current selected run. The
[grid runner](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/outputs/analysis/linear_grid_joint_n_lambda_limit/run_experiment.py:55)
retains its numerical settings and calculations.

The Q₁-only view preserves missing outcomes, all detected candidates, full search
reports, interval-minimum fields, and the extra fresh n=200 exact-scan checks.
It exports under `q1_selected_n/` in the shared local-Q-r run directory. Old
selection-only fallback checkpoints gain full reports by recomputing only the
missing reports and checking that the stored selections agree; existing
checkpoint records are not rewritten by this view. Its PDF is automatically
saved under the local-Q-r notebook's PDF directory. The
[tests](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/tests/test_notebook_cleanup.py:150)
cover report recovery, absent local minima, incomplete-selection rejection,
histogram counts, reference-check reuse, and PDF export.

## Supporting cleanup

The [routing tests](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/tests/test_perron.py:237)
now locate the first code cell and filter notebook command lines before parsing
Python. The unused `OLD_WORKFLOW` parameter plumbing was removed from the
[local-search adapter](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/outputs/analysis/local_qr_minimizers_vs_n/local_search.py:33).
The numerical `kde_tools/` package files remain unchanged.

The [main guide](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/README.md:160)
and affected folder guides now list the 16 active notebooks and explain the
merged views. [requirements.txt](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/requirements.txt)
records the direct dependency versions from the tested Python 3.12 environment.
No packages were installed or upgraded.

## Validation

- **122 tests passed**, with no failures, errors, or skips, using
  `/opt/anaconda3/bin/python3`. See [test_results.json](test_results.json) and
  [test_run.log](test_run.log).
- The merged single-sample CSV was compared against all **141 rows and seven
  columns** of the existing full n=200, N=50 export. Worst-pair indices were
  identical; the maximum absolute numerical difference was
  **2.220446049250313e-16**. The existing saved files were read without changing
  them. See [equivalence_checks.json](equivalence_checks.json).
- Q₁ local minima, objective values, interval minima, and boundary classifications
  were compared with the original runner from the recorded Git revision on
  **eight cases**: six canonical samples at n=100/200, a case without a local
  minimum, and a multiscale case with two minima. The compared values had
  **zero difference**. See [equivalence_checks.json](equivalence_checks.json).
- The preservation manifest covers **512 existing files**, including every file
  under the protected thesis directory and the KDE inputs, saved numerical data,
  images, automatic PDFs, and delivery PDFs. The final comparison found no
  missing or changed files. See [preservation_baseline.json](preservation_baseline.json)
  and [verification.json](verification.json).
- Static checks cover every remaining Python file and notebook code cell, the
  shared exporter in all 16 setup cells, documentation coverage, and references
  to the deleted code. See [verification.json](verification.json).

Full Monte Carlo experiments and thesis figure regeneration were not run.
The new regression checks use isolated small runs; the full single-sample
comparison reads the existing numerical cache. Unrelated historical diagnostics
and the larger joint-limit notebook were retained.
