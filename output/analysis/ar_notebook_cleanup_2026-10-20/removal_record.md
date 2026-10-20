# AR notebook cleanup — 20 October 2026

Removed the four notebooks explicitly selected by the user:

- `ar_minimum_width_exploration_temp.ipynb`
- `ar3_coefficient_space_descent.ipynb`
- `ar3_implied_support_width.ipynb`
- `ar3_stationarity_restarts_50.ipynb`

There are now **19 AR notebooks**. The deletion inventory contains **172 files,
38,218,526 bytes**: four notebooks, 157 result/figure/report files, six dedicated
support scripts or tests, and five bytecode files. The full paths, sizes, and
original checksums are in [removal_manifest.json](removal_manifest.json).

## Removed outputs and code

Removed the matching `outputs/csv`, `outputs/images`, and `outputs/pdfs` families
for coefficient-space descent, implied support width, and the old 50-process
stationarity-filtering experiment. The temporary exploration notebook contains
embedded figures but no current file-export calls; those embedded figures were
removed with the notebook. Its directory-creation code alone was not treated as
evidence that unrelated AR(2) figures belonged to it.

Also removed the old stationarity experiment's dedicated runner, verifier, and
report generator in `output/analysis/ar3_stationarity_restarts_50/`; the exclusively
used coefficient-space utility and its dedicated tests; and the second-difference
failure plots, report data, and reproducer in `output/analysis/ar_second_difference_failures/`,
which read the removed implied-width result table.

## Shared inputs required by retained studies

Kept only the common **simulation inputs** used by the three retained restart
studies, in [ar3_coefficient_recovery_50.npz](../../../Code/Code_AR_processes/inputs/ar3_coefficient_recovery_50.npz)
(**794,715 bytes**). This contains 50 series, their true AR coefficients, starting
directions, sample checksums, and seed settings. It contains no fitted coefficients,
method comparisons, descent histories, plots, or reports from the deleted studies.
The input arrays were extracted with their original floating-point values before
the old directories were deleted.

The seed family belongs to the retained
[ar3_minimum_width_k4.ipynb](../../../Code/Code_AR_processes/ar3_minimum_width_k4.ipynb),
Monte Carlo coefficient-recovery section. Reconstructing the seeds in the current
runtime agreed within 4.440892098500626e-14 for observations and
4.440892098500626e-16 for coefficients; all 50 banks of starting directions matched
exactly. Keeping the shared input values avoids changing existing fits through
rounding. Details: [verification.json](verification.json), `seed_reconstruction`.

The retained restart runners and verifiers now use
[coefficient_recovery_inputs.py](../../../Code/Code_AR_processes/ar_tools/coefficient_recovery_inputs.py).
Only their input provenance fields were updated in the retained configurations.
The optimizer, estimation rules, stopping rules, and numerical result files were
not changed. The warm-start runner now verifies its first- and second-difference
selections directly against retained source curves rather than the deleted table;
see [experiments.py](../../../Code/Code_AR_processes/ar_tools/experiments.py).
Notebook links to the deleted studies were removed or redirected.

## Verification

- All four notebooks, all inventoried output files, and all inventoried support
  files are absent. No remaining executable Python or notebook source references
  the removed families.
- All 50 shared series match the sample hashes recorded by each of the three
  retained restart studies; their generating coefficients and the ordered start
  banks also match exactly.
- Cached execution of the three retained restart studies accepts their updated
  configurations and reproduces their existing result tables and summaries exactly.
  Their independent validators pass after the old directories are deleted,
  including physical widths, coefficient errors, stationarity checks, trace
  stopping rules, selections, and independent full-data LP checks.
- The warm-start verification produces exactly the same selections and summary
  without the deleted derivative table. Its existing cache already records older
  `functions.py` and `descent.py` hashes. For this check only, a temporary cache copy
  used current hashes; the live cache configuration and its mismatch guard were
  left unchanged. This pre-existing source-version mismatch still needs a deliberate
  rerun or cache reconciliation before normal cached execution of that notebook.
- SHA-256 checks confirm **388 retained numerical output files remain byte-for-byte
  unchanged**. All 19 remaining notebooks and the affected Python modules parse.
- The AR test suite ran 55 tests: **54 passed, one failed**. The existing
  `test_notebooks_have_no_local_function_definitions` failure concerns `save_figure`
  in cell 2 of the unchanged `ar3_failed_coefficient_tracking.ipynb`. The same
  definition was verified in the committed notebook. No unrelated notebook
  refactoring was made to address it.
- No protected thesis file under `Abschlussarbeit_2021/` was changed.

The machine-readable verification details are in [verification.json](verification.json).
Historical GitHub synchronization evidence is retained as a dated record; its
mentions of old paths are not executable dependencies or retained experiment outputs.
