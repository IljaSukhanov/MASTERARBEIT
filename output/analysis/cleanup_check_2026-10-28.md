# Cleanup check — 28 October 2026

Cleanup completed after explicit user authorization on 28 October 2026: removed **45 files totaling 20.36 MiB** of file contents. The lists below record the original audit and the files that have now been removed. All files under `Abschlussarbeit_2021/` were excluded from cleanup recommendations under [AGENTS.md](/Users/iljasukhanov/MASTERARBEIT/AGENTS.md).

## Recommended candidates

| Candidate | Files | Logical size | Condition |
| --- | ---: | ---: | --- |
| Python bytecode caches outside the protected thesis | 35 | 0.47 MiB | All corresponding Python source files exist. |
| `.DS_Store` outside the protected thesis | 3 | 0.03 MiB | Folder metadata only; no project source. |
| Temporary LaTeX build records | 5 | 0.06 MiB | Keep the report source and final PDF. |
| Historical aggregate reference matrices | 1 | 19.07 MiB | Keep all 200 realization checkpoints. |
| Historical aggregate error results | 1 | 0.72 MiB | Keep all 200 realization checkpoints. |

Total candidate payload: **20.36 MiB** across 45 files. Actual reclaimed disk space can differ from file byte counts. Caches and temporary build records alone account for 0.56 MiB.

### Historical combined arrays

- [exact_grams.npy](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/outputs/data/g_rs_error_vs_k_sample_sizes/22b2ec46f6c2/exact_grams.npy) — combined reference matrices.
- [per_sample_results.npz](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/outputs/data/g_rs_error_vs_k_sample_sizes/22b2ec46f6c2/per_sample_results.npz) — combined errors and worst-pair indices.

Each of the 200 checkpoints `realizations/000.npz` through `realizations/199.npz` was compared with the corresponding combined-array slice. Every saved sample, reference matrix, error array, and worst-pair index matched exactly. The reference-array shape is `(200, 5, 50, 50)`; every checkpoint has configuration ID `22b2ec46f6c2`.

The [historical sample-size notebook](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/xx_notebooks_overview/g_rs_error_vs_k_sample_sizes.ipynb) reads the per-realization checkpoints into memory and writes the two combined arrays as exports. The retained [n=200 experiment](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/xx_notebooks_overview/g_rs_error_vs_k_repeated_samples.ipynb) reads its own, separate reference cache; that cache is not a removal candidate. The historical status is documented in the [KDE guide](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/README.md).

Keep the realization directory, `settings.json`, `samples.npy`, `bandwidths.npy`, summaries, validation records, and notebook. A later execution of the historical notebook will recreate the removed combined exports from its checkpoints.

### Temporary build records

- [ar_coefficient_estimation_results.aux](/Users/iljasukhanov/MASTERARBEIT/tmp/pdfs/ar_coefficient_estimation_results/ar_coefficient_estimation_results.aux)
- [ar_coefficient_estimation_results.fdb_latexmk](/Users/iljasukhanov/MASTERARBEIT/tmp/pdfs/ar_coefficient_estimation_results/ar_coefficient_estimation_results.fdb_latexmk)
- [ar_coefficient_estimation_results.fls](/Users/iljasukhanov/MASTERARBEIT/tmp/pdfs/ar_coefficient_estimation_results/ar_coefficient_estimation_results.fls)
- [ar_coefficient_estimation_results.log](/Users/iljasukhanov/MASTERARBEIT/tmp/pdfs/ar_coefficient_estimation_results/ar_coefficient_estimation_results.log)
- [ar_coefficient_estimation_results.out](/Users/iljasukhanov/MASTERARBEIT/tmp/pdfs/ar_coefficient_estimation_results/ar_coefficient_estimation_results.out)

Python automatically manages compiled module caches; see [Python tutorial §6.1.3](https://docs.python.org/3/tutorial/modules.html#compiled-python-files).

## Keep or archive deliberately

- **Current thesis assets and automatic PDF exports.** The new Figure 3.8 panel PDFs and their automatic-export counterparts are intentional. The export copies are required by [AGENTS.md](/Users/iljasukhanov/MASTERARBEIT/AGENTS.md).
- **Experiment data.** `Code/Code_kde/outputs/data/` occupies about 122 MiB on disk. It is ignored by Git under [.gitignore](/Users/iljasukhanov/MASTERARBEIT/.gitignore); ignored does not mean disposable. In particular, the local-Q-r results occupy about 61 MiB, including a 42.37 MiB checkpoint database. The [experiment runner](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/outputs/analysis/local_qr_minimizers_vs_n/run_experiment.py) uses that database to resume and the [active notebook](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/xx_notebooks_continuous_minimizers/local_qr_minimizers_vs_n.ipynb) links its CSV exports.
- **Files in `outputs/analysis/`.** Several are experiment engines imported by active notebooks; the folder name does not establish that they are obsolete.
- **Unique source in `tmp/`.** Keep [build_minimum_width_note.py](/Users/iljasukhanov/MASTERARBEIT/tmp/pdfs/build_minimum_width_note.py) and [minimum_width_linf_beta_experiment.tex](/Users/iljasukhanov/MASTERARBEIT/tmp/pdfs/minimum_width_linf_beta_experiment.tex); there is no second same-named source in this checkout. The script writes the minimum-width report. Other scripts and verification notes in `tmp/pdfs/` were also excluded from the deletion list.
- **Recovery material.** Keep `output/recovery/` and unresolved synchronization evidence. Their purpose is documented in [the recovery record](/Users/iljasukhanov/MASTERARBEIT/output/recovery/minimum_width_paragraph_2026-10-19/README.md).

## Duplicate exports

Byte-for-byte hashing of non-Git files larger than 1,000 bytes found 12 exact duplicate groups with 297,794 redundant bytes (0.28 MiB). This is an inventory result, not a blanket removal recommendation. Most copies are automatic PDFs or linked delivery versions. The much larger historical array duplication above is duplication of numerical content across different file formats and is additional to this byte-identical total.

## Disk use and Git

At the start of the check, `du` reported about 808 MiB for the checkout including `.git/`; `.git/` accounted for about 641 MiB. `git count-objects -vH` reported 143.27 MiB of loose objects, 497.12 MiB in six packs, and two temporary object files totaling 28 KiB.

Git repository maintenance is a separate option, not manual deletion of `.git/` contents. [Git’s official `gc` documentation](https://git-scm.com/docs/git-gc#_description) describes compression and other housekeeping, including pruning. No Git maintenance, pruning, history rewriting, or deletion of Git objects was performed, and no space-saving estimate for maintenance has been established.

## Full small-file candidate list

All paths below are outside the protected thesis. The five build records listed above are repeated here to make the complete list explicit.

- [Code/Code_kde/kde_tools/__pycache__/__init__.cpython-312.pyc](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/kde_tools/__pycache__/__init__.cpython-312.pyc) — 2,582 bytes
- [Code/Code_kde/kde_tools/__pycache__/bandwidth_histograms.cpython-312.pyc](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/kde_tools/__pycache__/bandwidth_histograms.cpython-312.pyc) — 7,153 bytes
- [Code/Code_kde/kde_tools/__pycache__/bandwidths.cpython-312.pyc](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/kde_tools/__pycache__/bandwidths.cpython-312.pyc) — 8,748 bytes
- [Code/Code_kde/kde_tools/__pycache__/continuous_minimizers.cpython-312.pyc](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/kde_tools/__pycache__/continuous_minimizers.cpython-312.pyc) — 14,517 bytes
- [Code/Code_kde/kde_tools/__pycache__/continuous_runs.cpython-312.pyc](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/kde_tools/__pycache__/continuous_runs.cpython-312.pyc) — 18,205 bytes
- [Code/Code_kde/kde_tools/__pycache__/density.cpython-312.pyc](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/kde_tools/__pycache__/density.cpython-312.pyc) — 5,084 bytes
- [Code/Code_kde/kde_tools/__pycache__/distribution_runs.cpython-312.pyc](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/kde_tools/__pycache__/distribution_runs.cpython-312.pyc) — 26,099 bytes
- [Code/Code_kde/kde_tools/__pycache__/experiments.cpython-312.pyc](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/kde_tools/__pycache__/experiments.cpython-312.pyc) — 7,713 bytes
- [Code/Code_kde/kde_tools/__pycache__/gaussian_mixtures.cpython-312.pyc](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/kde_tools/__pycache__/gaussian_mixtures.cpython-312.pyc) — 9,720 bytes
- [Code/Code_kde/kde_tools/__pycache__/kullback_leibler.cpython-312.pyc](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/kde_tools/__pycache__/kullback_leibler.cpython-312.pyc) — 5,222 bytes
- [Code/Code_kde/kde_tools/__pycache__/pdf_export.cpython-312.pyc](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/kde_tools/__pycache__/pdf_export.cpython-312.pyc) — 6,819 bytes
- [Code/Code_kde/kde_tools/__pycache__/perron.cpython-312.pyc](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/kde_tools/__pycache__/perron.cpython-312.pyc) — 13,642 bytes
- [Code/Code_kde/kde_tools/__pycache__/plotting_functions.cpython-312.pyc](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/kde_tools/__pycache__/plotting_functions.cpython-312.pyc) — 18,123 bytes
- [Code/Code_kde/kde_tools/__pycache__/qhat.cpython-312.pyc](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/kde_tools/__pycache__/qhat.cpython-312.pyc) — 12,100 bytes
- [Code/Code_kde/kde_tools/__pycache__/quadrature.cpython-312.pyc](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/kde_tools/__pycache__/quadrature.cpython-312.pyc) — 12,139 bytes
- [Code/Code_kde/kde_tools/__pycache__/similarities.cpython-312.pyc](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/kde_tools/__pycache__/similarities.cpython-312.pyc) — 7,368 bytes
- [Code/Code_kde/kde_tools/__pycache__/spectral_sweeps.cpython-312.pyc](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/kde_tools/__pycache__/spectral_sweeps.cpython-312.pyc) — 20,592 bytes
- [Code/Code_kde/kde_tools/fourier/__pycache__/__init__.cpython-312.pyc](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/kde_tools/fourier/__pycache__/__init__.cpython-312.pyc) — 997 bytes
- [Code/Code_kde/kde_tools/fourier/__pycache__/reconstruction.cpython-312.pyc](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/kde_tools/fourier/__pycache__/reconstruction.cpython-312.pyc) — 3,017 bytes
- [Code/Code_kde/kde_tools/fourier/__pycache__/similarities.cpython-312.pyc](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/kde_tools/fourier/__pycache__/similarities.cpython-312.pyc) — 7,466 bytes
- [Code/Code_kde/outputs/analysis/fast_l2_lambda_comparison/__pycache__/analyse_saved_results.cpython-312.pyc](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/outputs/analysis/fast_l2_lambda_comparison/__pycache__/analyse_saved_results.cpython-312.pyc) — 22,547 bytes
- [Code/Code_kde/outputs/analysis/local_qr_minimizers_vs_n/__pycache__/ise_reference.cpython-312.pyc](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/outputs/analysis/local_qr_minimizers_vs_n/__pycache__/ise_reference.cpython-312.pyc) — 8,842 bytes
- [Code/Code_kde/outputs/analysis/local_qr_minimizers_vs_n/__pycache__/local_search.cpython-312.pyc](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/outputs/analysis/local_qr_minimizers_vs_n/__pycache__/local_search.cpython-312.pyc) — 12,933 bytes
- [Code/Code_kde/outputs/analysis/spectral_lambda_jump_study/__pycache__/repeated_peak_switch_analysis.cpython-312.pyc](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/outputs/analysis/spectral_lambda_jump_study/__pycache__/repeated_peak_switch_analysis.cpython-312.pyc) — 35,756 bytes
- [Code/Code_kde/outputs/analysis/spectral_lambda_switch_distribution/__pycache__/switch_distribution_analysis.cpython-312.pyc](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/outputs/analysis/spectral_lambda_switch_distribution/__pycache__/switch_distribution_analysis.cpython-312.pyc) — 51,868 bytes
- [Code/Code_kde/tests/__pycache__/test_continuous_minimizers.cpython-312.pyc](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/tests/__pycache__/test_continuous_minimizers.cpython-312.pyc) — 7,857 bytes
- [Code/Code_kde/tests/__pycache__/test_continuous_runs.cpython-312.pyc](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/tests/__pycache__/test_continuous_runs.cpython-312.pyc) — 10,562 bytes
- [Code/Code_kde/tests/__pycache__/test_distribution_runs.cpython-312.pyc](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/tests/__pycache__/test_distribution_runs.cpython-312.pyc) — 14,400 bytes
- [Code/Code_kde/tests/__pycache__/test_kde_tools.cpython-312.pyc](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/tests/__pycache__/test_kde_tools.cpython-312.pyc) — 26,005 bytes
- [Code/Code_kde/tests/__pycache__/test_optional_fourier.cpython-312.pyc](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/tests/__pycache__/test_optional_fourier.cpython-312.pyc) — 10,307 bytes
- [Code/Code_kde/tests/__pycache__/test_perron.cpython-312.pyc](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/tests/__pycache__/test_perron.cpython-312.pyc) — 22,036 bytes
- [Code/Code_kde/tests/__pycache__/test_qhat.cpython-312.pyc](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/tests/__pycache__/test_qhat.cpython-312.pyc) — 6,762 bytes
- [Code/Code_kde/tests/__pycache__/test_spectral_sweeps.cpython-312.pyc](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/tests/__pycache__/test_spectral_sweeps.cpython-312.pyc) — 17,392 bytes
- [Code/Code_kde/tests/__pycache__/test_trapezoidal_js.cpython-312.pyc](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/tests/__pycache__/test_trapezoidal_js.cpython-312.pyc) — 11,962 bytes
- [Code/Code_kde/tests/__pycache__/validate_notebook_experiments.cpython-312.pyc](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/tests/__pycache__/validate_notebook_experiments.cpython-312.pyc) — 28,453 bytes
- [.DS_Store](/Users/iljasukhanov/MASTERARBEIT/.DS_Store) — 10,244 bytes
- [Code/.DS_Store](/Users/iljasukhanov/MASTERARBEIT/Code/.DS_Store) — 10,244 bytes
- [Code/Code_kde/.DS_Store](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/.DS_Store) — 10,244 bytes
- [tmp/pdfs/ar_coefficient_estimation_results/ar_coefficient_estimation_results.aux](/Users/iljasukhanov/MASTERARBEIT/tmp/pdfs/ar_coefficient_estimation_results/ar_coefficient_estimation_results.aux) — 2,258 bytes
- [tmp/pdfs/ar_coefficient_estimation_results/ar_coefficient_estimation_results.fdb_latexmk](/Users/iljasukhanov/MASTERARBEIT/tmp/pdfs/ar_coefficient_estimation_results/ar_coefficient_estimation_results.fdb_latexmk) — 16,346 bytes
- [tmp/pdfs/ar_coefficient_estimation_results/ar_coefficient_estimation_results.fls](/Users/iljasukhanov/MASTERARBEIT/tmp/pdfs/ar_coefficient_estimation_results/ar_coefficient_estimation_results.fls) — 18,508 bytes
- [tmp/pdfs/ar_coefficient_estimation_results/ar_coefficient_estimation_results.log](/Users/iljasukhanov/MASTERARBEIT/tmp/pdfs/ar_coefficient_estimation_results/ar_coefficient_estimation_results.log) — 27,810 bytes
- [tmp/pdfs/ar_coefficient_estimation_results/ar_coefficient_estimation_results.out](/Users/iljasukhanov/MASTERARBEIT/tmp/pdfs/ar_coefficient_estimation_results/ar_coefficient_estimation_results.out) — 0 bytes


## Execution and verification

- Completed: 2026-10-28T10:22:44+01:00.
- Removed 45 approved files, totaling 21,350,223 bytes (20.36 MiB); also removed 7 now-empty cache/build folders.
- Immediately before deletion, all 200 retained realization checkpoints were checked again against both combined arrays: samples, reference matrices, errors, and worst-pair indices matched exactly.
- After deletion, all 27 protected thesis files and all 1156 other remaining files matched their pre-cleanup hashes. This report was updated only after that verification.
- All 200 realization checkpoints remain. Current figure PDFs, source files, experiment databases, recovery records, and Git history were retained.
- No files outside the approved 45-file list were removed. References to removed files in the audit above are historical records, not links to existing files.

Removed empty directories:

- `Code/Code_kde/outputs/analysis/local_qr_minimizers_vs_n/__pycache__`
- `Code/Code_kde/outputs/analysis/spectral_lambda_jump_study/__pycache__`
- `Code/Code_kde/outputs/analysis/fast_l2_lambda_comparison/__pycache__`
- `Code/Code_kde/outputs/analysis/spectral_lambda_switch_distribution/__pycache__`
- `Code/Code_kde/kde_tools/fourier/__pycache__`
- `Code/Code_kde/tests/__pycache__`
- `tmp/pdfs/ar_coefficient_estimation_results`
