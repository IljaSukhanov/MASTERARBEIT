# Files removed — 20 October 2026

**302 files removed; total 28.934385 MB (28,934,385 bytes).**

Project folder: `/Users/iljasukhanov/MASTERARBEIT`. Sizes use decimal MB: 1 MB = 1,000,000 bytes.

This list covers this cleanup only. It does not include the four-notebook cleanup from the earlier turn. Counts use unique file paths and original file sizes. Removed empty directories are listed separately and do not contribute to the file-size total.

[CSV copy](removed_files.csv) · [Manifest with original checksums and verification](removal_manifest.json)

## Totals by category

| Category | Files | MB |
|---|---:|---:|
| Temporary PDF previews and build products | 76 | 13.954610 |
| Caches and desktop metadata | 121 | 1.804654 |
| Historical numbered KDE outputs | 35 | 0.978942 |
| Retired Hellinger checkpoints | 13 | 0.027126 |
| Old AR export folders | 4 | 0.205958 |
| Duplicate AR(2) figure | 1 | 0.169968 |
| Empty review placeholder | 1 | 0.000029 |
| Tuning run 50a9174d5186210f | 39 | 11.537012 |
| Old synchronization investigation | 12 | 0.256086 |
| **Total** | **302** | **28.934385** |

## Retained material and verification

- All 24 protected thesis files are unchanged by SHA-256 comparison; all 19 literal thesis graphics dependencies remain present.
- All 1776 other retained files are byte-for-byte unchanged, except the intentionally updated `.gitignore`.
- All 30 remaining notebooks parse. No experiment was rerun and no active source code was modified.
- Kept the other tuning run (`685def7e60c3548b`), current shared simulation inputs, active checkpoints/traces, thesis recovery snapshots, and the unique temporary sources/verification material.
- Added `.DS_Store` and `.pytest_cache/` to `.gitignore`.
- macOS recreated desktop metadata during the cleanup at `Code/.DS_Store`, `Code/Code_AR_processes/.DS_Store`, `output/recovery/.DS_Store`. Those regenerated copies were removed again and are not counted twice.

## Brief historical synchronization record

The removed synchronization investigation recorded a 19 October 2026 repair commit `0f15121fb694529de54c4ee4fe98dda450782e7f`, which reconciled local and GitHub history and imported the approved Overleaf chapter text. Its last saved status still described an Overleaf-side pull/refresh awaiting user action. This is preserved historical status, not a new check of the remote project.

## Complete removed-file list

Paths below are relative to the project folder stated above.

| # | Removed file | Bytes | MB |
|---:|---|---:|---:|
| 1 | `.DS_Store` | 16388 | 0.016388 |
| 2 | `.matplotlib-cache/fontlist-v390.json` | 125096 | 0.125096 |
| 3 | `.pytest_cache/.gitignore` | 37 | 0.000037 |
| 4 | `.pytest_cache/CACHEDIR.TAG` | 191 | 0.000191 |
| 5 | `.pytest_cache/README.md` | 302 | 0.000302 |
| 6 | `.pytest_cache/v/cache/lastfailed` | 107 | 0.000107 |
| 7 | `.pytest_cache/v/cache/nodeids` | 5803 | 0.005803 |
| 8 | `.pytest_cache/v/cache/stepwise` | 2 | 0.000002 |
| 9 | `Code/.DS_Store` | 14340 | 0.014340 |
| 10 | `Code/.matplotlib-cache/fontlist-v390.json` | 125346 | 0.125346 |
| 11 | `Code/Code_AR_processes/.DS_Store` | 14340 | 0.014340 |
| 12 | `Code/Code_AR_processes/.matplotlib-cache/fontlist-v390.json` | 125346 | 0.125346 |
| 13 | `Code/Code_AR_processes/__pycache__/ar3_descent_chain_lp.cpython-312.pyc` | 2392 | 0.002392 |
| 14 | `Code/Code_AR_processes/__pycache__/ar3_gradient_figure_2_4_examples.cpython-312.pyc` | 2344 | 0.002344 |
| 15 | `Code/Code_AR_processes/__pycache__/ar3_minimum_width_mse_vs_T.cpython-312.pyc` | 1833 | 0.001833 |
| 16 | `Code/Code_AR_processes/__pycache__/ar3_order_selection_warm_starts.cpython-312.pyc` | 2411 | 0.002411 |
| 17 | `Code/Code_AR_processes/__pycache__/ar3_population_breakpoint_selectors.cpython-312.pyc` | 2183 | 0.002183 |
| 18 | `Code/Code_AR_processes/ar_tools/__pycache__/__init__.cpython-312.pyc` | 255 | 0.000255 |
| 19 | `Code/Code_AR_processes/ar_tools/__pycache__/breakpoint_selectors.cpython-312.pyc` | 11434 | 0.011434 |
| 20 | `Code/Code_AR_processes/ar_tools/__pycache__/classical_selectors.cpython-312.pyc` | 14852 | 0.014852 |
| 21 | `Code/Code_AR_processes/ar_tools/__pycache__/coefficient_recovery_inputs.cpython-312.pyc` | 5682 | 0.005682 |
| 22 | `Code/Code_AR_processes/ar_tools/__pycache__/descent.cpython-312.pyc` | 18748 | 0.018748 |
| 23 | `Code/Code_AR_processes/ar_tools/__pycache__/experiment_settings.cpython-312.pyc` | 2424 | 0.002424 |
| 24 | `Code/Code_AR_processes/ar_tools/__pycache__/experiments.cpython-312.pyc` | 73356 | 0.073356 |
| 25 | `Code/Code_AR_processes/ar_tools/__pycache__/functions.cpython-312.pyc` | 9889 | 0.009889 |
| 26 | `Code/Code_AR_processes/ar_tools/__pycache__/lp_widths.cpython-312.pyc` | 10470 | 0.010470 |
| 27 | `Code/Code_AR_processes/ar_tools/__pycache__/plotting_functions.cpython-312.pyc` | 30894 | 0.030894 |
| 28 | `Code/Code_AR_processes/ar_tools/__pycache__/population_widths.cpython-312.pyc` | 20863 | 0.020863 |
| 29 | `Code/Code_AR_processes/ar_tools/__pycache__/support_geometry.cpython-312.pyc` | 8285 | 0.008285 |
| 30 | `Code/Code_AR_processes/ar_tools/__pycache__/tuning.cpython-312.pyc` | 8830 | 0.008830 |
| 31 | `Code/Code_AR_processes/outputs/.DS_Store` | 12292 | 0.012292 |
| 32 | `Code/Code_AR_processes/outputs/csv/.DS_Store` | 8196 | 0.008196 |
| 33 | `Code/Code_AR_processes/outputs/csv/ar3_single_path_geometric_descent/gamma0_beta_grid_configuration.json` | 708 | 0.000708 |
| 34 | `Code/Code_AR_processes/outputs/csv/ar3_single_path_geometric_descent/gamma0_beta_minimum_width_matrix.csv` | 1071 | 0.001071 |
| 35 | `Code/Code_AR_processes/outputs/csv/ar_projected_subgradient_eta0_beta_tuning/50a9174d5186210f/candidate_ranking.csv` | 7314 | 0.007314 |
| 36 | `Code/Code_AR_processes/outputs/csv/ar_projected_subgradient_eta0_beta_tuning/50a9174d5186210f/configuration.json` | 4643 | 0.004643 |
| 37 | `Code/Code_AR_processes/outputs/csv/ar_projected_subgradient_eta0_beta_tuning/50a9174d5186210f/ranking_by_sample_size.csv` | 4002 | 0.004002 |
| 38 | `Code/Code_AR_processes/outputs/csv/ar_projected_subgradient_eta0_beta_tuning/50a9174d5186210f/selected_schedule.json` | 718 | 0.000718 |
| 39 | `Code/Code_AR_processes/outputs/csv/ar_projected_subgradient_eta0_beta_tuning/50a9174d5186210f/trajectory_histories.csv` | 10842661 | 10.842661 |
| 40 | `Code/Code_AR_processes/outputs/csv/ar_projected_subgradient_eta0_beta_tuning/50a9174d5186210f/tuning_r000_T2000.csv` | 9583 | 0.009583 |
| 41 | `Code/Code_AR_processes/outputs/csv/ar_projected_subgradient_eta0_beta_tuning/50a9174d5186210f/tuning_r000_T500.csv` | 9474 | 0.009474 |
| 42 | `Code/Code_AR_processes/outputs/csv/ar_projected_subgradient_eta0_beta_tuning/50a9174d5186210f/tuning_r001_T2000.csv` | 9584 | 0.009584 |
| 43 | `Code/Code_AR_processes/outputs/csv/ar_projected_subgradient_eta0_beta_tuning/50a9174d5186210f/tuning_r001_T500.csv` | 9450 | 0.009450 |
| 44 | `Code/Code_AR_processes/outputs/csv/ar_projected_subgradient_eta0_beta_tuning/50a9174d5186210f/tuning_r002_T2000.csv` | 9521 | 0.009521 |
| 45 | `Code/Code_AR_processes/outputs/csv/ar_projected_subgradient_eta0_beta_tuning/50a9174d5186210f/tuning_r002_T500.csv` | 9448 | 0.009448 |
| 46 | `Code/Code_AR_processes/outputs/csv/ar_projected_subgradient_eta0_beta_tuning/50a9174d5186210f/tuning_r003_T2000.csv` | 9556 | 0.009556 |
| 47 | `Code/Code_AR_processes/outputs/csv/ar_projected_subgradient_eta0_beta_tuning/50a9174d5186210f/tuning_r003_T500.csv` | 9493 | 0.009493 |
| 48 | `Code/Code_AR_processes/outputs/csv/ar_projected_subgradient_eta0_beta_tuning/50a9174d5186210f/tuning_r004_T2000.csv` | 9609 | 0.009609 |
| 49 | `Code/Code_AR_processes/outputs/csv/ar_projected_subgradient_eta0_beta_tuning/50a9174d5186210f/tuning_r004_T500.csv` | 9489 | 0.009489 |
| 50 | `Code/Code_AR_processes/outputs/csv/ar_projected_subgradient_eta0_beta_tuning/50a9174d5186210f/tuning_r005_T2000.csv` | 9552 | 0.009552 |
| 51 | `Code/Code_AR_processes/outputs/csv/ar_projected_subgradient_eta0_beta_tuning/50a9174d5186210f/tuning_r005_T500.csv` | 9463 | 0.009463 |
| 52 | `Code/Code_AR_processes/outputs/csv/ar_projected_subgradient_eta0_beta_tuning/50a9174d5186210f/tuning_results.csv` | 124947 | 0.124947 |
| 53 | `Code/Code_AR_processes/outputs/csv/ar_projected_subgradient_eta0_beta_tuning/50a9174d5186210f/validation_paired_comparison.csv` | 1169 | 0.001169 |
| 54 | `Code/Code_AR_processes/outputs/csv/ar_projected_subgradient_eta0_beta_tuning/50a9174d5186210f/validation_r000_T2000.csv` | 867 | 0.000867 |
| 55 | `Code/Code_AR_processes/outputs/csv/ar_projected_subgradient_eta0_beta_tuning/50a9174d5186210f/validation_r000_T500.csv` | 863 | 0.000863 |
| 56 | `Code/Code_AR_processes/outputs/csv/ar_projected_subgradient_eta0_beta_tuning/50a9174d5186210f/validation_r001_T2000.csv` | 872 | 0.000872 |
| 57 | `Code/Code_AR_processes/outputs/csv/ar_projected_subgradient_eta0_beta_tuning/50a9174d5186210f/validation_r001_T500.csv` | 862 | 0.000862 |
| 58 | `Code/Code_AR_processes/outputs/csv/ar_projected_subgradient_eta0_beta_tuning/50a9174d5186210f/validation_r002_T2000.csv` | 869 | 0.000869 |
| 59 | `Code/Code_AR_processes/outputs/csv/ar_projected_subgradient_eta0_beta_tuning/50a9174d5186210f/validation_r002_T500.csv` | 865 | 0.000865 |
| 60 | `Code/Code_AR_processes/outputs/csv/ar_projected_subgradient_eta0_beta_tuning/50a9174d5186210f/validation_r003_T2000.csv` | 869 | 0.000869 |
| 61 | `Code/Code_AR_processes/outputs/csv/ar_projected_subgradient_eta0_beta_tuning/50a9174d5186210f/validation_r003_T500.csv` | 860 | 0.000860 |
| 62 | `Code/Code_AR_processes/outputs/csv/ar_projected_subgradient_eta0_beta_tuning/50a9174d5186210f/validation_r004_T2000.csv` | 870 | 0.000870 |
| 63 | `Code/Code_AR_processes/outputs/csv/ar_projected_subgradient_eta0_beta_tuning/50a9174d5186210f/validation_r004_T500.csv` | 866 | 0.000866 |
| 64 | `Code/Code_AR_processes/outputs/csv/ar_projected_subgradient_eta0_beta_tuning/50a9174d5186210f/validation_r005_T2000.csv` | 864 | 0.000864 |
| 65 | `Code/Code_AR_processes/outputs/csv/ar_projected_subgradient_eta0_beta_tuning/50a9174d5186210f/validation_r005_T500.csv` | 864 | 0.000864 |
| 66 | `Code/Code_AR_processes/outputs/csv/ar_projected_subgradient_eta0_beta_tuning/50a9174d5186210f/validation_results.csv` | 7566 | 0.007566 |
| 67 | `Code/Code_AR_processes/outputs/csv/ar_projected_subgradient_eta0_beta_tuning/50a9174d5186210f/validation_summary.csv` | 558 | 0.000558 |
| 68 | `Code/Code_AR_processes/outputs/csv/order_selection/.DS_Store` | 8196 | 0.008196 |
| 69 | `Code/Code_AR_processes/outputs/images/.DS_Store` | 10244 | 0.010244 |
| 70 | `Code/Code_AR_processes/outputs/images/ar3_single_path_geometric_descent/gamma0_beta_minimum_width_matrix.png` | 177107 | 0.177107 |
| 71 | `Code/Code_AR_processes/outputs/images/ar_projected_subgradient_eta0_beta_tuning/50a9174d5186210f/01_parameter_grid.png` | 118129 | 0.118129 |
| 72 | `Code/Code_AR_processes/outputs/images/ar_projected_subgradient_eta0_beta_tuning/50a9174d5186210f/02_fresh_sample_comparison.png` | 87783 | 0.087783 |
| 73 | `Code/Code_AR_processes/outputs/images/ar_projected_subgradient_eta0_beta_tuning/50a9174d5186210f/03_optimization_trajectories.png` | 155165 | 0.155165 |
| 74 | `Code/Code_AR_processes/outputs/images/experiment_overviews/AR2_experiment.png` | 169968 | 0.169968 |
| 75 | `Code/Code_AR_processes/outputs/images/order_selection/.DS_Store` | 12292 | 0.012292 |
| 76 | `Code/Code_AR_processes/outputs/pdfs/ar3_single_path_geometric_descent/fixed_ar3_process.pdf` | 27072 | 0.027072 |
| 77 | `Code/Code_AR_processes/outputs/pdfs/ar_projected_subgradient_eta0_beta_tuning/50a9174d5186210f/01_parameter_grid.pdf` | 22131 | 0.022131 |
| 78 | `Code/Code_AR_processes/outputs/pdfs/ar_projected_subgradient_eta0_beta_tuning/50a9174d5186210f/02_fresh_sample_comparison.pdf` | 16556 | 0.016556 |
| 79 | `Code/Code_AR_processes/outputs/pdfs/ar_projected_subgradient_eta0_beta_tuning/50a9174d5186210f/03_optimization_trajectories.pdf` | 19057 | 0.019057 |
| 80 | `Code/Code_AR_processes/tests/__pycache__/test_beta_innovations.cpython-312.pyc` | 9246 | 0.009246 |
| 81 | `Code/Code_AR_processes/tests/__pycache__/test_descent_diagnostics.cpython-312-pytest-7.4.4.pyc` | 5736 | 0.005736 |
| 82 | `Code/Code_AR_processes/tests/__pycache__/test_descent_diagnostics.cpython-312.pyc` | 5630 | 0.005630 |
| 83 | `Code/Code_AR_processes/tests/__pycache__/test_descent_warm_starts.cpython-312-pytest-7.4.4.pyc` | 5543 | 0.005543 |
| 84 | `Code/Code_AR_processes/tests/__pycache__/test_descent_warm_starts.cpython-312.pyc` | 5462 | 0.005462 |
| 85 | `Code/Code_AR_processes/tests/__pycache__/test_experiment_tools.cpython-312.pyc` | 8122 | 0.008122 |
| 86 | `Code/Code_AR_processes/tests/__pycache__/test_geometric_width_hull.cpython-312.pyc` | 6779 | 0.006779 |
| 87 | `Code/Code_AR_processes/tests/__pycache__/test_lp_widths.cpython-312.pyc` | 4163 | 0.004163 |
| 88 | `Code/Code_AR_processes/tests/__pycache__/test_notebook_tools.cpython-312.pyc` | 9863 | 0.009863 |
| 89 | `Code/Code_AR_processes/tests/__pycache__/test_population_breakpoint_selectors.cpython-312.pyc` | 3125 | 0.003125 |
| 90 | `Code/Code_AR_processes/tests/__pycache__/test_population_widths.cpython-312.pyc` | 4175 | 0.004175 |
| 91 | `Code/Code_AR_processes/tests/__pycache__/test_schedule_tuning.cpython-312.pyc` | 10275 | 0.010275 |
| 92 | `Code/Code_AR_processes/tests/__pycache__/test_script_tools.cpython-312.pyc` | 6473 | 0.006473 |
| 93 | `Code/Code_AR_processes/tests/__pycache__/test_support_geometry.cpython-312.pyc` | 4481 | 0.004481 |
| 94 | `Code/Code_AR_processes/tests/__pycache__/test_warm_start_experiment.cpython-312.pyc` | 2365 | 0.002365 |
| 95 | `Code/Code_kde/.DS_Store` | 14340 | 0.014340 |
| 96 | `Code/Code_kde/.matplotlib-cache/fontlist-v390.json` | 125346 | 0.125346 |
| 97 | `Code/Code_kde/.pytest_cache/.gitignore` | 37 | 0.000037 |
| 98 | `Code/Code_kde/.pytest_cache/CACHEDIR.TAG` | 191 | 0.000191 |
| 99 | `Code/Code_kde/.pytest_cache/README.md` | 302 | 0.000302 |
| 100 | `Code/Code_kde/.pytest_cache/v/cache/lastfailed` | 217 | 0.000217 |
| 101 | `Code/Code_kde/.pytest_cache/v/cache/nodeids` | 3287 | 0.003287 |
| 102 | `Code/Code_kde/.pytest_cache/v/cache/stepwise` | 2 | 0.000002 |
| 103 | `Code/Code_kde/__pycache__/_annotate_notebook_roles.cpython-312.pyc` | 12944 | 0.012944 |
| 104 | `Code/Code_kde/__pycache__/run_kl_reverse_sampling.cpython-312.pyc` | 5226 | 0.005226 |
| 105 | `Code/Code_kde/__pycache__/run_toeplitz_alpha_sampling.cpython-312.pyc` | 5175 | 0.005175 |
| 106 | `Code/Code_kde/kde_tools/.DS_Store` | 8196 | 0.008196 |
| 107 | `Code/Code_kde/kde_tools/__pycache__/__init__.cpython-312.pyc` | 2582 | 0.002582 |
| 108 | `Code/Code_kde/kde_tools/__pycache__/bandwidths.cpython-312.pyc` | 8701 | 0.008701 |
| 109 | `Code/Code_kde/kde_tools/__pycache__/continuous_minimizers._ise_pair_sums-35.py312.1.nbc` | 25952 | 0.025952 |
| 110 | `Code/Code_kde/kde_tools/__pycache__/continuous_minimizers._ise_pair_sums-35.py312.nbi` | 597 | 0.000597 |
| 111 | `Code/Code_kde/kde_tools/__pycache__/continuous_minimizers.cpython-312.pyc` | 13930 | 0.013930 |
| 112 | `Code/Code_kde/kde_tools/__pycache__/density.cpython-312.pyc` | 5084 | 0.005084 |
| 113 | `Code/Code_kde/kde_tools/__pycache__/experiments.cpython-312.pyc` | 7653 | 0.007653 |
| 114 | `Code/Code_kde/kde_tools/__pycache__/fft.cpython-312.pyc` | 2006 | 0.002006 |
| 115 | `Code/Code_kde/kde_tools/__pycache__/gaussian_mixtures.cpython-312.pyc` | 13398 | 0.013398 |
| 116 | `Code/Code_kde/kde_tools/__pycache__/kullback_leibler.cpython-312.pyc` | 5222 | 0.005222 |
| 117 | `Code/Code_kde/kde_tools/__pycache__/pdf_export.cpython-312.pyc` | 6786 | 0.006786 |
| 118 | `Code/Code_kde/kde_tools/__pycache__/perron.cpython-312.pyc` | 14509 | 0.014509 |
| 119 | `Code/Code_kde/kde_tools/__pycache__/plotting_functions.cpython-312.pyc` | 18123 | 0.018123 |
| 120 | `Code/Code_kde/kde_tools/__pycache__/qhat._pair_sums-37.py312.1.nbc` | 27464 | 0.027464 |
| 121 | `Code/Code_kde/kde_tools/__pycache__/qhat._pair_sums-37.py312.nbi` | 576 | 0.000576 |
| 122 | `Code/Code_kde/kde_tools/__pycache__/qhat.cpython-312.pyc` | 12100 | 0.012100 |
| 123 | `Code/Code_kde/kde_tools/__pycache__/quadrature.cpython-312.pyc` | 12139 | 0.012139 |
| 124 | `Code/Code_kde/kde_tools/__pycache__/similarities.cpython-312.pyc` | 7368 | 0.007368 |
| 125 | `Code/Code_kde/kde_tools/fourier/__pycache__/__init__.cpython-312.pyc` | 680 | 0.000680 |
| 126 | `Code/Code_kde/kde_tools/fourier/__pycache__/reconstruction.cpython-312.pyc` | 2970 | 0.002970 |
| 127 | `Code/Code_kde/kde_tools/fourier/__pycache__/similarities.cpython-312.pyc` | 1924 | 0.001924 |
| 128 | `Code/Code_kde/outputs/.DS_Store` | 14340 | 0.014340 |
| 129 | `Code/Code_kde/outputs/analysis/fast_l2_lambda_comparison/__pycache__/analyse_saved_results.cpython-312.pyc` | 22489 | 0.022489 |
| 130 | `Code/Code_kde/outputs/analysis/spectral_lambda_jump_study/__pycache__/continuous_bandwidth_switch.cpython-312.pyc` | 24997 | 0.024997 |
| 131 | `Code/Code_kde/outputs/analysis/spectral_lambda_jump_study/__pycache__/figure_3_3_requested_plot_layouts.cpython-312.pyc` | 11871 | 0.011871 |
| 132 | `Code/Code_kde/outputs/analysis/spectral_lambda_jump_study/__pycache__/numerical_artefact_checks.cpython-312.pyc` | 11057 | 0.011057 |
| 133 | `Code/Code_kde/outputs/analysis/spectral_lambda_jump_study/__pycache__/repeated_peak_switch_analysis.cpython-312.pyc` | 35734 | 0.035734 |
| 134 | `Code/Code_kde/outputs/analysis/spectral_lambda_jump_study/__pycache__/robust_switch_diagnostics.cpython-312.pyc` | 25524 | 0.025524 |
| 135 | `Code/Code_kde/outputs/analysis/spectral_lambda_jump_study/__pycache__/sample_geometry_analysis.cpython-312.pyc` | 24813 | 0.024813 |
| 136 | `Code/Code_kde/outputs/analysis/spectral_lambda_switch_distribution/__pycache__/check_closest_pair_limit.cpython-312.pyc` | 9532 | 0.009532 |
| 137 | `Code/Code_kde/outputs/analysis/spectral_lambda_switch_distribution/__pycache__/switch_distribution_analysis.cpython-312.pyc` | 51389 | 0.051389 |
| 138 | `Code/Code_kde/outputs/csv/01_equal_minima/equal_minima_hellinger_grid_consistency_repeat.csv` | 9282 | 0.009282 |
| 139 | `Code/Code_kde/outputs/csv/01_equal_minima/geometric_grid_001_2_selector_summary.csv` | 756 | 0.000756 |
| 140 | `Code/Code_kde/outputs/csv/01_equal_minima/geometric_grid_all_selected_kdes_summary.csv` | 499 | 0.000499 |
| 141 | `Code/Code_kde/outputs/csv/01_equal_minima/kl_stationary_equal_minima_summary.csv` | 588 | 0.000588 |
| 142 | `Code/Code_kde/outputs/csv/01_equal_minima/selector_comparison_equal_minima.csv` | 594 | 0.000594 |
| 143 | `Code/Code_kde/outputs/csv/02_grid_and_consistency_studies/hellinger_geometric_grid_stability.csv` | 9282 | 0.009282 |
| 144 | `Code/Code_kde/outputs/csv/03_three_minima_scenarios/all_selectors_three_minima_scenarios.csv` | 8587 | 0.008587 |
| 145 | `Code/Code_kde/outputs/csv/03_three_minima_scenarios/hellinger_on_three_minima_scenarios.csv` | 1612 | 0.001612 |
| 146 | `Code/Code_kde/outputs/csv/03_three_minima_scenarios/hellinger_three_scenarios_grid_stability.csv` | 82722 | 0.082722 |
| 147 | `Code/Code_kde/outputs/csv/04_bandwidth_scaling/gaussian_kspacing_selected_kdes_summary.csv` | 699 | 0.000699 |
| 148 | `Code/Code_kde/outputs/csv/04_bandwidth_scaling/spectral_bandwidth_vs_n_gaussian_adaptive_scaled_grid.csv` | 53992 | 0.053992 |
| 149 | `Code/Code_kde/outputs/csv/04_bandwidth_scaling/spectral_bandwidth_vs_n_gaussian_adaptive_scaled_grid_slopes.csv` | 176 | 0.000176 |
| 150 | `Code/Code_kde/outputs/csv/04_bandwidth_scaling/spectral_bandwidth_vs_n_gaussian_adaptive_scaled_grid_summary.csv` | 2637 | 0.002637 |
| 151 | `Code/Code_kde/outputs/csv/04_bandwidth_scaling/spectral_bandwidth_vs_n_gaussian_kspacing_grid.csv` | 55239 | 0.055239 |
| 152 | `Code/Code_kde/outputs/csv/04_bandwidth_scaling/spectral_bandwidth_vs_n_gaussian_kspacing_grid_slopes.csv` | 268 | 0.000268 |
| 153 | `Code/Code_kde/outputs/csv/04_bandwidth_scaling/spectral_bandwidth_vs_n_gaussian_kspacing_grid_summary.csv` | 2788 | 0.002788 |
| 154 | `Code/Code_kde/outputs/csv/04_bandwidth_scaling/spectral_bandwidth_vs_n_gaussian_scaled_grid.csv` | 41082 | 0.041082 |
| 155 | `Code/Code_kde/outputs/csv/04_bandwidth_scaling/spectral_bandwidth_vs_n_gaussian_scaled_grid_slopes.csv` | 174 | 0.000174 |
| 156 | `Code/Code_kde/outputs/csv/04_bandwidth_scaling/spectral_bandwidth_vs_n_gaussian_scaled_grid_summary.csv` | 1946 | 0.001946 |
| 157 | `Code/Code_kde/outputs/csv/04_bandwidth_scaling/spectral_bandwidth_vs_n_gaussian_spacing_range_grid.csv` | 29137 | 0.029137 |
| 158 | `Code/Code_kde/outputs/csv/04_bandwidth_scaling/spectral_bandwidth_vs_n_gaussian_spacing_range_grid_slopes.csv` | 344 | 0.000344 |
| 159 | `Code/Code_kde/outputs/csv/04_bandwidth_scaling/spectral_bandwidth_vs_n_gaussian_spacing_range_grid_summary.csv` | 2537 | 0.002537 |
| 160 | `Code/Code_kde/outputs/csv/04_bandwidth_scaling/spectral_bandwidth_vs_n_skewed_unimodal.csv` | 36962 | 0.036962 |
| 161 | `Code/Code_kde/outputs/csv/04_bandwidth_scaling/spectral_bandwidth_vs_n_skewed_unimodal_slopes.csv` | 308 | 0.000308 |
| 162 | `Code/Code_kde/outputs/csv/04_bandwidth_scaling/spectral_bandwidth_vs_n_skewed_unimodal_summary.csv` | 3936 | 0.003936 |
| 163 | `Code/Code_kde/outputs/csv/06_skewed_unimodal/skewed_unimodal_all_selectors_summary.csv` | 2511 | 0.002511 |
| 164 | `Code/Code_kde/outputs/csv/07_two_gaussian_scan/two_gaussian_minima_representatives.csv` | 548 | 0.000548 |
| 165 | `Code/Code_kde/outputs/csv/07_two_gaussian_scan/two_gaussian_two_minima_random_scan.csv` | 60428 | 0.060428 |
| 166 | `Code/Code_kde/outputs/csv/07_two_gaussian_scan/two_gaussian_two_minima_scan.csv` | 32772 | 0.032772 |
| 167 | `Code/Code_kde/outputs/data/.DS_Store` | 10244 | 0.010244 |
| 168 | `Code/Code_kde/outputs/data/experiment_1_joint_N_lambda_limit/.DS_Store` | 8196 | 0.008196 |
| 169 | `Code/Code_kde/outputs/data/gaussian_hellinger_bandwidth_distributions_vs_n/preview/ea67e9744ba1/n100/checkpoint.npz` | 1955 | 0.001955 |
| 170 | `Code/Code_kde/outputs/data/gaussian_hellinger_bandwidth_distributions_vs_n/preview/ea67e9744ba1/n1000/checkpoint.npz` | 1929 | 0.001929 |
| 171 | `Code/Code_kde/outputs/data/gaussian_hellinger_bandwidth_distributions_vs_n/preview/ea67e9744ba1/n200/checkpoint.npz` | 2148 | 0.002148 |
| 172 | `Code/Code_kde/outputs/data/gaussian_hellinger_bandwidth_distributions_vs_n/preview/ea67e9744ba1/n300/checkpoint.npz` | 1955 | 0.001955 |
| 173 | `Code/Code_kde/outputs/data/gaussian_hellinger_bandwidth_distributions_vs_n/preview/ea67e9744ba1/n400/checkpoint.npz` | 1950 | 0.001950 |
| 174 | `Code/Code_kde/outputs/data/gaussian_hellinger_bandwidth_distributions_vs_n/preview/ea67e9744ba1/n500/checkpoint.npz` | 1940 | 0.001940 |
| 175 | `Code/Code_kde/outputs/data/gaussian_hellinger_bandwidth_distributions_vs_n/preview/ea67e9744ba1/n600/checkpoint.npz` | 1937 | 0.001937 |
| 176 | `Code/Code_kde/outputs/data/gaussian_hellinger_bandwidth_distributions_vs_n/preview/ea67e9744ba1/n700/checkpoint.npz` | 1931 | 0.001931 |
| 177 | `Code/Code_kde/outputs/data/gaussian_hellinger_bandwidth_distributions_vs_n/preview/ea67e9744ba1/n800/checkpoint.npz` | 1930 | 0.001930 |
| 178 | `Code/Code_kde/outputs/data/gaussian_hellinger_bandwidth_distributions_vs_n/preview/ea67e9744ba1/n900/checkpoint.npz` | 1927 | 0.001927 |
| 179 | `Code/Code_kde/outputs/data/gaussian_hellinger_bandwidth_distributions_vs_n/preview/ea67e9744ba1/settings.json` | 465 | 0.000465 |
| 180 | `Code/Code_kde/outputs/data/gaussian_hellinger_bandwidth_distributions_vs_n/thesis/3bcd2e3fb7ed/n100/checkpoint.npz` | 6592 | 0.006592 |
| 181 | `Code/Code_kde/outputs/data/gaussian_hellinger_bandwidth_distributions_vs_n/thesis/3bcd2e3fb7ed/settings.json` | 467 | 0.000467 |
| 182 | `Code/Code_kde/outputs/images/.DS_Store` | 10244 | 0.010244 |
| 183 | `Code/Code_kde/outputs/images/02_grid_and_consistency_studies/bandwidth_choice.png` | 52410 | 0.052410 |
| 184 | `Code/Code_kde/outputs/images/02_grid_and_consistency_studies/bandwidth_consistency.png` | 63496 | 0.063496 |
| 185 | `Code/Code_kde/outputs/images/02_grid_and_consistency_studies/bandwidth_selection2.png` | 45975 | 0.045975 |
| 186 | `Code/Code_kde/outputs/images/02_grid_and_consistency_studies/chat_rate_controlled_hellinger_scaling_fast.png` | 255953 | 0.255953 |
| 187 | `Code/Code_kde/outputs/images/02_grid_and_consistency_studies/concistency_result2.png` | 44032 | 0.044032 |
| 188 | `Code/Code_kde/outputs/images/05_oracle_ise_curves/oracle_ise_curve_example.png` | 74670 | 0.074670 |
| 189 | `Code/Code_kde/outputs/pdfs/.DS_Store` | 8196 | 0.008196 |
| 190 | `Code/Code_kde/tests/__pycache__/test_continuous_minimizers.cpython-312.pyc` | 7857 | 0.007857 |
| 191 | `Code/Code_kde/tests/__pycache__/test_kde_tools.cpython-312-pytest-7.4.4.pyc` | 25436 | 0.025436 |
| 192 | `Code/Code_kde/tests/__pycache__/test_kde_tools.cpython-312.pyc` | 26005 | 0.026005 |
| 193 | `Code/Code_kde/tests/__pycache__/test_optional_fourier.cpython-312-pytest-7.4.4.pyc` | 4724 | 0.004724 |
| 194 | `Code/Code_kde/tests/__pycache__/test_optional_fourier.cpython-312.pyc` | 4620 | 0.004620 |
| 195 | `Code/Code_kde/tests/__pycache__/test_perron.cpython-312.pyc` | 20805 | 0.020805 |
| 196 | `Code/Code_kde/tests/__pycache__/test_qhat.cpython-312.pyc` | 6762 | 0.006762 |
| 197 | `Code/Code_kde/tests/__pycache__/test_trapezoidal_js.cpython-312.pyc` | 12035 | 0.012035 |
| 198 | `Code/Code_kde/tests/__pycache__/validate_notebook_experiments.cpython-312.pyc` | 19971 | 0.019971 |
| 199 | `output/.DS_Store` | 18436 | 0.018436 |
| 200 | `output/analysis/.DS_Store` | 8196 | 0.008196 |
| 201 | `output/analysis/ar3_conditional_restarts_2000_200/__pycache__/run_experiment.cpython-312.pyc` | 17328 | 0.017328 |
| 202 | `output/analysis/ar3_conditional_restarts_2000_200/__pycache__/verify_and_report.cpython-312.pyc` | 21702 | 0.021702 |
| 203 | `output/analysis/ar3_early_abort_200_12starts/__pycache__/plot_counterexamples.cpython-312.pyc` | 16589 | 0.016589 |
| 204 | `output/analysis/ar3_early_abort_200_12starts/__pycache__/run_experiment.cpython-312.pyc` | 20957 | 0.020957 |
| 205 | `output/analysis/ar3_early_abort_200_12starts/__pycache__/verify_and_report.cpython-312.pyc` | 27044 | 0.027044 |
| 206 | `output/analysis/ar3_single_start_restart_vs_horvath/__pycache__/run_experiment.cpython-312.pyc` | 19612 | 0.019612 |
| 207 | `output/analysis/ar3_single_start_restart_vs_horvath/__pycache__/verify_and_report.cpython-312.pyc` | 25958 | 0.025958 |
| 208 | `output/analysis/ar3_stationarity_restarts/__pycache__/run_analysis.cpython-312.pyc` | 25533 | 0.025533 |
| 209 | `output/analysis/github_overleaf_sync_2026-10-19/diagnosis.md` | 5097 | 0.005097 |
| 210 | `output/analysis/github_overleaf_sync_2026-10-19/evidence.json` | 7722 | 0.007722 |
| 211 | `output/analysis/github_overleaf_sync_2026-10-19/reconciliation/authenticated_git_objects.json` | 19677 | 0.019677 |
| 212 | `output/analysis/github_overleaf_sync_2026-10-19/reconciliation/import_verified_git_objects.py` | 5533 | 0.005533 |
| 213 | `output/analysis/github_overleaf_sync_2026-10-19/reconciliation/main_3_application_to_time_series.tex` | 57829 | 0.057829 |
| 214 | `output/analysis/github_overleaf_sync_2026-10-19/reconciliation/main_bibliography.bib` | 42012 | 0.042012 |
| 215 | `output/analysis/github_overleaf_sync_2026-10-19/reconciliation/overleaf_3_application_to_time_series.tex` | 58439 | 0.058439 |
| 216 | `output/analysis/github_overleaf_sync_2026-10-19/reconciliation/overleaf_bibliography.bib` | 45805 | 0.045805 |
| 217 | `output/analysis/github_overleaf_sync_2026-10-19/reconciliation/proposal.json` | 788 | 0.000788 |
| 218 | `output/analysis/github_overleaf_sync_2026-10-19/reconciliation/proposal.md` | 1566 | 0.001566 |
| 219 | `output/analysis/github_overleaf_sync_2026-10-19/reconciliation/proposed_thesis_sync.diff` | 8546 | 0.008546 |
| 220 | `output/analysis/github_overleaf_sync_2026-10-19/reconciliation/repair_status.md` | 3072 | 0.003072 |
| 221 | `output/analysis/gradient_bound_byproduct/__pycache__/check.cpython-312.pyc` | 12620 | 0.012620 |
| 222 | `output/analysis/gradient_bound_byproduct/__pycache__/diagnose_failures.cpython-312.pyc` | 9170 | 0.009170 |
| 223 | `output/analysis/proposition_8_review.md` | 29 | 0.000029 |
| 224 | `output/figures/.DS_Store` | 8196 | 0.008196 |
| 225 | `output/recovery/.DS_Store` | 6148 | 0.006148 |
| 226 | `tmp/.DS_Store` | 8196 | 0.008196 |
| 227 | `tmp/pdfs/ar_harmonics_v6/ar_support_harmonics_summary_v6.aux` | 2604 | 0.002604 |
| 228 | `tmp/pdfs/ar_harmonics_v6/ar_support_harmonics_summary_v6.log` | 28295 | 0.028295 |
| 229 | `tmp/pdfs/ar_harmonics_v6/ar_support_harmonics_summary_v6.out` | 2926 | 0.002926 |
| 230 | `tmp/pdfs/ar_harmonics_v6/ar_support_harmonics_summary_v6.pdf` | 296147 | 0.296147 |
| 231 | `tmp/pdfs/ar_harmonics_v6/build.log` | 7765 | 0.007765 |
| 232 | `tmp/pdfs/ar_harmonics_v6/final-build.log` | 7683 | 0.007683 |
| 233 | `tmp/pdfs/ar_harmonics_v6/final-page-01.png` | 171617 | 0.171617 |
| 234 | `tmp/pdfs/ar_harmonics_v6/final-page-02.png` | 180577 | 0.180577 |
| 235 | `tmp/pdfs/ar_harmonics_v6/final-page-03.png` | 167154 | 0.167154 |
| 236 | `tmp/pdfs/ar_harmonics_v6/final-page-04.png` | 173614 | 0.173614 |
| 237 | `tmp/pdfs/ar_harmonics_v6/final-page-05.png` | 200205 | 0.200205 |
| 238 | `tmp/pdfs/ar_harmonics_v6/final-page-06.png` | 205881 | 0.205881 |
| 239 | `tmp/pdfs/ar_harmonics_v6/final-page-07.png` | 169073 | 0.169073 |
| 240 | `tmp/pdfs/ar_harmonics_v6/final-page-08.png` | 218791 | 0.218791 |
| 241 | `tmp/pdfs/ar_harmonics_v6/final-page-09.png` | 214994 | 0.214994 |
| 242 | `tmp/pdfs/ar_harmonics_v6/final-page-10.png` | 275551 | 0.275551 |
| 243 | `tmp/pdfs/ar_harmonics_v6/original-page-1.png` | 123522 | 0.123522 |
| 244 | `tmp/pdfs/ar_harmonics_v6/page-01.png` | 171617 | 0.171617 |
| 245 | `tmp/pdfs/ar_harmonics_v6/page-02.png` | 180577 | 0.180577 |
| 246 | `tmp/pdfs/ar_harmonics_v6/page-03.png` | 167154 | 0.167154 |
| 247 | `tmp/pdfs/ar_harmonics_v6/page-04.png` | 173614 | 0.173614 |
| 248 | `tmp/pdfs/ar_harmonics_v6/page-05.png` | 200205 | 0.200205 |
| 249 | `tmp/pdfs/ar_harmonics_v6/page-06.png` | 205881 | 0.205881 |
| 250 | `tmp/pdfs/ar_harmonics_v6/page-07.png` | 169073 | 0.169073 |
| 251 | `tmp/pdfs/ar_harmonics_v6/page-08.png` | 218791 | 0.218791 |
| 252 | `tmp/pdfs/ar_harmonics_v6/page-09.png` | 214994 | 0.214994 |
| 253 | `tmp/pdfs/ar_harmonics_v6/page-10.png` | 275551 | 0.275551 |
| 254 | `tmp/pdfs/ar_review/ar1.png` | 35860 | 0.035860 |
| 255 | `tmp/pdfs/ar_review/ar2.png` | 40645 | 0.040645 |
| 256 | `tmp/pdfs/ar_review/ar3.png` | 50792 | 0.050792 |
| 257 | `tmp/pdfs/figure_3_5_N1000_bins5_verified.png` | 48436 | 0.048436 |
| 258 | `tmp/pdfs/figure_3_5_N1000_verified.png` | 43922 | 0.043922 |
| 259 | `tmp/pdfs/figure_3_5_original.png` | 52651 | 0.052651 |
| 260 | `tmp/pdfs/figure_3_5_thesis_replacement_verified.png` | 48436 | 0.048436 |
| 261 | `tmp/pdfs/minimum_width_linf_beta_experiment.aux` | 128 | 0.000128 |
| 262 | `tmp/pdfs/minimum_width_linf_beta_experiment.log` | 42065 | 0.042065 |
| 263 | `tmp/pdfs/minimum_width_linf_beta_experiment.out` | 0 | 0.000000 |
| 264 | `tmp/pdfs/minimum_width_linf_beta_experiment.pdf` | 84146 | 0.084146 |
| 265 | `tmp/pdfs/minimum_width_note/finalcheck-1.png` | 174168 | 0.174168 |
| 266 | `tmp/pdfs/minimum_width_note/finalcheck-2.png` | 330791 | 0.330791 |
| 267 | `tmp/pdfs/minimum_width_note/finalcheck-3.png` | 199715 | 0.199715 |
| 268 | `tmp/pdfs/minimum_width_note/latex-page-1.png` | 264189 | 0.264189 |
| 269 | `tmp/pdfs/minimum_width_note/latex-page-2.png` | 329843 | 0.329843 |
| 270 | `tmp/pdfs/minimum_width_note/latex-page-3.png` | 108061 | 0.108061 |
| 271 | `tmp/pdfs/minimum_width_note/page-1.png` | 206098 | 0.206098 |
| 272 | `tmp/pdfs/minimum_width_note/page-2.png` | 256391 | 0.256391 |
| 273 | `tmp/pdfs/minimum_width_note/page-3.png` | 288216 | 0.288216 |
| 274 | `tmp/pdfs/minimum_width_note/page-4.png` | 49468 | 0.049468 |
| 275 | `tmp/pdfs/mplconfig/fontlist-v390.json` | 125096 | 0.125096 |
| 276 | `tmp/pdfs/novelty_review/contact-1.jpg` | 519311 | 0.519311 |
| 277 | `tmp/pdfs/novelty_review/contact-2.jpg` | 426790 | 0.426790 |
| 278 | `tmp/pdfs/novelty_review/contact-3.jpg` | 521685 | 0.521685 |
| 279 | `tmp/pdfs/novelty_review/contact-4.jpg` | 131953 | 0.131953 |
| 280 | `tmp/pdfs/novelty_review/novelty_proposition_7_conjecture_1.aux` | 455 | 0.000455 |
| 281 | `tmp/pdfs/novelty_review/novelty_proposition_7_conjecture_1.log` | 28122 | 0.028122 |
| 282 | `tmp/pdfs/novelty_review/novelty_proposition_7_conjecture_1.out` | 0 | 0.000000 |
| 283 | `tmp/pdfs/novelty_review/novelty_proposition_7_conjecture_1.pdf` | 477469 | 0.477469 |
| 284 | `tmp/pdfs/novelty_review/rendered/page-01.png` | 334594 | 0.334594 |
| 285 | `tmp/pdfs/novelty_review/rendered/page-02.png` | 276858 | 0.276858 |
| 286 | `tmp/pdfs/novelty_review/rendered/page-03.png` | 335839 | 0.335839 |
| 287 | `tmp/pdfs/novelty_review/rendered/page-04.png` | 383369 | 0.383369 |
| 288 | `tmp/pdfs/novelty_review/rendered/page-05.png` | 357350 | 0.357350 |
| 289 | `tmp/pdfs/novelty_review/rendered/page-06.png` | 250264 | 0.250264 |
| 290 | `tmp/pdfs/novelty_review/rendered/page-07.png` | 264059 | 0.264059 |
| 291 | `tmp/pdfs/novelty_review/rendered/page-08.png` | 280921 | 0.280921 |
| 292 | `tmp/pdfs/novelty_review/rendered/page-09.png` | 319933 | 0.319933 |
| 293 | `tmp/pdfs/novelty_review/rendered/page-10.png` | 384219 | 0.384219 |
| 294 | `tmp/pdfs/novelty_review/rendered/page-11.png` | 371371 | 0.371371 |
| 295 | `tmp/pdfs/novelty_review/rendered/page-12.png` | 263367 | 0.263367 |
| 296 | `tmp/pdfs/novelty_review/rendered/page-13.png` | 312556 | 0.312556 |
| 297 | `tmp/pdfs/spectral_distribution_N1000_M1000_bins5_verified.png` | 50291 | 0.050291 |
| 298 | `tmp/pdfs/spectral_distribution_N1000_lambda_N3_2_M1000_verified.png` | 46253 | 0.046253 |
| 299 | `tmp/pdfs/spectral_distribution_N1000_lambda_N3_2_M100_verified.png` | 47283 | 0.047283 |
| 300 | `tmp/pdfs/width_harmonics/mpl-cache/fontlist-v390.json` | 125096 | 0.125096 |
| 301 | `tmp/pdfs/width_harmonics/page-1.png` | 147264 | 0.147264 |
| 302 | `tmp/pdfs/width_harmonics/page-2.png` | 94965 | 0.094965 |

## Removed empty directories (87)

- `.matplotlib/`
- `.matplotlib-cache/`
- `.pytest_cache/`
- `.pytest_cache/v/`
- `.pytest_cache/v/cache/`
- `Code/.matplotlib-cache/`
- `Code/Code_AR_processes/.matplotlib-cache/`
- `Code/Code_AR_processes/__pycache__/`
- `Code/Code_AR_processes/ar_tools/__pycache__/`
- `Code/Code_AR_processes/outputs/csv/ar3_single_path_geometric_descent/`
- `Code/Code_AR_processes/outputs/csv/ar_projected_subgradient_eta0_beta_tuning/50a9174d5186210f/`
- `Code/Code_AR_processes/outputs/csv/order_selection/`
- `Code/Code_AR_processes/outputs/csv/order_selection/ar2/`
- `Code/Code_AR_processes/outputs/images/ar3_single_path_geometric_descent/`
- `Code/Code_AR_processes/outputs/images/ar_projected_subgradient_eta0_beta_tuning/50a9174d5186210f/`
- `Code/Code_AR_processes/outputs/images/experiment_overviews/`
- `Code/Code_AR_processes/outputs/pdfs/ar3_single_path_geometric_descent/`
- `Code/Code_AR_processes/outputs/pdfs/ar_projected_subgradient_eta0_beta_tuning/50a9174d5186210f/`
- `Code/Code_AR_processes/tests/__pycache__/`
- `Code/Code_kde/.matplotlib-cache/`
- `Code/Code_kde/.pytest_cache/`
- `Code/Code_kde/.pytest_cache/v/`
- `Code/Code_kde/.pytest_cache/v/cache/`
- `Code/Code_kde/__pycache__/`
- `Code/Code_kde/kde_tools/__pycache__/`
- `Code/Code_kde/kde_tools/fourier/__pycache__/`
- `Code/Code_kde/outputs/analysis/fast_l2_lambda_comparison/__pycache__/`
- `Code/Code_kde/outputs/analysis/spectral_lambda_jump_study/__pycache__/`
- `Code/Code_kde/outputs/analysis/spectral_lambda_switch_distribution/__pycache__/`
- `Code/Code_kde/outputs/csv/`
- `Code/Code_kde/outputs/csv/01_equal_minima/`
- `Code/Code_kde/outputs/csv/02_grid_and_consistency_studies/`
- `Code/Code_kde/outputs/csv/03_three_minima_scenarios/`
- `Code/Code_kde/outputs/csv/04_bandwidth_scaling/`
- `Code/Code_kde/outputs/csv/05_oracle_ise_curves/`
- `Code/Code_kde/outputs/csv/06_skewed_unimodal/`
- `Code/Code_kde/outputs/csv/07_two_gaussian_scan/`
- `Code/Code_kde/outputs/csv/08_kl_distance_matrices/`
- `Code/Code_kde/outputs/data/gaussian_hellinger_bandwidth_distributions_vs_n/`
- `Code/Code_kde/outputs/data/gaussian_hellinger_bandwidth_distributions_vs_n/preview/`
- `Code/Code_kde/outputs/data/gaussian_hellinger_bandwidth_distributions_vs_n/preview/ea67e9744ba1/`
- `Code/Code_kde/outputs/data/gaussian_hellinger_bandwidth_distributions_vs_n/preview/ea67e9744ba1/n100/`
- `Code/Code_kde/outputs/data/gaussian_hellinger_bandwidth_distributions_vs_n/preview/ea67e9744ba1/n1000/`
- `Code/Code_kde/outputs/data/gaussian_hellinger_bandwidth_distributions_vs_n/preview/ea67e9744ba1/n200/`
- `Code/Code_kde/outputs/data/gaussian_hellinger_bandwidth_distributions_vs_n/preview/ea67e9744ba1/n300/`
- `Code/Code_kde/outputs/data/gaussian_hellinger_bandwidth_distributions_vs_n/preview/ea67e9744ba1/n400/`
- `Code/Code_kde/outputs/data/gaussian_hellinger_bandwidth_distributions_vs_n/preview/ea67e9744ba1/n500/`
- `Code/Code_kde/outputs/data/gaussian_hellinger_bandwidth_distributions_vs_n/preview/ea67e9744ba1/n600/`
- `Code/Code_kde/outputs/data/gaussian_hellinger_bandwidth_distributions_vs_n/preview/ea67e9744ba1/n700/`
- `Code/Code_kde/outputs/data/gaussian_hellinger_bandwidth_distributions_vs_n/preview/ea67e9744ba1/n800/`
- `Code/Code_kde/outputs/data/gaussian_hellinger_bandwidth_distributions_vs_n/preview/ea67e9744ba1/n900/`
- `Code/Code_kde/outputs/data/gaussian_hellinger_bandwidth_distributions_vs_n/thesis/`
- `Code/Code_kde/outputs/data/gaussian_hellinger_bandwidth_distributions_vs_n/thesis/3bcd2e3fb7ed/`
- `Code/Code_kde/outputs/data/gaussian_hellinger_bandwidth_distributions_vs_n/thesis/3bcd2e3fb7ed/n100/`
- `Code/Code_kde/outputs/data/gaussian_symmetric_divergence_bandwidth_distributions/thesis/074ff447ff12/`
- `Code/Code_kde/outputs/images/02_grid_and_consistency_studies/`
- `Code/Code_kde/outputs/images/05_oracle_ise_curves/`
- `Code/Code_kde/outputs/images/gaussian_hellinger_bandwidth_distributions_vs_n/`
- `Code/Code_kde/outputs/images/gaussian_hellinger_bandwidth_distributions_vs_n/preview/`
- `Code/Code_kde/outputs/images/gaussian_hellinger_bandwidth_distributions_vs_n/preview/ea67e9744ba1/`
- `Code/Code_kde/outputs/images/gaussian_hellinger_bandwidth_distributions_vs_n/thesis/`
- `Code/Code_kde/outputs/images/gaussian_hellinger_bandwidth_distributions_vs_n/thesis/3bcd2e3fb7ed/`
- `Code/Code_kde/outputs/images/gaussian_symmetric_divergence_bandwidth_distributions/increasing_sample_size/`
- `Code/Code_kde/outputs/images/gaussian_symmetric_divergence_bandwidth_distributions/increasing_sample_size/thesis/`
- `Code/Code_kde/outputs/images/gaussian_symmetric_divergence_bandwidth_distributions/increasing_sample_size/thesis/3d194af437f1/`
- `Code/Code_kde/outputs/pdfs/100_kde_test/`
- `Code/Code_kde/outputs/pdfs/gaussian_hellinger_bandwidth_distributions_vs_n/`
- `Code/Code_kde/outputs/pdfs/kde_mixture_bandwidth_selector_comparison/`
- `Code/Code_kde/outputs/pdfs/kl_divergence_doob_selector_experiments/`
- `Code/Code_kde/outputs/pdfs/marron_wand_15_gmm_mise_density/`
- `Code/Code_kde/outputs/pdfs/three_component_gmm_perron_vs_qhat/`
- `Code/Code_kde/tests/__pycache__/`
- `Code/__pycache__/`
- `output/analysis/ar3_conditional_restarts_2000_200/__pycache__/`
- `output/analysis/ar3_early_abort_200_12starts/__pycache__/`
- `output/analysis/ar3_single_start_restart_vs_horvath/__pycache__/`
- `output/analysis/ar3_stationarity_restarts/__pycache__/`
- `output/analysis/github_overleaf_sync_2026-10-19/`
- `output/analysis/github_overleaf_sync_2026-10-19/reconciliation/`
- `output/analysis/gradient_bound_byproduct/__pycache__/`
- `output/figures/`
- `tmp/pdfs/ar_review/`
- `tmp/pdfs/minimum_width_note/`
- `tmp/pdfs/mplconfig/`
- `tmp/pdfs/novelty_review/rendered/`
- `tmp/pdfs/proposition_8_recheck/`
- `tmp/pdfs/width_harmonics/mpl-cache/`
