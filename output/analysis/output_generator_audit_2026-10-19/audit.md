# CSV and figure generator audit — 19 October 2026

Base commit: `dc7cdfda66eac31dcf7f82d4a064b28fd2693ba0`; current working-tree deletions are respected. This audit inspected 89 current scripts/notebooks and 1221 remaining tracked CSV/image/PDF files. Notebook code cells were scanned; saved outputs and markdown were not treated as generators. Tests and the earlier GitHub-repair helper were excluded as output producers. No experiments were rerun.

## Strongest orphan candidates

These output families have lost their dedicated notebooks. Git history confirms the notebook deletion, and no current replacement export targeting the same family was located. This is evidence of a missing maintained reproducer, not a proof that equivalent numerical results could never be reconstructed from shared mathematical routines. The notebooks remain recoverable from Git history.

| Deleted generator | CSVs | PNG/images | PDF figures | Total MB |
|---|---:|---:|---:|---:|
| [100_kde_test.ipynb](https://github.com/IljaSukhanov/MASTERARBEIT/blob/8dee98b335ac505eab61f63fd429b4ab08adc0b9/Code/Code_kde/100_kde_test.ipynb) | 0 | 0 | 28 | 0.680 |
| [gaussian_hellinger_bandwidth_distributions_vs_n.ipynb](https://github.com/IljaSukhanov/MASTERARBEIT/blob/051e2bed669302314a391ea2141223f7f61cd5ba/Code/Code_kde/gaussian_hellinger_bandwidth_distributions_vs_n.ipynb) | 3 | 1 | 1 | 0.266 |
| [kde_mixture_bandwidth_selector_comparison.ipynb](https://github.com/IljaSukhanov/MASTERARBEIT/blob/4d7ba4a6adf2d407a05cfe0570f0a4117f742b19/Code/Code_kde/kde_mixture_bandwidth_selector_comparison.ipynb) | 0 | 0 | 2 | 0.070 |
| [kl_divergence_doob_selector_experiments.ipynb](https://github.com/IljaSukhanov/MASTERARBEIT/blob/4d7ba4a6adf2d407a05cfe0570f0a4117f742b19/Code/Code_kde/kl_divergence_doob_selector_experiments.ipynb) | 4 | 0 | 9 | 0.956 |
| [marron_wand_15_gmm_mise_density.ipynb](https://github.com/IljaSukhanov/MASTERARBEIT/blob/4d7ba4a6adf2d407a05cfe0570f0a4117f742b19/Code/Code_kde/marron_wand_15_gmm_mise_density.ipynb) | 0 | 0 | 15 | 1.338 |
| [three_component_gmm_perron_vs_qhat.ipynb](https://github.com/IljaSukhanov/MASTERARBEIT/blob/4d7ba4a6adf2d407a05cfe0570f0a4117f742b19/Code/Code_kde/three_component_gmm_perron_vs_qhat.ipynb) | 0 | 0 | 6 | 0.463 |

Total: **69 files, 3.774 MB** (7 CSVs, 1 raster/vector images and 61 PDF figures). The exact paths are in [orphan_candidates.csv](orphan_candidates.csv). The KL group includes its four CSV matrices under `outputs/csv/08_kl_distance_matrices/`.

## Additional cases needing review

The [additional list](additional_review_candidates.csv) contains 38 files, 1.184 MB:

- Old numbered KDE output folders `01_equal_minima` through `07_two_gaussian_scan` have no current output-family references in executable sources. Their exact historical producers have not all been attributed; dynamically assembled filenames prevent an absolute absence claim.
- Three AR outputs remain under the former `ar3_single_path_geometric_descent` folder name. The current [notebook](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/ar3_single_path_geometric_descent.ipynb) sets `EXPERIMENT_NAME = "ar3_geo"` and writes there instead. Two of the old figure files have identical counterparts in the new folder. These are stale-path candidates, rather than an experiment whose code was deleted.

## Cases that should not be called dead merely because a notebook disappeared

- The deleted `ar3_gradient_figure_2_4_examples.ipynb` notebook was replaced by current shared code: [CSV generation](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/ar_tools/experiments.py:399) and [figure generation](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/ar_tools/plotting_functions.py:325).
- The large tuning histories, coefficient-space histories and coefficient-tracking histories still have current notebooks: [tuning](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/ar_projected_subgradient_eta0_beta_tuning.ipynb), [coefficient-space descent](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/ar3_coefficient_space_descent.ipynb), [failed tracking](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/ar3_failed_coefficient_tracking.ipynb).
- Current KDE minimizer and switch studies still have their own generators, documented in [the KDE README](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/README.md:74).
- `outputs/csv/ar_width_comparison/width_curves.csv` has no current writer targeting that folder, but is still read by current analyses, including [diagnose.py](/Users/iljasukhanov/MASTERARBEIT/output/analysis/ar_order_without_penalties/diagnose.py:16). Its input dependency must be dealt with before removal.
- The thesis figures notebook supports changing its output directory through `AR_THESIS_FIGURES_DIR`. Therefore the lack of a literal `ar_thesis_figures_without_green` directory in its current source is not sufficient to label those figure copies unreproducible.

## EEG/COVID cleanup completed separately

At the user's explicit request, the remaining 14 EEG/COVID files (1.281 MB) and both dedicated directories were removed. The earlier five source-data deletions remain in place. No active AR/KDE script or thesis source reference to the removed EEG/COVID paths was found. General historical cleanup records were retained. The removal manifest is [stored outside the repository](/private/tmp/masterarbeit-cleanup-2026-10-19/remaining_eeg_covid_removal.json).

This generator audit itself deleted no additional outputs and changed no protected thesis source. It is a static audit; a located producer is not a tested promise that every historical parameter combination or external dependency is still available.
