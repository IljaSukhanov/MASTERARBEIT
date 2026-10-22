# Mean and maximum criteria in the AR tuning experiment

Updated on 22 October 2026. The notebook now uses two independent process-level contests on the same fits.

For each model/configuration, summarize achieved minimum widths across realizations with their arithmetic mean and their maximum. For each criterion, award full process-win credit to every configuration within its tolerance of the smallest summary across all four grids. Count process wins across models and retain all highest-count configurations. The maximum criterion minimizes the largest achieved width among the simulated realizations. Both tolerances are 1e-8 in the regenerated results.

The figure shows Mean and Max counts in each cell. Blue shading uses mean counts; red boxes mark global mean-count leaders, yellow boxes mark global max-count leaders, and grey boxes mark positive mean-count leaders within a panel. Red/yellow outlines are inset when they coincide.

Implementation: [notebook](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/xx_notebooks_tuning/ar_tuning_experiments.ipynb), [checks](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/tests/test_schedule_tuning.py:305).

Six checks passed, covering a reduced end-to-end experiment, independently selected mean/max co-winners, one-realization equivalence, cached setup refresh, and text archives. Both rebuilt figures were visually inspected, including overlapping red/yellow/grey boxes in the one-realization result.

## Rebuilt completed experiments

Both completed experiments were rebuilt from their saved individual fits. No descents were rerun. Every original mean-process count was checked against the old ranking and remained identical. Derived archives reference the original raw fit paths and hashes; original archives and figures were preserved.

| Experiment | Mean-criterion leaders | Max-criterion leaders |
|---|---|---|
| 100 × 1 | e1, diameter_normalized, c=0.3, H=50: 53 wins | e1, diameter_normalized, c=0.3, H=50: 53 wins |
| 100 × 10 | e1, diameter_normalized, c=0.1, H=100: 15 wins | e1, diameter_normalized, c=0.1, H=50: 55 wins; e1, diameter_normalized, c=0.2, H=50: 55 wins |

### 100 × 1

[Matrix PDF](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/pdfs/ar_tuning_experiments/hierarchical_process_wins_1e063eeb26083a68/01_hierarchical_100_processes_1_realizations_mean_and_max_process_win_counts_2x2.pdf) · [Matrix PNG](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/images/ar_tuning_experiments/hierarchical_process_wins_1e063eeb26083a68/01_hierarchical_100_processes_1_realizations_mean_and_max_process_win_counts_2x2.png)

[Mean ranking](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/txt/ar_tuning_experiments/hierarchical_process_wins_1e063eeb26083a68/20261022T193020_293087Z_9feb2417/process_win_ranking.txt) · [Maximum ranking](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/txt/ar_tuning_experiments/hierarchical_process_wins_1e063eeb26083a68/20261022T193020_293087Z_9feb2417/process_max_win_ranking.txt) · [Maximum leaders](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/txt/ar_tuning_experiments/hierarchical_process_wins_1e063eeb26083a68/20261022T193020_293087Z_9feb2417/highest_process_max_win_configurations.txt)

[Reuse provenance](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/txt/ar_tuning_experiments/hierarchical_process_wins_1e063eeb26083a68/20261022T193020_293087Z_9feb2417/reuse_details.txt) · [Referenced raw fits](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/txt/ar_tuning_experiments/hierarchical_process_wins_1e063eeb26083a68/20261022T193020_293087Z_9feb2417/reused_individual_fit_tables.txt) · [Completion](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/txt/ar_tuning_experiments/hierarchical_process_wins_1e063eeb26083a68/20261022T193020_293087Z_9feb2417/status.txt)

### 100 × 10

[Matrix PDF](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/pdfs/ar_tuning_experiments/hierarchical_process_wins_30a0701e000531f5/01_hierarchical_100_processes_10_realizations_mean_and_max_process_win_counts_2x2.pdf) · [Matrix PNG](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/images/ar_tuning_experiments/hierarchical_process_wins_30a0701e000531f5/01_hierarchical_100_processes_10_realizations_mean_and_max_process_win_counts_2x2.png)

[Mean ranking](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/txt/ar_tuning_experiments/hierarchical_process_wins_30a0701e000531f5/20261022T193025_272664Z_70c6e4c1/process_win_ranking.txt) · [Maximum ranking](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/txt/ar_tuning_experiments/hierarchical_process_wins_30a0701e000531f5/20261022T193025_272664Z_70c6e4c1/process_max_win_ranking.txt) · [Maximum leaders](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/txt/ar_tuning_experiments/hierarchical_process_wins_30a0701e000531f5/20261022T193025_272664Z_70c6e4c1/highest_process_max_win_configurations.txt)

[Reuse provenance](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/txt/ar_tuning_experiments/hierarchical_process_wins_30a0701e000531f5/20261022T193025_272664Z_70c6e4c1/reuse_details.txt) · [Referenced raw fits](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/txt/ar_tuning_experiments/hierarchical_process_wins_30a0701e000531f5/20261022T193025_272664Z_70c6e4c1/reused_individual_fit_tables.txt) · [Completion](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/txt/ar_tuning_experiments/hierarchical_process_wins_30a0701e000531f5/20261022T193025_272664Z_70c6e4c1/status.txt)

