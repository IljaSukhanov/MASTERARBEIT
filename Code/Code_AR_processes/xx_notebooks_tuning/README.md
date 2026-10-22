# Final AR tuning experiment

[ar_tuning_experiments.ipynb](ar_tuning_experiments.ipynb) contains one experiment.
Run all cells from top to bottom: setup, experiment settings, model/start
preparation, descents and aggregation, then count matrices.

The current settings are **100 Jones-sampled AR(3) processes with 10 innovation
realizations each**, using T=500 and burn-in 2,000. Four variants compare a
shared random-start bank and e1=(1,0,0,0), each with absolute gamma0=c and
diameter-normalized gamma0=c/L steps. Every variant uses the same models and
innovation paths. The random-start bank is drawn once; its realization-indexed
rows are reused across all processes and random-start configurations.
The experiment settings expose the counts, three independent seed streams,
process-win tolerance and worker count. [Notebook, Section 2](ar_tuning_experiments.ipynb),
[paired runner](../ar_tools/tuning.py), [simulation](../ar_tools/functions.py).

The grid uses c in {0.1,0.2,...,0.9} and H in {2,5,10,20,50,100,200,1000},
with beta=2^(-1/H): 72 configurations per variant, 288 in total and 288,000
descents at the current 100 × 10 experiment size. Each descent uses one start, a
2,000-iteration cap and patience 200, without stationarity checks or restarts.
Every strict width improvement is retained; patience resets after a cumulative
best-width decrease of delta=1e-8 from its last reference.
[Notebook settings](ar_tuning_experiments.ipynb), [optimizer](../ar_tools/descent.py).

For each process and configuration, compute the mean and maximum achieved
minimum width over its realizations. Compare each criterion independently
across all four grids together. Every configuration within
hierarchical_mean_win_tolerance=1e-8 of that process's lowest mean receives a
mean-process win; every configuration within hierarchical_max_win_tolerance=1e-8
of its smallest maximum receives a max-process win. Both use unrounded scores
and full credit for ties. The maximum describes the worst observed realization
among the simulated paths. Final rankings count process wins separately for
each criterion and retain all tied leaders. With one realization, the two
criteria coincide when their tolerances match. Total credits can exceed the
number of processes. Neither criterion selects by a pooled score over all paths.

Each cell in the 2×2 matrix figure displays Mean and Max process-win counts.
Blue shading follows the mean counts. Red outlines mark all overall leaders
for the mean criterion, yellow outlines mark all overall leaders for the maximum
criterion, and grey outlines mark each panel's highest positive mean count.
Overlapping outlines are inset so both criterion markers remain visible.
No coefficient MAE or line plots are shown.
[Notebook, Section 2](ar_tuning_experiments.ipynb).

The matrix figure is displayed and exported as PDF and PNG under
`../outputs/pdfs/ar_tuning_experiments/hierarchical_process_wins_<run-id>/`
and the corresponding `../outputs/images/` folder. Tables and text logs are
suppressed in the notebook. Each run creates a separate timestamped UTF-8
text archive under
`../outputs/txt/ar_tuning_experiments/hierarchical_process_wins_<run-id>/<timestamp>/`.
It retains settings, configuration grids, coefficients, exact seeds and
starting directions, retained series/clouds, every individual fit, process
means and maxima, both sets of co-winners and final rankings, and unrounded
mean/max figure data. The maximum ranking is saved in
`process_max_win_ranking.txt`, its leaders in
`highest_process_max_win_configurations.txt`, and its selection rule in
`max_selection_rule.txt`. Tables are
tab-separated and retain full floating-point precision. The archive includes
source snapshots, environment versions, a manifest and completion status.
Individual fits are exported one process at a time to bound memory usage;
interrupted runs preserve completed exports. Iteration trajectories and
discarded burn-in samples are not exported. [Export implementation](../ar_tools/tuning_exports.py).
