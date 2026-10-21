# AR coefficient scoring: MAE migration

All active AR coefficient comparisons now use the mean absolute error per
coefficient, averaged across replications for Monte Carlo summaries. The
shared definition is [coefficient_errors.py](../../../Code/Code_AR_processes/ar_tools/coefficient_errors.py).

The migration covers shared runners and settings, eight notebooks, tuning
co-winner counts, restart runners and report/plot generators, and the auxiliary
padded-coefficient audit. It replaces squared, root-squared, Euclidean, summed
absolute, and coefficient-norm-error comparisons with coefficient MAE. The
optimizer and estimator fitting objectives retain their previous behavior.

Old generated results keep their original scores. Changed notebook displays
were cleared. New MAE results use separate paths or metric-aware configuration
hashes, and replication caches require an explicit MAE schema. Full production
experiments were not rerun as part of the code migration.

## Verification

- 85 of 86 AR tests pass. The one pre-existing failure flags the local
  `save_figure` function in the failure-tracking notebook; it was present before
  this task. See [full test log](test_results_final.txt).
- All Python sources and all 17 AR notebooks parse.
- Complete tracking/failure notebook runs, paired MAE bootstrap summaries and
  plots, and four-method comparisons for AR orders 1–5 were exercised with
  temporary outputs. See [notebook checks](notebook_smoke.txt).
- All three restart scripts were exercised on an original retained process;
  MAEs were independently recomputed from the returned coefficients. See
  [restart checks](restart_smoke.txt).
- The optimizer body is unchanged, apart from renaming the experiment helper.
- Thesis source files were untouched. Three protected figure PDFs and the
  figure notebook's saved outputs changed independently while this task was
  active; those concurrent changes were left in place.

Detailed counts and the concurrent-change record: [verification.json](verification.json).
