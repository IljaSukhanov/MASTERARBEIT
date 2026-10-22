AR TUNING TEXT ARCHIVE

All data files are UTF-8 .txt documents. Tables are tab-separated with column
headers, no truncation, NaN for missing numeric values, and 17 significant
digits for floating-point values. Integer seeds are written as exact decimal
integers. Read tables with pandas.read_csv(path, sep='\t', float_precision='round_trip');
use dtype={'process_seed': 'uint64'} (and the same for other seed columns) when
an exact seed type is needed. Metadata documents use indented JSON syntax
inside .txt files. Matrix tables include their row labels in the first column.

settings.txt records the settings used when this archive was opened.
environment.txt records the Python and dependency versions. source/ contains
source snapshots, including notebook code without saved image outputs.
manifest.txt inventories the exported files, dimensions and SHA-256 hashes.
status.txt distinguishes an in-progress export from a completed section.
Each invocation of a section's settings cell creates a new timestamped archive;
rerunning its result or plot cells replaces that table, without appending rows.

Per-variant results.txt contains one row per configuration and realization:
candidate_id, step_rule, step_value (c), half_life (H), beta, actual gamma0,
minimum_width, best direction u_*, actual initial direction initial_u_*, model
coefficients phi_*, exact seeds, evaluation counts, stopping flags and all other
fields returned by the tuning runner. These widths are sufficient to recompute
mean, maximum, pooled win counts or process-level win counts without new fits.
If coefficient MAE is plotted, per_run_coefficient_mae is also exported; an
undefined coefficient estimate remains NaN. MAE never affects selection.

Per-variant cases.txt records realization metadata and within-variant winners.
replication identifies the realization within its input group; in Sections 1-5
it also identifies the coefficient draw. In Section 6 model_id identifies the
coefficient draw, and replication is local to that model. Candidate IDs repeat
across starting-direction variants: use initialization + candidate_id, together
with model_id (when present), replication and T, to identify individual fits.

inputs/ contains the retained simulated time series and the four-coordinate
point clouds, once per shared input group. They are reconstructed from the exact
saved coefficients, innovation seeds and burn-in using the snapshotted simulator;
their true-direction widths are checked against the runner's case records.
Discarded burn-in samples and per-iteration optimizer trajectories are not saved.
Initial directions are saved explicitly, as well as in each fit and case row.

Summary tables preserve unrounded values, all tied leaders, selection tolerances
and co-winner records. Full credit is awarded to ties, so credits can exceed the
number of cases. Section 6 averages widths within each model before awarding
process wins; it does not count individual-realization wins. Figure tables and
figure-path documents connect the numerical data to the saved PDF/PNG images.
An interrupted run retains completed variant/process files; it is not marked
complete. Earlier figures alone cannot recover discarded individual fit data.
