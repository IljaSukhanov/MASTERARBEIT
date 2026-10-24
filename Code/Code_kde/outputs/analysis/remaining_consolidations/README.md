# Remaining notebook consolidations

Completed on 2026-10-24:

1. The fast notebook's continuous Q-hat overlay and the continuous-minimizer
   notebook use [one runner](../../../kde_tools/continuous_runs.py) and
   `outputs/data/shared_continuous_minimizers/`. Matching actual sample prefixes,
   numerical settings, and objective versions share results. Historical imports
   also verify the documented sample generator and source hashes; producer
   versions remain attached to each selection. All 10,000 overlapping historical
   Q-hat/Q0 results match exactly after converting their endpoint-flag convention.
2. The Gaussian L²/ISE notebook is consolidated into
   [main notebook section 2](../../../xx_notebooks_overview/spectral_scale_selection_kde.ipynb).
   Choose `beta2_profile = "exact_gaussian"` and
   `beta2_gaussian_run_profile = "preview"` or `"thesis"`. Original per-repetition
   seeds, exact divergence, selected ISE values, CSV formats, and the overlaid
   and two individual plots are retained. The standalone notebook is removed.
3. The [joint-limit notebook](../../../xx_notebooks_asymptotics/experiment_1_joint_N_lambda_limit.ipynb)
   routes its repeated-sample and sample-size sections through
   [the shared spectral runner](../../../kde_tools/spectral_sweeps.py).
   Seeds, selections, summaries, plots, and eigenpair diagnostics are retained.
   New computations save diagnostics for all notebook views; older selection-only
   caches are upgraded only when diagnostics are requested, checking agreement
   with their saved selections.

The main sample-size sweep keeps its original seed independently of the selected
histogram profile. Automatic PDFs for the consolidated Gaussian plots now go
under `outputs/pdfs/spectral_scale_selection_kde/`. Historical saved outputs remain
in place; the full Monte Carlo experiments were not regenerated. Seven active
notebooks remain.

[Validation report](validation_report.json): 97 unit tests pass; reduced numerical
runs cover all three main histogram profiles, the joint notebook, fast sweeps and
continuous overlay, and continuous ISE/Q-r comparisons. Checks cover resuming,
cross-notebook cache reuse, CSV/PDF exports, refinement, and incompatible caches.

[Integration checks](integration_checks.json): reduced old/new joint selections,
counts and diagnostics agree exactly; the joint plot cells are unchanged. The
old/new exact Gaussian selections, ISE values and CSV files agree exactly. A real
historical import reuses all 12 requested selections without computation.
All 365 checked existing KDE result/figure files and protected thesis text
sources are unchanged. All seven active notebooks validate, compile, and enable
the automatic PDF exporter.
