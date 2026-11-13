# Completed maintenance records

Archived on 25 October 2026. These reports describe completed work. Their solver
versions, notebook counts, test counts, and rerun commands belong to the recorded
work and may differ from the current code. Current usage is documented in the
[KDE guide](../../../Code/Code_kde/README.md).

The original report text is retained below, with section headings and relative
links adjusted for this archive. Supporting data, validation records, scripts,
figures, and recovery files remain in their original folders.

- [Notebook consolidations](#notebook-consolidations)
- [Fourier quadrature migration](#fourier-quadrature-migration)
- [Perron figure regeneration](#perron-figure-regeneration)
- [Overleaf synchronization cleanup](#overleaf-synchronization-cleanup)

## Notebook consolidations

Original heading: **Remaining notebook consolidations**  
Original location: `Code/Code_kde/outputs/analysis/remaining_consolidations/README.md`

Completed on 2026-10-24:

1. The fast notebook's continuous Q-hat overlay and the continuous-minimizer
   notebook use [one runner](../../../Code/Code_kde/kde_tools/continuous_runs.py) and
   `outputs/data/shared_continuous_minimizers/`. Matching actual sample prefixes,
   numerical settings, and objective versions share results. Historical imports
   also verify the documented sample generator and source hashes; producer
   versions remain attached to each selection. All 10,000 overlapping historical
   Q-hat/Q0 results match exactly after converting their endpoint-flag convention.
2. The Gaussian L²/ISE notebook is consolidated into
   [main notebook section 2](../../../Code/Code_kde/xx_notebooks_overview/spectral_scale_selection_kde.ipynb).
   Choose `beta2_profile = "exact_gaussian"` and
   `beta2_gaussian_run_profile = "preview"` or `"thesis"`. Original per-repetition
   seeds, exact divergence, selected ISE values, CSV formats, and the overlaid
   and two individual plots are retained. The standalone notebook is removed.
3. The [joint-limit notebook](../../../Code/Code_kde/xx_notebooks_asymptotics/experiment_1_joint_N_lambda_limit.ipynb)
   routes its repeated-sample and sample-size sections through
   [the shared spectral runner](../../../Code/Code_kde/kde_tools/spectral_sweeps.py).
   Seeds, selections, summaries, plots, and eigenpair diagnostics are retained.
   New computations save diagnostics for all notebook views; older selection-only
   caches are upgraded only when diagnostics are requested, checking agreement
   with their saved selections.

The main sample-size sweep keeps its original seed independently of the selected
histogram profile. Automatic PDFs for the consolidated Gaussian plots now go
under `outputs/pdfs/spectral_scale_selection_kde/`. Historical saved outputs remain
in place; the full Monte Carlo experiments were not regenerated. Seven active
notebooks remain.

[Validation report](../../../Code/Code_kde/outputs/analysis/remaining_consolidations/validation_report.json): 97 unit tests pass; reduced numerical
runs cover all three main histogram profiles, the joint notebook, fast sweeps and
continuous overlay, and continuous ISE/Q-r comparisons. Checks cover resuming,
cross-notebook cache reuse, CSV/PDF exports, refinement, and incompatible caches.

[Integration checks](../../../Code/Code_kde/outputs/analysis/remaining_consolidations/integration_checks.json): reduced old/new joint selections,
counts and diagnostics agree exactly; the joint plot cells are unchanged. The
old/new exact Gaussian selections, ISE values and CSV files agree exactly. A real
historical import reuses all 12 requested selections without computation.
All 365 checked existing KDE result/figure files and protected thesis text
sources are unchanged. All seven active notebooks validate, compile, and enable
the automatic PDF exporter.

## Fourier quadrature migration

Original heading: **L2 migration to the thesis frequency quadrature**  
Original location: `Code/Code_kde/outputs/analysis/fourier_quadrature_migration/README.md`

Implemented on 23 October 2026. Protected thesis files and historical exported
figures/results were not modified. Full Monte Carlo experiments were not rerun.

### Change

The shared L2 implementation now forms the real weighted feature matrix

\[
B_{rk}=\sqrt{w_k}\,|P_n(\omega_k)|\exp(-h_r^2\omega_k^2/2),
\qquad G=BB^T,
\]

with composite trapezoidal weights including the factor 1/pi for integration
on the positive frequency axis. Half squared L2 distances are assembled from G.
This is an algebraic factorization of the existing thesis approximation, after
`eq:beta2_exact_formula02`; the mathematical basis is
[Parseval's identity, DLMF 1.14.7₅](https://dlmf.nist.gov/1.14.E7_5).
See [the implementation](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/kde_tools/fourier/similarities.py:13).

There are no inverse FFTs, spatial-density clipping or renormalization in this
L2 helper. `kde_family_fft` remains available for diagnostics requiring spatial
density values. The combined sample-size ISE/Q-hat diagnostic retains those
spatial values for its other criteria, while its L2 Gram uses direct quadrature.

Public names are `gaussian_kde_l2_gram_fourier`,
`beta2_divergence_matrix_fourier`, and `beta2_similarity_matrix_fourier`.
Old `_fft` L2 entry points are compatibility wrappers for the new calculation.
Existing half-width and point-count parameters preserve the previous frequency
spacing and cutoff. The Gram helper also accepts an explicit cutoff and node count.

Eight active notebooks now call the Fourier-quadrature API and identify the
method as `frequency_trapezoid_v1`. New checkpoint fingerprints/directories
separate these calculations from historical FFT runs. Cached notebook cell
outputs were cleared so old displays are not presented as new calculations.

### Validation

- All eight migrated notebook setup cells execute, and all code cells parse.
  [Setup report](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/outputs/analysis/fourier_quadrature_migration/setup_validation_report.json).
- Three small exact-formula checks give a maximum absolute L2-distance error
  of 1.86e-14 for the new implementation. All 40 selected-bandwidth comparisons
  agree between the new method, the previous reconstruction method, and a grid
  with halved frequency spacing. These cover eight sample/grid cases and five
  lambda values per case, including n=1000 and N=1000 cases.
  [Numerical report](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/outputs/analysis/fourier_quadrature_migration/migration_report.json).
- Local median distance-matrix construction speedups were 1.54x for
  n=200/N=201, 1.14x for n=1000/N=201, and 2.40x for n=200/N=1000.
  These timings exclude eigensolvers and notebook I/O and are specific to the
  recorded runtime; they are not whole-notebook speed guarantees.
- 69 unit tests pass, including the direct trapezoidal endpoint formula,
  narrow-bandwidth cutoff convergence, translation invariance, compatibility,
  and input checks. Four reduced notebook workflows pass, including the
  101-point Figure 3.3 selection path, checkpoint resuming, and rejection of
  legacy solver/L2-method checkpoints.
  [Notebook validation report](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/outputs/analysis/fourier_quadrature_migration/notebook_validation_report.json).

The unfiltered suite runs 71 test methods and still reports four pre-existing
subtest failures in two solver-routing test methods: two historical scripts and
two calls in the eigenvalue notebook use `eigh` for reference checks. An AST
comparison against HEAD confirms that the flagged calls are unchanged and no
new forbidden calls were introduced.
[Baseline audit](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/outputs/analysis/fourier_quadrature_migration/routing_baseline_audit.json).
The four-workflow report excludes those two known-failing routing methods only
in its invocation harness; the source routing tests remain unchanged.

### Use

Restart notebook kernels and run from setup. New sampling runs use the new
method identity; historical outputs have not been relabeled or automatically
regenerated. The migration checks do not certify every saved experiment or
arbitrary choices of frequency cutoff and spacing.

## Perron figure regeneration

Original heading: **Figures 3.1-3.4 after Perron centralization**  
Original location: `Code/Code_kde/outputs/analysis/perron_figures_3_1_to_3_4/README.md`

All four thesis figures were regenerated with their full existing notebook
settings and the shared solver `stabilized_lanczos_v1`. **No major changes
were detected.** Every selected bandwidth is unchanged. At 1200 pixels on
the longest side, all four rendered PDFs are pixel-identical to their saved
previous versions. The four refreshed PDFs occupy the same paths already
referenced by the thesis.

| Figure | Full comparison | Result |
| --- | --- | --- |
| 3.1 | Four Perron profiles, N=191, lambda=1,10,100,1000 | All four peaks unchanged; maximum component discrepancy below 5e-14 |
| 3.2 | 199 grid sizes, N=11,16,...,1001, four lambda values | 0 of 796 bandwidth selections changed |
| 3.3 | N=201, 101 lambda values from 0.01 to 10000 | 0 of 101 bandwidth selections changed |
| 3.4 | N=100,200,...,3000, six joint-limit regimes | 0 of 180 bandwidth selections changed |

The profile discrepancy includes the precision of recovering values from
PDF drawing coordinates. Bandwidth indices were recovered by calibrating
the PDF marker positions and matching each value to its original geometric
grid. The maximum mismatch before matching was below 2.4e-13, so the old
plotted bandwidth indices were unambiguous. This compares against the PDFs
actually referenced by the thesis, rather than stale notebook displays.

Both notebooks use the fixed normal sample of size 200 with seed 20260823,
the bandwidth interval [0.01,2], and the FFT grid of 8192 spatial nodes on
[-12,12]. Figure 3.4 retains the continuous reference h=0.1764236267. Its
largest recorded relative eigenpair residual was 2.1e-14. No thesis prose,
labels, captions, or references were edited.

The rerun completed in approximately 121 seconds. All 20 Perron and routing
checks passed afterward. Full numeric comparisons, raw solver diagnostics,
PDF hashes, and visual comparison results are recorded in
[comparison_report.json](../../../Code/Code_kde/outputs/analysis/perron_figures_3_1_to_3_4/comparison_report.json). Previous PDFs are saved
in [before/](../../../Code/Code_kde/outputs/analysis/perron_figures_3_1_to_3_4/before/).

The computation executes the current figure-producing cells from
[spectral_scale_selection_kde.ipynb](../../../Code/Code_kde/xx_notebooks_overview/spectral_scale_selection_kde.ipynb)
and [experiment_1_joint_N_lambda_limit.ipynb](../../../Code/Code_kde/xx_notebooks_asymptotics/experiment_1_joint_N_lambda_limit.ipynb)
without changing their settings or notebook files. The shared automatic PDF
exporter remains enabled; skipped unrelated figure slots retain their filenames.

To repeat the numerical rerun from the project root:

```bash
python3 Code/Code_kde/outputs/analysis/perron_figures_3_1_to_3_4/rerun_figures.py
```

The companion `extract_previous_figure_data.py` uses NumPy and pdfplumber
for recovering the baseline data from the preserved PDF snapshots. The
baseline extraction and visual comparison are independent of the new solver.

## Overleaf synchronization cleanup

Original heading: **Overleaf synchronization cleanup — 24 October 2026**  
Original location: `output/analysis/overleaf_cleanup_2026-10-24/README.md`

The scan covered all project files outside Git internals, including ignored outputs. Tracked files were reduced from approximately 72.38 MB to 57.03 MB. The detailed manifest records each affected path, its original checksum, and the reason.

- Removed 33 unreferenced PNG previews while retaining their PDF counterparts.
- Removed 42 cache and operating-system metadata files outside the protected thesis folder.
- Cleared saved displays in 9 notebooks; all source cells and notebook metadata remain unchanged.
- Kept 29 generated diagnostic CSV files on this computer and excluded them from GitHub/Overleaf. They can still be read by the existing analysis scripts.
- Deduplicated existing ignore rules and added the diagnostic-table exclusion.

Every protected thesis file, Python source, input file, and exported PDF retained its original checksum. All thesis graphic dependencies retain the same availability and checksums as before. No source-code or mathematical changes were made; numerical experiments were not rerun.

Original changed/deleted files are also saved in `/private/tmp/MASTERARBEIT-cleanup-2026-10-24-d9_89fp6/original-files.tar.gz`. Tracked originals remain recoverable from commit `7e919a1e0835e122be008b2e997a973e831e5a14`.

This reduces the current project snapshot; it does not remove old commits or establish the cause of Overleaf’s GitHub service error.

References: [Git ignore behavior](https://git-scm.com/docs/gitignore), [Overleaf GitHub sync limits](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/github-synchronization#github-synchronization-file-limits).
