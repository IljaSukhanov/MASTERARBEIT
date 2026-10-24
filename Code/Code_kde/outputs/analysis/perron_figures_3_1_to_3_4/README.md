# Figures 3.1-3.4 after Perron centralization

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
[comparison_report.json](comparison_report.json). Previous PDFs are saved
in [before/](before/).

The computation executes the current figure-producing cells from
[spectral_scale_selection_kde.ipynb](../../../xx_notebooks_overview/spectral_scale_selection_kde.ipynb)
and [experiment_1_joint_N_lambda_limit.ipynb](../../../xx_notebooks_asymptotics/experiment_1_joint_N_lambda_limit.ipynb)
without changing their settings or notebook files. The shared automatic PDF
exporter remains enabled; skipped unrelated figure slots retain their filenames.

To repeat the numerical rerun from the project root:

```bash
python3 Code/Code_kde/outputs/analysis/perron_figures_3_1_to_3_4/rerun_figures.py
```

The companion `extract_previous_figure_data.py` uses NumPy and pdfplumber
for recovering the baseline data from the preserved PDF snapshots. The
baseline extraction and visual comparison are independent of the new solver.
