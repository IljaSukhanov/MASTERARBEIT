# Ten Figure 2.4-style AR(3) examples

The ten-page PDF contains runs 1 through 10 from the existing 50-process AR(3)
experiment, in their original order. The examples were not selected by outcome.
Each run has its own coefficient vector, T=2000 observations, uniform [-1,1]
innovations, and burn-in 1000. Settings and input provenance are in
[configuration.json](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/csv/ar3_gradient_figure_2_4_examples/configuration.json).

The gradient optimizer is the existing `batched_geometric_width_curves` function:
three starts, 500 iterations, initial step 0.2, geometric decay 0.99, and the
previous best direction padded with zero as the first start for every k>=2.
All-valid observation windows are used, as in the earlier gradient experiment.
All ten runs were recomputed. The widths and directions agree exactly with the
saved padded-start results. Norms and reconstructed projected ranges were also
checked. See [verification.json](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/csv/ar3_gradient_figure_2_4_examples/verification.json).

## The four curves

- Blue: empirical directional width at the direction returned by gradient
  descent. This is a fitted width, not a certificate of the global empirical
  minimum.
- Orange: population directional width at that same fitted direction.
- Black: analytical population minimum-width plateau for k>=4.
- Green: empirical width at a population minimum-width direction. For k<=3,
  that direction is computed from a globally enumerated truncated support
  zonotope; for k>=4 it is the normalized AR recurrence direction.

The support-width formula and plateau follow Proposition `prop:minimum_width`
in [the thesis source](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/3_application_to_time_series.tex:279).
The pre-plateau population calculation is implemented in
[population_widths.py](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/ar_tools/population_widths.py),
with an omitted-tail width bound of 1e-8. Orange directional widths are evaluated
with an omitted-tail bound at most 1e-9. These bounds control truncation, not
floating-point arithmetic. The chosen history lengths and tail bounds are saved
in the data.

## Breakpoint markers

The red line and dot use the original descent-chain selector on the fitted raw
width curve. The dot is the empirical fitted width at the selected dimension.
The faint gray line marks the true breakpoint dimension k=4. The four width
curves retain their original meanings.

The shared selector is in
[breakpoint_selectors.py](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/ar_tools/breakpoint_selectors.py).
The selection summary also records the log-width transformation separately.

## Saved outputs

- [Ten-page PDF](/Users/iljasukhanov/MASTERARBEIT/output/pdf/ar3_gradient_figure_2_4_ten_examples.pdf)
- [All plotted values and directions](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/csv/ar3_gradient_figure_2_4_examples/figure_data.csv)
- [Coefficients and selector results](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/csv/ar3_gradient_figure_2_4_examples/selection_summary.csv)
- Ten individual PNGs and two contact-sheet previews in
  `Code/Code_AR_processes/outputs/images/ar3_gradient_figure_2_4_examples/`.

The thesis files were not modified.
