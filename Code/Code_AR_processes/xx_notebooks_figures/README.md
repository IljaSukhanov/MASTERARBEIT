# AR figure notebooks

| Notebook | What it is useful for |
|---|---|
| [ar1_sliding_window_embedding.ipynb](ar1_sliding_window_embedding.ipynb) | An introductory illustration of AR(1) geometry. Simulates a process, plots consecutive observations as a two-dimensional point cloud, and annotates the support strip, its center line, a horizontal chord, and the perpendicular strip width. Useful for explaining how a time series becomes a geometric width problem. |
| [ar_beta_sliding_window_2d.ipynb](ar_beta_sliding_window_2d.ipynb) | Three separate figures: five AR(1) time-series panels, their 2D sliding-window clouds with a common black dashed support boundary, and a five-color time-series overlay. Uses Beta innovations on the fixed interval [-1,1] for (1,1), (1/2,1/2), (2,2), (1/2,2), and (2,1/2). |
| [ar_thesis_figures_2_1_to_2_3.ipynb](ar_thesis_figures_2_1_to_2_3.ipynb) | Reproduces the three AR(1), AR(2), and AR(3) thesis illustrations. Plots attained empirical widths against embedding dimension, the population width at each returned direction, and the population plateau, and marks the selected elbow. Useful for regenerating the thesis's main examples of width-based order selection. |

Run each notebook from top to bottom. They locate the shared `../ar_tools/`
utilities automatically, including when launched from this folder.

The first two notebooks save PDF and PNG figures under their existing
`../outputs/pdfs/<notebook-name>/` and `../outputs/images/<notebook-name>/`
directories. The thesis-figure notebook saves `AR1_experiment`, `AR2_experiment`,
and `AR3_experiment` as PDF and PNG in
`../../../Abschlussarbeit_2021/LaTeX/images/` by default. Set the environment
variable `AR_THESIS_FIGURES_DIR` before starting it to choose another destination.

The Beta notebook exports each of these as PDF and PNG:

- `beta_ar1_time_series`: five separate time-series panels.
- `beta_sliding_window_clouds_2d`: five corresponding point clouds.
- `beta_ar1_time_series_overlay`: all five paths on one axis, in different colors.

Its fixed-support sampler is local to this illustration. `time_series_points`
controls the separate panels (default 300 observations); `overlay_points`
controls the overlay (default 100 to keep it readable). The clouds use the full
retained path, and all figures reuse the same simulated data.

The thesis-figure notebook's code currently selects the elbow using
`descent_chain`; its existing markdown still describes a LAD selector.
