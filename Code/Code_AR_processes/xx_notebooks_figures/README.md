# AR figure notebooks

| Notebook | What it is useful for |
|---|---|
| [ar1_sliding_window_embedding.ipynb](ar1_sliding_window_embedding.ipynb) | An introductory illustration of AR(1) geometry. Simulates a process, plots consecutive observations as a two-dimensional point cloud, and annotates the support strip, its center line, a horizontal chord, and the perpendicular strip width. Useful for explaining how a time series becomes a geometric width problem. |
| [ar_beta_sliding_window_2d.ipynb](ar_beta_sliding_window_2d.ipynb) | Visual comparison of embedding clouds for different Beta innovation shapes, including the uniform case. Keeps the AR coefficients, sample size, innovation mean, and innovation variance fixed, and overlays each distribution's theoretical support boundary. Useful for exploring how innovation shape affects the cloud and its support. |
| [ar_thesis_figures_2_1_to_2_3.ipynb](ar_thesis_figures_2_1_to_2_3.ipynb) | Reproduces the three AR(1), AR(2), and AR(3) thesis illustrations. Plots attained empirical widths against embedding dimension, the population width at each returned direction, and the population plateau, and marks the selected elbow. Useful for regenerating the thesis's main examples of width-based order selection. |

Run each notebook from top to bottom. They locate the shared `../ar_tools/`
utilities automatically, including when launched from this folder.

The first two notebooks save PDF and PNG figures under their existing
`../outputs/pdfs/<notebook-name>/` and `../outputs/images/<notebook-name>/`
directories. The thesis-figure notebook saves `AR1_experiment`, `AR2_experiment`,
and `AR3_experiment` as PDF and PNG in
`../../../Abschlussarbeit_2021/LaTeX/images/` by default. Set the environment
variable `AR_THESIS_FIGURES_DIR` before starting it to choose another destination.

The thesis-figure notebook's code currently selects the elbow using
`descent_chain`; its existing markdown still describes a LAD selector.
