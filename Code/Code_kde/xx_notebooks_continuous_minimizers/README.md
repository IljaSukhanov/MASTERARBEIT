# Continuous minimizers

Compare bandwidths obtained by continuously minimizing ISE and Q criteria.

| Notebook | Purpose |
| --- | --- |
| [continuous_ise_qhat_distributions_vs_n.ipynb](continuous_ise_qhat_distributions_vs_n.ipynb) | Paired sampling distributions of continuous ISE, Q₀, Q₁, and Q₁₃/₁₄ minimizers across sample sizes. |
| [local_energy_minimizers_n200.ipynb](local_energy_minimizers_n200.ipynb) | Interior local energy minima for 1,000 paired samples of size 200, for r=0, 13/14, and 1; reports missing and multiple minima. |
| [q1_local_minimizers_selected_n.ipynb](q1_local_minimizers_selected_n.ipynb) | Interior local Q₁ minima on (0.01,2) for n=100,1000,1500,2000. |
| [local_qr_minimizers_vs_n.ipynb](local_qr_minimizers_vs_n.ipynb) | Interior local Q₀, Q₁₃/₁₄, and Q₁ minima on (0.01,2), with the ISE-optimal bandwidth reference, for every n=100,200,…,2000; selected and complete overlaid histograms. |

Q₀ and the fast L² notebook's continuous Q-hat overlay now use the same
runner and `../outputs/data/shared_continuous_minimizers/` cache. Cache matching
uses actual sample prefixes and numerical search settings; validated historical
results retain their producer versions. The ISE, Q₁, and Q₁₃/₁₄ comparisons,
CSV exports, refinement checks, and figures remain here.

Run cells in order. Setup locates the shared helpers in `../kde_tools/` and
enables automatic PDF exports under `../outputs/pdfs/<notebook-name>/`.
Saved data, images, and checkpoints keep their existing locations in
`../outputs/`.

[All KDE notebooks and code guide](../README.md)
