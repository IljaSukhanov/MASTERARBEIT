# Continuous minimizers

Compare bandwidths obtained by continuously minimizing ISE and Q criteria.

| Notebook | Purpose |
| --- | --- |
| [continuous_ise_qhat_distributions_vs_n.ipynb](continuous_ise_qhat_distributions_vs_n.ipynb) | Paired sampling distributions of continuous ISE, Q₀, Q₁, and Q₁₃/₁₄ minimizers across sample sizes. |

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
