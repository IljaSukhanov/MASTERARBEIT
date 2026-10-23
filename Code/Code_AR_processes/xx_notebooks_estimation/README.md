# AR estimation notebooks

| Notebook | Purpose |
|---|---|
| [ar3_minimum_width_k4.ipynb](ar3_minimum_width_k4.ipynb) | Minimum-width coefficient recovery, comparisons with other known-order estimators, and experiments across orders and sample sizes. |
| [ar3_minimum_width_vs_T.ipynb](ar3_minimum_width_vs_T.ipynb) | Empirical widths, errors relative to population references, and coefficient accuracy as sample size increases, including repeated estimator comparisons. |

Run each notebook from top to bottom. They locate the shared
[`../ar_tools/`](../ar_tools/) utilities automatically, including when launched
from this folder. Existing experiment data and figures remain under
[`../outputs/`](../outputs/).

The move preserves the notebooks' experiment settings and saved results.
The new tuning configuration has not been applied to these experiments.
