# AR estimation notebooks

| Notebook | Purpose |
|---|---|
| [ar_coefficient_estimation_final.ipynb](ar_coefficient_estimation_final.ipynb) | Final paired coefficient benchmark: 100 Jones coefficient vectors per order p=1,...,5, each under five normalized Beta shapes, T=1000; 2,500 saved series, four estimators, checkpointed fitting, uncertainty summaries, and PDF/PNG figures. |

Run the notebook from top to bottom. It locates the shared
[`../ar_tools/`](../ar_tools/) utilities automatically, including when launched
from this folder. Existing experiment data and figures remain under
[`../outputs/`](../outputs/).

Use the final notebook for the thesis experiment. It first saves the original
series, coefficient vectors, seeds, condition metadata, and checksums under
[`../inputs/ar_coefficient_estimation_final/`](../inputs/ar_coefficient_estimation_final/).
Reruns verify and reload those inputs. Estimator outputs use separate data/fit
IDs under `../outputs/csv/ar_coefficient_estimation_final/`; figures use the
corresponding PDF and image directories. Settings or source changes get new IDs.

The final notebook uses the tuned e1 start, diameter-normalized scale c=0.1,
half-life H=100, cap 2000, patience 200, delta=1e-8, and no restarts.
The two older estimation notebooks have been removed; their saved experiment
data and figures remain available under `../outputs/`.
