# Lambda sensitivity

Inspect changes in the selected bandwidth as lambda varies.

| Notebook | Purpose |
| --- | --- |
| [spectral_lambda_jump_study.ipynb](spectral_lambda_jump_study.ipynb) | Lambda-sweep jumps, competing Perron peaks, and refined dominance crossings. |
| [spectral_lambda_switch_distribution.ipynb](spectral_lambda_switch_distribution.ipynb) | Sampling distributions of switch locations and severity, using coarse and fine lambda scans. |

Run cells in order. Setup locates the shared helpers in `../kde_tools/` and
enables automatic PDF exports under `../outputs/pdfs/<notebook-name>/`.
Saved data, images, and checkpoints keep their existing locations in
`../outputs/`.

[All KDE notebooks and code guide](../README.md)
