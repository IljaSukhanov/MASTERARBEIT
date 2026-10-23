# AR order-selection notebooks

| Notebook | Purpose |
|---|---|
| [ar_order_selection_final.ipynb](ar_order_selection_final.ipynb) | Final paired comparison of minimum width + descent chain, BIC, and PACF on the frozen 250-series subset. |
| [ar_order_minimum_width.ipynb](ar_order_minimum_width.ipynb) | Main paired comparison of order selectors. |
| [ar3_order_selection_warm_starts.ipynb](ar3_order_selection_warm_starts.ipynb) | Compare independent and padded warm starts for order selection. |
| [ar3_population_breakpoint_selectors.ipynb](ar3_population_breakpoint_selectors.ipynb) | Compare selectors using population width curves. |

The five order selectors used across these notebooks are BIC, PACF cutoff,
first difference, second difference, and descent chain.

The [frozen inputs for the final experiment](../inputs/ar_order_selection_final/030cde6f724e3c30/README.md)
contain 50 saved series per true order p=1,...,5, sampled from all five Beta
shapes in the coefficient-estimation dataset (250 series total). The final
notebook uses K=10 and the coefficient-study settings: initial step 0.1/L,
half-life 100, at most 2,000 evaluated states, patience 200, and one padded
previous-dimension start. Its [shared runner](../ar_tools/final_order_selection.py)
saves per-series checkpoints, all three selections, widths and directions,
numerical verification, and a Table 2.2-style matrix as CSV, Markdown and LaTeX
under `../outputs/csv/ar_order_selection_final/<dataset-id>/<run-id>/`.
The older notebooks retain their original experiment inputs.

Run each notebook from top to bottom. Shared utilities remain in `../ar_tools/`,
and existing experiment data and figures remain in `../outputs/`.
