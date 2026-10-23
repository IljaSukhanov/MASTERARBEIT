# AR order-selection notebooks

| Notebook | Purpose |
|---|---|
| [ar_order_minimum_width.ipynb](ar_order_minimum_width.ipynb) | Main paired comparison of order selectors. |
| [ar3_order_selection_warm_starts.ipynb](ar3_order_selection_warm_starts.ipynb) | Compare independent and padded warm starts for order selection. |
| [ar3_population_breakpoint_selectors.ipynb](ar3_population_breakpoint_selectors.ipynb) | Compare selectors using population width curves. |

The five order selectors used across these notebooks are BIC, PACF cutoff,
first difference, second difference, and descent chain.

Run each notebook from top to bottom. Shared utilities remain in `../ar_tools/`,
and existing experiment data and figures remain in `../outputs/`.
