# AR order-selection notebooks

| Notebook | Purpose |
|---|---|
| [ar_order_minimum_width.ipynb](ar_order_minimum_width.ipynb) | Main paired comparison of order selectors. |
| [ar_bic_order_selection.ipynb](ar_bic_order_selection.ipynb) | Separate BIC order-selection benchmark. |
| [ar3_order_selection_warm_starts.ipynb](ar3_order_selection_warm_starts.ipynb) | Compare independent and padded warm starts for order selection. |
| [ar3_population_breakpoint_selectors.ipynb](ar3_population_breakpoint_selectors.ipynb) | Compare selectors using population width curves. |
| [ar3_descent_chain_lp.ipynb](ar3_descent_chain_lp.ipynb) | LP tangent fitting and descent-chain diagnostics. |
| [ar_unshifted_residual_width_ratio.ipynb](ar_unshifted_residual_width_ratio.ipynb) | Explore a residual-span ratio selector. |

Run each notebook from top to bottom. Shared utilities remain in `../ar_tools/`,
and existing experiment data and figures remain in `../outputs/`.
