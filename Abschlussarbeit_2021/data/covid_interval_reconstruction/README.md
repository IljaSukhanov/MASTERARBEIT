# COVID interval-length reconstruction

This folder reconstructs

\[
L_i=r_i-s_i+1,
\]

where `s_i` is the first date on which county `i` has at least two cumulative
confirmed cases and `r_i` is the state's listed stay-at-home end date.

## Sources

- Johns Hopkins CSSE US confirmed-case time series:
  `time_series_covid19_confirmed_US.csv`
- Littler statewide stay-at-home order tracker cited by the COVID paper:
  `littler_statewide_orders.html`

Both source snapshots were retrieved on 2026-06-28.

## Reconstruction choices

- The JHU file is restricted to the 50 states and District of Columbia.
- Unassigned and out-of-state rows are removed.
- Chugach and Copper River, created by the 2020 split of Valdez-Cordova, are
  removed so that the county universe matches the paper's 3,142 entities.
- The first date shown in Littler's `Duration or End Date` column is used.
- When that column only gives a phased-reopening date, it is retained as an
  explicitly labelled proxy.
- Counties without a dated statewide end, or whose second case occurs after
  the selected end date, are excluded from the positive-span histogram.

This is a transparent reconstruction, not an exact reproduction. The paper
does not publish its processed county-level dates or its handling of states
without a statewide order.

Run `python3 build_li_histogram.py` from this folder to regenerate the CSVs,
summary, and figures.
