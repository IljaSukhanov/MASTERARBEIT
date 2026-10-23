# Frozen sample for the final order-selection experiment

This subset contains 250 original series: 50 sampled without replacement from
all 500 saved series at each true order p=1,...,5. All five Beta shapes are
pooled within each order. Shape counts are random, not fixed quotas.
The root seed is 2026102305, with a separate PCG64 stream seeded by
`SeedSequence([root_seed, order])` for each order. The exact selected source
IDs are retained, so future runs can load this sample without sampling again.

## Files and identities

- `dataset.npz`: `series` (250 x 1000), a compact `coefficients` bank,
  `source_sample_ids` (250), and `orders` (250). Load with `allow_pickle=False`.
- `series_metadata.csv`: one row per saved series. `subset_index` is its
  zero-based array row; `sample_id` is its new one-based ID. `source_sample_id`
  identifies the original coefficient-study series. `model_index` addresses
  the compact coefficient bank; `source_model_index` identifies the original
  bank row. The original shape, seeds, support, and series checksum are kept.
- `models.csv`: the coefficient vectors used by the sample, with compact and
  source model indices. Only the first `order` entries of a coefficient vector
  are active; later coordinates are zero padding.
- `shape_counts.csv`: observed shape counts and distinct coefficient-vector
  counts per true order.
- `manifest.json`: source checksums, seed, selection procedure, selected IDs,
  output checksums, and verification results.

The same coefficient vector can occur under multiple shapes because the
sampling unit is an individual saved series. Each source series is selected
at most once. Every observation and coefficient is copied exactly from
`../../ar_coefficient_estimation_final/1b5bb834ea4e8cc9/`; no process is simulated.
The source archive stays unchanged. Order-selection fits have not been run.

## Loading

```python
from pathlib import Path
import numpy as np
import pandas as pd

folder = Path("Code/Code_AR_processes/inputs/ar_order_selection_final/030cde6f724e3c30")  # from the project root
with np.load(folder / "dataset.npz", allow_pickle=False) as saved:
    series = saved["series"].copy()
    coefficients = saved["coefficients"].copy()
metadata = pd.read_csv(folder / "series_metadata.csv", float_precision="round_trip")
# Series row i has metadata.iloc[i]; its true coefficient vector is
# coefficients[int(metadata.iloc[i].model_index), :int(metadata.iloc[i].order)].
```
