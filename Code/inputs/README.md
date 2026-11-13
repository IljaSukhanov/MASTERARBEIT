# KDE experiment inputs

`qhat_reference_sample.csv` is the fixed 200-observation sample used by the
energy notebook's local-minimum control and joint-limit comparisons. `qhat_sampling.json` records
its identity and the seeds, block construction and hashes needed to regenerate
the paired Gaussian sample bank. These are inputs, not stored estimates or
optimization results.

Load them through [`kde_tools.experiment_inputs`](../kde_tools/experiment_inputs.py).
Search results belong under `outputs/` and can be regenerated independently.
