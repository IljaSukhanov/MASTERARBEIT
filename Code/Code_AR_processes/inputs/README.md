# Shared AR coefficient-recovery inputs

## Final coefficient-estimation experiment

[`ar_coefficient_estimation_final/<dataset-id>/`](ar_coefficient_estimation_final/)
holds the inputs prepared by the
[final estimation notebook](../xx_notebooks_estimation/ar_coefficient_estimation_final.ipynb).
Each default archive contains 500 Jones coefficient vectors and 2,500 original
float64 time series of length 1000: 100 vectors at each order p=1,...,5, with
the same vectors reused across Beta(1,1), Beta(0.5,0.5), Beta(2,2), Beta(1,2),
and Beta(2,1) innovations. Innovations have population mean zero and variance 1/3.

`dataset.npz` contains the numerical arrays, `models.csv` and
`series_metadata.csv` describe their identities and seeds, and `manifest.json`
records the design, generator fingerprint, versions, and file checksums.
Reruns verify and reload the saved data, including when the NumPy version
differs; corrupted inputs raise an error instead of being regenerated.
Changed designs are saved in a separate directory.
See [the shared implementation](../ar_tools/final_coefficient_estimation.py).

## Historical AR(3) inputs

`ar3_coefficient_recovery_50.npz` contains only the 50 simulated series, their
generating AR(3) coefficients, and the 12 starting directions per series used by
the retained restart comparisons. It contains no estimator results or descent
trajectories. These samples follow the 50-process seed family in
the Monte Carlo coefficient-recovery section of the removed notebook
`ar3_minimum_width_k4.ipynb`. Its source checksum is retained in
[coefficient_recovery_inputs.py](../ar_tools/coefficient_recovery_inputs.py).

The retained experiments load this file through
[coefficient_recovery_inputs.py](../ar_tools/coefficient_recovery_inputs.py).
Each sample is checked against its original binary SHA-256 checksum.

On 20 October 2026 the shared input values were separated from the removed
stationarity-filtering experiment. Reconstructing the seeds in the current
runtime agreed within `4.45e-14` for observations and `4.45e-16` for generating
coefficients, and reproduced every starting direction exactly. Keeping the
original input values avoids changing the existing fits because of rounding.
The cleanup verification is recorded in
[verification.json](../../../output/analysis/ar_notebook_cleanup_2026-10-20/verification.json).
