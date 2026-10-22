# Shared AR coefficient-recovery inputs

`ar3_coefficient_recovery_50.npz` contains only the 50 simulated series, their
generating AR(3) coefficients, and the 12 starting directions per series used by
the retained restart comparisons. It contains no estimator results or descent
trajectories. These samples follow the 50-process seed family in
[ar3_minimum_width_k4.ipynb](../ar3_minimum_width_k4.ipynb), its Monte Carlo
coefficient-recovery section.

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
