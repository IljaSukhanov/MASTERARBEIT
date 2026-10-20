# Fixed 12 starts versus stationarity-triggered starts

This corrects the earlier 5000-iteration, no-patience, 100-iteration-checkpoint experiment. The user specified `max_it=2000`, `improv_patience=200`, and checking stationarity **after each run ends**. There is no filtering of intermediate candidates in either mode.

## Prespecified comparison

- **Fixed 12 starts:** run all 12 original random directions and return the smallest-width completed-run result, even if it is nonstationary.
- **Stationarity-triggered starts:** run those same directions in their original order; inspect each completed run's best-width direction. Return the first stationary completed-run result. Restart from the next random direction only after a nonstationary result. Cap at 12 attempts (initial plus at most 11 restarts).
- Every attempt uses 2000 maximum evaluated states, stopping after 200 consecutive states without an incumbent-width improvement of more than `1e-10`. Step sizes reset on each attempt: `gamma_t=(0.2/diameter(P))*0.997**t`.
- If all 12 run results were nonstationary, mark the conditional policy failed and retain the unrestricted best result for error accounting. This did not occur in these 50 paths.
- The same 50 paths, true parameters, and ordered starting directions are loaded from `Code/Code_AR_processes/inputs/ar3_coefficient_recovery_50.npz`. Their original floating-point values are checked against sample hashes and independently against seed reconstruction. The true coefficients enter error reporting only.
- The conditional policy consumes a prefix of the fixed policy's attempted runs. Sharing those deterministic runs avoids duplicate computation in this paired analysis; reported costs count only the iterations the respective policy would consume. They are not wall-clock speed measurements.

The parameter validity criterion is companion spectral radius below 1, equivalent to the AR lag polynomial having roots outside the unit circle; see [statsmodels ArmaProcess.isstationary](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_process.ArmaProcess.isstationary.html) and the companion-root equivalence in [is_invertible, Notes](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.tools.is_invertible.html#notes). The implementation is in [run_experiment.py](/Users/iljasukhanov/MASTERARBEIT/output/analysis/ar3_conditional_restarts_2000_200/run_experiment.py), and the existing stopping behavior is in [descent.py](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/ar_tools/descent.py).

## Results across all 50 processes

| Measure | Fixed 12 starts | Stationarity-triggered starts |
|---|---:|---:|
| Stationary returned estimates | 46/50 | 50/50 |
| Mean coefficient MSE | 218.295726 | 0.0305072159 |
| Median coefficient MSE | 1.52543372e-05 | 3.84080411e-05 |
| Maximum coefficient MSE | 7680.88024 | 0.702085609 |
| Processes with MSE > 1 | 4 | 0 |
| Mean attempts | 12 | 1.66 |
| Total evaluated iterations | 600,005 | 78,908 |

Per process, the conditional policy improved MSE in 13, worsened it in 32, and tied in 5 cases (absolute tolerance `1e-10` plus relative tolerance `1e-6`). It consumed 86.85% fewer evaluated iterations. MSE means the mean squared error of the three coefficients, averaged across all 50 processes for the aggregate. All 50 processes remain in every summary.

These observed results show a tradeoff: the conditional rule avoids the four nonstationary outputs and reduces computation, but its first acceptable result often has less accurate coefficients than the result from 12 attempts. Stationary validity alone does not establish coefficient accuracy; the observed maximum MSE in the conditional mode was 0.702086. This interpretation is based on this paired experiment, not a general performance guarantee.

Fixed-mode nonstationary cases:

| replication | fixed_mse | conditional_mse | fixed_stationary | conditional_attempts | fixed_evaluations | conditional_evaluations |
| --- | --- | --- | --- | --- | --- | --- |
| 27 | 313.449134 | 0.269066214 | False | 6 | 10287 | 3518 |
| 35 | 7680.88024 | 0.000243088586 | False | 1 | 20701 | 2000 |
| 41 | 386.322549 | 0.0741436731 | False | 1 | 10354 | 894 |
| 50 | 2534.12517 | 0.392961295 | False | 4 | 7371 | 1154 |

Conditional estimates with coefficient MSE > 0.01:

| replication | selected_attempt | attempts_used | coefficient_mse | spectral_radius |
| --- | --- | --- | --- | --- |
| 5 | 2 | 2 | 0.702085609 | 0.925460471 |
| 50 | 4 | 4 | 0.392961295 | 0.915066848 |
| 27 | 6 | 6 | 0.269066214 | 0.991814066 |
| 43 | 1 | 1 | 0.079045668 | 0.991527393 |
| 41 | 1 | 1 | 0.0741436731 | 0.913427689 |

## Relationship to filtering

Filtering was an additional estimator in the earlier exploratory comparison: it retained the smallest-width **stationary intermediate iterate** across all attempted directions. That can rescue a discarded earlier candidate even if a run ends with a better-width nonstationary incumbent. The comparison here never does that. It only checks the result of a completed attempt to decide whether another attempt is required. Both modes still use the original width objective and return an actual completed run's unrestricted best-width direction.

Filtering intermediate iterates is outside the two-policy comparison here.

## Reproducibility and verification

[Configuration](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/csv/ar3_conditional_restarts_2000_200/configuration.json), [summary](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/csv/ar3_conditional_restarts_2000_200/summary.csv), [all 100 policy results](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/csv/ar3_conditional_restarts_2000_200/per_process_results.csv), [600 terminal attempt results](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/csv/ar3_conditional_restarts_2000_200/per_attempt_diagnostics.csv), [paired comparison](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/csv/ar3_conditional_restarts_2000_200/paired_process_comparison.csv), [validation](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/csv/ar3_conditional_restarts_2000_200/validation.json), [PDF figure](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/pdfs/ar3_conditional_restarts_2000_200/comparison.pdf).

Independent checks recompute all 600 widths and coefficient ratios on their original clouds, check spectral radii using characteristic-polynomial roots, verify all 100 selections and iteration counts, and inspect 48 complete traces including the 200-state patience condition. See [verification implementation](/Users/iljasukhanov/MASTERARBEIT/output/analysis/ar3_conditional_restarts_2000_200/verify_and_report.py). The thesis sources and original experiment notebook are untouched.
