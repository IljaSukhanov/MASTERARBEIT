# Twelve attempts with and without a stationarity check at iteration 200

## Frozen experiment

Use the same 50 coefficient-recovery AR(3) paths, same ordered bank of 12 random starting directions for each path, same geometric schedule `gamma_t=(0.2/diameter(P))*0.997**t`, `max_it=2000`, improvement patience 200, and width-improvement tolerance `1e-10`.

Both modes use **all 12 attempts** (the initial attempt plus 11 restarts). The comparison changes one attempt's stopping rule:

- Ordinary mode: each attempt stops at the iteration cap or after 200 consecutive evaluations without sufficient width improvement.
- Triggered mode: after exactly 200 evaluated states, recover phi from that attempt's **best-width incumbent**, and abort if the coefficients are undefined or the companion spectral radius is at least 1. Otherwise continue exactly the original trajectory, patience counter, and geometric schedule. There are no subsequent stationarity checks in this experiment.
- Aborted attempts are discarded from the final comparison. Among completed attempts, choose the smallest width **without a final stationarity filter**. Do not retain the best stationary intermediate iterate. If all 12 attempts were aborted, flag failure and retain the smallest-width aborted prefix only for explicit error accounting; this did not occur.
- A new attempt uses the next original random direction and resets its own step schedule and patience. The true coefficients never enter the abort or selection decision.

Iteration counts follow the existing optimizer: the initial state is evaluated iteration 1, and the check occurs after evaluation 200 (zero-based stored index 199), before another gradient update. Both modes have the same number of attempts and per-attempt cap; actual computation differs because early abortion is the purpose of the trigger. The reported savings concern evaluated iterations, not measured wall-clock speed.

This is a new exploratory test of the user's 200-iteration checkpoint, rather than the previous end-of-run stop-at-first-stationary policy or intermediate-iterate filtering experiment. The checkpoint was not tuned after seeing these results.

Implementation: [run_experiment.py](/Users/iljasukhanov/MASTERARBEIT/output/analysis/ar3_early_abort_200_12starts/run_experiment.py). The checkpoint runner is kept local to this experiment; shared optimizer defaults and all earlier outputs are preserved. Every triggered trajectory is checked against the existing [public optimizer](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/ar_tools/descent.py): a rejected attempt is its exact first 200 states and a surviving attempt is its exact full trajectory.

The parameter check uses companion spectral radius below 1; equivalently the autoregressive polynomial has its roots outside the unit circle. See [statsmodels ArmaProcess.isstationary](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_process.ArmaProcess.isstationary.html) and the reciprocal-root equivalence in [is_invertible, Notes](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.tools.is_invertible.html#notes).

## Results on all 50 paths

| Measure | Ordinary 12 starts | 12 starts with abort at 200 |
|---|---:|---:|
| Stationary returned estimates | 46/50 | 49/50 |
| Mean coefficient MSE | 218.295726 | 0.00660970123 |
| Median coefficient MSE | 1.52543372e-05 | 1.52543372e-05 |
| Maximum coefficient MSE | 7680.88024 | 0.269066214 |
| Processes with MSE > 1 | 4 | 0 |
| Total attempts | 600 | 600 |
| Attempts aborted at 200 | 0 | 179 |
| Total evaluated iterations | 600,005 | 432,283 |

MSE is the mean of the three squared coefficient errors for a process, then averaged across all 50 processes for the aggregate. None of the 50 paths was removed.

The triggered rule improved coefficient MSE on 6 paths, worsened it on 4, and tied on 40 (absolute tolerance `1e-10` plus relative tolerance `1e-6`). It saved 27.95% of iteration evaluations.

The mean error reduction is driven by avoiding the four original MSE-above-1 cases. The median is unchanged. These are observed results on the selected 50 paths, not a general guarantee.

Changed estimates:

| replication | fixed_mse | triggered_mse | fixed_stationary | triggered_stationary | fixed_evaluations | triggered_evaluations | aborted_attempts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 7.70859934e-05 | 8.09175941e-05 | True | True | 18023 | 4200 | 11 |
| 8 | 0.000389350568 | 0.00292214817 | True | True | 14574 | 2595 | 11 |
| 24 | 6.25930666e-05 | 6.02601942e-05 | True | True | 6210 | 5443 | 2 |
| 25 | 1.84464489e-06 | 3.16209558e-06 | True | True | 18637 | 5564 | 10 |
| 27 | 313.449134 | 0.269066214 | False | True | 10287 | 2760 | 11 |
| 32 | 2.2986456e-05 | 2.3399331e-05 | True | True | 23511 | 9600 | 8 |
| 35 | 7680.88024 | 0.00023944953 | False | True | 20701 | 9901 | 6 |
| 41 | 386.322549 | 0.00129831413 | False | True | 10354 | 6224 | 7 |
| 43 | 0.00667712053 | 0.000479349827 | True | False | 18097 | 2694 | 11 |
| 50 | 2534.12517 | 0.0543635474 | False | True | 7371 | 3331 | 7 |

## What happens inside the trigger

Of 179 aborted attempts, **47 would later return a stationary best estimate** if allowed to continue. The rule aborted 10 original winning starts, including 6 whose eventual best estimate would be stationary. These counterfactual continuation outcomes are diagnostics only; they were unavailable to the policy at the checkpoint.

A stationarity failure at 200 is therefore a heuristic for rejecting an unpromising direction. It is not sufficient evidence that the trajectory cannot recover. Conversely, passing the check does not ensure the later best-width incumbent will remain stationary: 1 passed attempts ended nonstationary. The triggered policy returned a nonstationary estimate on process 43: its selected start passed at 200 and later changed incumbent.

[Checkpoint examples](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/images/ar3_early_abort_200_12starts/02_checkpoint_examples.png) show an aborted failing direction, an aborted direction that would recover, and the later nonstationarity after a passed checkpoint. [Aborted but later recovered attempts](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/csv/ar3_early_abort_200_12starts/aborted_runs_that_would_recover.csv) and [passed but later nonstationary attempts](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/csv/ar3_early_abort_200_12starts/passed_checkpoint_but_nonstationary_final.csv) provide the complete diagnostics.

## Verification and outputs

[Configuration](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/csv/ar3_early_abort_200_12starts/configuration.json), [summary](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/csv/ar3_early_abort_200_12starts/summary.csv), [100 policy results](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/csv/ar3_early_abort_200_12starts/per_process_results.csv), [600 start diagnostics](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/csv/ar3_early_abort_200_12starts/per_start_diagnostics.csv), [paired comparison](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/csv/ar3_early_abort_200_12starts/paired_comparison.json), [independent validation](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/csv/ar3_early_abort_200_12starts/validation.json), [comparison PDF](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/pdfs/ar3_early_abort_200_12starts/01_comparison.pdf), [checkpoint PDF](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/pdfs/ar3_early_abort_200_12starts/02_checkpoint_examples.pdf).

The independent audit checks all 600 prefix decisions using polynomial roots, all 1200 saved candidate widths and coefficient errors, all 1200 trajectories' stopping counts, all selected estimates, and both modes' iteration costs. The implementation separately verified all 600 policy trajectories against the public optimizer and reproduced all 600 previous baseline results exactly. See [verification code](/Users/iljasukhanov/MASTERARBEIT/output/analysis/ar3_early_abort_200_12starts/verify_and_report.py).
