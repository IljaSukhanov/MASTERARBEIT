# One initial descent direction with conditional restarts versus Horváth L-infinity

## Exact policy used

Use the same original 50 coefficient-recovery AR(3) paths, each with 2000 observations after 1000 burn-in observations and Uniform[-1,1] innovations. The known order is 3. Both methods receive the same observations, with no true coefficients or innovation half-width supplied to fitting.

The descent uses **one initial random direction**. After 200 evaluated iterations, inspect the recovered coefficients of its best-width incumbent. If undefined or with companion spectral radius at least 1, discard that attempt and only then draw a new random direction. Reset the geometric schedule and patience for the new attempt. Each attempt uses `max_it=2000`, improvement patience 200, tolerance `1e-10`, and `gamma_t=(0.2/diameter(P))*0.997**t`.

Return the first attempt that passes the checkpoint and reaches its usual stopping point. Do not run additional directions to improve that result. Do not apply a final stationarity filter or restart if it becomes nonstationary after the single checkpoint. This implements the requested resampling-only-on-trigger rule exactly. There is an administrative 100-attempt safety guard; no process reached it (maximum actual attempts: 11). There is no 12-start allocation or cap.

The random stream is the original study's process-specific stream. Directions are generated one at a time, only when needed. The first 12 possible draws agree exactly with the earlier bank, but unused directions are not evaluated or drawn during the actual fit. Initial state is evaluated iteration 1; the check reads stored index 199. Both methods use the newest-first point cloud with 1997 rows.

## Horváth benchmark

The benchmark is the repository's existing **no-intercept** Horváth L-infinity fit:

$$\min_{\phi\in\mathbb{R}^3,\ h\geq0}h\quad\text{subject to}\quad|X_t-\phi_1X_{t-1}-\phi_2X_{t-2}-\phi_3X_{t-3}|\leq h.$$

Here only h is constrained nonnegative; the three coefficients are unrestricted. The residual centre is fixed at zero. This differs from the earlier free-centre Chebyshev benchmark. The implementation uses HiGHS and its default convex-hull constraint reduction, followed by residual feasibility checks on all observations and a primal/dual objective check. It imposes no stationarity constraint. Source: [classical_selectors.py, horvath_linf_coefficients](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/ar_tools/classical_selectors.py:204). Original reference: [Horváth (1989), AR models with uniformly distributed noise, pp. 396–401, publication record](https://dml.cz/handle/10338.dmlcz/104367). The exact benchmark formulation above is verified from the workspace implementation; the paper's full PDF could not be retrieved through the browsing tool during this comparison. Solver interface: [SciPy linprog documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html).

Your method still minimizes normalized directional width on the sphere, whereas the LP minimizes maximum absolute residual. The comparison below evaluates coefficient recovery against the same true coefficients, rather than treating these different optimization objectives as numerically interchangeable. Sources: [descent implementation](/Users/iljasukhanov/MASTERARBEIT/output/analysis/ar3_early_abort_200_12starts/run_experiment.py) and the LP implementation linked above.

## Results across all 50 processes

| Measure | Your descent with conditional resampling | Horváth L-infinity |
|---|---:|---:|
| Mean coefficient MSE | 0.0289855353 | 7.02453578e-06 |
| Median coefficient MSE | 3.41462058e-05 | 1.69103513e-06 |
| Maximum coefficient MSE | 0.702085609 | 9.61169422e-05 |
| Stationary returned estimates | 49/50 | 50/50 |
| Mean fit time | 28.473 ms | 5.962 ms |
| Median fit time | 25.461 ms | 5.899 ms |

Horváth has lower coefficient MSE on 44 paths, your descent on 6, with 0 ties (comparison tolerance: absolute `1e-10` plus relative `1e-6`). MSE is the average of the three squared coefficient errors, then averaged over all 50 processes for the aggregate. No processes were excluded.

Your method used 98 actual attempts: 50 initial directions plus 48 triggered resamples. 14 processes needed resampling; 36 finished from their first direction. Mean attempts were 1.96, maximum 11, and mean evaluated iterations including aborted attempts were 1220.36.

The ratio of mean per-process fit times is 4.776:1 (descent / LP); the median of the paired time ratios is 4.270:1. These are local implementation measurements, not claims about intrinsic algorithmic complexity or performance on other data.

On these 50 paths, Horváth performed better in both aggregate coefficient accuracy and measured fit time. The first run that passes the 200-iteration stationarity checkpoint can still yield a less accurate coefficient vector. Passing also does not guarantee the returned vector remains stationary: the descent's nonstationary returned estimate is process 43. The 200-iteration trigger's two limitations were previously documented in [the checkpoint examples](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/csv/ar3_early_abort_200_12starts/checkpoint_counterexamples.md).

Processes where descent has lower coefficient MSE:

| replication | descent_mse | horvath_mse | descent_fit_ms | horvath_fit_ms | descent_attempts | descent_stationary | horvath_stationary |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 5.2568614e-07 | 1.1980144e-06 | 32.393083 | 5.589459 | 1 | True | True |
| 6 | 2.2639439e-06 | 1.19674022e-05 | 24.860375 | 5.796709 | 2 | True | True |
| 11 | 2.15715567e-06 | 2.5622293e-06 | 31.443583 | 6.01179099 | 1 | True | True |
| 13 | 4.37541638e-06 | 9.44198595e-06 | 35.886625 | 5.66987501 | 1 | True | True |
| 14 | 8.01491857e-06 | 8.07689548e-06 | 42.257375 | 6.39870801 | 1 | True | True |
| 34 | 7.02010461e-07 | 2.16040546e-06 | 41.089916 | 5.63170799 | 3 | True | True |

Largest descent errors:

| replication | descent_mse | horvath_mse | descent_fit_ms | horvath_fit_ms | descent_attempts | descent_stationary | horvath_stationary |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 0.702085609 | 1.26571167e-06 | 23.924333 | 5.55062499 | 2 | True | True |
| 50 | 0.392961295 | 5.24516411e-05 | 22.943417 | 6.72729099 | 4 | True | True |
| 27 | 0.269066214 | 3.48015826e-07 | 34.604125 | 5.83454201 | 6 | True | True |
| 41 | 0.0741436731 | 2.19872624e-07 | 22.872625 | 6.19783399 | 1 | True | True |
| 8 | 0.00292214817 | 1.16995339e-06 | 38.863959 | 5.914792 | 8 | True | True |

## Timing protocol and limits

Each method was warmed up three times, then timed five times per path with the method order alternating. Report the median fit time per path, then the mean and median of those 50 medians. Each timing starts with the in-memory series. Descent includes point-cloud construction, full-cloud diameter computation, conditional random draws, all aborted attempts, the surviving attempt, the checkpoint, and trace recording. Horváth includes its cloud/hull construction, LP assembly and solve, and full-data feasibility/optimality checks. File I/O, saving traces, true-error scoring, and plotting are excluded from both. Repeated timing fits use the same deterministic random stream; they do not add independent starting-direction experiments.

BLAS libraries were limited to one thread; HiGHS uses the existing estimator's solver defaults. Environment and versions are saved in the configuration. Only the requested settings and original 50 paths were tested. No additional tuning was performed.

## Files and verification

[Configuration](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/csv/ar3_single_start_restart_vs_horvath/configuration.json), [summary](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/csv/ar3_single_start_restart_vs_horvath/summary.csv), [all 100 results](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/csv/ar3_single_start_restart_vs_horvath/per_process_results.csv), [actual descent attempts](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/csv/ar3_single_start_restart_vs_horvath/descent_attempts.csv), [500 timing samples](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/csv/ar3_single_start_restart_vs_horvath/timing_measurements.csv), [paired comparison](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/csv/ar3_single_start_restart_vs_horvath/paired_process_comparison.csv), [coefficient RMSE](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/csv/ar3_single_start_restart_vs_horvath/coefficient_rmse.csv), [validation](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/csv/ar3_single_start_restart_vs_horvath/validation.json), [plot PDF](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/pdfs/ar3_single_start_restart_vs_horvath/comparison.pdf).

The experiment reproduces each consumed direction's previous trajectory exactly and verifies it returns the first nonaborted attempt. An independent audit checks all actual abort decisions, stopping counts, coefficient errors and root diagnostics; solves all 50 LPs independently on the complete observation set; verifies hull-reduced estimates and full-data objective/feasibility; and recomputes all timing medians. See [run_experiment.py](/Users/iljasukhanov/MASTERARBEIT/output/analysis/ar3_single_start_restart_vs_horvath/run_experiment.py) and [verification code](/Users/iljasukhanov/MASTERARBEIT/output/analysis/ar3_single_start_restart_vs_horvath/verify_and_report.py).
