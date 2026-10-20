# Detailed examples of the 200-iteration checkpoint

The plotted quantity is the companion spectral radius of the **best-width incumbent's recovered coefficients** at each evaluated iteration. The boundary used by this experiment is radius 1. This parameter criterion and companion-root equivalence are documented in [statsmodels stationarity](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_process.ArmaProcess.isstationary.html) and [is_invertible, Notes](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.tools.is_invertible.html#notes). These curves describe the recorded experiment, not a claim about every possible AR fit.

## Aborted although it would recover: process 8, start 11

The radius is 1.198024540 at evaluated iteration 200, so the rule aborts. The saved ordinary trajectory becomes stationary at iteration 371 and remains stationary thereafter, ending at radius 0.955127999 at iteration 2000. Its final coefficient MSE is 0.000389350568. Only the solid orange segment was followed by the early-abort policy; the dashed blue continuation shows the ordinary run without that abortion.

![Aborted but later recovered](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/images/ar3_early_abort_200_12starts/03_aborted_run_that_recovers.png)

[PDF](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/pdfs/ar3_early_abort_200_12starts/03_aborted_run_that_recovers.pdf)

## Passed but later became nonstationary: process 43, start 2

The radius is 0.931681260 at iteration 200, so the attempt continues. At iteration 256 a new best-width incumbent is nonstationary. The best incumbent remains nonstationary through the final evaluated iteration 494, with radius 1.008298089. This attempt is the triggered method's selected result for process 43. The entire teal curve was followed by the policy.

![Passed but later nonstationary](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/images/ar3_early_abort_200_12starts/04_passed_run_that_becomes_nonstationary.png)

[PDF](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/pdfs/ar3_early_abort_200_12starts/04_passed_run_that_becomes_nonstationary.pdf)

The plots also show each true parameter radius as a black dashed horizontal line. Full and zoomed views use the same data; the full recovery plot uses a symmetric-log scale (linear below 1), while the other views use linear scales. Initial state is evaluated iteration 1, so the checkpoint is stored index 199. The final coefficient MSE of the nonstationary second example is 0.000479349827; parameter validity and coefficient error are distinct recorded diagnostics here.

Sources: [all plotted iterations](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/csv/ar3_early_abort_200_12starts/checkpoint_counterexamples_history.csv), [exact event values](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/csv/ar3_early_abort_200_12starts/checkpoint_counterexamples.json), [record 8](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/csv/ar3_early_abort_200_12starts/replications/process_008.json), [record 43](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/csv/ar3_early_abort_200_12starts/replications/process_043.json), [plot implementation](/Users/iljasukhanov/MASTERARBEIT/output/analysis/ar3_early_abort_200_12starts/plot_counterexamples.py). The plotted roots were independently cross-checked against characteristic-polynomial roots at every saved iteration. Existing fits and checkpoint decisions were not changed.
