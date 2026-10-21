# Experiment 4 compared with the original fixed random direction

## Matched design

Reuse the exact same 100 Jones-sampled coefficient vectors and the same innovation paths from Experiment 3 (T=500, burn-in 2000, coefficient seed 20261023, process seed 20261024). Compare e1=(1,0,0,0) with the **one fixed Gaussian-generated direction** used throughout Experiment 3:

`[0.04690430049805569, -0.8115615349421885, 0.31282302194507183, 0.4912327539346498]`

Both starts use the same 9 scales, 8 half-lives, 2000-evaluation cap, patience 200, zero width-improvement tolerance and **no stationarity-triggered restarts**. The random-start normalized grid was run separately for this comparison: the older normalized notebook section enabled restarts and is therefore not a comparison of initialization alone. There are 14,400 new random-start fits, paired with the 14,400 already completed e1 fits. All results retain every model.

Sources: [notebook](../../../Code/Code_AR_processes/xx_notebooks_tuning/ar_tuning_experiments.ipynb), [shared grid runner](../../../Code/Code_AR_processes/ar_tools/tuning.py), [optimizer](../../../Code/Code_AR_processes/ar_tools/descent.py), [coefficient MAE](../../../Code/Code_AR_processes/ar_tools/coefficient_errors.py).

## Same schedule, different initialization

Use the width-count-leading schedules selected in Experiment 4, and hold each schedule fixed for both starts. For absolute steps, scale is gamma0; for normalized steps, gamma0=scale/L. MAE averages the absolute errors of the three coefficients. The choice of these schedules used these same 100 models; this is a tuning-sample comparison, not independent validation.

| rule | initialization | scale | H | mean_mae | median_mae | max_mae | nonstationary | undefined | mean_evaluations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| absolute | fixed_random | 0.1 | 50 | 9.5336758 | 0.073195546 | 215.36733 | 35 | 0 | 1831.78 |
| absolute | e1 | 0.1 | 50 | 0.95164005 | 0.012807959 | 35.289122 | 10 | 0 | 1913.37 |
| diameter_normalized | fixed_random | 0.2 | 50 | 7.5857575 | 0.053039095 | 215.36733 | 33 | 0 | 1991.21 |
| diameter_normalized | e1 | 0.2 | 50 | 0.0472194 | 0.010249403 | 0.78399975 | 0 | 0 | 2000 |

Paired outcomes at these same settings (MAE ties within 1e-12, width ties within 1e-8):

| rule | scale | H | e1_lower_mae | random_lower_mae | mae_ties | e1_narrower | random_narrower | width_ties |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| absolute | 0.1 | 50 | 58 | 12 | 30 | 49 | 18 | 33 |
| diameter_normalized | 0.2 | 50 | 58 | 14 | 28 | 58 | 13 | 29 |

## Equal tuning budget for each initialization

Independently select every configuration with the largest width co-winner count within its own 72-configuration rule-specific grid. A width co-win means achieved width <= the model's best achieved grid width + 1e-8. These counts have different reference minima across initializations, so they are not head-to-head success rates. The coefficient errors and nonstationarity counts describe the resulting selected schedules on the same 100 models.

| rule | initialization | scale | H | width_cowins | mean_mae | median_mae | nonstationary |
| --- | --- | --- | --- | --- | --- | --- | --- |
| absolute | fixed_random | 0.3 | 50 | 45 | 15.876006 | 0.030562275 | 37 |
| absolute | e1 | 0.1 | 50 | 52 | 0.95164005 | 0.012807959 | 10 |
| diameter_normalized | fixed_random | 0.4 | 50 | 47 | 8.8632766 | 0.047400144 | 30 |
| diameter_normalized | fixed_random | 0.5 | 50 | 47 | 8.3982487 | 0.047400144 | 31 |
| diameter_normalized | e1 | 0.2 | 50 | 56 | 0.0472194 | 0.010249403 | 0 |

## Model-wise lowest-width diagnostics

This chooses a different schedule for each model and is not one common deployable schedule:

| rule | initialization | nonstationary | mean_mae | median_mae | worse_than_feasible_true_direction |
| --- | --- | --- | --- | --- | --- |
| absolute | fixed_random | 13 | 4.7415947 | 0.01326893 | 16 |
| absolute | e1 | 7 | 2.6104429 | 0.011715263 | 4 |
| diameter_normalized | fixed_random | 20 | 10.789209 | 0.017292396 | 18 |
| diameter_normalized | e1 | 3 | 0.22503146 | 0.0087735296 | 0 |

## Verification and limits

- Exact coefficient vectors, partial autocorrelations, seeds and reference widths match between both initializations for each step rule.
- All 14,400 newly computed widths were independently recomputed on the original paired point clouds; maximum absolute discrepancy: 7.1054273576010019e-15.
- The fresh absolute random-start grid differs from the historical saved Experiment 3 model-wise minima beyond the saved precision on 24/100 paths (maximum absolute difference 0.04566690557). The cause has not been established. This comparison uses the fresh baseline under the exact same current shared implementation and runtime as Experiment 4; it does not mix the older historical scores with the new e1 scores. The original historical notebook outputs are preserved.
- Every run uses one initial direction and at most 2000 evaluations. No restart or stationarity filter changes the width scores.
- All 27 protected thesis files are unchanged.
- This compares e1 with one particular fixed random direction, not a distribution of independently resampled random directions. It does not compare against best-of-many random starts.
