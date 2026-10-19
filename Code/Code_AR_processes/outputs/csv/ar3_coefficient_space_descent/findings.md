# Coefficient-space descent on the original AR(3) width

This is our derivation and numerical experiment on one fixed path.

Process: phi=[0.5, -0.2, 0.1], T=1000, burn-in=1000, innovations Uniform[-1,1], process seed=20261018.

**Finding:** the tested Euclidean coefficient updates do not improve the attained original width in this example. From zero, both methods reach the floating-point global reference to within 1.1e-14. The recovered coefficients are approximately (0.50066818, -0.20307294, 0.10251497). From the earlier random direction, the best grid gaps are 6.5999% for sphere descent and 7.2303% for Euclidean coefficient descent.

## Objective and update

The objective always retains its denominator: F(phi)=range(y-X phi)/sqrt(1+||phi||^2). Applying inverse radial projection to every sphere step is algebraically a change of coordinates. Ordinary Euclidean phi descent changes the step geometry while keeping this same objective. The derivation and both updates are in [the notebook, "The objective and the two coefficient updates"](../../../ar3_coefficient_space_descent.ipynb) and [coefficient_descent.py](../../../ar_tools/coefficient_descent.py).

Global empirical width reference (floating-point difference-body hull): 1.747611555746; associated coefficients: [0.5006681806450801, -0.2030729406195368, 0.10251496982549076]. Implementation: [lp_widths.py](../../../ar_tools/lp_widths.py), exact_empirical_width_low_dimension.

## Same schedule: gamma0=0.1, beta=0.997, 5000 evaluated states

| Starting point | Method | gamma0 | beta | Width | Gap to global reference | Coefficient error |
|---|---|---:|---:|---:|---:|---:|
| Earlier random direction | Sphere | 0.1 | 0.997 | 1.8629524383 | 6.59992% | 30.57811049 |
| Earlier random direction | Sphere expressed in phi | 0.1 | 0.997 | 1.8629524383 | 6.59992% | 30.57810801 |
| Earlier random direction | Euclidean phi | 0.1 | 0.997 | 2.8085454894 | 60.7077% | 20.83209791 |
| Zero coefficients | Sphere | 0.1 | 0.997 | 1.7476115640 | 4.70463e-07% | 0.00402689 |
| Zero coefficients | Sphere expressed in phi | 0.1 | 0.997 | 1.7476115654 | 5.51266e-07% | 0.00402689 |
| Zero coefficients | Euclidean phi | 0.1 | 0.997 | 1.7476115617 | 3.37971e-07% | 0.00402686 |

## Best of 45 schedules per method and starting point

| Starting point | Method | gamma0 | beta | Width | Gap to global reference | Coefficient error |
|---|---|---:|---:|---:|---:|---:|
| Earlier random direction | Sphere | 0.03 | 0.99 | 1.8629524361 | 6.59992% | 30.57810746 |
| Earlier random direction | Euclidean phi | 0.3 | 0.999 | 1.8739691010 | 7.2303% | 20.87404433 |
| Zero coefficients | Sphere | 0.003 | 0.995 | 1.7476115557 | 6.43929e-13% | 0.00402672 |
| Zero coefficients | Euclidean phi | 0.003 | 0.995 | 1.7476115557 | 4.44089e-13% | 0.00402672 |

Each pair uses the same 5000-state budget. Selection uses only the original width; generating coefficients are used only for diagnostic errors. Sources: [schedule grid](schedule_grid.csv), [selected results](selected_results.csv), [configuration](configuration.json).

## Coordinate check

The largest projective round-trip error is 3.78e-16. For independently run sphere and mapped-sphere updates, the largest difference between best widths is 1.41e-09. The largest discrepancy between current widths along these full numerical trajectories is 0.199377; active-set switches can separate floating-point trajectories despite the algebraic coordinate equivalence. Source: [coordinate equivalence checks](coordinate_equivalence_checks.csv).

## Limits

One fixed path, two starting points, geometric steps and a finite grid. A grid-edge winner leaves room for further tuning. Failure of these coefficient schedules does not exclude better schedules or other processes. The coefficient chart excludes u0=0, and small |u0| gives large initial coefficients. The earlier random start has |u0|=0.04690430 and phi0 approximately (17.3025,-6.6694,-10.4731). No stationarity constraint is imposed on the estimates. These limitations follow from the settings, coordinate formulas, and saved runs; they are not a general impossibility result.

## Sources

- Boyd, Duchi, Pilanci and Vandenberghe, [Subgradients (2022), sections 3.3–3.4, p. 7, and section 5, pp. 10–12](https://stanford.edu/class/ee364b/lectures/subgradients_notes.pdf): affine/max subgradient rules and the generalized-gradient interpretation. The experiment-specific substitution and gradient formulas are our derivation.

- [Original sphere optimizer](../../../ar_tools/descent.py), projected_subgradient_empirical_width; [simulator and newest-first embedding](../../../ar_tools/functions.py), simulate_uniform_ar and sliding_window_point_cloud.
