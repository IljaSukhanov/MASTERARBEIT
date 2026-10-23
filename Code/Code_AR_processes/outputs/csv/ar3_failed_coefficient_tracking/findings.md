# Inside a failed AR(3) coefficient recovery

Fixed process: true coefficients [0.5, -0.2, 0.1], T=1000, burn-in=1000, Uniform[-1,1] innovations, process seed 20261018. Original random direction seed 20261019; geometric schedule gamma0=0.1, beta=0.997; 5000 evaluated states.

**Finding:** the tracked coefficients become very large while the accepted width continues decreasing. At iteration 8, the coefficient error of the best-width estimate is 5.004268. At iteration 25, a smaller accepted width comes with error 178.267625. The first-coordinate magnitude at that update is 0.00562669. Source: [diagnostic landmarks](diagnostic_milestones.csv) and [accepted width updates](failed_best_updates.csv).

| Iteration | Best width | Signed first coordinate | phi_1 | phi_2 | phi_3 | Coefficient L2 error |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3.00594201 | 0.04690430 | 17.30250 | -6.66939 | -10.47309 | 20.87982 |
| 8 | 2.07142247 | 0.17757380 | 4.63620 | -2.91995 | 0.83240 | 5.00427 |
| 25 | 1.92619331 | -0.00562669 | -159.08660 | 75.63889 | -23.56277 | 178.26762 |
| 4999 | 1.86295244 | 0.03211422 | 27.49300 | -14.01412 | 4.04570 | 30.57811 |

Of 73 accepted width improvements, 32 increase coefficient error relative to the generating vector. This is a diagnostic count from [the accepted updates](failed_best_updates.csv), not a criterion used by the algorithm.
## End of run and optimization reference
The final best width is 1.862952439466, 6.599915% above the global floating-point reference 1.747611555746. The final tracked coefficients are (27.49300123, -14.01412340, 4.04569640), with error 30.57810722 and |u0|=0.03211422. The acute angle to the reference direction is 59.5201 degrees. Sources: [summary](summary.csv), [configuration](configuration.json), [global reference implementation](../../../ar_tools/lp_widths.py), exact_empirical_width_low_dimension.
The same-sample global reference gives coefficients (0.50066818, -0.20307294, 0.10251497). The zero-start control gives (0.50066821, -0.20307305, 0.10251509), with coefficient error 0.00402689. This supports the diagnosis of an unsuccessful direction search in the failed run, rather than a claim that this sample cannot recover the coefficients. Sources: [summary](summary.csv) and [configuration](configuration.json).
## Inside the tracking
The closest coefficient vector among all visited current directions occurs at iteration 26, with error 3.486295. Its width is 2.10299673, above the then-best 1.92619331, so the best-width tracker does not accept it. It is still far from the generating coefficients. This landmark is identified after fitting using the true vector and is not proposed as an estimator. Source: [all recorded states](iteration_history.csv).
For a unit direction (a,b), the ratio phi=-b/a has norm sqrt(1-a^2)/|a|. Its differential is -(delta b + phi delta a)/a. These formulas explain why small first coordinates produce large estimates and amplify perturbations. They are our direct algebraic derivations in the removed notebook `ar3_failed_coefficient_tracking.ipynb`, section "What happens at the accepted improvements?"; the norm identity is checked numerically. The ratio is implemented in [descent.py](../../../ar_tools/descent.py). A full sign reversal of a direction leaves both its width and recovered coefficients unchanged; the singularity concerns a zero first coordinate.
The tracker is passive: all recorded directions and widths were checked for exact equality with the same run without coefficient tracking. The saved run reproduces the earlier failure. Sources: the removed executed notebook `ar3_failed_coefficient_tracking.ipynb` and [earlier failure](../ar3_best_coefficient_tracking/summary.csv).
## Limits
One fixed path and two fixed starts. The global reference uses floating-point convex hulls. The diagnostics do not certify that the failed final direction is a local minimizer or prove convergence. No estimate, step or start is selected by true coefficient error. No claim about failure frequency across processes or initializations is made.
