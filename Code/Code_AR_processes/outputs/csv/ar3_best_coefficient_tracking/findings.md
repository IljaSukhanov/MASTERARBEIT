# Best-coefficient tracking during the unchanged AR(3) sphere descent

This implements the user-supplied algorithm with two corrections: update the best width along with the best direction and coefficients, and use gamma_t consistently in the tangent step.

Example: true phi=[0.5, -0.2, 0.1], T=1000, process seed=20261018, burn-in=1000; innovations Uniform[-1,1].

Schedule: gamma0=0.1, beta=0.997; 5000 evaluated states; no patience stopping; strict best-width comparison.

**Finding:** recording coefficients at each new best width leaves all recorded directions and widths exactly unchanged. The final tracked coefficients equal a single inverse projection of the returned best direction exactly for both starts.

| Starting point | Best width | phi_1 | phi_2 | phi_3 | Updates after initialization |
|---|---:|---:|---:|---:|---:|
| Zero coefficients | 1.7476115640 | 0.50066821 | -0.20307305 | 0.10251509 | 68 |
| Earlier random direction | 1.8629524395 | 27.49300123 | -14.01412340 | 4.04569640 | 73 |

The equality is an inference from the update itself and is also checked numerically: phi is never used to update a direction or choose a step. It is computed only when the best direction changes.
A zero first direction coordinate makes the coefficient estimate undefined; the implementation records NaN and continues the sphere search. It does not clip small first coordinates.
Sources: [implementation](../../../ar_tools/descent.py), projected_subgradient_empirical_width, record_best_coefficients; [notebook](../../../ar3_best_coefficient_tracking.ipynb), sections "Update being tested" and "Run the original descent with and without coefficient tracking"; [saved summary](summary.csv); [coefficient updates](coefficient_updates.csv); [all iterations](iteration_history.csv).
