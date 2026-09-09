# Preliminary degree-two harmonic results

All quantities use normalized sphere measure and the half-width harmonic convention.
The main curves reuse the three thesis realizations at T=2,000, with 98,304 pooled directions per dimension.

## At the known dimension k=p+1

- AR(1): min(Q2)/Q0 = -0.4185 population and -0.3973 empirical. Full raw-matrix error = 24.7%; normalized-matrix error = 8.2%. Empirical versus population Q2 minimizing directions differ by 1.91 degrees. The population Q2 direction differs from the AR recurrence normal by 11.03 degrees.
- AR(2): min(Q2)/Q0 = -0.5388 population and -0.5608 empirical. Full raw-matrix error = 31.5%; normalized-matrix error = 21.1%. Empirical versus population Q2 minimizing directions differ by 2.73 degrees. The population Q2 direction differs from the AR recurrence normal by 8.74 degrees.
- AR(3): min(Q2)/Q0 = -0.5542 population and -0.5227 empirical. Full raw-matrix error = 72.9%; normalized-matrix error = 40.2%. Empirical versus population Q2 minimizing directions differ by 3.16 degrees. The population Q2 direction differs from the AR recurrence normal by 10.13 degrees.

## Monte Carlo integration

- AR(1), k=2, M=32,768: population matrix RMSE/Q0 is 0.0142 for the direct rule and 0.0029 for the centered rule (5.0 times smaller RMSE).
- AR(2), k=3, M=32,768: population matrix RMSE/Q0 is 0.0284 for the direct rule and 0.0068 for the centered rule (4.2 times smaller RMSE).
- AR(3), k=4, M=32,768: population matrix RMSE/Q0 is 0.0555 for the direct rule and 0.0111 for the centered rule (5.0 times smaller RMSE).

## Sample-size effects

- AR(1), k=2: mean normalized-matrix error is 13.0% at T=500 and 9.9% at T=16,000. At the larger T, raw-matrix error remains 23.7% and mean direction error to population Q2 is 1.45 degrees.
- AR(2), k=3: mean normalized-matrix error is 11.0% at T=500 and 11.2% at T=16,000. At the larger T, raw-matrix error remains 33.8% and mean direction error to population Q2 is 2.06 degrees.
- AR(3), k=4: mean normalized-matrix error is 34.6% at T=500 and 22.9% at T=16,000. At the larger T, raw-matrix error remains 60.3% and mean direction error to population Q2 is 2.42 degrees.

Monte Carlo precision, finite-sample shape accuracy, and AR order recovery are separate questions. Centering addresses integration noise, and normalization removes multiplicative scale. Neither guarantees recovery of the population shape or a breakpoint at p+1.

Direction comparisons above use the known p+1. These are preliminary experiments on three fixed coefficient vectors, not an order-selection validation. Error bars in the sample-size study include both path variation and Monte Carlo variation. The empirical reference in the integration study is itself estimated numerically.
