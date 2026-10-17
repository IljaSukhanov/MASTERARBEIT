# Why the spectral run stopped early

These controlled reruns use the process and settings from the single-process coefficient notebook. Only the indicated patience, step multiplier, or seed construction changes. The notebook itself was not modified by this diagnostic.

The original spectral width is 1.6822386. Its first update increases the width to 2.7763717. With the original schedule, the first sufficiently improved width appears at evaluation 327; the patience counter stops the original run at evaluation 201.

| Spectral run | Best width | Coefficient MSE | Evaluations |
|---|---:|---:|---:|
| Patience 200 | 1.682238615 | 1.041327246 | 201 |
| Patience 500 | 1.307708924 | 0.000949169 | 7478 |
| Full 10,000 evaluations | 1.307708924 | 0.000949169 | 10000 |

Thus premature stopping is the main cause of the poor reported fitted spectral result in this instance. This is a controlled finding for the current process, not a general ranking of initialization methods.

There is also a distinction between initialization and the final objective. The sample spectral direction has projection variance 0.082797 and width 1.682239. The true AR recurrence direction has larger projection variance 0.142299 but smaller width 1.304045. These directly computed values show that least variance and least range order the two directions differently. The spectral initializer minimizes squared projections; see Tsakiris and Vidal (2018), §6.1 and Algorithm 1, https://www.jmlr.org/papers/volume19/17-436/17-436.pdf.

Centering the sample changes the spectral direction by only about 0.00248 degrees and does not prevent stopping at evaluation 201. The sample spectral direction is about 0.282 degrees from the population smallest-eigenvalue direction. The population covariance was computed from the AR state equation by solving the discrete Lyapunov equation; the maximum residual is 3.55e-15. These additional controls do not point to centering or sample eigenvector estimation as the main issue.

`comparison.csv` records all nine controlled runs, `width_traces.csv` their traces, and `diagnostic.json` the projection statistics and algebraic checks. The PNG and PDF illustrate the early stopping effect. The suggested next comparison is a common fixed evaluation cap, with patience disabled, to study initialization separately from this stopping rule.
