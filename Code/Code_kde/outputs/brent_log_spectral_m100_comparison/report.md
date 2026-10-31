# Brent in log h versus spectral selection

Fresh timing run using the same 100 independent samples of 200 observations from N(0,1), with bandwidths in [0.01, 2]. The spectral selector uses geometric grids with N = 100, 200, 400 and lambda_N = N^1.9.

Each method was timed once per sample, giving 100 calls per method. Method order was randomized for each sample. Numerical libraries used one thread, and all four methods were initialized before timing. Each timed call prepared its own quantities from the raw sample. Spectral times include grid creation, full Fourier divergence-matrix construction, the stabilized Perron eigensolve, and peak selection. Brent times include pair-distance preparation, bounded minimization of Q_0(exp(z)), and comparison with the two endpoints.

Imports, initialization, sample generation, reference searches, validation, plotting, and saving were outside the timed calls. All recorded times are retained. SE means the sample standard deviation of the 100 times divided by sqrt(100). Timing variability was appreciable in this rerun; the medians are provided as supplementary information.

| Method | Mean time (ms) | SE of mean time (ms) | Median time (ms) | Mean selected h | Mean absolute relative bandwidth error (%) |
|---|---:|---:|---:|---:|---:|
| Brent in log h | 3.8523 | 0.3047 | 2.9339 | 0.27214918 | 0.00000098 |
| Spectral N=100 | 16.7567 | 1.4640 | 12.5146 | 0.26197061 | 5.79394777 |
| Spectral N=200 | 21.6763 | 0.9249 | 17.9968 | 0.26697326 | 2.96956214 |
| Spectral N=400 | 49.0901 | 2.3854 | 40.5315 | 0.26968822 | 1.54758400 |

Bandwidth errors are relative to the numerical minimum of Q_0 on the closed interval for each sample. The references passed scan-resolution checks; they are numerical references, rather than an exhaustive mathematical certification. Brent in log h matched all 100 references within an absolute log-bandwidth tolerance of 1e-6. For the spectral selector, 52/100, 82/100, and 95/100 selections were within 5% of the reference for N = 100, 200, and 400, respectively.

All 300 spectral peak indices were unchanged when the number of Fourier integration intervals was doubled. Three N=100 spot checks also agreed with analytic Gaussian overlaps. The maximum spectral relative residual was 1.138e-15.

[Detailed configuration](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/outputs/brent_log_spectral_m100_comparison/configuration.json) · [All 400 calls](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/outputs/brent_log_spectral_m100_comparison/results.csv) · [Aggregate statistics](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/outputs/brent_log_spectral_m100_comparison/aggregate_results.csv) · [Validation and summary](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/outputs/brent_log_spectral_m100_comparison/summary.json)

![Timing and paired bandwidths](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/outputs/brent_log_spectral_m100_comparison/brent_log_spectral_m100_comparison.png)

[Bounded Brent implementation](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/experiments/qhat0_interval_example/whole_interval_brent_comparison.py:27) · [Experiment implementation](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/experiments/qhat0_interval_example/brent_spectral_m100_comparison.py)
