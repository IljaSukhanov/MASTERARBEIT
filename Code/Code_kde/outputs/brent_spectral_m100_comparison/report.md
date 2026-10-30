# Bounded Brent versus the spectral selector

We compared the methods on 100 independent standard-Gaussian samples, each containing 200 observations. Every method used the same samples and the interval [0.01, 2]. The samples are the first 100 rows of the existing reproducible sample bank (seed 20260830).

Bounded Brent minimized the analytic sample energy Q_0 either in h or in z = log h, with xatol = 2e-12 in the corresponding coordinate. Both interval endpoints were evaluated and compared with the returned candidate. The spectral selector used geometric bandwidth grids with N = 100, 200, 400 and lambda_N = N^1.9; its divergence matrices used the shared Fourier quadrature and its eigenvectors used the shared stabilized solver.

Each method was initialized once before measurement. We then timed one fresh calculation per sample and method, randomizing the method order separately for every sample. Times include sample-specific preparation, including pair distances for Brent and fresh matrix construction plus eigensolution for spectral selection. Imports, sample generation, numerical reference searches, validation, plotting and saving were excluded. Numerical libraries used one thread.

The runtime reported below is the arithmetic mean over the 100 timed calls. Bandwidth means and standard deviations summarize the 100 separately selected bandwidths.

| Method | Mean time (ms) | SE of mean time (ms) | Mean selected h | SD of selected h | Mean absolute relative reference error (%) | Within 5% of reference |
|---|---:|---:|---:|---:|---:|---:|
| Brent in h | 3.003 | 0.048 | 0.279621 | 0.058168 | 5.57357 | 97/100 |
| Brent in log h | 2.758 | 0.040 | 0.272149 | 0.058612 | 9.78426e-07 | 100/100 |
| Spectral N=100 | 12.297 | 0.026 | 0.261971 | 0.050311 | 5.79395 | 52/100 |
| Spectral N=200 | 17.683 | 0.040 | 0.266973 | 0.053712 | 2.96956 | 82/100 |
| Spectral N=400 | 39.480 | 0.050 | 0.269688 | 0.055833 | 1.54758 | 95/100 |

Brent in log h agreed with the checked analytic interval reference on all 100 samples, using a log-bandwidth tolerance of 1e-6. Brent in h agreed on 97 samples and returned a higher-energy local candidate on the other three. The mean absolute relative difference between the spectral bandwidths and the Brent-in-log-h bandwidths decreased as N increased in this experiment. Spectral mean runtimes were respectively 4.46, 6.41 and 14.32 times the mean Brent-in-log-h runtime for N = 100, 200 and 400.

All numerical reference searches passed their scan-resolution checks. All 300 spectral peak selections agreed after doubling the Fourier frequency resolution. Three additional comparisons with analytic Gaussian overlaps also agreed. Reference search stability is numerical evidence, not a global-minimum certificate. SciPy describes bounded Brent as a local minimization algorithm.

These observations concern these implementations, the specified Gaussian sample size, grids and interval. The 100 samples were reused from the prior sample bank; this is a timing and bandwidth comparison rather than a new confirmation batch for choosing alpha.

![Runtime and paired bandwidth comparison](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/outputs/brent_spectral_m100_comparison/brent_spectral_m100_comparison.png)

Sources and reproducibility:

- [SciPy 1.13.1: minimize_scalar, Notes](https://docs.scipy.org/doc/scipy-1.13.1/reference/generated/scipy.optimize.minimize_scalar.html#notes).
- [Experiment source](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/experiments/qhat0_interval_example/brent_spectral_m100_comparison.py).
- [Complete settings and validation](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/outputs/brent_spectral_m100_comparison/summary.json).
- [All 500 individual selections and timings](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/outputs/brent_spectral_m100_comparison/results.csv).
- [Aggregate measurements](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/outputs/brent_spectral_m100_comparison/aggregate_results.csv).
- [PDF figure](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/outputs/pdfs/brent_spectral_m100_comparison/001_bounded_brent_versus_spectral_selection.pdf).
