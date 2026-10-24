# Analysis of saved fast L2 lambda experiments

This is an original descriptive analysis of the existing completed runs. No simulations were rerun. All 40 checkpoints are complete (M=1000). Selection CSVs, frequency CSVs and checkpoint indices agree. The manifests agree in every setting except λ: standard-normal data, 201 geometric bandwidths from 0.01 to 2, FFT spatial grid 8192 on [-12,12], seed 20260830. The same seeded samples and their prefixes are used across λ and n in the [notebook](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/xx_notebooks_bandwidth_distributions/fast_l2_bandwidth_distributions_vs_n_lambda.ipynb).

## Empirical findings

Mean selected bandwidths, calculated from the saved selections:

| n | λ=1 | λ=10 | λ=100 | λ=1000 |
| --- | --- | --- | --- | --- |
| 100 | 0.099 | 0.138 | 0.256 | 0.296 |
| 500 | 0.267 | 0.177 | 0.121 | 0.183 |
| 1000 | 0.438 | 0.372 | 0.093 | 0.145 |

The existing histograms show two separated regions of selected bandwidth at intermediate n for λ=1 and λ=10. The fraction selecting h>0.25 quantifies the shift, but the threshold is an analyst-chosen diagnostic and is not a formal test of bimodality:

| n | λ=1 | λ=10 |
| --- | --- | --- |
| 400 | 29.6% | 13.9% |
| 500 | 51.6% | 25.7% |
| 600 | 71.2% | 41.4% |
| 700 | 85.2% | 59.2% |
| 1000 | 98.8% | 89.5% |

The same delayed transition for λ=10 remains visible with thresholds 0.20 and 0.30; all three thresholds are recorded in distribution_statistics.csv. These cross-repetition histograms do not establish that individual Perron vectors have two competing maxima.

For λ=100 and λ=1000, the mean bandwidth decreases at every tested step in n. The empirical 10th–90th percentile intervals narrow from approximately [0.200,0.321] to [0.077,0.112] for λ=100, and from [0.222,0.367] to [0.127,0.162] for λ=1000. None of the 40 points selects either grid endpoint.

Increasing λ is not a monotone bandwidth reduction. At n=1000, raising λ from 100 to 1000 increases mean h by 0.051701. Paired comparisons and their Monte Carlo standard errors are recorded separately.

## Interpretation and limits

The experiment demonstrates a strong interaction between sample size and λ for this Gaussian model and this fixed grid. A plausible mechanism is a change in the similarities that determine the spectral ranking: [similarities.py](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/kde_tools/similarities.py) constructs S_ij=exp(-λ D_ij), with D equal to half the squared L2 distance. This interpretation does not by itself explain the observed transition or establish a theorem.

Concentration of selected bandwidths measures reproducibility of the selector, not density-estimation accuracy. This analysis does not calculate realized density errors. A Gaussian fixed-bandwidth MISE reference is computed using [gaussian_mise_oracle_bandwidth](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/kde_tools/bandwidths.py:120); closeness of a data-dependent bandwidth to that reference is not a risk comparison. For general guidance on separating performance measures and reporting Monte Carlo uncertainty, see Morris, White and Crowther (2019), [Sections 5.2–5.3](https://onlinelibrary.wiley.com/doi/full/10.1002/sim.8086).

No asymptotic convergence rate or consistency conclusion follows from n=100,...,1000. In particular, the [draft joint-limit conjecture](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:820) varies N and λ together; the present experiment holds N=201 fixed.

As an additional original deterministic calculation, replacing each empirical KDE by its population-smoothed Gaussian density in the shared [population matrix helper](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/kde_tools/similarities.py:50) gives selected grid bandwidths approximately λ=1: h=0.466, λ=10: h=0.430, λ=100: h=0.260, λ=1000: h=0.149. These are model references, not additional Monte Carlo observations or density-error optima. In particular, the observed downward trend at λ=100 should not be extrapolated indefinitely; its population-smoothed reference is h≈0.260. No thesis or notebook source was edited.

## Reproduce

Run `python3 Code/Code_kde/outputs/analysis/fast_l2_lambda_comparison/analyse_saved_results.py` from the project root. It reads only the completed runs and writes analysis CSVs and this summary. The comparison figure is automatically exported to the existing notebook's PDF folder by [pdf_export.py](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/kde_tools/pdf_export.py). Mean Monte Carlo SE is the sample standard deviation divided by sqrt(M). Approximate 95% Monte Carlo intervals use ±1.96 SE; they describe simulation uncertainty in an estimated mean, not the spread of individual selected bandwidths (Morris et al., Table 6).

Source runs:

- λ=1: [saved run](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/outputs/data/fast_l2_bandwidth_distributions_vs_n_lambda/thesis/028fa8664f9c)
- λ=10: [saved run](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/outputs/data/fast_l2_bandwidth_distributions_vs_n_lambda/thesis/1de0f2734d87)
- λ=100: [saved run](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/outputs/data/fast_l2_bandwidth_distributions_vs_n_lambda/thesis/4e41e6f126cf)
- λ=1000: [saved run](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/outputs/data/fast_l2_bandwidth_distributions_vs_n_lambda/thesis/d7062cba5620)
