# Distribution of the selected interior local energy minimizer, n=200

This note distinguishes empirical finite-sample fits, an exact equivariance
calculation, and a possible route to an asymptotic theorem. It establishes
no exact named distribution or asymptotic normality theorem for this selector.

## Statistic and the no-minimum outcome

The saved experiment uses 1,000 paired standard-normal samples of size 200.
For each r, it selects the interior local minimum with the lowest energy
when several exist. A sample without an interior minimum returns a separate
failure symbol, denoted by bottom. It does not return zero or a bandwidth
endpoint. The full probability law can therefore be written

    Law(T_n,r) = p_n,r delta_bottom + (1-p_n,r) F_n,r,

where F_n,r is the distribution conditional on the existence of a local
minimum. This decomposition follows directly from conditioning on the
existence event. The reported failure rates estimate p_n,r; an observed
zero rate does not prove that the population probability is zero.

## Empirical characterization

| r | Observed no-minimum fraction | Conditional mean | Conditional SD | Skewness | Best AIC family among those tested | Normal delta AIC |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| 0 | 0.0% | 0.264492 | 0.061894 | -0.060 | normal | 0.00 |
| 13/14 | 5.0% | 0.426772 | 0.088068 | 0.785 | skew_normal | 121.50 |
| 1 | 10.4% | 0.440142 | 0.090189 | 0.812 | skew_normal | 119.49 |

The candidate families are normal, lognormal with location fixed at zero,
gamma with location fixed at zero, and skew-normal. Every family is fitted
to the available selected minima only. Ten-fold predictive log likelihood
uses sample-id modulo ten to define folds. Its model ranking agrees with AIC:
normal for r=0 and skew-normal for r=13/14 and r=1. The normal and skew-normal
scores for r=0 are very close; the symmetric approximation is adequate as a
description without introducing an extra shape parameter.

The normal fit for r=0 has mean 0.264492 and SD 0.061863. The skew-normal
parameterization is location + scale * Z_alpha, with standardized density
2 phi(z) Phi(alpha*z). This is the parameterization in the
[SciPy skew-normal documentation, Notes](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skewnorm.html).
For r=13/14 the fitted (location, scale, shape) is
(0.320187, 0.138233, 3.934652); for r=1 it is
(0.330339, 0.142063, 4.087891).

These are descriptive approximations, not identification of the exact law.
The saved KS distances are descriptive; no ordinary KS p-values are used
with parameters estimated from the same observations. Parameter fitting
must be incorporated into a calibrated goodness-of-fit test; see
[SciPy goodness_of_fit, algorithm and parameter handling](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.goodness_of_fit.html).

## Exact location-scale equivariance: original derivation

For transformed observations Y_i=a+b X_i, b>0, the KDE definition implies

    fhat_h^Y(y) = b^(-1) fhat_(h/b)^X((y-a)/b).

Differentiate with respect to h and substitute y=a+bx in the squared L2
norm. The derivative contributes b^(-2) and dy contributes b, giving

    Qhat_r^Y(h) = b^(-(1+2r)) Qhat_r^X(h/b).

The positive constant preserves the ordering of the local-minimum energies.
Hence, for the unrestricted local-minimum rule used here,

    T_r(a+bX) = b T_r(X),

with existence and multiplicity preserved. For normal observations this
means that the conditional distribution of T_r/sigma and the no-minimum
probability depend on n and r, but not on the normal mean or scale.
This conclusion is a direct algebraic derivation from the
[KDE definition, Section 6.1.1](https://homepage.divms.uiowa.edu/~luke/classes/STAT7400-2023/_book/density-estimation-and-smoothing.html)
and the project definition of Q_r in
[Proposition prop:main_thm](../../Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex).
It requires an equivariant search domain; arbitrary fixed endpoints would
break the statement. A numerical check of the energy scaling identity is
saved in distribution_scaling_check.json.

## Possible asymptotic characterization: not a proved theorem here

Let q_n,r(h)=E Qhat_n,r(h), and let h_n,r be a simple interior local minimum
of q_n,r on a specified branch. For a sample minimum on the same stable
branch, a mean-value expansion of the score gives

    hhat_n,r-h_n,r = -Qhat'_n,r(h_n,r) / Qhat''_n,r(h_tilde).

If the empirical curvature can be replaced by q''_n,r(h_n,r), this suggests

    hhat_n,r-h_n,r approximately -Qhat'_n,r(h_n,r) / q''_n,r(h_n,r).

This is a Taylor linearization, not an asymptotic normality result. A limit
law would require a suitable central limit theorem for the derivative of
this Gaussian-pair criterion, control of the shrinking h_n,r, and stability
of the selected local branch. Ordinary fixed-parameter root-n normality
must not be assumed. Branch switches, small curvature, and the no-minimum
outcome also require attention. For the general score/Hessian method, see
[Yen-Chi Chen, M-estimation notes, Theorems 1 and 9](https://faculty.washington.edu/yenchic/26A_stat581/Lec3_Mest.html).

The pair-sum structure needed for that analysis is implemented in
Code/Code_kde/kde_tools/qhat.py and its r transformation in
Code/Code_kde/kde_tools/continuous_minimizers.py. Related bandwidth criteria
are analyzed through U-statistics in
[Scott and Terrell, Biased and Unbiased Cross-Validation in Density Estimation (1987)](https://repository.rice.edu/items/005dda79-7726-417e-8374-f098523e6f7f);
that paper does not establish a theorem for this particular Q_r selector.

## Reproducibility

- Numerical fit workflow: Code/Code_kde/outputs/analysis/local_energy_minimizers_n200/characterize_distribution.py
- Raw selected minima and every candidate: Code/Code_kde/outputs/data/local_energy_minimizers_n200/
- Descriptive statistics: distribution_descriptive.csv
- Fit parameters, AIC and predictive scores: distribution_fits.csv
- Per-observation predictive scores: distribution_predictive_scores.csv

