# Distance between the local-bandwidth and ISE-bandwidth distributions

The comparison uses the existing 1,000 paired standard-normal samples at
each n=100,200,...,2000. The local selectors use the lowest-energy detected
interior minimum on (0.01,2), and ISE is minimized numerically on [0.01,2].
See the experiment's [settings](../../Code/Code_kde/outputs/data/local_qr_minimizers_vs_n/settings.json)
and [ISE reference validation](../../Code/Code_kde/outputs/data/local_qr_minimizers_vs_n/ise_reference_validation.json).

## Metric and treatment of missing minima

The primary metric is Wasserstein-1 distance on the bandwidth h itself:

\[
W_1(F,G)=\int_{\mathbb R}|F(h)-G(h)|\,dh.
\]

Here F and G are cumulative distribution functions. The distance has
bandwidth units, and a smaller value means closer distributions. This is
the one-dimensional CDF identity documented in
[SciPy, wasserstein_distance, Notes](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html#notes).

For a local selector, counts are normalized by the number of available
local minima. Therefore F is conditional on the existence of a detected
local minimum. G uses all 1,000 ISE selections. The missing-outcome fraction
is recorded separately; no artificial numerical bandwidth or penalty is
assigned to failure. At low n, selection conditioning matters: this is a
comparison of the available-bandwidth law with the full ISE law, rather than
a comparison of two estimators that always return a bandwidth.

For the 201 common histogram centres h_j, we also independently evaluate
the exact discrete CDF integral, sum_j |F_j-G_j|(h_(j+1)-h_j). It agrees with
the SciPy weighted calculation within 1e-12 for every comparison.

Secondary descriptive metrics are W1 on log h and the maximum absolute CDF
gap. The latter is the two-sided KS distance; its definition is given in
[SciPy, ks_2samp, Notes](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html#notes).
We use the distance only, with no KS p-values. Samples are paired, and the
histogram representation is discrete; no independent-continuous-sample
significance test is claimed.

## Empirical result

Among the three tested values r=0,13/14,1, the r=13/14 distribution is closest
under all three metrics at every one of the twenty tested sample sizes.
This is a descriptive finding for this experiment, not an optimization over
all possible r or a universal finite-sample theorem.

| n | W1, Q0 | W1, Q13/14 | W1, Q1 |
|---:|---:|---:|---:|
| 100 | 0.123309 | 0.062470 | 0.067748 |
| 1000 | 0.080811 | 0.014245 | 0.023492 |
| 1500 | 0.075826 | 0.010568 | 0.016772 |
| 2000 | 0.072896 | 0.009902 | 0.014507 |

Sources: [all distances and missing fractions](../../Code/Code_kde/outputs/data/local_qr_minimizers_vs_n/distribution_distance_to_ise.csv)
and [comparison validation](../../Code/Code_kde/outputs/data/local_qr_minimizers_vs_n/distribution_distance_validation.json).
For the four sample sizes with saved individual ISE bandwidths, an unbinned
calculation gives the same winning r. The largest raw-versus-binned distance
difference is 0.000334; see the [raw checks](../../Code/Code_kde/outputs/data/local_qr_minimizers_vs_n/distribution_distance_raw_checks.csv).

The CSV also reports W1 divided by the binned mean ISE bandwidth. This
relative scale is useful when comparing sample sizes, because absolute
bandwidths themselves shrink with n. At n=2000 the relative distances are
approximately 31.18%, 4.24%, and 6.20% for Q0, Q13/14, and Q1, respectively.

## Connection with the leading-order bandwidth calculation

The following is our algebraic comparison of the two displayed formulas,
not a theorem about the random selector's distribution. Dividing
[h_AMQ,r](../../Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex#L895)
by [h_AMISE](../../Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex#L95)
gives

\[
\frac{h_{\mathrm{AMQ},r}(n)}{h_{\mathrm{AMISE}}(n)}
=\left[
\frac n{n-1}\frac{1+2r}{4-2r}\frac{\|L\|_2^2}{\|K\|_2^2}
\right]^{1/5}.
\]

For the Gaussian kernel, the norm ratio is 3/4, using the constants recorded
in the [same section](../../Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex#L903).
At r=13/14, ((1+2r)/(4-2r))(3/4)=1, so the leading bandwidth constants match.
The entire bracket in this algebraic ratio is raised to the fifth root.

This concerns the minimizer of the leading approximation to the expected
energy and the AMISE bandwidth. It does not by itself identify the sampling
law of a selected sample local minimum or of the sample-specific ISE oracle.
Marginal distribution closeness also does not establish closeness of the two
bandwidths on each paired sample, or minimum ISE regret. Those would require
separate paired-error or density-loss comparisons.

## Reproduction

Run [compare_to_ise.py](../../Code/Code_kde/outputs/analysis/local_qr_minimizers_vs_n/compare_to_ise.py).
It reads the existing experiment, writes the distance tables, and exports
the comparison figure automatically as PDF. No simulation or thesis-source
file is changed.
