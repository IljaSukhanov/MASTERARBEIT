# Order recovery without penalty sequences

These notes distinguish verified numerical findings, mathematical deductions made here, and conditional results that still require an asymptotic argument for the empirical minimum-width estimator. They do not claim novelty.

The central finding is that the low L-method accuracy does not establish that order recovery requires a penalty sequence. There is an actual AR(3) population example where both tested L-method variants choose the wrong order. There are also substantial optimization errors in some computed width curves. Under a known, exact innovation support, a minimum-feasible-order rule performs well without any penalty or shrinking threshold.

## 1. What the population width proposition identifies

Assume a causal, minimal-order AR(p) model with independent innovations having full support [-1,1], and fix an embedding limit K with p <= K-2. The population proposition in [the thesis, Proposition labelled `prop:minimum_width`](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/3_application_to_time_series.tex:316) gives a constant tail beginning at p+1 and strictly larger widths before that tail. In particular,

$$d_k=m_k-m_{k+1},\qquad d_p>0,\qquad d_k=0\quad(k>p).$$

Earlier drops need not all be strictly positive. Neither the statement nor its proof requires the pre-plateau sequence to be affine. Therefore the population order is the location of the last nonzero drop, not necessarily the minimizer of a two-line approximation error. This is a deduction from the proposition's stated plateau and strict separation properties.

## 2. An actual population counterexample to both L-method fits

Consider the stationary AR(3) parameter vector

$$\phi=(1.5416770721146058,-0.8888876908549326,0.16319326164037884).$$

Its largest companion-root modulus is approximately 0.69342947. The population width calculation gives

$$m=(13.09302613,4.73694176,1.51876071,0.97665360,0.97665360,\ldots).$$

For the independent, overlapping least-squares fits tested in the previous comparison, with segment-size-weighted RMSE, the scores are:

| Tail model | Candidate order 2 | Correct order 3 |
|---|---:|---:|
| Affine | 0.436739 | 0.717378 |
| Constant | 0.460666 | 0.717378 |

Both select order 2. This is a population calculation, so increasing sample size cannot repair this particular population fitting criterion. The last genuine drop is positive, but the large and unequal earlier drops make an earlier two-line approximation cheaper.

Sources: [parameter and width values](/Users/iljasukhanov/MASTERARBEIT/output/analysis/ar_order_without_penalties/population_counterexample.json), [all candidate scores](/Users/iljasukhanov/MASTERARBEIT/output/analysis/ar_order_without_penalties/population_counterexample_scores.csv), and [calculation](/Users/iljasukhanov/MASTERARBEIT/output/analysis/ar_order_without_penalties/population_counterexample.py). The parameter vector comes from replication 2 of the existing AR(3) seed family; it is not fitted to manufacture a desired width curve.

The calculation uses the support representation w(u)=2 sum_j |g_j dot u| from [the thesis proof](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/3_application_to_time_series.tex:350). For k=1 it sums absolute impulse coefficients. For k=2 and 3 it enumerates facet normals of the finite-generator zonotope: perpendiculars to generators in dimension two, and cross products of noncollinear generator pairs in dimension three. The reason is that the support function is positive and linear on each spherical cell of its sign arrangement; along a great-circle arc within a cell its second derivative equals minus its value. A minimum therefore lies at a cell vertex, where k-1 independent generator projections vanish. This is a derivation made here, rather than an external theorem being attributed to the numerical code.

For this model 57 generators suffice. A partial-fraction representation psi_j=sum_i r_i z_i^j gives |psi_j| <= C rho^j, with C=sum_i |r_i| and rho=max_i |z_i|. The omitted width is bounded by 2 sqrt(k) C rho^(N-k+1)/(1-rho). The maximum tail truncation bound for k<=3 is 7.77e-8. Floating-point arithmetic is not interval-certified, but the score gap is orders of magnitude larger than the truncation bound. Segment RMSE is Lipschitz in the input vector: perturbing every width by at most epsilon changes each segment RMSE, and hence the weighted score, by at most epsilon. Thus the tail truncation uncertainty cannot account for the observed score reversal.

## 3. The computed curves also contain optimization errors

The mathematical empirical minimum is the infimum over all unit directions. A particular feasible direction gives an upper bound on that infimum. For the first ten AR(3) simulations, 32 of the 70 computed widths at dimensions 4 through 10 are larger than the widths of the padded true recurrence direction. They therefore cannot be the empirical minima. The comparison with the true direction is an oracle diagnostic, not a proposed estimator.

For replication 1 at k=4:

- Original 500-iteration result: 2.154225.
- Feasible true recurrence direction: 1.422520.
- Data-only LP and nested warm starts, with 1,000 iterations: 1.421508.

The revised starts use a minimax-regression direction, the previous dimension's best direction padded with zero, the smallest covariance eigenvector, and two random directions. They do not use true coefficients or the innovation bound. They improve this numerical miss but do not certify a global minimum.

Sources: [oracle audit](/Users/iljasukhanov/MASTERARBEIT/output/analysis/ar_order_without_penalties/optimizer_audit.csv), [data-only initialization audit](/Users/iljasukhanov/MASTERARBEIT/output/analysis/ar_order_without_penalties/initialization_audit_widths.csv), and [audit selections](/Users/iljasukhanov/MASTERARBEIT/output/analysis/ar_order_without_penalties/initialization_audit_selections.csv). On the first ten AR(3) curves, the drop-to-tail-energy selector improves from 4/10 to 5/10 correct selections. Both L-method variants remain at 4/10. This small audit does not establish an improved general selection rate.

Moreover, the original approximate curves have at least one upward step in 48/50 AR(1), 47/50 AR(2), and 45/50 AR(3) cases. The exact empirical minima, with the present nested sliding-window construction, are nonincreasing: padding a k-dimensional direction with zero gives a feasible direction in dimension k+1, and its projected point set is a subset of the old set. Thus upward steps are further evidence of numerical error. Source data: [the 150 width curves](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/csv/ar_shared_breakpoint_l_methods/width_curves.csv); embedding definition: [functions.py](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/functions.py:174). This nesting argument is a deduction made here.

The existing consistency statement concerns exact empirical minima. It does not supply a guarantee for a solver with a fixed iteration budget and a fixed number of starts. See [the corollary](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/3_application_to_time_series.tex:416) and [the numerical optimizer](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/descent.py:131).

## 4. A width-only candidate with no penalty sequence

A direct candidate is

$$\widehat d_k=\widehat m_k-\widehat m_{k+1},\qquad
R_T(q)=\frac{\widehat d_q^2}{\sum_{j=q+1}^{K-1}\widehat d_j^2},\qquad
\widehat p_T=\operatorname*{arg\,max}_{1\le q\le K-2}R_T(q).$$

Use the explicit conventions positive/0=infinity and 0/0=0, and break ties toward the smaller order. There is no added ridge, penalty, or threshold sequence. The method searches orders at least one; it does not address detecting white noise.

At the population curve the true index p is the unique index with a positive numerator and zero denominator. For q<p the denominator includes d_p^2>0; for q>p both quantities vanish. This is why it targets the last genuine drop rather than the largest absolute drop. It imposes no affine head.

On the original approximate width curves, this candidate recovers 49/50, 42/50, and 28/50 orders. Two other exploratory variants, using successive tail SSE or variance ratios, recover respectively (47,42,26) and (48,42,26). All variants are reported to avoid presenting only the best exploratory score. Source: [summary.csv](/Users/iljasukhanov/MASTERARBEIT/output/analysis/ar_order_without_penalties/summary.csv) and [all ratio scores](/Users/iljasukhanov/MASTERARBEIT/output/analysis/ar_order_without_penalties/ratio_candidate_scores.csv). These are exploratory results on the already-used curves, not independent validation or global-minimum calculations.

The use of a ratio to separate signal from noise has an established analogue in [Ahn and Horenstein (2013), abstract and eigenvalue-ratio estimator](https://doi.org/10.3982/ECTA8968). Their factor-model result is not a theorem for AR width curves.

### Conditional consistency argument made here

For fixed K and fixed p>=1, suppose the estimated widths are consistent and that there exists a_T -> 0 such that

$$(\widehat d_{p+1},\ldots,\widehat d_{K-1})/a_T\Rightarrow Z.$$

Suppose every suffix of Z is nonzero almost surely. This condition ensures that the post-plateau denominators do not degenerate relative to the numerators. It is an additional assumption, not a consequence of width consistency alone.

Then R_T(p) diverges in probability: its numerator tends to d_p^2>0 while its denominator tends to zero. For q<p, R_T(q) has a finite population limit, because its denominator includes d_p^2. For q>p, both numerator and denominator scale as a_T^2, and their ratio is tight by the nondegeneracy condition. With finitely many candidates this implies Pr(hat p_T=p)->1.

The scale a_T appears only in the analysis, not in the algorithm. No rate needs to be chosen by the user. A proof for the actual AR empirical widths still needs to establish the joint tail-error behavior and bound numerical optimization error. Mere almost-sure convergence of every width is insufficient: different vanishing rates can make a spurious post-plateau ratio dominate. This is a conditional result and a research route, not a completed consistency theorem for the implemented estimator.

## 5. A stronger constructive answer with known exact innovation support

If the innovation support is known to be exactly [-h,h], define, using the same response times for every candidate,

$$\Theta_{q,T}=\left\{b\in\mathbb R^q:
\left|X_t-\sum_{j=1}^q b_jX_{t-j}\right|\le h\ \text{for every observed }t\right\},$$

$$\widehat p_T=\min\{q: \Theta_{q,T}\ne\varnothing\}.$$

For order zero the predictor is zero. These are fixed-support feasibility constraints, not a penalty or a shrinking threshold. The constraints are linear and are checked with linear programming. This is a set-membership construction; the general feasible-parameter-set viewpoint is described in [Lauricella and Fagiano (2020), Section III-D, Eq. (22)](https://arxiv.org/pdf/2001.07628). The specific AR order argument below is a derivation made here, not a result attributed to that paper.

The rule uses the time series and the known sharp innovation bound. It does not use only the spherical minimum-width curve, so its accuracy is not evidence that the L-method has been repaired. An upper bound larger than the actual support radius is not sufficient for the consistency argument below; the bound must be sharp and the innovations must have full support on the stated interval.

For the same 150 series, with h=1:

| True order | Correct | Under | Over |
|---|---:|---:|---:|
| 1 | 49/50 | 1 | 0 |
| 2 | 50/50 | 0 | 0 |
| 3 | 49/50 | 1 | 0 |

Source: [exploratory summary](/Users/iljasukhanov/MASTERARBEIT/output/analysis/ar_order_without_penalties/summary.csv), [fitted residual radii and feasibility](/Users/iljasukhanov/MASTERARBEIT/output/analysis/ar_order_without_penalties/known_bound_radii.csv). The two errors have final coefficients approximately 0.0045687442 and -0.0031779546. Their lower-order models are still compatible with the observed sample and the specified noise support. Candidate orders are 0 through 8; in the line-fit comparison the candidate set started at 1.

An independent validation uses 100 new processes per true order, fixed seed families 20260918+p with a separate seed component 123, T=2000, burn-in 1000, and the same Jones coefficient sampler. There is no tuning on the validation outcomes. The rule recovers 99/100, 100/100, and 100/100 orders, totaling 299/300. Sources: [validation summary](/Users/iljasukhanov/MASTERARBEIT/output/analysis/ar_order_without_penalties/validation_summary.csv), [all validation selections](/Users/iljasukhanov/MASTERARBEIT/output/analysis/ar_order_without_penalties/validation_selections.csv), and [validation procedure](/Users/iljasukhanov/MASTERARBEIT/output/analysis/ar_order_without_penalties/validate_feasibility.py). This is validation for one simulation design, not a uniform finite-sample guarantee.

### Consistency deduction under the stated model

Assume a stable, causal, minimal-order AR(p) with phi_p!=0, independent innovations of full support [-h,h], and a stationary ergodic trajectory. Use a fixed common starting time M at least as large as the largest candidate order, and let T grow.

1. The true coefficient vector is feasible for every T, because its residuals are the actual innovations. Hence the selected order never exceeds p in exact arithmetic.
2. For q<p and any fixed b, let B(z)=1-sum_{j=1}^q b_jz^j and A(z)=1-sum_{j=1}^p phi_jz^j. The residual filter is B(z)/A(z), with coefficients a_0=1 and at least one nonzero a_j for j>=1. Otherwise B=A, which is impossible when q<p and phi_p!=0. Its residual support radius is h sum_j |a_j|>h. This support calculation follows from independent full-support innovations and the causal convolution representation; compare [the thesis support calculation](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/3_application_to_time_series.tex:350).
3. No fixed lower-order b can therefore satisfy the residual bound on the whole population support. Almost-sure density of the observed embeddings in their support implies it cannot satisfy the bound for the infinite observed trajectory. The density property is established under stationarity and ergodicity in [the thesis lemma](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/3_application_to_time_series.tex:400).
4. To rule out a sequence of changing lower-order coefficients b_T, use compactness. The lag-vector support has nonempty interior for independent full-support innovations, so a finite observed set contains q linearly independent lag vectors almost surely. If a lower-order model is feasible, the constraints at those q response times bound its coefficients through the inverse of that lag matrix. The resulting feasible set is closed and bounded. All subsequent feasible sets are nested closed subsets of it. If every finite-sample set were nonempty, compactness would give an element in their infinite intersection, contradicting step 3.
5. Thus, for each q<p, a finite random T_q exists beyond which it is infeasible. There are finitely many q. Eventually every smaller order is rejected and the true order remains feasible, yielding hat p_T=p almost surely for all sufficiently large T.

This deduction supplies a concrete penalty-free route under exact known support. It gives no deterministic sample size at which all small nonzero final coefficients can be recovered, and it does not apply when the sharp support bound is unknown. The finite-sample overlap of feasible lower-order and true-order models is visible in the two exploratory underselections.

## 6. What follows for the geometric project

The justified conclusions are:

- The current two-line population criterion is wrong for at least the numerically checked actual AR(3) example; its poor recovery is not evidence that a penalty sequence is unavoidable.
- Some reported widths are not even minima relative to a known feasible competitor. Their consistency should not be assumed from the exact-minimum consistency theorem.
- A width-only ratio avoids the affine-head assumption and is a legitimate penalty-free candidate, with a conditional consistency argument. Its performance on the present approximate curves remains limited.
- With known sharp support, feasible-model order selection supplies a much stronger penalty-free benchmark and an almost-sure consistency deduction under the stated assumptions.

For a result using only geometric minimum widths and no known support scale, the remaining mathematical task is to establish relative tail-error control for a ratio or another calibrated plateau rule, together with an adequate optimization-error bound. The known-bound feasibility method establishes that the simulated trajectories contain useful order information; it does not establish that the current width-only summary preserves enough of that information at T=2000.
