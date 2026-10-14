# Plugging an estimated innovation support into the feasibility selector

The mathematical arguments below are deductions made in this investigation, not published AR-order theorems or claims of novelty. The numerical results are exploratory comparisons on previously used simulation sets. No thesis source was modified.

## 1. A concrete rule with unknown support scale

Assume a stationary, stable, causal, minimal AR(p), independent innovations Uniform[-h,h] with h>0 unknown, and a fixed candidate limit M>=max(p,1). The innovation support is symmetric about the known center zero. All candidate models use the same response times t=M+1,...,T; write n=T-M and L_t=(X_{t-1},...,X_{t-M}).

Fit a pilot AR(M) by ordinary least squares:

$$\widetilde b_n=\operatorname*{arg\,min}_{b\in\mathbb R^M}
\sum_t(X_t-b^TL_t)^2.$$

Use its largest absolute residual to estimate the support radius and width:

$$\widehat h_n=\max_t|X_t-\widetilde b_n^TL_t|,
\qquad\widehat W_n=2\widehat h_n.$$

Then select the smallest feasible order:

$$\widehat p_n=\min\left\{0\le q\le M:
\exists b\in\mathbb R^q,\quad
\left|X_t-\sum_{j=1}^q b_jX_{t-j}\right|\le\widehat h_n
\text{ for every response time }t\right\}.$$

No penalty, inflation constant, ridge, or threshold sequence is used. M is the fixed search limit, not an estimated true order. The pilot itself is feasible at order M by construction, so the search always has a feasible candidate. Using the same data for estimation and selection is permitted by the argument below; it does not assume an independent pilot sample.

This is a time-series residual and feasible-parameter-set method. It does not use only the spherical minimum-width curve. The general approach of estimating a noise bound before model identification has a precedent in [Lauricella and Fagiano (2020), Section III-C, Procedures 1 and 2, pp. 4–5](https://arxiv.org/pdf/2001.07628). Their model is a deterministic system with bounded measurement noise and an external input; their results are not the AR innovation theorem derived here.

## 2. What any plug-in support estimate must satisfy

Let

$$r_{q,n}=\inf_{b\in\mathbb R^q}\max_t
\left|X_t-\sum_{j=1}^q b_jX_{t-j}\right|.$$

A sufficient condition for a general plug-in rule is

$$\widehat h_n\longrightarrow h\text{ in probability},
\qquad \Pr(r_{p,n}\le\widehat h_n)\longrightarrow1.$$

The second condition keeps the true order feasible. A stronger sufficient condition is Pr(hat h_n>=h)->1, because the true coefficients have residuals bounded by h at every sample size.

Ordinary consistency alone is insufficient for the discontinuous feasibility decision. For example, if p=0 and the innovations themselves are observed, max_t|X_t|-1/n (truncated at zero) consistently estimates h but is always smaller than the empirical residual radius at order zero. It rejects the correct order at every sample size. This counterexample is an original deduction, not a cited theorem.

For q<p the population optimized radius is strictly larger than h. For a fixed b, the residual filter B/A has leading coefficient one and at least one additional nonzero coefficient, so its support radius is h||B/A||_1>h. The support argument is the same convolution calculation used in [the thesis proof, label `prop:minimum_width`](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/3_application_to_time_series.tex:350). The strict gap persists after optimization: bounded residual radii force bounded coefficients using finitely many independent lag vectors; a minimizing subsequence therefore has a convergent subsequence, and a radius equal to h would contradict the fixed-b statement. Dense observed embeddings, established in [the thesis lemma `lem:ergodic_implies_dense`](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/3_application_to_time_series.tex:400), imply r_{q,n} converges to this optimized population radius. Thus a consistent h estimate rejects all smaller orders eventually in probability. It remains to show that it accepts p.

## 3. Consistency argument for the pilot maximum-residual rule

Under the Uniform[-h,h] model and fixed M>=max(p,1), the construction in Section 1 satisfies

$$\Pr(\widehat p_n=p)\longrightarrow1.$$

This is consistency in probability, not a claim of eventual almost-sure order recovery. The proof is given here to make the assumptions and the one-sided coverage mechanism reviewable.

### A. Pilot coefficient limit

Let b_0=(phi_1,...,phi_p,0,...,0) and delta_n=tilde b_n-b_0. Write Gamma=E[L_tL_t^T]. The lag-vector support has nonempty interior: the first M innovation generators form a triangular matrix with diagonal psi_0=1. It contains a ball about zero because the innovations are symmetric and have full interval support. Therefore Gamma is positive definite. Stable causal AR processes with bounded innovations have bounded lag vectors and are stationary ergodic; compare [the thesis causal representation and stationarity remark](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/3_application_to_time_series.tex:411).

The least-squares identity is

$$\sqrt n\,\delta_n=
\left(n^{-1}\sum_tL_tL_t^T\right)^{-1}
\left(n^{-1/2}\sum_tL_t\varepsilon_t\right).$$

The first factor converges to Gamma^{-1} by ergodicity. The summands L_t epsilon_t are martingale differences because L_t is measurable before epsilon_t is drawn and epsilon_t is independent, centered, and has variance sigma^2=h^2/3. Their normalized conditional covariance converges to sigma^2 Gamma; boundedness implies the conditional Lindeberg condition. Applying the martingale CLT to each scalar projection gives

$$\sqrt n\,\delta_n\Rightarrow
N(0,\sigma^2\Gamma^{-1}).$$

The precise CLT used is stated in [Péter Major, *Central limit theorems for martingales*, Section 1, p. 3, conditions (a)–(c), equations (1.2)–(1.3)](https://www.renyi.hu/~major/probability/martingale.pdf), with a proof adapting Brown (1971). This source supports the CLT, not the subsequent support or order deductions. In particular, the Gaussian limit has no atom at zero and delta_n->0 in probability.

### B. Consistency of the estimated radius

If ||L_t||<=R, then

$$\left|\widehat h_n-\max_t|\varepsilon_t|\right|
\le R\|\delta_n\|.$$

The innovation sample maximum converges to h, so hat h_n->h in probability. This establishes ordinary support-scale consistency. We next establish that the estimate is also an upper bound with probability tending to one.

### C. The known-bound feasible set is much tighter than the pilot error

Let Theta_{M,n}(h) be the set of order-M coefficients whose observed residuals all lie in [-h,h]. We show that, for a fixed constant C,

$$\Pr\left(\Theta_{M,n}(h)\subseteq
\{b:\|b-b_0\|\le Cn^{-2/3}\}\right)\longrightarrow1.$$

The exponent 2/3 is only a device in this proof. It is never used by the algorithm.

Choose rho>0 such that the lag support contains B(0,rho), and let R bound all lag norms. Choose a finite net of unit directions u_i with mesh at most rho/(8R). Each lag region V_i=B((rho/2)u_i,rho/8) has positive probability. For any direction u within that mesh of u_i and any L_t in V_i,

$$u^TL_t\ge\rho/4.$$

Set eta_n=n^{-2/3}. For every V_i, with probability tending to one there is an observed t with L_t in V_i and epsilon_t in [-h,-h+eta_n]. To see this, ergodicity gives at least c_i n visits to V_i with probability tending to one, for some fixed c_i>0. The visit times are predictable from past innovations. At these visit times, the new innovations retain independent uniform distributions. Thus the probability of no boundary hit among the first floor(c_i n) visits is

$$\left(1-\frac{\eta_n}{2h}\right)^{\lfloor c_i n\rfloor}
\le \exp\left(-\frac{\eta_n\lfloor c_i n\rfloor}{2h}\right)
\longrightarrow0.$$

A finite union bound supplies boundary hits for every net direction simultaneously. Independence of the entire lag sequence and the current innovations is not assumed; only the fresh innovation's conditional independence at predictable visit times is used.

For any delta=ru with r>4eta_n/rho, choose the corresponding net direction and boundary hit. The residual at b_0+delta then satisfies

$$\varepsilon_t-\delta^TL_t
\le -h+\eta_n-r\rho/4<-h.$$

It violates the bound. Hence every feasible delta has norm at most 4eta_n/rho on this event. This proves the set containment with C=4/rho, uniformly over all coefficients, including coefficients selected using the same data.

### D. One-sided coverage and order recovery

If hat h_n<=h, the pilot tilde b_n belongs to Theta_{M,n}(h). Consequently

$$\Pr(\widehat h_n\le h)
\le o(1)+\Pr(\|\delta_n\|\le Cn^{-2/3})
\longrightarrow0.$$

The last limit follows from the nondegenerate root-n Gaussian limit: the event is ||sqrt n delta_n||<=Cn^{-1/6}, whose radius tends to zero. Therefore Pr(hat h_n>h)->1. The true order is feasible with probability tending to one. Section 2 supplies rejection of every underfitted order. With finitely many candidate orders this yields Pr(hat p_n=p)->1.

The mechanism is specific: fitting error in the least-squares pilot makes its extreme residual slightly larger than the true innovation endpoint, while that error vanishes. The rule estimates the bound from this maximum; it does not add a manually chosen slack. This does not mean that every support estimator has the same property.

## 4. Numerical checks on existing simulations

The pilot limit and selection limit are both M=8. No intercept is fitted, matching the zero-centered model. Each series has T=2000 observations after a burn-in of 1000, with Jones-sampled stable coefficients and innovations Uniform[-1,1]. The true h=1 is used only to generate the benchmark data and to report diagnostics; it is not supplied to either plug-in estimator. The feasibility constraints use numerical solver tolerances of 1e-9 and a residual verification tolerance of 1e-8; these are numerical tolerances, not statistical thresholds.

| Method | Original AR(1), /50 | Original AR(2), /50 | Original AR(3), /50 |
|---|---:|---:|---:|
| Known h=1 | 49 | 50 | 49 |
| Pilot OLS maximum absolute residual | 46 | 46 | 44 |
| Pilot OLS uniform moment estimate | 25 | 23 | 28 |
| Pilot order-8 minimax radius | 0 | 0 | 0 |

The maximum-residual rule has only underselections (4,4,6), with no overselected or infeasible cases. Its median estimated radii are approximately 1.0644, 1.0615, and 1.0715, and every estimate exceeds 1 in these 150 cases. Its wider interval admits a few weak lower-order models; this is the finite-sample price of estimating the bound this way. Asymptotic consistency does not eliminate ambiguity for very small final coefficients at a fixed sample size.

The previously used 300-series validation set was also reused for this new comparison. It is not new independent validation of the plug-in rule:

| Method | Reused AR(1), /100 | Reused AR(2), /100 | Reused AR(3), /100 |
|---|---:|---:|---:|
| Known h=1 | 99 | 100 | 100 |
| Pilot OLS maximum absolute residual | 92 | 89 | 90 |
| Pilot OLS uniform moment estimate | 37 | 46 | 46 |

The maximum-residual rule again has only underselections and no overselected or infeasible cases. All 300 estimated radii exceed 1. The table's correct counts for the moment estimator exclude 135 cases where none of the candidate orders is feasible. In the original set it has 60 infeasible cases. All outcomes, including infeasibility, are reported in [the summary](/Users/iljasukhanov/MASTERARBEIT/output/analysis/ar_order_without_penalties/estimated_support_summary.csv).

Data and reproducibility: [all selections and estimates](/Users/iljasukhanov/MASTERARBEIT/output/analysis/ar_order_without_penalties/estimated_support_selections.csv), [summary](/Users/iljasukhanov/MASTERARBEIT/output/analysis/ar_order_without_penalties/estimated_support_summary.csv), [procedure](/Users/iljasukhanov/MASTERARBEIT/output/analysis/ar_order_without_penalties/estimate_support_plugin.py), [original simulation configuration](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/csv/ar_width_comparison/configuration.json), and [reused validation seeds](/Users/iljasukhanov/MASTERARBEIT/output/analysis/ar_order_without_penalties/validate_feasibility.py).

## 5. Two estimates that must be distinguished from the pilot maximum

The uniform moment estimate sqrt(3 mean_t tilde epsilon_t^2) uses Var(epsilon)=h^2/3, from [NIST, Uniform Distribution, Section 1.3.6.6.2, Common Statistics](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3662.htm). That identity is specific to a uniform distribution; variance does not identify the support width of an arbitrary bounded distribution. A moment estimate can fall below the radius needed for the observed sample, explaining infeasible cases. Consistent variance estimation does not itself establish the one-sided coverage needed here.

The minimax pilot instead estimates h by r_{M,n}. Since the candidate spaces are nested, r_{q,n}>=r_{M,n} for all q<=M. Plugging r_{M,n} into the selector accepts a lower order only if its optimized residual radius is exactly equal to the largest-order radius. In all 150 original simulations the selected order is M=8. This estimate is consistent under the correctly specified model but is circular for this particular selection rule. The LP comparison uses a 1e-8 tolerance for equality of separately solved numerical optimum values.

## 6. Scope of the result

- The support being estimated is that of the innovations, not the marginal support of X_t. The causal convolution generally changes the marginal support; compare [the thesis support convolution](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/3_application_to_time_series.tex:350).
- The construction assumes the support center is zero. For an unknown interval [a,b], width alone does not specify the absolute-residual bound: one must handle the center, for example through a free intercept in each feasibility model. The theorem here does not cover that altered algorithm.
- The proof uses uniform boundary probability eta/(2h). It extends to centered innovation densities bounded below in neighborhoods of both endpoints, with the probability bound changed accordingly and the nondegenerate finite-variance pilot CLT checked. It is not a theorem for every full-support bounded innovation distribution; very thin endpoint probabilities may change the coverage argument.
- The statement uses a fixed known upper limit M>=p, stable minimal order, exact mathematical feasibility, and consistency in probability. It is not a uniform finite-sample guarantee, an almost-sure eventual recovery result, or a claim about numerical spherical-minimum-width optimization.
- The root-n pilot error, the auxiliary n^{-2/3}, and the unknown constants rho and c_i appear only in the proof. The implemented rule uses none of them.
