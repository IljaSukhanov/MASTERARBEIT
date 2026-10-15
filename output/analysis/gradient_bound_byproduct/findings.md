# Estimating the innovation bound from the gradient direction

This is an exploratory derivation and a diagnostic on existing simulations,
not a published consistency theorem or a claim of novelty. No thesis source,
gradient implementation, or active selector was changed. The original
descent-chain selector is unchanged.

## 1. Direct residual estimate: no second regression

Assume a stationary stable causal AR(p), independent innovations with full
support [-h,h], known center zero, and a fixed candidate limit M>=p. Let
Y_t=(X_t,...,X_{t-M}), using common response times t=M+1,...,T.
The gradient returns a unit direction u and projections z_t=Y_t^T u.
These are already calculated by
[descent.py](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/ar_tools/descent.py:82).

When u_0 is nonzero, define

$$\widehat b_j=-u_j/u_0,\qquad
\widehat e_t=z_t/u_0=X_t-\sum_{j=1}^M\widehat b_jX_{t-j}.$$

A direct estimate of the symmetric innovation radius is

$$\widehat h_{\max}=\max_t|\widehat e_t|
=\frac{\max_t|z_t|}{|u_0|},\qquad
\widehat W=2\widehat h_{\max}.$$

This identity is an original algebraic application of the coefficient recovery
in [the thesis, Remark `rem:minimizer_estimates_coefficients`](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/3_application_to_time_series.tex:302).
It requires no regression refit. Half the achieved width gives the related
estimate ptp(z)/(2|u_0|); it differs from max|z|/|u_0| by the absolute residual
midpoint. A symmetric interval about known zero therefore requires the
maximum-absolute version to cover these particular residuals.

## 2. Why division by the first coordinate fails after overembedding

The population minimizers include shifts of the AR equation:

$$U(B)=\pm B^r A(B)/\|A\|_2,\qquad
A(B)=1-\phi_1B-\cdots-\phi_pB^p.$$

Here B is the lag variable and ||A||_2 is the Euclidean coefficient norm.
For r>0, u_0=0, so the direct ratio is undefined even at a correct population
minimum. In finite samples, a nearly shifted direction has a tiny first
coordinate and an unstable ratio. These shifts are explicitly allowed by
[the thesis, Proposition `prop:minimum_width`, equation `eq:minimizers`](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/3_application_to_time_series.tex:279).

One way to resolve the shift without choosing a coefficient threshold is
minimum-phase normalization. Root reflection is a standard filter operation
that preserves frequency-response magnitude; see
[Julius O. Smith, *Introduction to Digital Filters*, section "Minimum-Phase/Allpass Decomposition"](https://www.dsprelated.com/freebooks/filters/Minimum_Phase_Allpass_Decomposition.html).
The following application to the AR gradient output is our proposed
construction, rather than a theorem from that reference.

Factor U(B) using the lag-variable convention. For every root a with |a|<1,
replace its factor (B-a) by (1-conj(a)B). Keep the factors whose roots have
|a|>=1. The identity

$$|B-a|=|1-\overline a B|,\qquad |B|=1,$$

shows that this preserves |U| on the unit circle. In particular, a zero root
replaces a pure delay B by 1. This is the lag-variable version of reflecting
zeros; the direction of reflection is reversed when using z=B^{-1}.
Choose the real output v with v_0>0 and unit coefficient norm. For exact
population minimizing directions of a stable AR process, the construction
gives v=A/||A||_2 independently of the shift and sign.

Use the resulting residuals and radius

$$\widehat e_t^{\rm can}=\frac{Y_t^Tv}{v_0},\qquad
\boxed{\widehat h_{\rm can}=\max_t|\widehat e_t^{\rm can}|}.$$

This needs polynomial roots and one projection calculation, with no second
regression. The root operation preserves frequency magnitude; it does **not**
preserve the finite-sample range, maximum absolute residual, or innovation
distribution of an arbitrary imperfect direction. Recompute the residuals.
Unit-circle roots are boundary cases and do not establish strict AR stability.

An even more direct achieved-width diagnostic is available. Put g=v_0. At an
exact population minimizer g=1/||A||_2 and m=2h/||A||_2, so h=m/(2g).
The corresponding sample quantities are ptp(z)/(2g) or max|z|/g. They retain
the original gradient projections, but do not guarantee that the canonical
residuals fit inside the estimated interval. They are not the same estimator
as h_can in finite samples.

For specifically Uniform[-h,h] innovations, the residual second moment gives
another candidate, sqrt(3 mean(e_hat_can^2)). The uniform identity
Var(epsilon)=h^2/3 follows from
[NIST/SEMATECH e-Handbook, section 1.3.6.6.2, "Common Statistics"](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3662.htm).
Applying that identity to fitted residuals is an additional estimation step,
not a guarantee that they are innovations or that their extremes are covered.

## 3. Check on all 50 saved AR(3) runs

The diagnostic uses the saved **Padded warm start** directions at M=8 (k=9),
T=2000, with innovations Uniform[-1,1]. No gradient fit was rerun. Every
candidate order uses the same 1992 response times t=9,...,2000, and there is
no intercept. The support estimate is then plugged into the smallest feasible
order q in {0,...,8}; all residuals must lie in the estimated symmetric bound.
The order search uses linear feasibility programs, which are separate from
the computation of the gradient-derived bound. The true bound is supplied
only to the labelled benchmark, not to a gradient estimate. M=8 matches the
earlier pilot comparison; it was not selected using these outcomes.

| Bound estimate | Median h estimate | Minimum–maximum | Correct order /50 | Under | Over | No feasible order |
|---|---:|---:|---:|---:|---:|---:|
| Direct projection maximum / absolute first coordinate | 46.238 | 0.998–1727.207 | 5 | 44 | 1 | 0 |
| Canonical residual maximum | 1.050 | 0.998–5.322 | 32 | 17 | 1 | 0 |
| Canonical uniform moment | 1.008 | 0.984–3.174 | 29 | 9 | 3 | 9 |
| Original projection maximum / canonical gain g | 1.026 | 0.996–5.055 | 31 | 17 | 2 | 0 |
| Known h=1, benchmark | 1.000 | 1.000–1.000 | 49 | 1 | 0 | 0 |
| Separate OLS maximum residual, comparison | 1.072 | 1.024–1.140 | 44 | 6 | 0 | 0 |

The machine-readable values are in [summary.csv](/Users/iljasukhanov/MASTERARBEIT/output/analysis/gradient_bound_byproduct/summary.csv)
and every replication is in [selections.csv](/Users/iljasukhanov/MASTERARBEIT/output/analysis/gradient_bound_byproduct/selections.csv).
These are reused development simulations, not fresh validation. The earlier
40/50 result concerned a different selector and optimization procedure; it
should not be compared as though only the bound estimator had changed.

The canonical maximum lies below the true radius in 1/50 runs; the moment
estimate does so in 13/50. Conversely, some canonical estimates substantially
overstate the radius and permit lower-order models. A plausible median does
not resolve either problem. This experiment does not support replacing the
44/50 OLS pilot by the current gradient-only pilot.

Verification in [check.py](/Users/iljasukhanov/MASTERARBEIT/output/analysis/gradient_bound_byproduct/check.py)
includes all possible exact shifted AR(3) population directions at k=9 for
all 50 saved coefficient vectors. The largest canonical coefficient error
was 1.62e-15. On a 1024-point unit-circle grid, the largest frequency-magnitude
change in the empirical filters was 3.96e-15. Input hashes and settings are in
[verification.json](/Users/iljasukhanov/MASTERARBEIT/output/analysis/gradient_bound_byproduct/verification.json).

## 4. What this proves, and what remains

Here is a conditional original argument for estimating h. Suppose the
canonical coefficients b_hat=-v[1:]/v_0 converge in probability to the
true order-M coefficient vector b_0=(phi_1,...,phi_p,0,...,0). Bounded
innovations and stable causality give a deterministic finite lag bound R.
Then

$$\left|\max_t|\widehat e_t^{\rm can}|-\max_t|\varepsilon_t|\right|
\le R\|\widehat b-b_0\|.$$

Since the independent innovations have full support [-h,h], their absolute
sample maximum converges to h. The inequality therefore establishes
h_hat_can -> h in probability **under the stated coefficient-consistency
assumption**. It does not establish that the existing fixed-budget gradient
algorithm satisfies that assumption. Exact recovery of the population
minimizers in Section 2 is an algebraic identity, not an optimizer-convergence
proof.

For smallest-feasible-order selection, ordinary bound consistency is not
enough. Also establish

$$\Pr\{r_{p,T}\le\widehat h_T\}\longrightarrow1,\qquad
r_{p,T}=\inf_{b\in\mathbb R^p}\max_t|X_t-b^TL_t^{(p)}|.$$

See the original plug-in argument and counterexample in
[estimated_support_findings.md, section 2](/Users/iljasukhanov/MASTERARBEIT/output/analysis/ar_order_without_penalties/estimated_support_findings.md:35).
The canonical maximum guarantees feasibility of its own order-M filter on
the observed sample. It does not automatically guarantee feasibility at the
true smaller order p. No one-sided coverage proof for this gradient-derived
construction is claimed here.

## 5. Why the gradient-derived bound gives only 32/50

The failure audit is in
[diagnose_failures.py](/Users/iljasukhanov/MASTERARBEIT/output/analysis/gradient_bound_byproduct/diagnose_failures.py),
with per-run values in
[failure_diagnostics.csv](/Users/iljasukhanov/MASTERARBEIT/output/analysis/gradient_bound_byproduct/failure_diagnostics.csv)
and counts in
[failure_summary.json](/Users/iljasukhanov/MASTERARBEIT/output/analysis/gradient_bound_byproduct/failure_summary.json).
All calculations use the same saved directions, common response times, and
feasibility definition as Section 3. No selector or gradient setting changes.

For each q, let r_q be the smallest achievable maximum absolute residual
among order-q coefficients on these observations. Because the candidate
models are nested, the smallest-feasible-order rule returns exactly order 3
if and only if

$$r_3\le\widehat h<r_2.$$

This is an original direct consequence of the rule, not an external theorem.
Minimax linear programs compute r_2 and r_3; the interval condition matches
all 50 saved classifications. Thus closeness to the true h=1 is useful only
relative to the particular sample's r_2-r_3 gap.

The 18 failures split as follows:

* **16 additional underselections relative to the known-bound benchmark.**
  In every one, r_2>1 but h_hat>=r_2. Replacing the correct bound by the
  gradient estimate admits an AR(2) or still smaller model.
* **One underselection already present with known h=1 (run 40).** Its
  order-2 radius is 0.998116652, so an AR(2) fits all observed constraints
  even with the correct bound. Here phi_3=-0.003177955, consistent with a
  weak third-lag signal; the exact cause established by the constraints is
  finite-sample feasibility of the lower order.
* **One overselection (run 37).** The estimate 0.997979733 is smaller than
  r_3=0.998629260, so even the best order-3 coefficients cannot satisfy the
  estimated interval. The rule first becomes feasible at order 6.

Examples, with values rounded only for display:

| Run | r_3 | r_2 | Estimated h | Selected order |
|---|---:|---:|---:|---:|
| 2 | 0.998675 | 1.219937 | 1.463870 | 2 |
| 26 | 0.997969 | 1.122240 | 5.321531 | 0 |
| 32 | 0.998577 | 1.017792 | 1.063583 | 2 |
| 37 | 0.998629 | 1.280382 | 0.997980 | 6 |

Run 32 shows why a modest bound error can matter: its useful interval is
approximately [0.99858,1.01779). An estimate 6.36% above the true bound is
already too large for this series.

### Fit quality and the width objective

There is also evidence against attributing every failure to too few gradient
iterations. For each saved process, form all six shifted, normalized **true**
AR(3) directions at k=9 and evaluate their empirical widths on the same
observations. Any smaller such witness width certifies that the achieved
gradient width is not globally minimal. Five of the 16 additional underfits
have such a certificate. In the other 11, the achieved gradient width is
already smaller than the empirical width of **every** shifted true direction.
Truth is used only for this diagnostic, not by the estimator.

For example, in run 26 the achieved original width is 1.104512, compared with
1.261992 for the narrowest shifted true direction. Nonetheless, the canonical
direction has v_0=0.111439872, and its implied residual bound is 5.321531.
Consequently a small geometric sample width does not certify a good
innovation-bound estimate. The original objective minimizes a unit-direction
range; the bound construction additionally changes the filter and divides
its projections by v_0. The exact population identity does not control those
finite-sample operations for an arbitrary returned direction.

This witness check does not determine the global empirical minimizer or
prove that more accurate optimization could never help. It establishes why
"just minimize the width more accurately" is not yet a justified complete
explanation or remedy for the observed bound-estimation failures.
