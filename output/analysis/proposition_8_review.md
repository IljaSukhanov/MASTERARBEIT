# Review of Proposition 8

Rechecked against the latest thesis source on 11 October 2026. No thesis files were changed.

Current status of the numbered PDF checklist: **items 1-4 are resolved; item 6 needs one sentence corrected; items 5 and 7-14 remain open**. The integer-count argument now shifts both endpoints correctly. The tail sums, scaled limsup, and final Gaussian-tail constant are also correct; only the sentence at line 802 still claims boundedness of the unscaled T_N(A). Item numbers in this status refer to the PDF checklist, not the grouped subsection numbers in this longer review.

The proposition reviewed is the row-sum limit in [the bandwidth-selection chapter, line 765](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:765). The calculations below are my derivations from the thesis's Gaussian Fourier formula and grid definition; cited external results provide the general Taylor and Laplace tools.

## Verdict and scope

The asserted limit is correct for every fixed finite sample of real observations, with fixed sample size, fixed positive grid endpoints, and fixed grid exponent. The proof currently contains errors beyond the two red remainder statements. They are repairable without changing the asserted limit. A complete argument is given below.

**Checklist item 1 - resolved.** The proposition now specifies fixed data X_1,...,X_n, fixed positive grid endpoints, and a fixed exponent r. This correctly fixes the sample and its size before N tends to infinity. Sampling from a density is unnecessary for this deterministic statement conditional on the observations. The current argument does not establish a joint limit with n = n(N), or with endpoints tending to zero. This scope follows from the proof below and the [current statement](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:766).

**Checklist item 2 - resolved.** The proposition now defines P_2 := P intersect L^2(R) and takes D : P_2 x P_2 -> R_{>=0}. This fixes the finite-domain issue for the L^2 divergence. The thesis's [definition of P](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:160) still correctly includes probability densities that need not be square integrable. The Gaussian KDEs used here belong to P_2, as follows from their [Fourier representation](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:358).

## Repairs beyond the red statements

### 1. Grid-spacing lemma: resolved (PDF checklist item 3)

The [lemma at line 544](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:544) and its proof now establish the claimed conclusion. The statement explicitly fixes positive endpoints. The proof uses the powered grid identity, c_N = O(delta_{ij}^{r,N}), the correct o(N) assumption, and the O((delta_{ij}^{r,N})^2) remainder. The incorrect auxiliary N c_N calculation and misleading explanation of o(N) have been removed. The O(1) condition is correctly described as stronger at line 564.

In particular, Delta_{r,N} is a constant multiple of 1/(N-1), so the o(N) index condition implies delta_{ij}^{r,N} -> 0 and then c_N -> 0. The fixed upper endpoint also justifies absorbing h_{i_N} into the final remainder constant. No further change is needed for checklist item 3.

For a uniform version of the bound useful in Proposition 8, my derivation sets u = h^r/r and g(u) = (ru)^(1/r). On the fixed transformed interval,

\[
g'(u)=h^{1-r},\qquad g''(u)=(1-r)h^{1-2r}.
\]

The second derivative is bounded because h stays in [a,b], where a = h_min > 0 and b = h_max. Taylor's theorem therefore gives, uniformly in i,j,

\[
h_j-h_i=h_i^{1-r}(u_j-u_i)+O((u_j-u_i)^2).
\]

This is my application of Taylor's theorem to the [grid definition](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:517). It supports the uniform quadratic estimate independently of which local grid indices are chosen. For the scalar remainder formula, see [DLMF Section 1.4(vi), equations (1.4.35)-(1.4.36)](https://dlmf.nist.gov/1.4#vi). The thesis proof still needs the separate uniform KDE remainder check recorded in PDF checklist item 8.

### 2. Correct the global lower bound

The [calculation at lines 785 onward](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:785) has a sign problem, a bandwidth/index typo, an exponent error after squaring, and an incorrect conversion constant.

Using the thesis's [Fourier convention and P_n](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:358), the corrected calculation is

\[
\left|e^{-s^2\omega^2/2}-e^{-h^2\omega^2/2}\right|
=\int_{\min(h,s)}^{\max(h,s)}t\omega^2e^{-t^2\omega^2/2}\,dt
\ge |s-h|a\omega^2e^{-b^2\omega^2/2}.
\]

Consequently,

\[
D(h,s)\ge\frac{a^2}{4\pi}|s-h|^2
\int_{\mathbb R}|P_n(\omega)|^2\omega^4e^{-b^2\omega^2}\,d\omega.
\]

Since |u(s)-u(h)| <= a^(r-1)|s-h|, a valid constant for D(h,s) >= c|u(s)-u(h)|^2 is

\[
c=\frac{a^{4-2r}}{4\pi}
\int_{\mathbb R}|P_n(\omega)|^2\omega^4e^{-b^2\omega^2}\,d\omega>0.
\]

Positivity follows directly because P_n is continuous and P_n(0)=1; the integrand is positive on a sufficiently small punctured interval around zero. These are my corrected estimates, not a quotation of an external theorem.

### 3. Normalize the tail consistently

You define T_N(A) as an unscaled sum at [line 782](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:782). The ensuing Gaussian-tail estimate is valid for epsilon_N T_N(A), not for T_N(A) itself. With epsilon_N = sqrt(lambda_N)Delta_{r,N}, write

\[
\epsilon_N T_N(A)
\le 2\epsilon_N\sum_{k=\lfloor A/\epsilon_N\rfloor+1}^{\infty}
e^{-c\epsilon_N^2k^2}
\le 2\int_{A-\epsilon_N}^{\infty}e^{-ct^2}\,dt
\]

for sufficiently large N. Thus

\[
\limsup_{N\to\infty}\epsilon_N T_N(A)
\le2\int_A^\infty e^{-ct^2}\,dt
\le\frac{e^{-cA^2}}{cA}.
\]

**Checklist item 6 - one sentence remains.** The epsilon_N factors at lines 798-799, the scaled limsup at line 804, and the final constant 1/(cA) are now correct. Once the lower-bound calculation in item 5 is repaired, the displayed tail argument is valid for sufficiently large N, when epsilon_N < A. At line 802, replace "T_N(A) is asymptotically bounded" by "epsilon_N T_N(A) is asymptotically bounded for every fixed A > 0". The estimate does not show that the unscaled T_N(A) is asymptotically bounded. The passage to the displayed limsup uses epsilon_N -> 0. These are my checks of the current sums and the decreasing-Gaussian integral comparison.

### 4. Use absolute values in the remainder estimate

At [lines 812 and 822 onward](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:812), divide by |tau_{N,j}| when the increment is nonzero, and use

\[
|\rho_{N,j}|\le
|\delta_{ij}^{r,N}|h_{i_N}^{1-r}
\|\partial_h\hat f_h|_{h=h_{i_N}}\|_2\|r_{N,j}\|_2
+\tfrac12\|r_{N,j}\|_2^2.
\]

The signed delta or signed inner product cannot bound an absolute value. At j=i_N, all the remainders vanish; no division is needed. The corrected bound follows directly from your formula for rho and the norm inequality for an inner product.

### 5. Give the correct reason for the exponential error tending to zero

The sentence at [line 832](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:832) attributes this to lambda_N/N^2 tending to zero. On I_N(A), the relevant estimate is instead lambda_N(delta_{ij}^{r,N})^2 <= A^2 together with the uniform remainder bound. The mesh condition makes epsilon_N tend to zero and is used for the sum approximation. The explicit exponential estimate is given below.

### 6. Correct the final passage to the full limit

The [equality at line 863](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:863) is false for fixed A: the full row sum does not have the same limit as the truncated sum. Set

\[
L_A=\int_{-A}^{A}e^{-t^2\hat Q_r(h_0)}\,dt.
\]

After establishing convergence of the local sum and the corrected tail bound, write

\[
L_A\le\liminf_N\epsilon_NR_{i_N}^{(N)}
\le\limsup_N\epsilon_NR_{i_N}^{(N)}
\le L_A+2\int_A^{\infty}e^{-ct^2}\,dt.
\]

Now let A tend to infinity. This proves the claimed limit without assuming its existence beforehand. This is my repair of the limit argument using the nonnegative tail.

### 7. Remaining bookkeeping

- **Checklist item 4 - resolved.** At [line 778](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:778), both endpoints now have the shift i_N. Since i_N is an integer, it cancels from the cardinality calculation. The floor/ceiling formula and the final bound are correct. These are my calculations from the local-window definition; the [official floor and ceiling definitions](https://docs.python.org/3/library/math.html#math.floor) give the integer endpoints.
- Before the Riemann-sum step, note that sqrt(lambda_N)(u_{i_N}-u_1) and sqrt(lambda_N)(u_N-u_{i_N}) both tend to infinity. Hence I_N(A) is not truncated by the grid endpoints for fixed A and large N. This follows from h_{i_N} tending to an interior h_0.
- Use one notation for delta_{ij}^{r,N} and tau_{N,j} throughout; currently delta_N, delta_{N,j}, tau_N and r_{N.j} appear without consistent definitions.
- State why Q_r is continuous and strictly positive; the calculation below supplies both properties. The existing derivative lemma establishes differentiability, but its statement does not supply the uniform Taylor bound needed at moving base points.

For item 4, set B = A/epsilon_N. The current calculation is equivalent to the following form, with the row indices explicitly restricted to 1,...,N:

\[
I_N(A)=\{j\in\{1,\ldots,N\}:|j-i_N|\le B\},\qquad
\#I_N(A)\le\#\{k\in\mathbb Z:-B\le k\le B\}
=\lfloor B\rfloor-\lceil-B\rceil+1
=2\lfloor B\rfloor+1\le2B+1.
\]

Equivalently, the shifted set is {j in Z : i_N-B <= j <= i_N+B}. Its count is floor(i_N+B)-ceil(i_N-B)+1 = 2 floor(B)+1 because i_N is an integer. This is now the interval used in the thesis. No further correction is needed for item 4.

## Fill the red statements by citation and a short applicability check

Keep the current proof structure if you prefer. You do not need to reprove Banach-space Taylor's theorem. The verified statement in [BYU Math 346, Theorem 6.6.9, with the norm bound in Corollary 6.6.14](https://uamte.math.byu.edu/~bakker/Math346/Lectures/M346Lec06.pdf) provides an integral remainder for C^k maps between Banach spaces. Applying it with k=2 gives a uniform quadratic remainder when the second derivative is bounded on the relevant interval.

Your existing Zeidler reference has a Taylor section: Section 4.5, beginning on page 243, as verified in the [publisher's table of contents, printed page x](https://link.springer.com/content/pdf/bfm:978-1-4612-0821-1/1). I verified this locator, but not the full theorem text in that section. The BYU result above is the statement checked for this review. You can use Zeidler instead after matching its theorem to the needed norm estimate in your copy.

### Applicability to this KDE family

Define

\[
u=h^r/r,\quad g(u)=(ru)^{1/r},\quad F(u)=\hat f_{g(u)}.
\]

This makes u_j-u_{i_N} = delta_{ij}^{r,N} exactly, by the [grid definition](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:517). It bypasses the spacing lemma within Proposition 8.

For the standard Gaussian kernel, your [Fourier formula](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:358) gives

\[
\mathcal F[\partial_h\hat f_h](\omega)
=-h\omega^2P_n(\omega)e^{-h^2\omega^2/2},
\quad
\mathcal F[\partial_h^2\hat f_h](\omega)
=(h^2\omega^4-\omega^2)P_n(\omega)e^{-h^2\omega^2/2}.
\]

Here |P_n| <= 1. For h in [a,b], the absolute values are bounded by polynomial multiples of exp(-a^2 omega^2/2), which are square integrable. The same domination applies to the difference quotients on a slightly larger positive interval. Thus the derivatives exist in L^2 and depend continuously on h. Composing with g shows F is C^2 on a neighborhood of the compact transformed interval and

\[
B:=\sup_{u\in[u_1,u_N]}\|F''(u)\|_2<\infty,
\qquad
M:=\sup_{u\in[u_1,u_N]}\|F'(u)\|_2<\infty.
\]

These are my applicability checks from the Fourier representation, not extra regularity assumptions on the unknown density f. Explicitly, F' = h^(1-r) partial_h fhat_h and F'' = (1-r)h^(1-2r) partial_h fhat_h + h^(2-2r) partial_h^2 fhat_h.

The same representation gives

\[
\hat Q_r(h)=\frac12\|F'(h^r/r)\|_2^2
=\frac{h^{4-2r}}{4\pi}
\int_{\mathbb R}|P_n(\omega)|^2\omega^4e^{-h^2\omega^2}\,d\omega>0.
\]

Its positivity follows from P_n(0)=1 and continuity; its continuity follows from the displayed uniform domination. In particular, the denominator in the proposition is nonzero for every fixed finite sample.

### First red statement

The cited Taylor bound gives

\[
\hat f_{h_j}-\hat f_{h_{i_N}}
=\delta_{ij}^{r,N}F'(u_{i_N})+r_{N,j},
\qquad
\|r_{N,j}\|_2\le\frac B2|\delta_{ij}^{r,N}|^2.
\]

For j in I_N(A), |delta_{ij}^{r,N}| <= A/sqrt(lambda_N). Therefore the requested estimate holds with

\[
a_N(A)=\frac{BA}{2\sqrt{\lambda_N}}\longrightarrow0.
\]

This is the first red statement, uniformly over all j in I_N(A), with an explicit rate.

### Second red statement

Your corrected norm expansion then yields

\[
|\rho_{N,j}|\le(\delta_{ij}^{r,N})^2
\bigl(Ma_N(A)+\tfrac12a_N(A)^2\bigr).
\]

Consequently,

\[
\sup_{j\in I_N(A)}\lambda_N|\rho_{N,j}|
\le A^2\bigl(Ma_N(A)+\tfrac12a_N(A)^2\bigr)=:b_N(A)\to0,
\]

and

\[
\sup_{j\in I_N(A)}|\eta_{N,j}|
\le e^{b_N(A)}-1\to0.
\]

These are elementary consequences of the cited Taylor estimate; they do not need a separate KDE-specific theorem or a second long proof.

Together with the repaired tail estimate, continuity/positivity of Q_r, the local Riemann sum, and the corrected liminf/limsup argument above, these estimates complete the proof of Proposition 8.

## Optional shorter proof using Laplace's method

If you want to replace the local-sum calculation as well, a Laplace citation can shorten the proof. [NIST DLMF Section 2.3(iii), equations (2.3.14)–(2.3.17)](https://dlmf.nist.gov/2.3#iii) gives the asymptotic expansion for integrals with a minimum of the phase. Splitting at an interior quadratic minimum and applying the formula on each side gives the familiar Gaussian leading term. DLMF also points to Olver, *Asymptotics and Special Functions*, pages 80–88. A citation for an integral does not, by itself, justify your discrete sum or a moving minimum.

Here are the remaining checks, derived for this setting.

Let v=u_{i_N}, p_v(u)=(1/2)||F(u)-F(v)||_2^2, and q_{lambda,v}(u)=exp(-lambda p_v(u)). The Fourier formula shows p_v is strictly decreasing for u<v and strictly increasing for u>v: for each nonzero frequency, the squared difference between exp(-g(u)^2 omega^2/2) and exp(-g(v)^2 omega^2/2) has that behavior. Integration preserves the monotonicity. This also follows from the [Gaussian divergence result preceding Proposition 8](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:412).

Thus 0 <= q <= 1 and its total variation on [u_1,u_N] is at most 2. Comparing each left-rectangle area to its integral gives an error at most Delta_{r,N} times this variation; adding the final endpoint contributes at most Delta_{r,N}. Hence, uniformly in lambda and v,

\[
\left|\Delta_{r,N}\sum_{j=1}^{N}q_{\lambda,v}(u_j)
-\int_{u_1}^{u_N}q_{\lambda,v}(u)\,du\right|
\le3\Delta_{r,N}.
\]

Multiplication by sqrt(lambda_N) makes this error tend to zero by the assumed mesh condition.

For the integral, p_v has a unique zero at u=v, and

\[
p_v(v+z)=\hat Q_r(g(v))z^2+O(|z|^3)
\]

uniformly for v near u_0=h_0^r/r. The minimum stays away from the endpoints, and the positive coefficient tends to Q_0=Q_r(h_0). To justify uniformity rather than merely cite a fixed-phase theorem, for any 0<eta<Q_0 choose a common neighborhood where

\[
(Q_0-\eta)z^2\le p_v(v+z)\le(Q_0+\eta)z^2.
\]

The global bound p_v(u) >= c(u-v)^2 makes the remaining integral exponentially negligible after multiplication by sqrt(lambda). Gaussian upper and lower bounds then give limits bounded respectively by sqrt(pi/(Q_0-eta)) and sqrt(pi/(Q_0+eta)); letting eta decrease to zero establishes

\[
\sqrt{\lambda_N}\int_{u_1}^{u_N}e^{-\lambda_Np_{u_{i_N}}(u)}\,du
\longrightarrow\sqrt{\frac{\pi}{\hat Q_r(h_0)}}.
\]

Combining this with the rectangle estimate proves Proposition 8. This is an application of the cited Laplace result with the discrete approximation and moving-parameter checks made explicit.

## The red interpretation following the proposition

The [red sentence at line 871](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:871) is justified as an asymptotic comparison of row sums at fixed interior limiting bandwidths. For two such sequences with limits h and s, Proposition 8 implies

\[
\frac{R_{i_N}^{(N)}}{R_{k_N}^{(N)}}
\longrightarrow\sqrt{\frac{\hat Q_r(s)}{\hat Q_r(h)}}.
\]

Use the same functional notation, or explain that beta=2(1-r) identifies Q_{n,beta}(h)=(1/2)h^beta||partial_h fhat_h||_2^2 with Q_r(h). The slow-change interpretation refers to the transformed coordinate u=h^r/r, since Q_r=(1/2)||F'(u)||_2^2. Calling this a curvature functional is misleading: its displayed definition uses the first derivative, i.e. squared speed, rather than a second derivative.

Proposition 8 establishes row-sum asymptotics. It does not yet prove concentration of the Perron vector at a minimizer, convergence of a row-sum maximizer over the whole grid, or a joint statistical bandwidth-consistency result. These conclusions require further arguments; the thesis itself records the missing Perron connection in the [TODO at line 878](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:878). The pointwise interior result also does not give the stated full-Gaussian coefficient for boundary rows, where only one side of the local peak is available.
