# Proposition 8: remaining fixes

Updated on 12 October 2026. Only outstanding actions are retained, using the original checklist numbers: **7, 8, 10, 11, 13, 14**. The thesis source was not edited.

NOW means work needed while temporarily assuming the red estimate. RED means work proving that estimate. BOTH means shared or split work.

The temporary assumption (R) is: for every fixed A>0, a_N(A)>=0 is independent of j and tends to zero, with

\[
\|r_{N,j}\|_2\le |\delta_{ij}^{r,N}|a_N(A)
\quad\text{for every }j\in I_N(A).
\]

Work order: item 10 (including the constant needed in 7), remaining details in 11, downstream notation in 13, and the row-sum interpretation in 14. Then finish RED work in 8 and the remaining parts of 7/13. The Perron claim in 14 needs a separate theorem.

All findings and correction formulas below are original checks of [the thesis chapter](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:718); cited sources provide the general analysis results.

## 7. Remainder inequalities - BOTH

**NOW:** Define the M_0 at [line 840](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:840) to bound h^(1-r)||Psi'(h)||_2 as well as the unweighted derivative at line 826. Item 10 gives one constant covering both.

**RED:** At [line 818](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:818), use |tau_N,j| in the quotient when tau_N,j is nonzero. At j=i_N, the increment and remainders are zero; handle this separately. The integral argument avoids division entirely.


## 8. First red estimate and general integral formula - PARTIAL / RED

The new argument at [lines 822-830](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:822) uses the right approach, but does not yet prove (R).

### General formula before Lemma 7

At [line 726](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:726), replace the current integrand d(Psi(u+th)-dPsi(u))[h] by

\[
r_u(h)=\int_0^1\bigl(d\Psi(u+th)-d\Psi(u)\bigr)[h],dt.
\]

The current expression mixes a function value with a derivative operator. State that Psi is C^1 in the operator norm on an open neighborhood of the entire segment {u+th:0<=t<=1}, which lies in U. C^1 on an open convex U is a sufficient alternative. Mere Frechet differentiability is not the hypothesis being used here.

The formula is my derivation from the Banach-valued fundamental theorem of calculus applied to t -> Psi(u+th): [Zeidler, Applied Functional Analysis: Main Principles and Their Applications, §4.4, Corollary 5, p. 242](https://www.scribd.com/document/681807107/Applied-Functional-Analysis-Main-Principles-and-Their-Applications-Zeidler-Compress). Taking the norm inside the integral uses Proposition 4(ii), Eq. (33), p. 241, of that same source. The bound at line 728 is correct under these hypotheses.

### Lost factor and j-independent sequence

The grid-correction term at line 829 should retain its square:

\[
\|r_{N,j}\|_2\le L|\delta|\mu(|\tau_{N,j}|)+C_gM_0|\delta|^2,
\qquad L=b^{1-r}.
\]

The current looser bound with C_g M_0 |delta| is not false when |delta|<=1, but its coefficient C_g M_0 does not tend to zero in general. It therefore cannot prove the claimed small-o remainder. After factoring out |delta|, the second term inside the parentheses must still contain |delta|. Also, mu(|tau_N,j|) depends on j, so it cannot be the final definition of a sequence independent of j.

Define the modulus on the fixed positive interval J=[a,b]:

\[
\mu(s)=\sup_{\substack{h,k\in J\\|h-k|\le s}}\|\Psi'(h)-\Psi'(k)\|_2.
\]

Item 10 supplies continuity of Psi', hence uniform continuity on J and mu(s)->0. For this compact-interval fact, see [Zeidler, §4.4, Standard Example 3, p. 240](https://www.scribd.com/document/681807107/Applied-Functional-Analysis-Main-Principles-and-Their-Applications-Zeidler-Compress). The modulus is nondecreasing by its definition. For all local indices, |delta|<=A/sqrt(lambda_N) and |tau|<=L|delta|, so the corrected uniform choice is

\[
\boxed{a_N(A)=L\mu\!\left(\frac{LA}{\sqrt{\lambda_N}}\right)
+\frac{C_gM_0A}{\sqrt{\lambda_N}}\longrightarrow0.}
\]

This is my direct estimate from the thesis's u/r decomposition. It requires no second derivative of Psi.

### Uniform grid constants and preparatory lemma

The two inequalities at line 824 are valid uniformly, but cite or give their compact-interval justification. For v=h^r/r and G(v)=(rv)^(1/r), direct differentiation gives G'=h^(1-r) and G''=(1-r)h^(1-2r). The scalar mean value theorem and Taylor theorem yield

\[
|\tau|\le b^{1-r}|\delta|,
\quad
|\tau-h_{i_N}^{1-r}\delta|\le C_g|\delta|^2,
\quad
C_g=\frac{1-r}{2}\max\{a^{1-2r},b^{1-2r}\}.
\]

These bounds actually hold for every pair of grid points. See [DLMF Eq. (1.4.11)](https://dlmf.nist.gov/1.4#E11) and [§1.4(vi), Taylor's theorem](https://dlmf.nist.gov/1.4#vi). Only G is differentiated twice here; the KDE route uses C^1.

The earlier Lemma 7 proof still needs two repairs: line 749 must use the interval with ordered endpoints min(h,h+tau), max(h,h+tau) for negative tau; line 750 must use <= when enlarging that interval to (h/2,3h/2). These are deferred RED setup corrections.

## 10. Derivative bound and energy properties - OPEN / BOTH

The bound M_0 at [line 826](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:826), the weighted M_0 at line 840, and continuity of Q_r at line 859 remain asserted without their analytic justification. Restrict the bound to h in the fixed J=[a,b]. Lemma 7 proves existence of the derivative, but does not state or prove continuity of h -> Psi'(h).

Here is my Gaussian applicability check using the [Fourier representation at lines 358-366](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:358):

\[
\mathcal F[\Psi'(h)](\omega)=-h\omega^2P_n(\omega)e^{-h^2\omega^2/2},
\qquad
|\mathcal F[\Psi'(h)](\omega)|\le b\omega^2e^{-a^2\omega^2/2}.
\]

The envelope is square integrable. Pointwise continuity in h and dominated convergence applied to squared differences, followed by Parseval, establish L^2 continuity. For the general convergence theorem, see [Hunter, Measure Theory, Theorem 4.24, p. 49](https://www.math.ucdavis.edu/~hunter/measure_theory/measure_notes_ch4.pdf#page=11).

Let B_0=sup_J||Psi'(h)||_2, which is finite. To keep the thesis's current use of the same M_0 in both places, define

\[
M_0:=\max\{1,b^{1-r}\}B_0.
\]

Then both ||Psi'(h)||_2<=M_0 and h^(1-r)||Psi'(h)||_2<=M_0 for h in J. Thus the current b_N(A)=M_0 a_N(A)+(1/2)a_N(A)^2 remains valid. Alternatively, retain separate weighted and unweighted constants, but using one sufficiently large constant is simpler. The same Fourier computation gives

\[
\hat Q_r(h)=\frac{h^{4-2r}}{4\pi}\int_{\mathbb R}|P_n(\omega)|^2\omega^4e^{-h^2\omega^2}\,d\omega>0.
\]

Continuity follows from the same domination; positivity follows because P_n is continuous, P_n(0)=1, and the integrand is positive on a punctured neighborhood of zero. This one first-derivative check supplies the two M_0 bounds, the modulus needed for (R), and continuity/positivity of Q_r. No assumption on unknown f is needed.

## 11. Riemann-sum step - PARTIAL / NOW

Finish these details:

1. Explicitly state
   \[
   \epsilon_N=\frac{b^r-a^r}{r}\frac{\sqrt{\lambda_N}}{N-1}\to0
   \]
   by lambda_N/N^2->0.
2. Define m_N=floor(A/epsilon_N). Replace m_n and the unindexed epsilon by m_N and epsilon_N.
3. Your width k epsilon_N-(k-1)epsilon_N with the value g(k epsilon_N) at [line 876](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:876) is a valid RIGHT-endpoint rule. Its rectangle interval is
   \[
   [-(m_N+1)\epsilon_N,\ m_N\epsilon_N],
   \]
   so use these actual endpoints, both tending to -A and A respectively. Alternatively, a left-endpoint rule uses width (k+1)epsilon_N-k epsilon_N and matches the currently stated endpoint pair. Either rule works; choose one consistently.
4. In the first endpoint identity at line 872, use delta_{i_N,1} or omit the generic delta_{ij} and keep the explicit powered-bandwidth formula.

For the current right rule, let g(t)=exp(-t^2 Q_r(h_0)). My direct bound is

\[
\left|\epsilon_N\sum_{k=-m_N}^{m_N}g(k\epsilon_N)
-\int_{-A}^{A}g(t)\,dt\right|
\le(2A+\epsilon_N)\omega_g(\epsilon_N)+2\epsilon_N\to0,
\]

where omega_g is the modulus of continuity on [-A-1,A+1] and N is large enough that epsilon_N<=1. The first term bounds the rectangle error; the second bounds the endpoint pieces because 0<g<=1. This is the needed application of [DLMF §1.4(v), Eq. (1.4.18)](https://dlmf.nist.gov/1.4#E18), with endpoint errors checked explicitly. No new assumption is required.

## 13. Consistent notation - OPEN / BOTH

At [lines 813-856](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:813), use one notation for delta and tau, r_N,j instead of r_N.j, eta_N,j instead of eta_Nj, rho_N,j instead of rho_Nj, and S^(N) consistently. Still define m_N=floor(A/epsilon_N), replace m_n and unindexed epsilon in lines 876/879, and fix the generic delta_{ij} in the endpoint identity at line 872. Use r_u(h) in the quotient at line 722. Define M_0 so both uses are valid, as specified in item 10; keep the current b_N(A) convention.

Downstream notation can be completed under (R). The u/tau notation can be completed with RED work. These are source checks, not additional mathematical hypotheses.

## 14. Interpretation and Perron claim - OPEN

The [red sentence at line 894](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:894) should state an asymptotic comparison of fixed interior limiting bandwidths, rather than an unqualified finite-grid implication. My consequence of Proposition 8 is

\[
\frac{R_{i_N}^{(N)}}{R_{k_N}^{(N)}}\to
\sqrt{\frac{\hat Q_r(s)}{\hat Q_r(h)}}
\quad\text{when }h_{i_N}\to h,\ h_{k_N}\to s
\]

for h,s in the fixed interior. Use Q_r consistently, or explain beta=2(1-r) to identify Q_{n,beta} with it. With v=h^r/r and F(v)=Psi((rv)^(1/r)), Q_r(h)=(1/2)||F'(v)||_2^2, so the slow-change interpretation concerns this transformed coordinate. This is squared speed, as a direct consequence of the displayed definition, rather than curvature.

The [heading at line 714](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:714) claims Perron-vector concentration; the [TODO at line 901](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:901) still records its missing connection. Proposition 8 gives pointwise interior row-sum asymptotics, not that separate eigenvector theorem or a result about maximizers over the whole grid. Proving (R) alone will not establish Perron concentration. The existing Laplace TODO at line 903 can be removed after the present proof is completed if no alternative proof is intended; invoking Laplace is not an extra requirement for this argument.
