# Proposition 8: remaining fixes

Updated on 12 October 2026. Only outstanding actions are retained, using the original checklist numbers: **7, 8, 10, 11, 13, 14**. The thesis source was not edited. Line references refer to the chapter inspected for this revision.

NOW means work for **13 October 2026**, temporarily assuming continuity (C). RED means the analytic proof of continuity in Lemma 7, left for later. BOTH means an item containing both kinds of work.

**(C)** The derivative map
\[
d\Psi:(0,\infty)\longrightarrow\mathcal L(\mathbb R,L^2(\mathbb R))
\]
is continuous in the **operator norm** for the KDE under consideration. This is exactly the [red continuity statement in Lemma 7](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:774). Its analytic proof is the **only deferred prerequisite** for Proposition 8.

The uniform remainder estimate in Proposition 8 is **no longer a temporary assumption**. Prove it tomorrow from (C), the integral remainder, and the grid bounds. Derive the modulus limit, derivative bounds, and continuity of the energy from (C); do not postpone them as independent unproved statements.

Work order: item 7 and the NOW setup in 8, item 10, the uniform remainder proof in 8, then 11, 13, and the interpretation in 14. Only the RED continuity proof is deferred. The separate Perron theorem is outside Proposition 8.

All findings and correction formulas below are original checks of [the thesis chapter](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:718); cited sources provide the general analysis results. The earlier integral-remainder integrand issue is resolved and has been removed from this open list.

## 7. Remainder inequalities and derivative-proof details - NOW

**NOW:** At [line 755](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:755), use the interval with ordered endpoints \(\min\{h,h+\tau\}\), \(\max\{h,h+\tau\}\), so negative \(\tau\) is covered.

At line 756, replace the equality by \(\le\) when enlarging that interval to \((h/2,3h/2)\).

At [line 833](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:833), use \(|\tau_{N,j}|\) in any nonzero-increment quotient. Prefer the uniform integral bound from item 8, which avoids division and the moving-base-point issue.

Explicitly handle \(j=i_N\): \(\delta_{N,j}=\tau_{N,j}=0\), and \(u_{N,j}=r_{N,j}=\rho_{N,j}=0\). The remainder inequalities hold without division.

## 8. First red estimate and Lemma 7 setup - BOTH

**RED:** Prove the red continuity statement in Lemma 7 at line 774. This is the only proof prerequisite left for later.

Return only to the analytic proof of continuity of \(d\Psi\). The operator-norm identity in item 8 reduces this to \(L^2\)-continuity of \(\partial_h\hat f_h\). For the Gaussian case, the dominated-convergence route discussed in this chat is available; see [Hunter, Measure Theory, Theorem 4.24, p. 49](https://www.math.ucdavis.edu/~hunter/measure_theory/measure_notes_ch4.pdf#page=11). If proving the statement at the full generality of Lemma 7, check the chosen argument against that lemma's kernel hypotheses. This scope check belongs to the same deferred continuity task.

### Modulus and integral remainder - NOW

**NOW:** Fix \(J=[a,b]\subset(0,\infty)\) and define one modulus:
\[
\mu_J(s):=\sup_{\substack{x,y\in J\\|x-y|\le s}}
\|d\Psi(x)-d\Psi(y)\|_{\mathrm{op}},\qquad s\ge0.
\]
The scalar-domain derivative formula gives the exact identity
\[
\|d\Psi(x)-d\Psi(y)\|_{\mathrm{op}}
=\|\Psi'(x)-\Psi'(y)\|_2.
\]

Under (C), invoke uniform continuity on the compact interval to state \(\mu_J(s)\to0\) as \(s\downarrow0\). The modulus is nonnegative and nondecreasing by definition. This is a standard consequence, not another deferred proof. Add the requested source at [line 781](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:781): [Zeidler, Applied Functional Analysis, §4.4, Standard Example 3, p. 240](https://www.scribd.com/document/681807107/Applied-Functional-Analysis-Main-Principles-and-Their-Applications-Zeidler-Compress), which invokes uniform continuity for a continuous Banach-valued function on a compact interval.

At [lines 778-779](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:778), use the operator norm for derivative operators. Replace the second equality by **\(\le\)** when enlarging the supremum from the line segment to all close pairs in \(J\). Avoid reusing the fixed base point as a bound variable. Obtain
\[
\|r_h(\tau)\|_2
\le|\tau|\sup_{t\in[0,1]}
\|d\Psi(h+t\tau)-d\Psi(h)\|_{\mathrm{op}}
\le|\tau|\mu_J(|\tau|).
\]

Make the closing sentence at line 781 conditional on (C) while its proof is deferred; the current sentence says continuity has already been established.

The integral formula uses [Zeidler, §4.4, Corollary 5, p. 242](https://www.scribd.com/document/681807107/Applied-Functional-Analysis-Main-Principles-and-Their-Applications-Zeidler-Compress); the norm inequality is Proposition 4(ii), Eq. (33), p. 241. Differentiability together with (C) gives the required \(C^1\) hypothesis.

### Uniform grid constants and the j-independent remainder - NOW

**NOW:** Define \(\delta_{N,j}:=(j-i_N)\Delta_{r,N}\) and \(\tau_{N,j}:=h_j-h_{i_N}\). Supply explicit constants for the [grid bounds at line 839](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:839). For \(v=h^r/r\) and \(G(v)=(rv)^{1/r}\), direct differentiation gives \(G'=h^{1-r}\), \(G''=(1-r)h^{1-2r}\), hence
\[
|\tau_{N,j}|\le L|\delta_{N,j}|,\qquad
|\tau_{N,j}-h_{i_N}^{1-r}\delta_{N,j}|
\le C_g|\delta_{N,j}|^2,
\]
\[
C_g=\frac{1-r}{2}\max\{a^{1-2r},b^{1-2r}\}.
\]
These constants work for every pair of grid points. The standard scalar results are [DLMF Eq. (1.4.11), mean value theorem](https://dlmf.nist.gov/1.4#E11), and [§1.4(vi), Taylor's theorem](https://dlmf.nist.gov/1.4#vi).

Retain the **square** in the grid-correction term at [line 844](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:844):
\[
\|r_{N,j}\|_2
\le L|\delta_{N,j}|\mu_J(|\tau_{N,j}|)
+C_gM_0|\delta_{N,j}|^2.
\]

For every \(j\in I_N(A)\), use \(|\delta_{N,j}|\le A/\sqrt{\lambda_N}\) and monotonicity of \(\mu_J\) **before defining** the coefficient. Set
\[
\boxed{a_N(A):=
L\mu_J\!\left(\frac{LA}{\sqrt{\lambda_N}}\right)
+\frac{C_gM_0A}{\sqrt{\lambda_N}}\longrightarrow0.}
\]
This proves \(\|r_{N,j}\|_2\le|\delta_{N,j}|a_N(A)\) uniformly over the local indices, with a coefficient independent of \(j\). Once written, the [red statement at line 835](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:835) is proved conditional only on (C).

Connect this to the existing \(b_N(A)=M_0a_N(A)+\tfrac12a_N(A)^2\) and exponential-error estimates at [lines 852-871](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:852). Their algebra is already in place; the proved remainder estimate and the correctly defined \(M_0\) supply the missing inputs.

## 10. Derivative bound and energy properties - OPEN / NOW

**NOW:** Set \(J=[a,b]=[h_{\min},h_{\max}]\), \(L=b^{1-r}\), and define
\[
B_0:=\sup_{h\in J}\|\Psi'(h)\|_2<\infty,\qquad
M_0:=\max\{1,L\}B_0.
\]
Finiteness follows from (C) and compactness. This one constant supplies both bounds needed at [lines 841 and 855](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:841):
\[
\|\Psi'(h)\|_2\le M_0,\qquad
h^{1-r}\|\Psi'(h)\|_2\le M_0,\qquad h\in J.
\]

State that \(\hat Q_r(h)=\tfrac12h^{2(1-r)}\|\Psi'(h)\|_2^2\) is continuous under (C), by continuity of the norm and scalar factor. This justifies [line 874](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:874); no separate continuity proof is needed tomorrow.

Prove strict positivity for the Gaussian KDE. My computation from the [chapter's Fourier representation and Parseval formula](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:358) gives
\[
\hat Q_r(h)=\frac{h^{4-2r}}{4\pi}
\int_{\mathbb R}|P_n(\omega)|^2\omega^4e^{-h^2\omega^2}\,d\omega>0.
\]
Here \(P_n(\omega)=n^{-1}\sum_{j=1}^ne^{-iX_j\omega}\) is continuous and \(P_n(0)=1\), so the integrand is strictly positive in a punctured neighborhood of zero. This proves positivity without proving the deferred continuity statement. The Fourier derivative used is \(\mathcal F[\Psi'(h)](\omega)=-h\omega^2P_n(\omega)e^{-h^2\omega^2/2}\); justify its identification from the existing \(L^2\) derivative and Fourier representation if it is not yet recorded.

## 11. Riemann-sum step - PARTIAL / NOW

Finish these details:

1. Explicitly state
\[
\epsilon_N=\frac{b^r-a^r}{r}\frac{\sqrt{\lambda_N}}{N-1}
\longrightarrow0
\]
from \(\lambda_N/N^2\to0\).

2. Define \(m_N=\lfloor A/\epsilon_N\rfloor\); replace \(m_n\) and the unindexed \(\epsilon\) at [lines 891-894](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:891).

3. Keep the current right-endpoint rectangles, of width \(k\epsilon_N-(k-1)\epsilon_N\). Their union is **\([-(m_N+1)\epsilon_N,m_N\epsilon_N]\)**. Use these endpoints consistently; they tend to \(-A\) and \(A\).

4. At [line 887](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:887), replace the generic \(\delta_{ij}^{r,N}\) in the identity for \(t_{N,1}\) by \(\delta_{N,1}\), or keep only the explicit powered-bandwidth formula. Endpoint divergence ensures that all indices \(-m_N,\ldots,m_N\) are available for large \(N\).

For an explicit error bound, let \(g(t)=e^{-t^2\hat Q_r(h_0)}\) and let \(\omega_g\) be its modulus on \([-A-1,A+1]\). My rectangle and endpoint estimate is
\[
\left|\epsilon_N\sum_{k=-m_N}^{m_N}g(k\epsilon_N)
-\int_{-A}^{A}g(t)\,dt\right|
\le(2A+\epsilon_N)\omega_g(\epsilon_N)+2\epsilon_N
\longrightarrow0.
\]
This uses continuity of \(g\), \(0<g\le1\), and \(\epsilon_N\le1\) for large \(N\). Reference: [DLMF §1.4(v), Eq. (1.4.18)](https://dlmf.nist.gov/1.4#E18). The existing tail bound and final liminf/limsup argument at lines 818-825 and 896-905 need no new argument once these local steps are completed.

## 13. Consistent notation - OPEN / NOW

**NOW:** Use \(\delta_{N,j}\) (or consistently \(\delta_{ij}^{r,N}\)) and \(\tau_{N,j}\) throughout; replace unindexed \(\delta_N\), \(\tau_N\), and the variant \(\delta_{i,j}^{r,N}\).

Replace \(r_{N.j}\), \(\eta_{Nj}\), and \(\rho_{Nj}\) by \(r_{N,j}\), \(\eta_{N,j}\), and \(\rho_{N,j}\). Use \(S^{(N)}\) consistently.

Use \(\mu_J\) consistently in the lemma and proposition. Use \(r_h(\tau)\) instead of \(r_u(\tau)\) in the lemma statement at line 738.

Remove the duplicated words in the definition of continuous differentiability at line 722. Use operator norms for operators and \(L^2\) norms for function values.

Keep explicit that the little-o estimates are uniform over all \(j\in I_N(A)\), for each **fixed** \(A\). The displayed \(a_N(A)\) and \(b_N(A)\) provide this uniformity.

## 14. Interpretation and Perron claim - OPEN

**NOW:** Qualify the [sentence after Proposition 8 at line 909](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:909) as an asymptotic comparison of fixed interior limits. My consequence of the proposition is
\[
\frac{R_{i_N}^{(N)}}{R_{k_N}^{(N)}}
\longrightarrow\sqrt{\frac{\hat Q_r(s)}{\hat Q_r(h)}}
\quad\text{if }h_{i_N}\to h,\ h_{k_N}\to s,\qquad h,s\in(a,b).
\]

Use \(\hat Q_r\) consistently, or explain \(\beta=2(1-r)\) to identify it with \(\hat Q_{n,\beta}\). With \(v=h^r/r\) and \(F(v)=\Psi((rv)^{1/r})\), direct differentiation gives \(\hat Q_r(h)=\tfrac12\|F'(v)\|_2^2\). The slow-change interpretation concerns this transformed coordinate; the functional is half the squared speed of the curve.

Make clear that the [Perron-concentration heading](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:714) and [Perron TODO](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:916) refer to a **separate theorem outside Proposition 8**. Its proof is not a prerequisite for tomorrow's Proposition 8 checklist; its unfinished status must remain visible.

Resolve the Laplace-method TODO at line 918 as a writing decision: it is unnecessary for the completed local-expansion/Riemann-sum proof. Keep it only if it refers to a separate intended argument.

Tomorrow's completion target: **Proposition 8 has a complete proof conditional on (C), with its notation and row-sum interpretation finished. The only deferred proof prerequisite is the red continuity statement in Lemma 7.** The modulus limit, constants, energy properties, uniform remainder, exponential error, Riemann sum, and final limit are all accounted for.

The separate Perron-vector theorem remains outside this completion target and must not be presented as a consequence already proved by Proposition 8.
