# Guided exercise: from row sums to Perron-vector concentration

This exercise is tailored to the current statement of `prop:main_thm` and its appendix proof. Fix the observations throughout: the limit concerns the grid size $N$ and the penalty $\lambda_N$, with $\lambda_N\to\infty$ and $\lambda_N/N^2\to0$. The additional uniqueness assumption below is an assumption for this exercise, not a claim about every dataset. [Your proposition](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:781); [your proof](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/9999_appendix.tex:325).

**The purpose.** Prove two distinct conclusions: the location of a largest Perron entry approaches a minimizer of $\hat Q_r$, and the sum-normalized Perron vector puts asymptotically all its mass in every neighborhood of that minimizer. The argument below is a derivation for your setting, not a theorem attributed wholesale to an external source. A related variational proof of peak convergence already appears in your [earlier research note, proof Steps 3–4](/Users/iljasukhanov/MASTERARBEIT/output/pdf/novelty_proposition_7_conjecture_1.tex:238); use the current thesis for the current hypotheses.

**How to use it.** First take the three estimates in Exercise 2 as given and solve Exercises 3–6. Then return to Exercise 2 to finish the analytic justification. [Separate solution notes](/Users/iljasukhanov/MASTERARBEIT/output/analysis/perron_concentration_exercise/solutions.md) are available when needed.

**1. Normalize the matrix and formulate the goal.**

Use your grid coordinate

$$
t_i=\psi_r(h_i),\qquad \psi_r(h)=(h^r-1)/r,\qquad
\Delta_N=\Delta_{r,N},\qquad [a,b]=\psi_r([h_{\min},h_{\max}]).
$$

Here $0<r\le1$, as in the current proposition. Define

$$
c_N=\sqrt{\lambda_N}\Delta_N,\qquad A_N=c_NS^{(N)},\qquad
a_{i,N}=\sum_j(A_N)_{ij},\qquad \theta_N=\rho(A_N),
$$

$$
q(t)=\hat Q_r(\psi_r^{-1}(t)),\qquad
m(t)=\sqrt{\pi/q(t)},\qquad M=\max_{t\in[a,b]}m(t).
$$

These definitions use the normalization and energy from [your row-sum proposition](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:781) and [your grid definition](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:527). Assume $q$ is continuous and strictly positive, and initially assume it has a unique minimizer $t_*\in(a,b)$. Write $h_*=\psi_r^{-1}(t_*)$.

Let $p^{(N)}>0$ solve $A_Np^{(N)}=\theta_Np^{(N)}$, with $\sum_i p_i^{(N)}=1$. Existence and uniqueness of this positive normalization follow from Perron's theorem because $A_N$ is entrywise positive. [Meckes, *Lecture notes on matrix analysis*, Theorem 8.10, p. 67](https://case.edu/artsci/math/mwmeckes/matrix-analysis.pdf#page=67).

Show that multiplication by $c_N>0$ preserves the eigenvectors and that

$$
\theta_N=c_N\rho(S^{(N)}).
$$

Our targets are

$$
\theta_N\to M,\qquad h_{k_N}\to h_*
\quad\text{for every }k_N\in\arg\max_i p_i^{(N)},
$$

and, for every fixed $\delta>0$,

$$
\sum_{|h_i-h_*|\ge\delta}p_i^{(N)}\to0.
$$

**2. Extract precisely the estimates needed from your proof.**

Establish the following three statements. Define $x_+=\max(x,0)$.

$$
\tag{I}
\sup_{i:t_i\in K}|a_{i,N}-m(t_i)|\to0
\quad\text{for every compact }K\subset(a,b),
$$

$$
\tag{U}
\eta_N:=\max_i(a_{i,N}-m(t_i))_+\to0,
$$

$$
\tag{L}
T_N(d):=\sup_i\sum_{j:|t_j-t_i|\ge d}(A_N)_{ij}\to0
\quad\text{for every fixed }d>0.
$$

*Hint for (I).* Your proposition applies to every index sequence converging to an interior point. If uniform convergence failed on $K$, choose bad indices and a convergent subsequence of their grid locations. Use the proposition and continuity of $m$.

*Hint for (U).* Repeat the local/tail split in the appendix uniformly in the central index $i$. For the upper bound, extend the truncated local Gaussian sum to all integer offsets. The compact bandwidth interval makes the derivative-remainder bounds uniform. First let $N\to\infty$ for a fixed rescaled cutoff, then let that cutoff tend to infinity.

*Hint for (L).* Your proof already establishes a constant $c>0$, independent of $i,j,N$, such that $D_{ij}\ge c|t_j-t_i|^2$. Deduce

$$
T_N(d)\le c_NN e^{-c\lambda_Nd^2}\to0,
$$

using $N\Delta_N=O(1)$. [The Gaussian lower bound and tail argument in your appendix](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/9999_appendix.tex:339).

**Boundary checkpoint.** Do not replace (U) by uniform equality $a_{i,N}=m(t_i)+o(1)$ on the closed interval. At an endpoint, the same Gaussian calculation integrates only half of the limiting Gaussian, giving $m(a)/2$ or $m(b)/2$. The one-sided estimate (U) is sufficient. This is a derivation from the truncated Gaussian integral; compare the explicit boundary formula in the [earlier note](/Users/iljasukhanov/MASTERARBEIT/output/pdf/novelty_proposition_7_conjecture_1.tex:185).

**3. Obtain the upper bound for the Perron eigenvalue.**

Prove

$$
\theta_N\le\max_i a_{i,N}\le M+\eta_N.
$$

The first inequality is the maximum row-sum bound. [Meckes, Theorem 8.4, p. 66](https://case.edu/artsci/math/mwmeckes/matrix-analysis.pdf#page=66).

*Hint.* Choose $k_N$ maximizing $p_i^{(N)}$, and compare every $p_j^{(N)}$ with $p_{k_N}^{(N)}$ in the eigenvector equation. Retain the stronger intermediate inequality $\theta_N\le a_{k_N,N}$: you will reuse it in Exercise 5.

**4. Use a localized Rayleigh quotient for the matching lower bound.**

Fix $\varepsilon>0$. Choose a nonzero continuous function $g\ge0$, with compact support in an interior interval where $m(t)>M-\varepsilon$. Let $g_N=(g(t_i))_{i=1}^N$. Keep this function fixed while $N\to\infty$.

(a) Starting from

$$
(A_Ng_N)_i-a_{i,N}g(t_i)
=\sum_j(A_N)_{ij}\bigl(g(t_j)-g(t_i)\bigr),
$$

split the sum at $|t_j-t_i|=d$. If $\omega_g(d)=\sup_{|s-t|\le d}|g(s)-g(t)|$, prove

$$
\sup_i|(A_Ng_N)_i-a_{i,N}g(t_i)|
\le (M+\eta_N)\omega_g(d)+2\|g\|_\infty T_N(d).
$$

Take $N\to\infty$ first and $d\downarrow0$ second.

(b) Use (I) on the support of $g$, then Riemann sums in the **uniform coordinate $t$**, to obtain

$$
\frac{g_N^TA_Ng_N}{g_N^Tg_N}
\longrightarrow
\frac{\int_a^b m(t)g(t)^2\,dt}{\int_a^b g(t)^2\,dt}
\ge M-\varepsilon.
$$

(c) Because $A_N$ is real symmetric and its Perron eigenvalue is its largest eigenvalue,

$$
\theta_N=\max_{x\ne0}\frac{x^TA_Nx}{x^Tx}.
$$

This is Rayleigh–Ritz. [Meckes, Theorem 3.1, p. 20](https://case.edu/artsci/math/mwmeckes/matrix-analysis.pdf#page=20). Combine (b), Exercise 3, and finally $\varepsilon\downarrow0$, to conclude $\boxed{\theta_N\to M}$.

*Diagnostic question.* Why is $x=e_i$ a poor test vector here? Why does $x=(1,\ldots,1)^T$ generally yield only an average of $m$? The test vector must occupy a small but fixed neighborhood containing many interacting grid points.

**5. Prove convergence of the peak location.**

For any maximizing index $k_N$, complete

$$
\theta_Np_{k_N}^{(N)}
=\sum_j(A_N)_{k_Nj}p_j^{(N)}
\le a_{k_N,N}p_{k_N}^{(N)}.
$$

Divide by the positive maximal component and use (U) to show

$$
0\le M-m(t_{k_N})\le M+\eta_N-\theta_N\to0.
$$

*Hint.* For every neighborhood $V$ of $t_*$, continuity, compactness, and uniqueness imply $\max_{[a,b]\setminus V}m<M$, provided the complement is nonempty. Conclude $t_{k_N}\to t_*$, then $h_{k_N}\to h_*$. This is a consequence derived here from Exercises 2–4.

**6. Prove concentration of the entire sum-normalized vector.**

(a) Sum all coordinates of $A_Np^{(N)}=\theta_Np^{(N)}$. Use symmetry to prove the exact identity

$$
\boxed{\theta_N=\sum_i a_{i,N}p_i^{(N)}.}
$$

Explain precisely where row sums replace column sums.

(b) For a fixed neighborhood $V$ of $t_*$ with nonempty complement, define

$$
\gamma_V=M-\max_{[a,b]\setminus V}m>0,
\qquad P_N(V^c)=\sum_{i:t_i\notin V}p_i^{(N)}.
$$

Use (a) and (U) to show

$$
\theta_N\le (M+\eta_N)(1-P_N(V^c))
+(M-\gamma_V+\eta_N)P_N(V^c),
$$

and therefore

$$
\boxed{P_N(V^c)\le\frac{M+\eta_N-\theta_N}{\gamma_V}\to0.}
$$

Take $V=\{t:|\psi_r^{-1}(t)-h_*|<\delta\}$ to obtain the target in bandwidth coordinates. Empty complements give zero mass immediately. This finite-$N$ bound and its limiting conclusion are derived here from the eigenvector equation and the preceding exercises.

**7. Optional: concentration with squared components.**

Normalize the same Perron direction as $\sum_i u_i^2=1$. Expand the square to verify

$$
\theta_N=u^TA_Nu
=\sum_i a_{i,N}u_i^2
-\frac12\sum_{i,j}(A_N)_{ij}(u_i-u_j)^2.
$$

The sum-of-squares term is the weighted graph Dirichlet form; see [Chung, *Spectral Graph Theory*, Sections 1.1 and 1.4, especially (1.1) and (1.13)](https://fanchung.ucsd.edu/cbms.pdf). Here the identity can be verified directly, including the harmless diagonal terms.

Discard this nonnegative term and repeat Exercise 6 to get

$$
\sum_{i:t_i\notin V}u_i^2
\le\frac{M+\eta_N-\theta_N}{\gamma_V}\to0.
$$

**Final checks.** Explain why these arguments give concentration in every fixed neighborhood, but do not establish a shrinking-neighborhood rate or concentration on one particular grid point. If minimizers are not unique, replace $\{t_*\}$ by $\arg\min q$; the conclusions become concentration near that set and convergence of the peak's distance to that set. No allocation among tied minimizers is determined. For an endpoint minimizer, choose the test function on an interior interval where $m>M-\varepsilon$, which exists by continuity; the same proof works with (U).
