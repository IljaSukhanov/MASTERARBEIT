# Solution notes: Perron-vector concentration

These notes complete the [guided exercise](/Users/iljasukhanov/MASTERARBEIT/output/analysis/perron_concentration_exercise/exercise.md). The application-specific argument is a derivation from the current [row-sum proposition](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:781) and [appendix estimates](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/9999_appendix.tex:325). The external matrix facts used are [Meckes, Theorems 3.1, 8.4, and 8.10](https://case.edu/artsci/math/mwmeckes/matrix-analysis.pdf). No convergence rate or statistical limit in sample size is assumed or proved.

**1. Scaling and positivity.** Since $c_N>0$, the equations $S_Np=\rho(S_N)p$ and $A_Np=c_N\rho(S_N)p$ are equivalent. Multiplication scales every eigenvalue by $c_N$; hence $\theta_N=c_N\rho(S_N)$. The Gaussian KDE curve on the fixed positive bandwidth interval is continuous and has a continuous $L^2$ derivative. The appendix's bound $D(h,k)\ge c|\psi_r(h)-\psi_r(k)|^2$, combined with its local expansion, implies $q(t)\ge c>0$. Thus $m$ is finite, positive, and continuous. These are consequences of the [current derivative lemma](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:766) and [appendix lower bound](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/9999_appendix.tex:339). Uniqueness of the minimizer remains the exercise's additional assumption.

**2. The three analytic estimates.**

For (I), suppose there were a compact $K\subset(a,b)$, a number $\epsilon>0$, and indices $i_k$ at sizes $N_k\to\infty$ with $t_{i_k}\in K$ and $|a_{i_k,N_k}-m(t_{i_k})|\ge\epsilon$. Pass to a subsequence with $t_{i_k}\to t_0\in K$. The row-sum result gives $a_{i_k,N_k}\to m(t_0)$, and continuity gives $m(t_{i_k})\to m(t_0)$, a contradiction. The original proposition applies along subsequences by completing the missing grid sizes with indices nearest $t_0$.

For (U), write $z_{ij}=\sqrt{\lambda_N}(t_j-t_i)=c_N(j-i)$. The uniform derivative estimates in the appendix give, for every fixed cutoff $L>0$,

$$
D_{ij}=q(t_i)(t_j-t_i)^2+\rho_{ij,N},\qquad
\sup_{i,j:|z_{ij}|\le L}\lambda_N|\rho_{ij,N}|\to0.
$$

Consequently, because $c_N\#\{j:|z_{ij}|\le L\}\le2L+c_N$,

$$
c_N\sum_{|z_{ij}|\le L}e^{-\lambda_ND_{ij}}
=c_N\sum_{|z_{ij}|\le L}e^{-q(t_i)z_{ij}^2}+o_L(1)
$$

uniformly in $i$. Extending the right side to all integer offsets bounds it above by

$$
c_N\sum_{k\in\mathbb Z}e^{-q(t_i)c_N^2k^2}
\le c_N+\int_{\mathbb R}e^{-q(t_i)z^2}\,dz
=c_N+m(t_i).
$$

The inequality follows by comparing the positive, decreasing Gaussian on each half-line with its integral. The tail bound from the appendix is uniform in $i$:

$$
c_N\sum_{|z_{ij}|>L}e^{-\lambda_ND_{ij}}
\le2\int_{L-c_N}^{\infty}e^{-cz^2}\,dz
$$

for all sufficiently large $N$, with $c>0$ from the global quadratic lower bound. Therefore

$$
\limsup_{N\to\infty}\eta_N
\le2\int_L^{\infty}e^{-cz^2}\,dz.
$$

Let $L\to\infty$. This proves (U) without using full Gaussian mass at endpoint rows.

For (L), directly estimate

$$
T_N(d)\le c_NNe^{-c\lambda_Nd^2}
\le C\sqrt{\lambda_N}e^{-c\lambda_Nd^2}\to0.
$$

Only $N\Delta_N=O(1)$ was used in the second inequality. These arguments extend the bounds already present in the [appendix proof](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/9999_appendix.tex:325); they do not follow from a bare pointwise row-sum statement for arbitrary matrices.

**3. Upper eigenvalue bound.** Positivity and $p_j\le p_k$ at a maximal coordinate give

$$
\theta_Np_k=\sum_j(A_N)_{kj}p_j\le a_{k,N}p_k.
$$

Divide by $p_k>0$. In particular, $\theta_N\le\max_i a_{i,N}\le M+\eta_N$. This is also a direct verification of the upper bound in [Meckes, Theorem 8.4, p. 66](https://case.edu/artsci/math/mwmeckes/matrix-analysis.pdf#page=66).

**4. Lower eigenvalue bound.** Let $g$ be as in the exercise. The near/far split gives

$$
\sup_i|(A_Ng_N)_i-a_{i,N}g(t_i)|
\le(M+\eta_N)\omega_g(d)+2\|g\|_\infty T_N(d).
$$

For fixed $d>0$, take the upper limit as $N\to\infty$, then take $d\downarrow0$. Uniform continuity of $g$ makes the bound vanish. On the compact support of $g$, (I) then yields

$$
\sup_{i:g(t_i)\ne0}|(A_Ng_N)_i-m(t_i)g(t_i)|\to0.
$$

Multiplying by $g(t_i)\Delta_N$ and summing gives

$$
\Delta_N g_N^TA_Ng_N\to\int_a^b m(t)g(t)^2\,dt,
\quad
\Delta_N g_N^Tg_N\to\int_a^b g(t)^2\,dt>0.
$$

The accumulated error vanishes because $\Delta_N\sum_i|g(t_i)|$ is bounded. Rayleigh–Ritz now implies

$$
\liminf_N\theta_N\ge
\frac{\int m g^2}{\int g^2}\ge M-\varepsilon.
$$

Together with the upper bound and arbitrary $\varepsilon>0$, this proves $\theta_N\to M$. [Variational characterization: Meckes, Theorem 3.1, p. 20](https://case.edu/artsci/math/mwmeckes/matrix-analysis.pdf#page=20).

The single-coordinate test has Rayleigh quotient $(A_N)_{ii}=c_N\to0$, so it misses the limit $M>0$. The constant vector has quotient $N^{-1}\sum_i a_{i,N}$, which tends to $(b-a)^{-1}\int_a^b m(t)\,dt$. To justify this last claim, first exclude endpoint strips of width $d>0$, apply (I) on the remaining interval, bound the strips using (U), and finally let $d\downarrow0$. With a unique maximizer on a nontrivial interval, this spatial average is strictly smaller than $M$. Both observations are direct calculations in this setting.

**5. Peak location.** The sharper inequality from Exercise 3 gives

$$
0\le M-m(t_{k_N})\le M+\eta_N-\theta_N\to0.
$$

If $t_{k_N}$ failed to converge to $t_*$, a subsequence would remain outside some neighborhood of $t_*$. On that compact complement, continuity and uniqueness give a strictly positive gap below $M$, contradicting the display. Continuity of $\psi_r^{-1}$ transfers the convergence to $h_{k_N}\to h_*$.

**6. Sum-normalized mass.** Symmetry gives $\sum_i(A_N)_{ij}=a_{j,N}$. Sum the eigenvector equation:

$$
\theta_N=\theta_N\sum_i p_i
=\sum_{i,j}(A_N)_{ij}p_j
=\sum_j a_{j,N}p_j.
$$

For a neighborhood $V$ with nonempty compact complement, set $\gamma_V=M-\max_{V^c}m>0$. Estimate every row by $M+\eta_N$, improving this by $\gamma_V$ outside $V$. Then

$$
\theta_N\le M+\eta_N-\gamma_V\sum_{t_i\notin V}p_i,
$$

which is the desired mass bound. Taking the preimage of a bandwidth neighborhood proves the stated concentration in $h$. This derivation uses symmetry essentially: for an arbitrary nonsymmetric matrix, summing the right-eigenvector equation produces column sums instead.

**7. Squared mass and interpretation.** For $\|u\|_2=1$, symmetry and expansion of the square give

$$
\frac12\sum_{i,j}(A_N)_{ij}(u_i-u_j)^2
=\sum_i a_{i,N}u_i^2-u^TA_Nu\ge0.
$$

This is the weighted Dirichlet-form identity, consistent with [Chung, Sections 1.1 and 1.4](https://fanchung.ucsd.edu/cbms.pdf). Since $u^TA_Nu=\theta_N$, the same gap estimate proves concentration of $u_i^2$. This proof is separate from the sum-normalized argument; a dimension-dependent norm comparison is unnecessary.

The estimates concern fixed neighborhoods. They provide no bound on how quickly $M+\eta_N-\theta_N$ vanishes, so they do not establish a localization rate. A neighborhood may contain increasingly many grid points, so neighborhood concentration does not imply mass one on a single grid index. With several minimizers, the compact-complement argument applies to any neighborhood of the entire minimizing set. If a maximizer of $m$ is at the boundary, the Rayleigh test can still be supported in a fixed interior interval with $m>M-\varepsilon$; continuity supplies that interval. These extensions follow from the same proof.
