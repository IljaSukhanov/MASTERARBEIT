# Extending the minimum-width approach: mean width and spherical harmonics

This is a separate explanatory note, dated 9 September 2026. The thesis sources were read but not changed. The starting point is `Abschlussarbeit_2021/LaTeX/chapters/3_application_to_time_series.tex`, particularly the minimum-width proposition, its empirical consistency argument, and the draft harmonic formula under “Other convex--geometric functionals”.

The main conclusion is that mean width, the degree-two component, and its minimum all have natural population and empirical versions with consistency for fixed embedding dimension. The exact order-detection plateau of minimum width does not generally extend to these new quantities.

## 1. The geometry already in the thesis

For a causal AR(p) process with innovation support [-1,1], use the backward window

\[
Y_t^{(k)}=(X_t,X_{t-1},\ldots,X_{t-k+1})^\top,
\qquad
g_j^{(k)}=(\psi_j,\psi_{j-1},\ldots,\psi_{j-k+1})^\top,
\]

with negative-index impulse coefficients set to zero. Its support is

\[
\mathcal Z^{(k)}=\sum_{j\ge0}[-g_j^{(k)},g_j^{(k)}].
\]

The summable impulse response makes this an absolutely convergent Minkowski sum. The directional width is

\[
w_k(u)=2\sum_{j\ge0}|u^\top g_j^{(k)}|,\quad \|u\|=1.
\]

The empirical version is the range of the projected windows:

\[
\widehat w_{k,T}(u)=\max_t u^\top Y_t^{(k)}-\min_t u^\top Y_t^{(k)}.
\]

Minimum width asks for the thinnest projection. Your population formula is

\[
m_k=\frac{2}{\sqrt{1+\sum_{i=1}^p\phi_i^2}},\quad k\ge p+1.
\]

At smaller dimensions the minimum is strictly larger. That plateau is special: a direction representing the AR equation cancels all innovations except one.

## 2. Mean width and the normalization of Q0

Throughout, \(\sigma_k\) is the uniform **probability** measure on \(\mathbb S^{k-1}\), so its total mass is one. An integral against it is an average over directions. If using unnormalized surface area instead, divide by \(2\pi^{k/2}/\Gamma(k/2)\).

Define population and empirical mean widths by

\[
\overline w_k=\int w_k(u)\,d\sigma_k(u),\qquad
\widehat{\overline w}_{k,T}=\int\widehat w_{k,T}(u)\,d\sigma_k(u).
\]

To match the harmonic formula currently in the thesis, expand the **half-width**

\[
f_k(u)=\tfrac12w_k(u),\qquad \widehat f_{k,T}(u)=\tfrac12\widehat w_{k,T}(u).
\]

For the origin-symmetric population support, \(f_k=h_{\mathcal Z^{(k)}}\). For the empirical cloud, half-width is the support function of

\[
\tfrac12(K_{k,T}+(-K_{k,T})),\qquad K_{k,T}=\operatorname{conv}(\mathcal P_{k,T}).
\]

Consequently the empirical construction works without assuming that a finite cloud is centrally symmetric. It also removes translations automatically.

The degree-zero component is simply the constant average:

\[
Q_{0,k}=\int f_k\,d\sigma_k=\tfrac12\overline w_k,
\qquad
\widehat Q_{0,k,T}=\tfrac12\widehat{\overline w}_{k,T}.
\]

Thus \(\min_u Q_{0,k}(u)=Q_{0,k}\), which provides no additional information. Minimum width instead minimizes the full directional function. If you prefer Q0 itself to equal mean width, expand w directly and multiply every Q and every matrix below by two.

Write

\[
r_j=\|g_j^{(k)}\|,\quad v_j=g_j^{(k)}/r_j,\quad R_k=\sum_jr_j,
\quad c_k=\mathbb E|U_1|=\frac{\Gamma(k/2)}{\sqrt\pi\,\Gamma((k+1)/2)},
\]

where U is uniform on the sphere; omit zero generators when defining v. Rotation invariance gives \(\int|u^\top g|d\sigma_k(u)=c_k\|g\|\), so

\[
\boxed{\overline w_k=2c_kR_k,\qquad Q_{0,k}=c_kR_k.}
\]

For example, \(c_1=1\), \(c_2=2/\pi\), and \(c_3=1/2\). Interchanging the sum and integral is justified by nonnegativity, and the result is finite by absolute summability.

## 3. Fourier analysis from the beginning

In two dimensions write a unit direction as \(u(\theta)=(\cos\theta,\sin\theta)\). The map \(\theta\mapsto f_k(u(\theta))\) is an ordinary periodic curve: the half-width as you turn around the cloud.

Fourier analysis expresses a periodic curve as a sum of standard waves. Each wave contributes its own pattern and amplitude. The coefficients are obtained by averaging the curve against that wave. For an even directional function, \(f(u)=f(-u)\), turning by 180 degrees leaves the value unchanged, and only the even angular frequencies remain:

\[
f(\theta)=Q_0+a_2\cos(2\theta)+b_2\sin(2\theta)
+a_4\cos(4\theta)+b_4\sin(4\theta)+\cdots.
\]

Here

\[
Q_0=\frac1{2\pi}\int_0^{2\pi}f(\theta)d\theta,
\quad a_2=\frac1\pi\int_0^{2\pi}f(\theta)\cos(2\theta)d\theta,
\quad b_2=\frac1\pi\int_0^{2\pi}f(\theta)\sin(2\theta)d\theta.
\]

The factors follow from the identities that sine and cosine average to zero and their squares average to one half. Waves of different frequencies have zero average product; this property is called orthogonality. It lets us extract one frequency without interference from the others.

The second component is

\[
Q_2(\theta)=a_2\cos(2\theta)+b_2\sin(2\theta)
=\sqrt{a_2^2+b_2^2}\cos(2\theta-\delta).
\]

It describes a broad pattern of relatively wide and relatively narrow directions. Its minimum is \(-\sqrt{a_2^2+b_2^2}\). The angle \(\delta\) specifies its orientation. Higher components add more detailed angular structure.

For larger embedding dimensions the directions lie on \(\mathbb S^{k-1}\). Spherical harmonics supply the corresponding collection of standard patterns. The expansion is

\[
f_k=Q_{0,k}+Q_{2,k}+Q_{4,k}+\cdots
\]

in the mean-square (L2) sense. The subscript 2 denotes harmonic degree, not AR order or embedding dimension. Q2 is a function of direction, not one Fourier coefficient. Angular frequency here concerns rotation around the cloud, rather than frequency over time in the original signal.

Mathematically a degree-n spherical harmonic is the restriction to the sphere of a homogeneous degree-n polynomial with Laplacian zero. In degree two, any such function has the particularly simple form

\[
\boxed{Q_{2,k}(u)=u^\top A_ku,\qquad A_k=A_k^\top,\quad\operatorname{tr}(A_k)=0.}
\]

Indeed, the Laplacian of \(x^\top A_kx\) is \(2\operatorname{tr}(A_k)\). Also \(\int u^\top A_ku\,d\sigma_k=\operatorname{tr}(A_k)/k\), so the trace condition removes the constant component. In two dimensions, \(A=\left(\begin{smallmatrix}a_2&b_2\\b_2&-a_2\end{smallmatrix}\right)\) recovers the trigonometric expression above. No table of three-dimensional harmonics is needed for degree two in general dimension.

The standard harmonic framework, orthogonality, addition formula, and Funk-Hecke theorem can be cited from [Dai and Xu, Spherical Harmonics, Sections 1, 2 and 6](https://arxiv.org/pdf/1304.2585). The AR-specific formulas below are derived from your generator representation.

## 4. Deriving the population Q2

For any continuous half-width function f, its degree-two projection is given by

\[
A=\frac{k(k+2)}2\int f(u)\left(uu^\top-\frac1kI_k\right)d\sigma_k(u).
\]

One way to prove this is to use the sphere moments

\[
\mathbb E(U_iU_j)=\frac{\delta_{ij}}k,\qquad
\mathbb E(U_iU_jU_aU_b)=\frac{\delta_{ij}\delta_{ab}+\delta_{ia}\delta_{jb}+\delta_{ib}\delta_{ja}}{k(k+2)}.
\]

For a symmetric trace-zero B, these give

\[
\int(u^\top Au)(u^\top Bu)d\sigma_k(u)=\frac{2\operatorname{tr}(AB)}{k(k+2)}.
\]

The displayed formula for A therefore makes the residual \(f-Q_0-u^\top Au\) orthogonal to every quadratic harmonic. In plain terms, Q0 + Q2 is the best constant-plus-quadratic approximation in average squared error over directions.

Now define the generator matrix

\[
S_k=\sum_{j:r_j>0}\frac{g_j^{(k)}(g_j^{(k)})^\top}{r_j}
=\sum_jr_jv_jv_j^\top,\qquad \operatorname{tr}(S_k)=R_k.
\]

The entries describe how generator length is distributed among directions. This matrix is positive semidefinite and the sum converges because the operator norm of each summand is r_j. It differs from the covariance of the observed windows.

Rotation invariance and one-dimensional sphere moments give

\[
\int|u^\top v|\,uu^\top d\sigma_k(u)=\frac{c_k}{k+1}(I_k+vv^\top),\qquad \|v\|=1.
\]

To check the constant, rotate v to the first coordinate axis. Then \(\mathbb E|U_1|^3=2c_k/(k+1)\); the remaining diagonal entries are each \(c_k/(k+1)\), and the off-diagonal entries vanish by sign symmetry.

Substitution gives the explicit population matrix and component:

\[
\boxed{A_k=\frac{c_k k(k+2)}{2(k+1)}\left(S_k-\frac{R_k}{k}I_k\right),}
\]

\[
\boxed{Q_{2,k}(u)=\frac{c_k(k+2)}{2(k+1)}\left(k\,u^\top S_ku-R_k\right).}
\]

The link to the existing draft formula is exact under the probability-measure convention:

\[
P_0^{(k)}(t)=1,\quad P_2^{(k)}(t)=\frac{kt^2-1}{k-1},\quad
N(k,2)=\frac{(k-1)(k+2)}2,
\]

\[
\lambda_{0,k}=c_k,\qquad \lambda_{2,k}=\frac{c_k}{k+1},
\quad Q_{2,k}(u)=\lambda_{2,k}N(k,2)\sum_j r_jP_2^{(k)}(u^\top v_j).
\]

For k=2 the polynomial is \(2t^2-1\), interpreted through the circle/Chebyshev convention; do not evaluate a degenerate unnormalized Gegenbauer expression at parameter zero. The matrix formulas work for every k >= 2. On \(\mathbb S^0\), width is constant, so there is no nonzero degree-two component to investigate.

For general innovation support [a,b], multiply the population half-width quantities above by \((b-a)/2\). The midpoint only translates the support. Beta distributions with the same full support interval have the same population geometry, while the finite-sample behavior can differ.

## 5. What exactly should “min Q2” mean?

Let \(\lambda_{\min}(A)\) denote the smallest eigenvalue of a symmetric matrix A. The spectral theorem implies that, among unit u, \(u^\top Au\) is minimized at an eigenvector for that eigenvalue. Hence

\[
\boxed{q_{2,k}^{\min}:=\min_{\|u\|=1}Q_{2,k}(u)=\lambda_{\min}(A_k)
=\frac{c_k(k+2)}{2(k+1)}\bigl(k\lambda_{\min}(S_k)-R_k\bigr).}
\]

Because A has trace zero, this number is nonpositive. Unless A is zero, it is strictly negative. It is the deepest negative degree-two correction to the average half-width, not a geometric width on its own. The average of Q2 is always zero, so averaging Q2 is not a useful additional scalar statistic.

The comparable approximation to minimum width is instead

\[
\boxed{m_k^{[2]}:=\min_u2(Q_{0,k}+Q_{2,k}(u))
=2(Q_{0,k}+\lambda_{\min}(A_k)).}
\]

It generally differs from m_k, and its minimizing direction generally differs from the AR cancellation direction. A useful deterministic bound is

\[
|m_k-m_k^{[2]}|\le2\|f_k-Q_{0,k}-Q_{2,k}\|_\infty.
\]

Mean-square optimality by itself does not guarantee a small uniform error, and therefore does not guarantee accurate minima. The truncated function is a quadratic approximation of the support function; it should not automatically be identified with the support function of an ellipsoid or another convex body.

Optional scalar summaries of degree-two anisotropy are

\[
-\lambda_{\min}(A_k)/Q_{0,k},\qquad
\|Q_{2,k}\|_{L^2}=\sqrt{\frac{2\operatorname{tr}(A_k^2)}{k(k+2)}}.
\]

The first is scale invariant and the second measures the total degree-two amplitude. Normalization by Q0 removes overall scale, but not all effects of changing dimension. Both vanish when A=0, which does not imply a spherical support: a cube has zero Q2 by symmetry and still has nonconstant width, encoded in higher degrees.

## 6. Empirical quantities and numerical computation

Use the empirical half-width itself, without estimating AR coefficients or a generator representation:

\[
\boxed{\widehat A_{k,T}=\frac{k(k+2)}2\int\widehat f_{k,T}(u)
\left(uu^\top-\frac1kI_k\right)d\sigma_k(u).}
\]

Then

\[
\widehat Q_{2,k,T}(u)=u^\top\widehat A_{k,T}u,\quad
\widehat q_{2,k,T}^{\min}=\lambda_{\min}(\widehat A_{k,T}),\quad
\widehat m_{k,T}^{[2]}=2(\widehat Q_{0,k,T}+\lambda_{\min}(\widehat A_{k,T})).
\]

For numerical integration choose M directions U_l uniformly on the sphere, independently of the data. One construction is to draw a vector of k independent standard normal numbers and divide it by its Euclidean length. With \(d_l=\widehat w_{k,T}(U_l)\), calculate

\[
\widehat{\overline w}_{k,T,M}=\frac1M\sum_ld_l,\quad
\widehat Q_{0,k,T,M}=\frac1{2M}\sum_ld_l,
\]

\[
\boxed{\widehat A_{k,T,M}=\frac{k(k+2)}{4M}\sum_ld_l
\left(U_lU_l^\top-\frac1kI_k\right).}
\]

Compute its smallest eigenvalue and eigenvector. The minimization is then exact for the computed matrix; the remaining approximations are finite-data error and angular-integration error. For k=2, use an equally spaced angular grid. In higher dimensions, quadrature or random directions work; sampling spherical latitude uniformly does not produce uniform directions on a sphere.

The empirical integral and its numerical approximation are distinct. Increasing T addresses sampling error; increasing M addresses directional integration error. A fixed finite collection of directions generally retains integration error as T grows. A precise consistency statement is obtained by taking T -> infinity for the exact empirical integral, followed by M -> infinity for its numerical approximation. Joint limits require control of both errors.

To parallel the cross-evaluation curves in the current notebook, let u* and uhat minimize Q2 and its empirical estimate. Report

\[
Q_2(u^*),\quad\widehat Q_2(\widehat u),\quad Q_2(\widehat u),\quad\widehat Q_2(u^*).
\]

Their guaranteed comparisons are \(Q_2(u^*)\le Q_2(\widehat u)\) and \(\widehat Q_2(\widehat u)\le\widehat Q_2(u^*)\). Harmonic projection need not preserve pointwise order, so there is no general upper/lower ordering between population and empirical Q2. For the mean width, the empirical value is at most the population value whenever the sample cloud lies in the true support.

## 7. Consistency follows from the existing Hausdorff argument

Fix k. Under the stationary ergodic compact-support assumptions already used in the thesis,

\[
\delta_{k,T}:=d_H(K_{k,T},\mathcal Z^{(k)})\longrightarrow0\quad\text{almost surely}.
\]

The support-function identity gives

\[
\|\widehat w_{k,T}-w_k\|_\infty\le2\delta_{k,T},\qquad
\|\widehat f_{k,T}-f_k\|_\infty\le\delta_{k,T}.
\]

It follows immediately that

\[
|\widehat{\overline w}_{k,T}-\overline w_k|\le2\delta_{k,T},\qquad
|\widehat Q_{0,k,T}-Q_{0,k}|\le\delta_{k,T}.
\]

Since \(\|uu^\top-I/k\|_{\rm op}=(k-1)/k\) for k >= 2,

\[
\|\widehat A_{k,T}-A_k\|_{\rm op}
\le\frac{(k-1)(k+2)}2\delta_{k,T}=N(k,2)\delta_{k,T}.
\]

The variational characterization of eigenvalues then yields

\[
\|\widehat Q_{2,k,T}-Q_{2,k}\|_\infty\le N(k,2)\delta_{k,T},\qquad
|\widehat q_{2,k,T}^{\min}-q_{2,k}^{\min}|\le N(k,2)\delta_{k,T}.
\]

Finally,

\[
|\widehat m_{k,T}^{[2]}-m_k^{[2]}|\le2(1+N(k,2))\delta_{k,T}.
\]

Thus every displayed empirical functional converges almost surely to its population counterpart for fixed k. A finite collection of candidate dimensions can be handled simultaneously. Growing dimension requires additional work because both support coverage and the displayed constants depend on k.

The minimizing direction converges up to sign when the smallest eigenvalue is simple and separated from the rest. If it is repeated, there is no unique direction to estimate; the appropriate target is a minimizing eigenspace. Scalar consistency still holds.

## 8. Why there is no automatic order-detection theorem

For \(X_t=0.6X_{t-1}+Z_t\) and innovation support [-1,1], exact geometric-tail summation gives:

| k | Mean width | min Q2 | Minimum of 2(Q0+Q2) | True minimum width |
|---:|---:|---:|---:|---:|
| 2 | 4.985339 | -1.103150 | 2.779040 | 1.714986 |
| 3 | 5.217420 | -1.204953 | 2.807514 | 1.714986 |
| 4 | 5.504916 | -1.255093 | 2.994730 | 1.714986 |
| 5 | 5.806258 | -1.299849 | 3.206560 | 1.714986 |

The order is one, so the true minimum width is constant from k=2. Mean width, min Q2, and the quadratic minimum all continue changing. This is a concrete counterexample to transferring the same plateau theorem.

The example has the exact two-dimensional half-width

\[
f_2(\theta)=|\cos\theta|+\frac{|0.6\cos\theta+\sin\theta|}{0.4}.
\]

This follows because \(g_0=(1,0)\), and all subsequent generators are positive multiples of (0.6,1). It provides an independent angular-integral check of Q0 and the matrix formula. The full-width minimizing angle is about 149.04 degrees; the Q2 minimizing angle is about 139.11 degrees.

The accompanying two-page PDF shows the width decomposition, one empirical sample with T=2,000 after burn-in 1,000 (seed 20260909), and the population curves across dimensions. Empirical angular integrals use 32,768 equally spaced directions. This is an illustration, not a simulation study of estimation rates or order-selection accuracy.

Even the white-noise support [-1,1]^k has mean width \(2kc_k\), which changes with k although its minimum width is always two. Its Q2 is identically zero because \(S_k=I_k\) and \(R_k=k\). This also illustrates the influence of dimension and the information lost by keeping only degree two.

For a thesis extension, establish population formulas and consistency first. Then compare mean width, min Q2, and normalized degree-two summaries empirically across dimensions. Treat reliable AR order recovery from these summaries as a separate question needing an identifiable population feature and an estimator/penalty analysis.

## 9. Small corrections to the existing minimum-width exposition

These observations are confined to the passages read for this extension; they are not a full audit. No corrections were applied to the thesis.

1. The equality remark after Young's inequality should use \(\|q_\phi\|_2\|a\|_1\), as in the proof, rather than \(\|q_\phi\|_2\|a\|_2\). The single-impulse equality condition is the one for the l2-l1 bound. Distinct shifts of a nonzero finite-support q cannot be collinear, which justifies the equality characterization used in the proof.
2. At k=p+1, if the minimizing direction is \(u^*=\pm(1,-\phi_1,\ldots,-\phi_p)/\|q_\phi\|_2\), coefficient recovery is \(\phi_i=-u_i^*/u_0^*\). Multiplying its last coordinates by \(\|u^*\|_2\) does not recover the coefficients, because this norm is one.
3. The stated Hausdorff nesting lemma needs \(A\subseteq B\subseteq Z\), rather than only \(A\subseteq B\). With A=Z={0} and B={0,10}, the current general statement fails. The extra inclusion does hold for the empirical cloud, its convex hull, and the true convex support, so the intended convergence application remains valid.
4. Keep the segment convention explicit: uniform [-1,1] innovations produce sums of [-g_j,g_j], while a sum of [0,g_j] has half the width. Likewise \(w=2h\) requires symmetry about the origin. Using f=w/2 as above avoids translation issues for asymmetric Beta supports and empirical clouds.

## References and reproducibility

- The thesis's generator, minimum-width and Hausdorff-consistency arguments are the starting point; the new formulas in this note are direct calculations from that setup.
- [F. Dai and Y. Xu, Spherical Harmonics](https://arxiv.org/pdf/1304.2585): Definitions and orthogonality in Section 1; dimension formula in Corollary 1.4; addition formula and Funk-Hecke theorem in Section 2; circle formulas in Section 6. Their surface-measure convention must be normalized when using the probability-measure formulas here.
- [D. Chafaï, The Funk-Hecke formula](https://djalil.chafai.net/blog/2021/05/22/the-funk-hecke-formula/): a probability-measure formulation of sphere projection integrals. The constants c_k and lambda_2 above can be checked by substituting the absolute-value kernel into these integrals.
- `build_example.py` reproduces the figures and `example_values.json`, with independent checks of the generator formulas against angular integration and of lambda_2 against one-dimensional quadrature in dimensions 2, 3, 4 and 8.
