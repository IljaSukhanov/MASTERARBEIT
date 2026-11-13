# Large-sample local extrema of the Gaussian KDE scale energy

Date: 29 October 2026.

This note gives a direct asymptotic argument for the sample functional. It
extends the earlier expectation calculations in [report.md](report.md).
The argument below is our derivation, not a theorem quoted from a paper,
and is not a claim of novelty. It does not modify the thesis.

Assume throughout that the observations are i.i.d.
\(N(\mu,\sigma^2)\), with \(0<\sigma<\infty\), that the KDE kernel is the
standard Gaussian, and that \(r=13/14\). Conclusions about a selected
minimum refer to a specified local branch, not an unconstrained global
minimum on \((0,\infty)\).

The sample functional is defined in
[the thesis, Proposition `prop:main_thm`](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:800).
The Gaussian derivative-norm pair representation underlying the argument
is given in [Botev, Grotowski and Kroese (2010), Section 5, Eq. (26)](https://arxiv.org/pdf/1011.2602#page=15).
The general problem of estimating integrated squared density derivatives
is studied by [Hall and Marron (1987), pp. 109–115](https://doi.org/10.1016/0167-7152(87)90083-6);
their abstract does not state the extrema conclusions derived below.

## 1. A fixed positive bandwidth has a deterministic limit

Let \(a=1+2r=20/7\), \(b=4-2r=15/7\), and
\(C=3/(16\sqrt\pi)\). For every compact interval
\(J\subset(0,\infty)\),

\[
\widehat Q_{n,r}\longrightarrow q_{r,f}
\quad\text{in }C^2(J)\text{ in probability},
\]

where

\[
q_{r,f}(h)=\frac12h^b\|(K_h*f)''\|_2^2
=C\frac{h^b}{(\sigma^2+h^2)^{5/2}}.
\]

One elementary justification is to write the squared derivative norm as
a pair sum. On a fixed positive bandwidth interval, the kernel and its
first three bandwidth derivatives are uniformly bounded. Independence
and counting pairs that share an observation give variances \(O(1/n)\).
The expectation differs from the displayed limit by \(O(1/n)\) in
\(C^3(J)\). The one-dimensional inequality used in Section 2 below
upgrades the derivative variance bounds to uniform bounds.

The logarithmic derivative of the limit is

\[
\frac{h q_{r,f}'(h)}{q_{r,f}(h)}
=b-5\frac{h^2}{\sigma^2+h^2}.
\]

It has one, nondegenerate, stationary point: a maximum at

\[
\boxed{h_{\mathrm{max},\infty}
=\sigma\sqrt{b/a}=\frac{\sqrt3}{2}\sigma
\approx0.8660254\sigma.}
\]

Uniform convergence of the first two derivatives implies that, with
probability tending to one, there is exactly one sample maximum in any
sufficiently small fixed neighborhood of this point, and its location
converges to the displayed limit. On any fixed compact interval away from
this maximum, there are no stationary points with probability tending to
one. In a compact interval containing the maximum in its interior, this
is the only stationary point with probability tending to one.

Thus a meaningful sample minimum cannot converge to a positive finite
bandwidth under these assumptions.

## 2. The shrinking bandwidth scale retains a local minimum

Let \(\tau_n=n^{-1/5}\) and rescale the sample function:

\[
F_n(t)=n^{b/5}\widehat Q_{n,r}(\tau_nt).
\]

The exact expectation from [report.md, Section 5](report.md) becomes

\[
\mathbb E F_n(t)=C\left[t^{-a}
+\frac{n-1}{n}\frac{t^b}{(\sigma^2+\tau_n^2t^2)^{5/2}}\right].
\]

Consequently, on every compact \(I\subset(0,\infty)\), the expectation
and its first two derivatives converge uniformly to those of

\[
F(t)=C\left[t^{-a}+\sigma^{-5}t^b\right].
\]

The deterministic error is \(O(n^{-2/5})\). The first term is the
deterministic diagonal contribution of the sample pair sum; the second
is the limiting off-diagonal contribution.

Here is the stochastic control needed to transfer this statement to
sample minima. Write the rescaled pair kernel as

\[
J_{n,t}(x,y)=\frac12 n^{b/5}(\tau_nt)^b
K_{\sqrt2\tau_nt}^{(4)}(x-y).
\]

For \(k=0,1,2,3\), its derivatives have the form
\(\partial_t^kJ_{n,t}(x,y)=n\psi_{k,t}((x-y)/\tau_n)\), where the
Gaussian-polynomial shapes \(\psi_{k,t}\) have uniformly bounded squared
integrals for \(t\in I\). The difference \(X-Y\) has a bounded density,
so

\[
\sup_{t\in I}\mathbb E|\partial_t^kJ_{n,t}(X,Y)|^2
=O(n^2\tau_n).
\]

The conditional means are uniformly bounded in \(n,t,x\). To see the
cancellation that matters here, integrate by parts four times:

\[
\mathbb E[J_{n,t}(x,Y)]
=\frac12t^b(K_{\sqrt2\tau_nt}*f^{(4)})(x).
\]

The Gaussian density and all its derivatives are bounded, and the same
uniform bound holds after up to three \(t\)-derivatives on \(I\).
The diagonal is deterministic. In the variance of the off-diagonal pair
average, disjoint pairs have zero covariance, one-shared-index pairs are
controlled by these conditional means, and two-shared-index pairs by
the squared-kernel bound. Therefore

\[
\sup_{t\in I}\operatorname{Var}(F_n^{(k)}(t))
=O(n^{-1})+O(\tau_n)=O(n^{-1/5}),\qquad 0\le k\le3.
\]

For any continuously differentiable function \(g\) on an interval of
length \(L>0\), integration of
\(|g(t)|^2\le2|g(s)|^2+2L\int|g'|^2\) over \(s\) gives

\[
\|g\|_\infty^2\le\frac2L\int_I|g|^2+2L\int_I|g'|^2.
\]

Applying this inequality to the centered derivatives, followed by the
variance bounds, proves

\[
\boxed{\|F_n-F\|_{C^2(I)}=O_p(n^{-1/10}).}
\]

The limit has a unique minimum at

\[
t_* = \sigma(a/b)^{1/5}=\sigma(4/3)^{1/5}.
\]

Moreover \(F''(t)>0\) for every \(t>0\), because \(a>0\) and
\(b>1\). Choose any fixed \(0<A<t_*<B<\infty\). Uniform convergence
of the first derivative preserves its negative sign at \(A\) and positive
sign at \(B\), and uniform convergence of the second derivative preserves
strict convexity on \([A,B]\). Therefore, with probability tending to
one, the sample function has exactly one local minimum in
\([An^{-1/5},Bn^{-1/5}]\). Denote it by \(\widehat h_n^{\rm loc}\)
(define it arbitrarily on the exceptional event). Then

\[
\boxed{\frac{\widehat h_n^{\rm loc}}
{\sigma(4/(3n))^{1/5}}\xrightarrow{P}1.}
\]

The derivative error and the strictly positive limiting curvature also
give the upper stochastic rate

\[
\frac{\widehat h_n^{\rm loc}}
{\sigma(4/(3n))^{1/5}}-1=O_p(n^{-1/10}).
\]

The denominator is the Gaussian AMISE bandwidth: compare
[Botev et al., Appendix A, Eqs. (37)–(38)](https://arxiv.org/pdf/1011.2602#page=30),
with \(\|f''\|_2^2=3/(8\sqrt\pi\sigma^5)\).
This is a result about a particular local minimum of the realized
criterion, rather than merely a minimizer of its expectation.

## 3. Can other sample extrema escape to large bandwidths?

There is a useful deterministic bound. Put
\(s_n^2=n^{-1}\sum_i(X_i-\overline X_n)^2\) and
\(P_n(\omega)=n^{-1}\sum_i e^{-i\omega X_i}\). The inequality
\(\cos z\ge1-z^2/2\) gives

\[
|P_n(\omega)|^2\ge1-s_n^2\omega^2,
\qquad |P_n(\omega)|^2\le1.
\]

Using the Fourier representations of \(R_2\) and \(R_3\), and setting
\(y=h\omega\), it follows that

\[
B_n(h)=h^2R_3(h)/R_2(h)
\ge\frac52-\frac{35s_n^2}{4h^2}.
\]

For a positive lower bound on the numerator, this follows by bounding
the denominator above by \(\int y^4e^{-y^2}dy\); for a negative lower
bound it follows from \(B_n\ge0\). The constants use
\(\int y^6e^{-y^2}/\int y^4e^{-y^2}=5/2\) and the analogous ratio
\(35/4\) for the eighth and fourth moments.

Since stationary points require \(B_n=15/14\), every stationary
bandwidth satisfies

\[
\boxed{h^2\le\frac{49}{8}s_n^2.}
\]

The sample variance converges to \(\sigma^2\) in probability. Combining
this bound with Section 1 shows that for every \(\epsilon>0\),

\[
\mathbb P\{\text{there is a sample local minimum at }h\ge\epsilon\}
\longrightarrow0.
\]

In this sense all sample local minima move toward zero. The small-scale
argument in Section 2 guarantees a unique minimum within each fixed
rescaled interval around \(t_*\). It does not establish that there are
no additional extrema at other shrinking scales. In particular, it is
not a theorem that the whole sample criterion has exactly one minimum
and one maximum for all sufficiently large \(n\).

## 4. Interpretation and limits

For every finite \(n\), the sample criterion diverges at zero and has
no unconstrained global minimum. In the population limit its value
tends to zero as \(h\downarrow0\). The finite-sample local minimum is
retained by rescaling the bandwidth by \(n^{1/5}\); it disappears from
every fixed interval bounded away from zero.

These conclusions use Gaussian sampling in addition to a Gaussian
kernel. With a different sampling density, the fixed-bandwidth limit is
still \(\tfrac12h^{4-2r}\|(K_h*f)''\|_2^2\), but its extrema can differ.
The Gaussian sampling results do not establish a universal statement
about all densities or arbitrary rules choosing among sample minima.

For reference, exact *expected* stationary bandwidths with \(\sigma=1\)
are approximately:

| n | Expected local minimum | Expected local maximum |
|---:|---:|---:|
| 92 | 0.63975043 | 0.68766475 |
| 200 | 0.44241319 | 0.81689961 |
| 1,000 | 0.28806778 | 0.85739977 |
| 10,000 | 0.17276887 | 0.86518300 |
| 1,000,000 | 0.06712401 | 0.86601700 |

These deterministic values illustrate the expectation; they are not
simulated sample minima or empirical confidence intervals.
