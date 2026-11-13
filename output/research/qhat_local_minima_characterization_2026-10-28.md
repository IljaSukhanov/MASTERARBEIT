# Local minima of the Gaussian KDE scale energy

Date: 2026-10-28.

This note contains direct derivations and new computational checks for the
functional defined in the thesis. It does not attribute the results below to
an external theorem or make a claim of novelty. No thesis sources were edited.

## Scope and sources

Let the kernel be the standard normal density, let the sample be fixed and
finite, and set

\[
\widehat Q_r(h)=\tfrac12 h^{2(1-r)}\|\partial_h\widehat f_h\|_2^2
              =\tfrac12 h^{4-2r}\|\widehat f_h''\|_2^2,
\qquad h>0.
\]

The definition is in
[the thesis, Proposition prop:main_thm](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:800).
The equality follows from the Gaussian heat equation; see
[Botev, Grotowski and Kroese (2010), Section 2, Eq. (2)](https://people.smp.uq.edu.au/DirkKroese/ps/AOS799.pdf).
For the Gaussian density-derivative pair formula, see the same paper,
Section 5, Eq. (26). Workspace implementations are
[qhat.py](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/kde_tools/qhat.py:27)
and
[continuous_minimizers.py](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/kde_tools/continuous_minimizers.py:49).

All statements about minima below refer to interior minima on (0,infinity).
An interval endpoint selected by a constrained minimization is a different
object. A local minimum of the expected criterion is also different from a
local minimum of a realized sample criterion.

## 1. Characterization of a sample local minimum

Write R_j(h)=||partial_x^j fhat_h||_2^2. Gaussian decay justifies
differentiation and integration by parts, giving

\[
R_j'(h)=2h\langle\partial_x^j\widehat f_h,
                         \partial_x^{j+2}\widehat f_h\rangle
       =-2hR_{j+1}(h).
\]

Define

\[
B(h)=h^2\frac{R_3(h)}{R_2(h)},\qquad c=2-r.
\]

Then

\[
\widehat Q_r'(h)=h^{3-2r}R_2(h)[c-B(h)].
\]

Thus a stationary point satisfies B(h)=c. A local minimum is characterized
by B crossing c downwards: B>c just before it and B<c just after it. A
local maximum corresponds to the opposite crossing. The Gaussian sample
criterion is real analytic and nonconstant in the range r>-1/2, so this
sign characterization also covers degenerate strict extrema.

For a nondegenerate stationary point h_*, the following equivalent tests
characterize a local minimum:

\[
B'(h_*)<0,
\]

\[
h_*^4\frac{R_4(h_*)}{R_2(h_*)}>(2-r)(3-r).
\]

Indeed,

\[
B'(h)=2h\frac{R_3}{R_2}
 -2h^3\left[\frac{R_4}{R_2}-\left(\frac{R_3}{R_2}\right)^2\right],
\]

and at a stationary point

\[
\widehat Q_r''(h_*)
=-h_*^{3-2r}R_2(h_*)B'(h_*)
=2h_*^{2-2r}\left[h_*^4R_4(h_*)-(2-r)(3-r)R_2(h_*)\right].
\]

If this second derivative is zero, the second-derivative test is
inconclusive; use the sign change or higher derivatives.

## 2. Existence and finiteness

For every fixed finite Gaussian sample,

\[
B(h)\longrightarrow\tfrac52
\quad\text{as }h\downarrow0\text{ and as }h\to\infty.
\]

To see this, the squared derivative norm of a single Gaussian component is

\[
\|\partial_x^j K_h\|_2^2
=\frac{(2j-1)!!}{2^{j+1}\sqrt\pi}\,h^{-2j-1}.
\]

At small h, distinct-location cross terms vanish exponentially; coincident
observations can be grouped and give a common multiplicity factor in
R_2 and R_3. At large h, all component shifts become negligible relative
to h and again give a common factor. The ratio of the j=3 and j=2
constants is 5/2.

Consequently, for -1/2<r<2,

\[
\boxed{\widehat Q_r\text{ has a local minimum}
\quad\Longleftrightarrow\quad
\inf_{h>0}B(h)<2-r.}
\]

Proof: the derivative is negative near both endpoints. A dip of B below
2-r creates a region of positive derivative, preceded by a local minimum
and followed by a local maximum. Conversely, an analytic strict local
minimum requires the derivative to change from negative to positive.

Existence is not automatic. If all observations coincide, B(h)=5/2 for
every h and Qhat_r is strictly decreasing when r>-1/2. For r>=2 it is
strictly decreasing for every sample.

There are only finitely many stationary points for a fixed sample when
r>-1/2. The derivative is negative near both ends, so its zeros lie in a
compact subinterval of (0,infinity). A nonzero real-analytic function has
only finitely many zeros on such a compact interval. In particular, there
are finitely many local minima, but their number need not be one.

## 3. Numerical experiments

The separate empirical local-minimum distribution study and its old counts are
outside the retained thesis scope. The [energy notebook](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/notebooks/continuous_ise_qhat_distributions_vs_n.ipynb)
includes a fixed-sample local-minimum control using the shared search. The
derivations and the explicit deterministic example below do not depend on the
removed empirical counts.

## 4. An explicit counterexample to uniqueness at r=13/14

Take the 25 distinct observations

\[
X_{ij}=a_i+\frac{a_j}{100},\qquad
(a_1,\ldots,a_5)=(-3,-1,0,1,3),\qquad 1\le i,j\le5.
\]

Evaluating the exact Gaussian pair criterion gives the following extrema:

| Type | h (approximately) | Qhat_r(h) (approximately) |
|---|---:|---:|
| Local minimum | 0.01594054444 | 213.1183036 |
| Local maximum | 0.02001689455 | 219.2190081 |
| Local minimum | 1.59380893052 | 0.002056988092 |
| Local maximum | 2.00177346892 | 0.002116001868 |

At the two minima, independent spatial integration gave
B=1.07142857142857=15/14, and second derivatives approximately
3,757,020.50 and 0.00362950, respectively. The spatial derivative norms
were integrated over subintervals around the Gaussian components to
resolve the narrow-bandwidth case.

Existence of two minima can be established without relying on numerical
root locations. Put u_ij=(X_i-X_j)^2/(4h^2), and define

\[
p(u)=u^2-3u+\tfrac34,\qquad
g(u)=2u^3-11u^2+\tfrac{21}{2}u-\tfrac34,
\]

\[
S_r(h)=\sum_{i,j}e^{-u_{ij}}[g(u_{ij})-2r p(u_{ij})].
\]

The derivative Qhat_r'(h) is S_r(h) times a strictly positive factor.
Interval arithmetic with exact rational inputs and 60 decimal digits
gave enclosures contained in the following deliberately widened bounds:

| h | An enclosure for S_(13/14)(h) | Sign of Qhat_r'(h) |
|---|---|---|
| 0.01 | [-24.989, -24.988] | negative |
| 0.018 | [1.779, 1.780] | positive |
| 0.03 | [-38.262, -38.261] | negative |
| 1 | [-124.700, -124.699] | negative |
| 1.8 | [8.905, 8.906] | positive |
| 3 | [-191.279, -191.278] | negative |

In particular, there is at least one local minimum in each of the disjoint
intervals (0.01,0.018) and (1,1.8). These strict derivative signs persist
under sufficiently small perturbations of the observations, so the
example is not dependent on exact ties or a flat stationary point.

## 5. A uniqueness result for the EXPECTED energy under Gaussian sampling

Add the assumption X_i are i.i.d. N(mu,sigma^2), sigma>0. Write
q_(n,r)(h)=E Qhat_r(h), a=1+2r and b=4-2r=5-a. For -1/2<r<2,

\[
q_{n,r}(h)
=\frac{3}{16\sqrt\pi}\left[
\frac1n h^{-a}
+\frac{n-1}{n}h^b(\sigma^2+h^2)^{-5/2}\right].
\]

This is exact. The diagonal term uses the squared norm of K_h'', and
the off-diagonal term is the squared norm of the second derivative of
K_h*f, which is Gaussian with variance sigma^2+h^2.

Set s=h^2/(sigma^2+h^2), so s increases from 0 to 1 with h. Then

\[
q_{n,r}'(h)
=\frac{3}{16\sqrt\pi\,n}h^{-a-1}
\left[-a+(n-1)s^{5/2}(b-5s)\right].
\]

On the interval where it is positive, F(s)=s^(5/2)(b-5s) has exactly
one maximum, at s=b/7, because

\[
F'(s)=\tfrac52 s^{3/2}(b-7s),\qquad
\max_{0<s<1}F(s)=2(b/7)^{7/2}.
\]

It follows that the exact expected energy has exactly one local minimum
and one local maximum if and only if

\[
\boxed{n>1+\frac{1+2r}{2[(4-2r)/7]^{7/2}}.}
\]

Below this threshold it is strictly decreasing. At equality it is still
strictly decreasing, with one horizontal stationary inflection point.

For r=13/14 the threshold is n>91.00545534774447. Therefore:

- n<=91: no local minimum of the expected energy;
- n>=92: exactly one local minimum of the expected energy, and one local maximum.

For n=200 and r=13/14, their locations are approximately

\[
h_{\min}=0.4424131861\,\sigma,\qquad
h_{\max}=0.8168996065\,\sigma.
\]

Independent integration of the Gaussian pair expectation agreed with
the closed-form expected energy to relative error below 2e-15 at the
four checked bandwidths 0.2, 0.4424131861, 0.8 and 1.5 for sigma=1.

This proves uniqueness for the expected criterion under Gaussian
sampling. It does not imply uniqueness for the random sample criterion;
the deterministic counterexample above shows the
distinction. No consistency theorem for a selected sample-minimum branch
is claimed here.
