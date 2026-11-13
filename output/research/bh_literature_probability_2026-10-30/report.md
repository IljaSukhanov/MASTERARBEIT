# The Gaussian KDE ratio B: literature, expectations, and probability of extrema

Date: 30 October 2026.

This note distinguishes verified literature precedents from calculations derived
for this project. The deductions below are not attributed as theorems to the
cited papers and are not claims of novelty. No protected thesis files were edited.

## 1. Setup and verified literature connections

Let \(X_1,\ldots,X_n\) be a fixed sample, let \(K\) be the standard normal
density, and write
\[
R_j(h)=\|\partial_x^j\widehat f_h\|_2^2,\qquad
B_n(h)=h^2 R_3(h)/R_2(h).
\]
For extrema, assume throughout \(-1/2<r<2\), and put \(c=2-r>0\).
The existing [sample-extrema derivation](../qhat_local_minima_characterization_2026-10-28.md)
shows that
\[
\mathcal E_n:=\{\widehat Q_r\text{ has an interior local minimum and maximum on }(0,\infty)\}
=\{\inf_{h>0}B_n(h)<c\}.
\]

The following literature connections can be verified directly:

* [Lindeberg (1998), Section 9.2, Eqs. (82)–(83), manuscript printed p. 39](https://people.kth.se/~tony/papers/cvap198.pdf#page=41)
  defines normalized Gaussian derivative energies and relates them to spectral
  moments. In dimension one, with his \(\gamma=1\) and \(t=h^2\),
  \(E_j(t)=t^jR_j(\sqrt t)\). Substitution gives the exact identification
  \(B_n(h)=E_3(h^2)/E_2(h^2)\). This identification is our algebraic use
  of his definitions, not a claim that he states the KDE threshold theorem.
* [Ghisi, Gobbino and Haraux, arXiv manuscript, Section 3.2, Eq. (3.16)](https://arxiv.org/pdf/1402.5355#page=16)
  define the classical Dirichlet quotient
  \(\|A^{1/2}u\|^2/\|u\|^2\). With \(u=\widehat f_h''\) and
  \(A=-\partial_x^2\), integration by parts identifies
  \(B_n(h)/h^2=\langle u,Au\rangle/\langle u,u\rangle\).
  We use the quotient definition, not the applicability of every PDE theorem
  in that paper to an empirical KDE on the real line.
* [Botev, Grotowski and Kroese (2010), Section 5, Eq. (26)](https://arxiv.org/pdf/1011.2602#page=15)
  give the exact Gaussian pair representation of \(R_j\).
  [Hall and Marron (1987), pp. 109–115](https://www.sciencedirect.com/science/article/pii/0167715287900836)
  study estimation of integrated squared density derivatives.
* [Scott and Terrell (1987), technical report, Section 1.1, Eq. (1.1)](https://repository.rice.edu/bitstream/1911/101613/1/TR87-02.pdf)
  explicitly use \(R(g)=\int g(x)^2\,dx\) as a measure of roughness.
  Thus R2 and R3 are the roughness functionals of the second and third
  derivatives, respectively. This supports the descriptive name
  "dimensionless roughness ratio" for B when defined explicitly.
* A subsequent, more targeted terminology search found a direct naming
  precedent: [Coleman-Smith, Petersen and Wolpert (2013), "Classification
  of initial state granularity via 2D Fourier expansion", Eq. (8),
  manuscript p. 7](https://arxiv.org/pdf/1204.5774#page=7).
  They use the term "roughness ratio" in connection with
  \[
  \mathcal R_\ell(g)^2
  =\frac{H_1(g)^2}{L_2(g)^2}-1
  =\ell^2\frac{\langle-\Delta g,g\rangle}{\langle g,g\rangle}.
  \]
  The same definition appears in [Petersen, Coleman-Smith and Wolpert
  (2013), "Quantifying Initial State Fluctuations in Heavy Ion
  Collisions", p. 801, Eq. (7)](https://www.actaphys.uj.edu.pl/fulltext?page=797&series=Sup&vol=6).
  In one dimension on the real line, with \(g=\widehat f_h''\), \(\ell=h\),
  and the Gaussian decay that removes integration-by-parts boundary terms,
  the displayed squared quantity becomes exactly
  \(h^2\|\widehat f_h'''\|_2^2/\|\widehat f_h''\|_2^2=B_n(h)\).
  This is our specialization of their quotient structure. Their application
  concerns fluctuation fields in heavy-ion physics on a disk, rather than
  Gaussian KDE bandwidth selection. In particular, their square/root
  convention should be stated when comparing names.

Also, the project identity \(R_2'=-2hR_3\) gives
\[
B_n(h)=-\tfrac12\,\frac{d\log R_2(h)}{d\log h}.
\]
Thus B is a scaled Dirichlet quotient, a ratio of normalized spectral
energies, and a negative logarithmic slope.

The searches checked here did not verify a universally established KDE name
or an earlier paper stating precisely this empirical KDE ratio together with
\(\inf_hB_n(h)<2-r\) as an extrema-existence criterion. However, the targeted
search above did verify published use of "roughness ratio" for the directly
related scaled Dirichlet quotient. Therefore the earlier statement that no
standard name had been verified should be read narrowly as concerning the
particular KDE quantity, not the absence of the terminology in mathematics
or physics. These limited search findings are not proof of originality.

## 2. Exact expectations of derivative norms

Now assume \(X_i\) are i.i.d. with density \(f\). Put \(g_h=K_h*f\).
Separating diagonal and off-diagonal terms in the pair representation gives
the exact identity
\[
\mu_j(h):=\mathbb E R_j(h)
=\frac{c_j}{n h^{2j+1}}+\frac{n-1}{n}\|g_h^{(j)}\|_2^2,\qquad
c_j=\frac{(2j-1)!!}{2^{j+1}\sqrt\pi}.
\]
For fixed positive h the Gaussian derivative kernels are bounded and
integrable, so this calculation is justified without smoothness of f.
In particular, \(c_2=3/(8\sqrt\pi)\), \(c_3=15/(16\sqrt\pi)\).

Define the deterministic ratio
\[
\overline B_n(h)=h^2\frac{\mathbb E R_3(h)}{\mathbb E R_2(h)}.
\]
It is not in general \(\mathbb E B_n(h)\). Indeed,
\(\overline B_n=\mathbb E[R_2B_n]/\mathbb E R_2\): it weights samples by
their R2 value. Independent one-dimensional quadrature for n=2,
Gaussian N(0,1) observations, h=0.7 gives approximately
\(\mathbb EB_n=2.4289449485\), while
\(\overline B_n=2.4020180685\); see [checks.json](checks.json).
No simple exact finite-sample formula for the expectation of the ratio
was verified in this search.

The deterministic ratio is nevertheless exactly the right quantity for
the *expected energy*:
\[
\frac{d}{dh}\mathbb E\widehat Q_r(h)
=h^{3-2r}\mu_2(h)[c-\overline B_n(h)].
\]
This is a statement about the expected curve, not about the probability
that a random sample curve has an extremum.

## 3. Gaussian sampling: explicit deterministic ratio and expected extrema

If additionally \(f=N(\mu,\sigma^2)\), with \(\sigma>0\), then
\[
\mu_j(h)=c_j\left[
\frac1{nh^{2j+1}}+\frac{n-1}{n}(\sigma^2+h^2)^{-(j+1/2)}
\right].
\]
Consequently, with \(s=h^2/(\sigma^2+h^2)\),
\[
\boxed{\overline B_n(h)=\frac52
\frac{1+(n-1)s^{7/2}}{1+(n-1)s^{5/2}}.}
\]
The [earlier expected-energy derivation, Section 5](../qhat_local_minima_characterization_2026-10-28.md)
then gives exactly one expected local minimum and one expected local maximum
if and only if
\[
n>1+\frac{1+2r}{2[(4-2r)/7]^{7/2}}.
\]
For r=13/14 this means n>=92. This is not a probability-one statement
about realized sample extrema.

## 4. A finite-sample lower bound on the probability of extrema

At any fixed test bandwidth \(h_0>0\), define the random contrast
\[
T_n(h_0)=cR_2(h_0)-h_0^2R_3(h_0).
\]
Since R2 is positive, \(T_n(h_0)>0\) implies \(B_n(h_0)<c\), and hence
\(\mathcal E_n\). Write \(m_n=\mathbb ET_n(h_0)\) and
\(v_n=\operatorname{Var}(T_n(h_0))\). Whenever \(m_n>0\),
\[
\boxed{\mathbb P_f(\mathcal E_n)
\ge\mathbb P_f(T_n(h_0)>0)
\ge\frac{m_n^2}{m_n^2+v_n}.}
\]
This is the one-sided variance inequality (often called Cantelli's
inequality); see [Hoeffding (1963), p. 13, Eq. (1.6)](https://www.cs.rpi.edu/academics/courses/spring06/random/hoefding.pdf).
It also follows directly by bounding
\(\mathbb P\{Y\le-m_n\}\le(v_n+a^2)/(m_n+a)^2\), where
\(Y=T_n-m_n\), and choosing \(a=v_n/m_n\).

Here are explicit moment formulas to make this bound computable. Define
\[
A_{j,h}(x,y)=(-1)^jK_{\sqrt2h}^{(2j)}(x-y),\qquad
H_h(x,y)=cA_{2,h}(x,y)-h^2A_{3,h}(x,y).
\]
Then \(T_n=n^{-2}\sum_{i,j}H_h(X_i,X_j)\). Its diagonal
\[
d_h=H_h(x,x)=\frac{cc_2-c_3}{h^5}
\]
is constant. For independent \(X,Y\sim f\), put
\[
\theta_h=\mathbb EH_h(X,Y),\quad
\zeta_1=\operatorname{Var}(\mathbb E[H_h(X,Y)\mid X]),\quad
\zeta_2=\operatorname{Var}(H_h(X,Y)).
\]
Direct covariance counting of overlapping pairs, for n>=2, yields
\[
m_n=\frac{d_h}{n}+\frac{n-1}{n}\theta_h,
\qquad
v_n=\frac{4(n-1)(n-2)}{n^3}\zeta_1+
     \frac{2(n-1)}{n^3}\zeta_2.
\]
These variance formulas are derived here by expanding the pair sum:
disjoint pairs are independent, pairs sharing one observation contribute
zeta1, and identical/reversed pairs contribute zeta2.
They also show \(v_n=O(1/n)\) for every fixed positive h.

For Gaussian sampling all of these moments are Gaussian-polynomial
integrals. With r=13/14 and \(h_0=3\sigma/5\), exact algebraic moment
expressions, evaluated numerically and checked independently by quadrature,
give the following lower bounds:

| n | Lower bound for at least one minimum–maximum pair |
|---:|---:|
| 200 | 0.61545192 |
| 1,000 | 0.98801929 |
| 1,500 | 0.99396219 |
| 2,000 | 0.99614148 |

See [checks.json](checks.json) for the exact moment expressions and values.
These are analytical lower bounds evaluated numerically, not estimated
probabilities or certified interval-arithmetic enclosures. They concern
the whole domain h>0, not a prescribed finite search interval, and guarantee
existence rather than uniqueness. The choice h0 uses the model scale sigma;
it is a proof device and does not define a data-based selector.

## 5. A general large-sample existence result

The following is an original deduction for this project.
Assume i.i.d. sampling from a density f on the real line and
\[
0<\|f''\|_2^2<\infty
\]
with the second derivative understood weakly. Keep the Gaussian kernel
and \(-1/2<r<2\). Then
\[
\boxed{\mathbb P_f(\mathcal E_n)\longrightarrow1.}
\]

Proof: for fixed h>0, the bounded pair kernels give, by the exact expectation
formula and the overlapping-pair variance argument,
\[
R_j(h)\xrightarrow{P}\|(K_h*f)^{(j)}\|_2^2,\qquad j=2,3.
\]
Thus
\[
B_n(h)\xrightarrow{P}
B_f(h):=h^2\frac{\|(K_h*f)'''\|_2^2}{\|(K_h*f)''\|_2^2}.
\]
The denominator is strictly positive: otherwise the smoothed density
would be affine and integrable with mass one, which is impossible.
To control small h, Plancherel gives
\[
B_f(h)=
\frac{\int (h^2\omega^2)e^{-h^2\omega^2}
                 \omega^4|\widehat f(\omega)|^2\,d\omega}
     {\int e^{-h^2\omega^2}
                 \omega^4|\widehat f(\omega)|^2\,d\omega}.
\]
The denominator tends to \(2\pi\|f''\|_2^2>0\), with the Fourier
convention used in the project. In the numerator,
\(h^2\omega^2e^{-h^2\omega^2}\to0\) pointwise and is at most \(1/e\).
Dominated convergence therefore gives \(B_f(h)\to0\) as h decreases to zero.
Choose a *fixed* sufficiently small h0 with \(B_f(h_0)<c\). Then
\(\mathbb P(B_n(h_0)<c)\to1\), and this event implies \(\mathcal E_n\).
Equivalently, the contrast has a positive limiting mean and variance O(1/n);
the bound in Section 4 gives a conservative failure bound O(1/n).

This proves the existence of at least one pair with high probability. It
does not establish uniqueness, an optimal bandwidth location, or validity
of any particular rule for selecting among minima.

For Gaussian f specifically,
\[
B_f(h)=\frac52\frac{h^2}{\sigma^2+h^2}.
\]
This fixed-h population curve is not the finite-sample expectation of B.
It increases from 0 to 5/2. There is no contradiction with the finite-sample
limit \(B_n(h)\to5/2\) as h decreases to zero: the orders of the limits in
n and h differ. The earlier [shrinking-scale Gaussian derivation](../gaussian_qhat_properties_2026-10-29/large_sample_extrema.md)
additionally identifies a local minimum on the n^(-1/5) scale.
