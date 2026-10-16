# Intrinsic volumes of the full AR(3) stability region

These are our boundary calculations for the closure of the full coefficient
region, in the standard Euclidean metric of `(phi_1, phi_2, phi_3)`. They are not
estimates from the 500 sampled points. The two intermediate values are numerical
evaluations of explicit integrals, not literature quotations or claimed closed forms.

| Intrinsic volume | Value |
|---|---:|
| V0 | 1, exactly |
| V1 | 9.832991784702918, numerical |
| V2 | 13.568717996370772, numerical |
| V3 | 16/3, exactly |

The boundary surface area is `2 V2 = 27.137435992741544`.

## Convention and sources

We use the normal-cycle/Lipschitz–Killing extension to compact semialgebraic sets.
The normalization agrees with the convex Steiner formula

\[
\operatorname{Vol}(K+rB^3)
=V_3(K)+2V_2(K)r+\pi V_1(K)r^2+\tfrac{4\pi}{3}V_0(K)r^3
\]

when the convex Steiner formula applies. This display fixes the normalization;
we do not assert this offset identity for our nonconvex region.

- [Dutertre, *Euler characteristic and Lipschitz-Killing curvatures of closed
  semi-algebraic sets*, §§2–3](https://arxiv.org/pdf/1002.0767): normal cycles,
  curvature measures, and the compact Gauss–Bonnet identity.
- [Solanes and Trillo, *Tube formulas for valuations in complex space forms*,
  §2.2, equation (7)](https://link.springer.com/article/10.1007/s00208-024-02929-2):
  intrinsic-volume normalization by the Steiner formula.
- [Edelsbrunner and Pausinger, *Approximation and convergence of the intrinsic
  volume*, §1, pp. 2–3](https://pub.ista.ac.at/~edels/Papers/2016-03-FirstIntVolume.pdf):
  total mean curvature and the signed edge-angle contribution.
- [Kirschenhofer and Thuswaldner, *Distribution results on polynomials with
  bounded roots*, §1, equation (1.1)](https://arxiv.org/pdf/1609.06947): independent
  general-dimensional volume formula, giving `V3 = 16/3`.
- Workspace parametrization: [the existing Jones sampler](../../../Code/Code_AR_processes/ar_tools/functions.py),
  function `jones_ar_coefficients`. All geometric calculations below are our
  derivations from its recursion.

## Exact region and its three boundary patches

Write `(x,y,t)=(phi_1,phi_2,phi_3)`. If the first two partial autocorrelations are
`a,b`, the sampler gives

\[
x=a(1-b)-tb,\qquad y=b-ta(1-b),\qquad t=\kappa_3.
\]

Thus each section at fixed `-1 <= t <= 1` is exactly the triangle

\[
(x,y)\in\operatorname{conv}\{(t-2,2t-1),\ (t+2,-2t-1),\ (-t,1)\}.
\]

This section description includes the degenerate endpoint sections correctly.
In particular, it prevents spurious rays that can be introduced by naively
extending strict interior inequalities to `t=+/-1`.

The four vertices are

\[
A=(-3,-3,-1),\quad B=(1,1,-1),\quad
C=(-1,1,1),\quad D=(3,-3,1).
\]

The boundary consists of:

1. Triangle `BCD` in `x+y+t=1`, outward normal `(1,1,1)/sqrt(3)`;
2. Triangle `ABC` in `-x+y-t=1`, outward normal `(-1,1,-1)/sqrt(3)`;
3. The saddle patch
   \[
   r(s,t)=(s+t,-1-st,t),\qquad -2\le s\le2,\quad-1\le t\le1.
   \]

Each planar triangle has area `4 sqrt(3)`. The saddle's outward normal is

\[
n(s,t)=\frac{(-t,-1,t-s)}{W(s,t)},\qquad
W(s,t)=\sqrt{1+t^2+(t-s)^2}.
\]

## V0, V2, and V3

Shrinking all polynomial roots by `u` gives the contraction
`(x,y,t) -> (u x,u^2 y,u^3 t)` of this closed region to the origin.
Consequently its Euler characteristic, and therefore `V0`, is 1.

The area element on the saddle is `W(s,t) ds dt`, giving

\[
V_2=4\sqrt3+\frac12\int_{-1}^{1}\int_{-2}^{2}W(s,t)\,ds\,dt.
\]

The section triangle has area `4(1-t^2)`, so

\[
V_3=\int_{-1}^{1}4(1-t^2)\,dt=\frac{16}{3}.
\]

## V1, including all five edges

Use `H = div(n)/2`, positive on a sphere with its outward normal. The
piecewise-smooth normal-cycle expression, in this normalization, is

\[
\pi V_1=\int_{\partial K_{\rm smooth}}H\,dA
+\frac12\sum_{e}\int_e\theta_e\,d\ell,
\]

where `theta_e` is the signed exterior angle between adjacent outward normals.
The interiors of all five boundary edges in this region have positive exterior
angle. The planar faces have zero smooth mean curvature. On the saddle,

\[
H=\frac{1+st}{W(s,t)^3},\qquad
M_{\rm smooth}=\int_{-1}^{1}\int_{-2}^{2}
\frac{1+st}{1+t^2+(t-s)^2}\,ds\,dt.
\]

The ridge `BC` has length `2 sqrt(2)` and angle `acos(-1/3)`.
The two edges `s=+/-2` and the two edges `t=+/-1` on the saddle are symmetric
in pairs. Combining them, including the factor `1/2`, gives

\[
\begin{aligned}
M_{\rm edges}={}&\sqrt2\arccos(-1/3)\\
&+\sqrt6\int_{-1}^{1}
\arccos\!\left(\frac{-\sqrt3}{\sqrt{3+2(t-1)^2}}\right)dt\\
&+\sqrt2\int_{-2}^{2}
\arccos\!\left(\frac{-1-s}{\sqrt{3[2+(s-1)^2]}}\right)ds.
\end{aligned}
\]

Numerically,

\[
M_{\rm smooth}=4.188999563583965,\quad
M_{\rm edges}=26.70225519004751,\quad
V_1=(M_{\rm smooth}+M_{\rm edges})/\pi.
\]

The use of signed mean curvature matters: it is not the integral of `|H|`.

## Numerical verification

`calculate.py` evaluates these integrals by adaptive quadrature at tolerances
`1e-8` and `1e-11`, with agreement better than `1e-8`. Reported quadrature errors
are numerical estimates, not certified interval bounds.

An independent conforming triangulation of the saddle and both planar faces
checks surface area, volume, and signed edge-angle mean curvature. Every mesh
is closed, has opposite orientations across each shared edge, and has boundary
Euler characteristic 2.

| Mesh parameter n | Triangles | V1 | V2 | V3 |
|---:|---:|---:|---:|---:|
| 20 | 1,722 | 9.832443056 | 13.577202273 | 5.340000000 |
| 40 | 6,642 | 9.832854618 | 13.570839062 | 5.335000000 |
| 80 | 26,082 | 9.832957494 | 13.569248263 | 5.333750000 |
| 160 | 103,362 | 9.832983212 | 13.568850563 | 5.333437500 |
| Boundary integrals | — | 9.832991785 | 13.568717996 | 5.333333333 |

The mesh errors decrease by about a factor of four under each refinement.
Richardson extrapolation from the final two meshes, `(4 value_160-value_80)/3`,
agrees with the boundary integrals to approximately `8e-11` for V1 and `2e-11`
for V2. This is a numerical consistency check, not a rigorous error bound.
Complete results, intermediate contributions, and quadrature error estimates
are in `results.json`.
