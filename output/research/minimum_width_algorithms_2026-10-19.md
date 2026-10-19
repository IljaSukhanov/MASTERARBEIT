# Algorithms for Euclidean minimum width: dimension, guarantees, and running time

Literature checked on **19 October 2026**. This note concerns one enclosing slab containing all input points. It was created outside the protected thesis sources.

## Scope and notation

For a nonempty finite set \(P=\{p_1,\ldots,p_n\}\subset\mathbb R^d\), the problem is

\[
w^*(P)=\min_{\|u\|_2=1}w_P(u),\qquad
w_P(u)=\max_i\langle u,p_i\rangle-\min_i\langle u,p_i\rangle.
\]

This is the minimum-width enclosing slab, equivalently twice the minimum possible maximum perpendicular distance to an affine hyperplane. See [Chan (2002), Sections 1 and 3](https://tmc.web.engr.illinois.edu/apx.pdf). It matches [the empirical objective in this project](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/3_application_to_time_series.tex:315).

Here \(n\) counts embedded points and \(d\) is embedding dimension. In the thesis, embedding dimension is called \(k\). These are not multiple-slab or trimmed-enclosure problems.

Unless stated otherwise, the bounds below are geometric arithmetic-operation bounds with **fixed \(d\)**. They suppress constants depending on dimension; they are not full bit-complexity or floating-point error analyses. The exact-runtime slack \(\delta>0\), approximation tolerance \(0<\varepsilon<1\), and additional fixed exponent slack \(\alpha>0\) are distinct.

If the affine dimension of \(P\) is less than \(d\), its ambient minimum width is zero: choose a unit normal to its affine hull, so all projections coincide. This is a direct deduction from the definition. The positive-width discussion assumes full affine dimension.

## 1. Strongest general exact bounds verified

| Ambient dimension | Exact upper bound | Algorithm/source | Qualification |
|---|---|---|---|
| \(d=1\) | \(\Theta(n)\) | Scan for minimum and maximum | Direct algorithm and input-reading lower bound. |
| \(d=2\) | \(O(n\log n)\) | Convex hull plus rotating calipers | Optimal worst-case bound in the standard comparison/algebraic setting. |
| \(d=3\) | \(O(n^{3/2+\delta})\), for every fixed \(\delta>0\) | Agarwal and Sharir (1996) | Randomized **expected** running time; exact geometric optimum. |
| \(d=4\) | \(O(n^2)\) | Chan (2002), Section 3 | Fixed dimension; lifted feasible-polytope construction. |
| \(d=5,6\) | \(O(n^3)\) | Same general construction | Substitution into \(O(n^{\lceil d/2\rceil})\). |
| \(d=7,8\) | \(O(n^4)\) | Same | Same. |
| \(d=9,10\) | \(O(n^5)\) | Same | Same. |
| Any fixed \(d\ge4\) | \(O(n^{\lceil d/2\rceil})\) | Chan (2002) | A polynomial for each fixed dimension, not a dimension-independent polynomial. |

Primary sources: [Houle and Toussaint (1988), Section 2.B and Theorem 2.2](https://cgm.cs.mcgill.ca/~godfried/publications/width.pdf); [Agarwal and Sharir (1996), abstract](https://link.springer.com/article/10.1007/BF02712871); [Chan (2002), Sections 1 and 3](https://tmc.web.engr.illinois.edu/apx.pdf).

The same dimension-dependent exact bounds are explicitly reported in [Ahn and Bae (2025), Introduction, p. 2](https://www.sciencedirect.com/science/article/pii/S0304397525005213). A [2026 paper, Introduction](https://drops.dagstuhl.de/storage/00lipics/lipics-vol368-forc2026/html/LIPIcs.FORC.2026.18/LIPIcs.FORC.2026.18.html) also reports the planar and three-dimensional bounds. I did not find a subsequent improvement to the general exact bounds above. This is a best-verified literature overview, not a proof that no newer result exists.

**Upper bound does not mean proven optimality.** The planar worst-case result is optimal. No matching optimality claim is made here for the three-dimensional or higher-dimensional exponents.

### Two dimensions: hull and rotating calipers

Compute the ordered convex hull, then rotate a pair of parallel supporting lines while maintaining antipodal contacts. If the hull has \(h\) vertices and is already available in cyclic order, the width phase is \(O(h)\). An output-sensitive hull construction yields \(O(n\log h)\) total time for full-dimensional planar inputs. [Houle and Toussaint, Sections 2.B–2.C, Theorem 2.2](https://cgm.cs.mcgill.ca/~godfried/publications/width.pdf).

### Three dimensions: useful alternatives to the fastest bound

| Method | Bound | What is useful about it |
|---|---|---|
| Hull plus antipodal vertex-face and edge-edge pairs | \(O(n\log n+I)\), where \(I\) counts antipodal edge-edge pairs; worst case quadratic | A more direct geometric baseline. |
| Parametric searching, Chazelle et al. (1993) | \(O(n^{8/5+\delta})\), deterministic | A verified deterministic subquadratic alternative; not asserted here to be the current best deterministic result. |
| Randomized geometric optimization, Agarwal–Sharir (1996) | \(O(n^{3/2+\delta})\), expected | Strongest exact general 3D bound verified in this search. |

Sources: [Houle–Toussaint, Theorems 3.1–3.3](https://cgm.cs.mcgill.ca/~godfried/publications/width.pdf); [Chazelle et al., abstract](https://link.springer.com/article/10.1007/BF02573973); [Agarwal–Sharir, abstract](https://link.springer.com/article/10.1007/BF02712871).

Checking only distances associated with hull facets misses a necessary class of candidates in 3D: two opposite supporting planes can touch edges whose directions differ. Theorem 3.1 of Houle–Toussaint explicitly includes both vertex-face and edge-edge pairs.

### Fixed higher dimensions: a lifted polytope

Chan's Section 3 normalizes the slab's unnormalized projection range to one. In equivalent notation, define

\[
F=\{(v,a)\in\mathbb R^{d+1}:a\le\langle v,p_i\rangle\le a+1,\quad i=1,\ldots,n\}.
\]

For a full-dimensional cloud,

\[
w^*(P)=\frac{1}{\max_{(v,a)\in F}\|v\|_2}.
\]

The constraints are linear, but maximizing the convex norm is not an ordinary convex minimization problem. The exact construction enumerates the lifted feasible polytope, with \(O(n^{\lceil d/2\rceil})\) complexity, and optimizes over its vertices. [Chan (2002), Section 3, opening formulation](https://tmc.web.engr.illinois.edu/apx.pdf).

This is not the same operation as merely building the original \(d\)-dimensional hull and scanning its facet normals.

## 2. Approximation algorithms

The desired guarantee can be expressed as a direction whose full-data width satisfies

\[
w^*(P)\le w_P(\hat u)\le(1+\varepsilon)w^*(P).
\]

This is a multiplicative global-width guarantee. Small width error alone is not a coefficient-error guarantee for the recovery map \(\hat\phi_j=-\hat u_j/\hat u_0\) used [in the thesis](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/3_application_to_time_series.tex:311).

### A relatively direct guaranteed method: a cover of directions and multiple LPs

The Duncan–Goodrich–Ramos approach, reformulated by Chan, solves \(O(\varepsilon^{-(d-1)/2})\) fixed-dimensional linear programs over the same lifted constraints. Each LP maximizes a linear objective \(\langle b,v\rangle\) for a direction \(b\) in a suitable spherical cover. The total bound is

\[
O\!\left(n\varepsilon^{-(d-1)/2}\right)
\]

for fixed \(d\), using linear-time fixed-dimensional LP algorithms. [Chan (2002), Section 3, Algorithm 1](https://tmc.web.engr.illinois.edu/apx.pdf).

The LPs adapt the fitted normal to the data. Simply evaluating widths along a coarse collection of directions is a different algorithm and lacks this aspect-ratio-independent relative guarantee. See also [the 2026 exposition, Section 2.2.1](https://drops.dagstuhl.de/storage/00lipics/lipics-vol368-forc2026/html/LIPIcs.FORC.2026.18/LIPIcs.FORC.2026.18.html).

### Strongest end-to-end approximation bound verified

**Arya, da Fonseca, and Mount (ESA 2018), Theorem 3**, give a width approximation in

\[
\boxed{
O\!\left(n\log(1/\varepsilon)
+\varepsilon^{-(d-1)/2-\alpha}\right)
}
\]

for fixed \(d\) and any fixed \(\alpha>0\). Their construction approximates the difference body through approximate convex intersection and Minkowski-sum machinery. [Theorem 3, Section 1.3, and the final argument on p. 3:12](https://drops.dagstuhl.de/storage/00lipics/lipics-vol112-esa2018/LIPIcs.ESA.2018.3/LIPIcs.ESA.2018.3.pdf).

The direction of a closest bounding hyperplane of the outer difference-body approximation supplies an approximate slab direction; the actual slab endpoints can then be obtained by a scan of the input. This follows from the paper's halfspace construction and its width argument.

The additive accuracy-dependent term specializes as follows. The same \(n\log(1/\varepsilon)\) term is present in every row.

| \(d\) | Accuracy-dependent term |
|---|---|
| 2 | \(\varepsilon^{-1/2-\alpha}\) |
| 3 | \(\varepsilon^{-1-\alpha}\) |
| 4 | \(\varepsilon^{-3/2-\alpha}\) |
| 5 | \(\varepsilon^{-2-\alpha}\) |
| 6 | \(\varepsilon^{-5/2-\alpha}\) |
| 7 | \(\varepsilon^{-3-\alpha}\) |
| 8 | \(\varepsilon^{-7/2-\alpha}\) |
| 9 | \(\varepsilon^{-4-\alpha}\) |
| 10 | \(\varepsilon^{-9/2-\alpha}\) |

These rows are arithmetic substitutions into Theorem 3. For a fixed accuracy and dimension, the bound is linear in \(n\). Constants can depend strongly on \(d\) and \(\alpha\); this is not a measured wall-clock recommendation. The theorem is stated as an ordinary time bound, without an expected-time qualifier.

I found no later general width-approximation bound improving this one. I checked later literature searches and [the authors' publication list, including 2024–2026 work](https://www.cse.ust.hk/~arya/publications.html). I do not assert a matching lower bound for computing a single approximate width value.

### Kernels are useful, but their construction time is not the full solver time

An \(\varepsilon\)-kernel \(Q\subseteq P\) preserves all directional widths:

\[
(1-\varepsilon)w_P(u)\le w_Q(u)\le w_P(u).
\]

[Agarwal, Har-Peled, and Varadarajan (2004), Section 2, Theorem 3.7, Section 6.1](https://graphics.stanford.edu/courses/cs468-06-winter/Papers/har-peled-JACM2004%20fitting.pdf).

The later [Arya–da Fonseca–Mount (2017), Theorem 1.1](https://drops.dagstuhl.de/storage/00lipics/lipics-vol077-socg2017/LIPIcs.SoCG.2017.10/LIPIcs.SoCG.2017.10.pdf) builds a kernel of size \(O(\varepsilon^{-(d-1)/2})\) in \(O(n\log(1/\varepsilon)+\varepsilon^{-(d-1)/2-\alpha})\) time. A global width computation on that kernel still costs additional time. The 2018 paper explicitly distinguishes this issue from its end-to-end improvement in Section 1.3.

Direct deductions, assuming the kernel guarantee and an exact global solution on \(Q\), are

\[
(1-\varepsilon)w^*(P)\le w^*(Q)\le w^*(P),
\qquad
w_P(u_Q)\le\frac{w^*(P)}{1-\varepsilon}.
\]

An arbitrary local optimization run on \(Q\) does not establish the exact value \(w^*(Q)\).

## 3. Variable dimension and problem representation

The exact fixed-dimensional table does not imply a polynomial-time algorithm in \(n\) and \(d\) together. For rational Euclidean inputs with dimension part of the input, deciding whether squared width is below a rational threshold is NP-hard even for simplices; membership in NP gives the corresponding NP-completeness statement. [Gritzmann and Klee (1993), Theorems 1.3 and 5.5, printed pp. 171 and 193](https://doi.org/10.1007/BF01581243).

Their Theorem 1.8, printed p. 172, gives polynomial-time computation of squared width in each fixed Euclidean dimension. Their model is the binary/Turing model; distinguish it from the operation counts in the geometric table.

Symmetry and input representation also matter: symmetric polytopes given by bounding inequalities have an easier width computation than symmetric polytopes given by points in variable dimension. The latter case remains NP-hard; see their Theorems 1.1 and 1.5. These are worst-case results for general polytopes, not a hardness theorem for the AR-generated subclass.

## 4. What this means for the existing project

The following are workspace findings and implementation assessments, not literature running-time theorems.

1. **An exact geometric construction is already present for \(d\le4\).** The function [exact_empirical_width_low_dimension](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/ar_tools/lp_widths.py:72) forms pairwise differences of hull vertices, builds their convex hull, and takes its nearest facet to the origin. This uses the difference-body identity \(w_P(u)=h_{K-K}(u)\), \(K=\operatorname{conv}(P)\), also used by the 2018 paper. It is a global floating-point construction, not a rational arithmetic certificate.

2. **That implementation is not Chan's \(O(n^2)\) four-dimensional algorithm.** If the original hull has \(h\) vertices, it explicitly creates \(h^2\) difference vectors before another hull computation. Its running time and memory therefore depend on those intermediate objects. No timing comparison was performed for this note.

3. **The current subgradient loop has a different type of cost and guarantee.** Each iteration computes all projections and extrema: [descent.py, main loop](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/ar_tools/descent.py:100). Counting the displayed matrix-vector product and scans gives \(O(nd)\) work per iteration, hence \(O(Tnd)\) for \(T\) iterations per start. This operation count is our code analysis; it supplies no global approximation guarantee. The thesis's stated convergence discussion assumes a sufficiently close starting point: [source](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/3_application_to_time_series.tex:452).

4. **The existing LP iteration is also distinct from the global LP-cover method.** [lp_width_descent](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/ar_tools/lp_widths.py:12) adapts the LP normalization direction iteratively. It does not enumerate a spherical cover with a prescribed covering radius. Its mere use of LPs does not make it Chan's globally guaranteed approximation scheme.

### Implementation priorities: assessment

- For \(d=2\), a hull-plus-calipers exact baseline is the first method I would implement or verify.
- For \(d=3,4\), first profile the existing difference-body routine on representative small and moderate clouds. Use it to distinguish optimization error from sampling effects where numerically reliable.
- For \(d=4\), investigate the lifted \(5\)-dimensional polytope approach if the difference-body construction becomes too expensive.
- For \(d=5\) through \(10\), a complete LP-cover implementation is a comparatively direct candidate for a global relative-error benchmark. Its cover must have a proved covering radius; random restarts alone do not provide that deterministic guarantee.
- Treat the 2018 approximation result as the strongest verified theoretical benchmark, with implementation complexity assessed separately.

## 5. Search and verification record

The exact bounds were checked against original sources and recent slab papers, including Ahn–Bae (2025). The 2026 FORC paper corroborates the low-dimensional exact bounds and provides a recent exposition of the classical LP approximation scheme. The 2017 kernel theorem and 2018 width theorem were checked separately to avoid confusing coreset construction with end-to-end minimization. Later author publication lists and targeted searches did not reveal an improved general width bound.

The companion BibTeX file is [minimum_width_algorithms_2026-10-19.bib](/Users/iljasukhanov/MASTERARBEIT/output/research/minimum_width_algorithms_2026-10-19.bib). No claim about fastest practical implementation is made: that requires measurements on the project's \(n,d,\varepsilon\), hull sizes, and hardware.
