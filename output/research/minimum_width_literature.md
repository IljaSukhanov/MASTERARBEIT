# Minimum width of a point cloud: literature positioning

Research checked on 18 October 2026. This is a separate research note; no file under `Abschlussarbeit_2021/` was changed.

## Match to the thesis

The TODO is in [the literature-positioning section](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/3_application_to_time_series.tex:729). The empirical objective is defined at [lines 315 onward](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/3_application_to_time_series.tex:315).

For a finite nonempty point set \(P\subset\mathbb R^k\), the relevant problem is

\[
m(P)=\min_{\|u\|_2=1}\left(\max_{x\in P}\langle u,x\rangle-\min_{x\in P}\langle u,x\rangle\right).
\]

It is called the **width of a point set** or **minimum-width enclosing slab**. It equals the width of its convex hull. See [Chan (2002), Sections 1 and 3, author PDF](https://tmc.web.engr.illinois.edu/apx.pdf) and [Agarwal et al. (2004), Sections 2 and 6.1, author PDF](https://graphics.stanford.edu/courses/cs468-06-winter/Papers/har-peled-JACM2004%20fitting.pdf).

## Recommended papers

1. **Houle, M. E. and Toussaint, G. T. (1988). Computing the Width of a Set. IEEE TPAMI 10(5), 761–765.** [DOI](https://doi.org/10.1109/34.6790) · [Author PDF](https://www-cgrl.cs.mcgill.ca/~godfried/publications/width.pdf).
   Foundational exact algorithms in two and three dimensions: planar width in \(O(n\log n)\), and spatial width in \(O(n\log n+I)\), with worst-case quadratic antipodal-edge count \(I\). Section 3.C, Theorem 3.3 of the linked manuscript explicitly treats minimax plane fitting. This directly supports the hyperplane-fitting interpretation already in the thesis. The conference predecessor appeared at SoCG 1985, [pages 1–7](https://doi.org/10.1145/323233.323234).

2. **Chan, T. M. (2002). Approximating the Diameter, Width, Smallest Enclosing Cylinder, and Minimum-Width Annulus. IJCGA 12(1–2), 67–85.** [DOI](https://doi.org/10.1142/S0218195902000748) · [Author PDF](https://tmc.web.engr.illinois.edu/apx.pdf).
   Read Section 3, “Width.” It formulates slab fitting with linear enclosure constraints and a normalized, nonconvex objective, then gives a direction-discretization approach using multiple LPs with a multiplicative approximation guarantee. Section 1 also discusses an exact \(O(n^{\lceil k/2\rceil})\) approach for fixed \(k\ge4\). Chan credits the underlying width approximation to Duncan, Goodrich and Ramos (1997); this is not a claim that Chan originated it.

3. **Agarwal, P. K., Har-Peled, S. and Varadarajan, K. R. (2004). Approximating Extent Measures of Points. JACM 51(4), 606–635.** [DOI](https://doi.org/10.1145/1008731.1008736) · [Author PDF](https://graphics.stanford.edu/courses/cs468-06-winter/Papers/har-peled-JACM2004%20fitting.pdf).
   The strongest general reference here for arbitrary **fixed** dimension. Small subsets preserve directional widths and therefore approximate minimum width. Consult Sections 2–3 and Section 6.1; the linked manuscript states Theorems 3.7 and 6.1. The dimension is held fixed in its running-time claims. This paper already appears in [the thesis bibliography](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/bibliography.bib:173), under key `10.1145/1008731.1008736`.

4. **Varadarajan, K. R., Venkatesh, S., Ye, Y. and Zhang, J. (2007). Approximating the Radii of Point Sets. SIAM J. Comput. 36(6), 1764–1776.** [DOI](https://doi.org/10.1137/050627472) · [Published paper on an author's site](https://webhome.cs.uvic.ca/~srinivas/papers/radii.pdf).
   Useful for the distinction between fixed and variable ambient dimension: the introduction discusses NP-hardness when dimension is part of the input. Theorem 1 gives a polynomial-time \(O(\sqrt{\log n})\) approximation for outer radii. Their definition uses maximum distance to a fitted flat; for hyperplanes this is **half** the thesis's slab width. This factor follows directly from the definitions, despite the paper also using the word “width.”

5. **Chazelle, B., Edelsbrunner, H., Guibas, L. J. and Sharir, M. (1993). Diameter, Width, Closest Line Pair, and Parametric Searching. DCG 10, 183–196.** [DOI](https://doi.org/10.1007/BF02573973) · [Authors' institutional publication record](https://collaborate.princeton.edu/en/publications/diameter-width-closest-line-pair-and-parametric-searching-2/).
   Optional historical reference for exact computation: a deterministic \(O(n^{8/5+\delta})\) algorithm for three-dimensional width, for each fixed \(\delta>0\), is stated in the abstract. This is a historical bound, not a claim about the current fastest algorithm.

## Positioning recommendation (assessment)

These sources establish prior work for computing width and for minimax hyperplane fitting. I would position the thesis around the use of that established functional for AR sliding-window embeddings, the population order-identification result, and empirical consistency; see [the population proposition](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/3_application_to_time_series.tex:288) and [the consistency subsection](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/3_application_to_time_series.tex:315). This search does not establish novelty of those AR-specific results.

For interpreting computation, keep the globally minimized empirical width separate from a value returned by the projected subgradient method. The thesis's stated convergence result requires a suitably close initialization ([discussion at line 452](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/3_application_to_time_series.tex:452)); the cited geometry algorithms address global exact or approximation guarantees.

Suggested paragraph for later insertion, subject to a separate explicit thesis-edit request:

> Computing the minimum width of a finite point set is a classical problem in computational geometry, also formulated as finding a minimum-width enclosing slab. Exact algorithms in two and three dimensions were developed by Houle and Toussaint \cite{houle1988width}, while approximation methods in fixed dimension are discussed by Chan \cite[Section 3]{chan2002approximating} and Agarwal, Har-Peled and Varadarajan \cite{10.1145/1008731.1008736}. The connection with minimax hyperplane fitting is also established in this literature. Here, the minimum-width functional is used to study AR sliding-window point clouds, with emphasis on its population structure and the consistency of its empirical counterpart.

The accompanying [BibTeX supplement](/Users/iljasukhanov/MASTERARBEIT/output/research/minimum_width_literature.bib) contains all five papers. Retain the existing Agarwal entry rather than adding a duplicate to the thesis bibliography.
