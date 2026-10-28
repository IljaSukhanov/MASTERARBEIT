# Minimum-width and AR-estimation precedents

Compiled on 25 October 2026. This consolidates the relevant literature identified in the project and this conversation: **28 principal references and two additional bibliographic leads**. It is not an exhaustive survey of every minimum-zone or geometric-fitting paper. No protected thesis source or bibliography was edited.

The two papers requested explicitly are **[1] Malyscheff et al. (2002)** and **[2] Gosavi and Phatakwala (2006)**. Entries distinguish direct numerical precedents, established geometric algorithms, general optimization theory, adjacent statistical work, and possible future-method references. These are different kinds of precedent; their inclusion does not mean that they contain the complete AR estimator proposed in the thesis.

## Scope and connection to the thesis

The target is one slab containing **all** input points:

\[
w^*(P)=\min_{\|u\|_2=1}\left[\max_{x\in P}\langle x,u\rangle-\min_{x\in P}\langle x,u\rangle\right].
\]

See [Chan, Section 3](https://tmc.web.engr.illinois.edu/apx.pdf) and [the empirical objective in the thesis](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/3_application_to_time_series.tex:317). Minimum-zone straightness in the plane and flatness in three dimensions are the same geometric enclosure criterion when perpendicular distances are used; vertical-residual fitting, circularity, cylinders, and multiple slabs require separate comparisons. See [Malyscheff et al., Section 4, equations (4)-(6), pp. 63-64](https://doi.org/10.1016/S0360-8352(02)00003-7).

The AR application additionally uses the minimizing direction to recover coefficients by coordinate ratios; see [the coefficient-recovery remark](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/3_application_to_time_series.tex:313). **No exact precedent for the full combination of sphere-based projected subgradient width minimization and this AR coefficient estimator was verified in this review. This is a search finding, not a proof of priority.**

## Reading priorities

| Purpose | Start with |
|---|---|
| Prior gradient-based numerical computation of width | [1], [2] |
| Earlier numerical minimum-zone methods | [3]-[6], [9] |
| Related neural/interval and support-vector approaches | [7], [11] |
| Geometry of the objective and alternative formulations | [8], [10], [13], [17] |
| Guaranteed geometric approximation and dimension dependence | [17]-[22] |
| General convergence framework for the sphere method | [23] |
| Nearby AR and semiparametric estimation literature | [24], [25] |
| Adaptive optimization proposed for future work | [26], [27] |
| Enclosure with a specified number of points | [28] |

The relevance assessments below are editorial judgments. The evidence notes describe the depth of source inspection; a paper checked through its abstract is not treated as if its complete algorithm or proof had been audited.

## A. Direct and related numerical minimum-zone literature

### 1. Malyscheff, A. M., Trafalis, T. B., and Raman, S. (2002)

**From support vector machine learning to the determination of the minimum enclosing zone.** *Computers & Industrial Engineering*, **42**(1), 59-74. DOI: [10.1016/S0360-8352(02)00003-7](https://doi.org/10.1016/S0360-8352(02)00003-7). BibTeX: `malyscheff2002minimumzone`.

- **Relevance:** direct gradient-based precedent for the same slab-width objective in straightness/flatness. It maximizes the squared normal length subject to linear enclosure constraints, then repeatedly solves a linearized LP. It is not the sphere-based projected subgradient iteration used in the thesis.
- **Read:** Section 4, equations (4)-(6), pp. 63-64; Section 6 and Table 2, pp. 66-69; Section 7, p. 73.
- **Evidence:** complete user-supplied PDF read, with visual inspection of the decisive equations and algorithm. User-supplied source filename: `1-s2.0-S0360835202000037-main.pdf`.
- **Qualification:** its zero-step termination criterion is not a global-optimality certificate; an independent counterexample is recorded below. Its reported comparison uses three straightness datasets. It supplies no AR-estimation or statistical-consistency result.

### 2. Gosavi, A., and Phatakwala, S. (2006)

**A Finite-Differences Derivative-Descent Approach for Estimating Form Error in Precision-Manufactured Parts.** *Journal of Manufacturing Science and Engineering*, **128**(1), 355-359. DOI: [10.1115/1.2124989](https://doi.org/10.1115/1.2124989). [Author PDF](https://web.mst.edu/_disabled/gosavia/paper_asme.pdf). BibTeX: `gosavi2006finitedifferences`.

- **Relevance:** direct numerical descent precedent. It uses finite-difference approximations of steepest descent and includes straightness and flatness among its applications. Equation (4) defines the span of deviations; for signed normal deviations, this is the directional-width criterion.
- **Read:** Section 2.1 and equation (4), pp. 355-356; algorithm in Section 3, p. 357; numerical comparisons in Section 4, pp. 357-358.
- **Evidence:** original paper text and relevant formulas inspected.
- **Qualification:** finite differences differ from the thesis's analytic extremal-pair subgradient and tangent projection. The paper explicitly discusses the absence of a general guarantee of reaching the global optimum in Section 3. Its stronger empirical performance relative to Nelder-Mead concerns sphericity/cylindricity, not a universal superiority for flatness.

### 3. Murthy, T. S. R., and Abdin, S. Z. (1980)

**Minimum zone evaluation of surfaces.** *International Journal of Machine Tool Design and Research*, **20**(2), 123-136. DOI: [10.1016/0020-7357(80)90024-4](https://doi.org/10.1016/0020-7357(80)90024-4). [Publisher abstract](https://www.sciencedirect.com/science/article/pii/0020735780900244). BibTeX: `murthy1980minimumzone`.

- **Relevance:** early numerical-search precedent, including flatness; the abstract identifies Monte Carlo, normal least-squares, simplex, and spiral-search techniques.
- **Evidence:** publisher abstract. Full algorithm not audited. Cite it for numerical minimum-zone search, not specifically projected gradient descent.

### 4. Kanada, T., and Suzuki, S. (1993)

**Evaluation of minimum zone flatness by means of nonlinear optimization techniques and its verification.** *Precision Engineering*, **15**(2), 93-99. DOI: [10.1016/0141-6359(93)90343-9](https://doi.org/10.1016/0141-6359(93)90343-9). [Publisher abstract](https://www.sciencedirect.com/science/article/abs/pii/0141635993903439). BibTeX: `kanada1993minimumzoneflatness`.

- **Relevance:** direct numerical flatness minimization using downhill simplex and repetitive bracketing, with convergence criteria and comparison to least squares.
- **Evidence:** publisher abstract. These are not evidence of the particular projected subgradient algorithm.
- **Citation correction:** do not confuse this paper with their separate 1993 paper on computing techniques for straightness; the flatness article is pp. 93-99.

### 5. Carr, K., and Ferreira, P. (1995)

**Verification of form tolerances part I: Basic issues, flatness, and straightness.** *Precision Engineering*, **17**(2), 131-143. DOI: [10.1016/0141-6359(94)00017-T](https://doi.org/10.1016/0141-6359(94)00017-T). [Publisher record](https://www.sciencedirect.com/science/article/pii/014163599400017T). BibTeX: `carr1995verification`.

- **Relevance:** foundational optimization formulation of minimum-zone flatness and straightness; an important comparison for the sequential LP formulation in [1].
- **Evidence:** publisher abstract verifies the geometric minimax criterion; the relationship of the formulations is displayed in [1], equations (5)-(6). The original complete algorithm was not audited here.
- **Citation correction:** the publisher gives **1995**, not the 1996 year printed in [1]'s bibliography.

### 6. Cheraghi, S. H., Lim, H. S., and Motavalli, S. (1996)

**Straightness and flatness tolerance evaluation: an optimization approach.** *Precision Engineering*, **18**(1), 30-37. DOI: [10.1016/0141-6359(95)00033-X](https://doi.org/10.1016/0141-6359(95)00033-X). [Publisher abstract](https://www.sciencedirect.com/science/article/pii/014163599500033X). BibTeX: `cheraghi1996straightnessflatness`.

- **Relevance:** nonlinear geometric optimization reduced through a linear search to LP subproblems. Useful as an alternative numerical formulation and benchmark reference.
- **Evidence:** publisher abstract. It claims exact evaluation; that claim and its assumptions have not been independently audited here. It should not be described as projected subgradient descent.

### 7. Suen, D. S., and Chang, C. N. (1997)

**Application of neural network interval regression method for minimum zone straightness and flatness.** *Precision Engineering*, **20**(3), 196-207. DOI: [10.1016/S0141-6359(97)00028-7](https://doi.org/10.1016/S0141-6359(97)00028-7). [Authors' institutional record and abstract](https://ir.lib.ncu.edu.tw/handle/987654321/26988?locale=en-US). BibTeX: `suen1997intervalregression`.

- **Relevance:** related iterative learning approach using an interval-bias adaptive linear neural network and an LMS learning algorithm for straightness/flatness.
- **Evidence:** institutional abstract; full objective and update not audited. This is a useful lead when assessing broader learning/gradient-related precedent, but not a verified match to the thesis's sphere algorithm.

### 8. Kaiser, M. J., and Krishnan, K. K. (1998)

**Geometry of the minimum zone flatness functional: planar and spatial case.** *Precision Engineering*, **22**(3), 174-183. DOI: [10.1016/S0141-6359(98)00005-1](https://doi.org/10.1016/S0141-6359(98)00005-1). [Publisher abstract and introduction](https://www.sciencedirect.com/science/article/abs/pii/S0141635998000051). BibTeX: `kaiser1998geometry`.

- **Relevance:** geometric study and visualization of straightness/flatness functionals and their dependence on direction. Useful background for understanding and illustrating the optimization landscape.
- **Evidence:** publisher abstract, introduction, notation, and section previews. Not a verified precedent for projected subgradient descent or AR estimation.

### 9. Orady, E., Li, S., and Chen, Y. (2000)

**Evaluation of Minimum Zone Straightness by a Nonlinear Optimization Method.** *Journal of Manufacturing Science and Engineering*, **122**(4). DOI: [10.1115/1.1285880](https://doi.org/10.1115/1.1285880). [Original-paper text indexed from its PDF](https://citeseerx.ist.psu.edu/document?doi=2788a1598e969dc803feb32bbc16958d07b19421&repid=rep1&type=pdf). BibTeX: `orady2000straightness`.

- **Relevance:** nonlinear numerical optimization for minimum-zone straightness, compared to least-squares and convex-hull methods.
- **Evidence:** original abstract and introductory page; full algorithm not audited. The publisher-page range was not independently confirmed in this compilation and is deliberately omitted from the BibTeX entry.

### 10. Zhu, X., and Ding, H. (2002)

**Flatness tolerance evaluation: an approximate minimum zone solution.** *Computer-Aided Design*, **34**(9), 655-664. DOI: [10.1016/S0010-4485(01)00137-3](https://doi.org/10.1016/S0010-4485(01)00137-3). [Publisher abstract and section previews](https://www.sciencedirect.com/science/article/abs/pii/S0010448501001373). BibTeX: `zhu2002flatness`.

- **Relevance:** explicitly connects point-set width to the origin-centered inradius of the convex hull of pairwise differences, then develops exact and approximate geometric/LP approaches. Particularly relevant to the support-function formulation and possible numerical baselines.
- **Read:** definition of width, Theorem 1, and the exact/approximate algorithm sections.
- **Evidence:** publisher abstract, introduction, and theorem/algorithm previews. The paper's phrase "almost exact" is not treated here as a dimension-independent global approximation guarantee.

### 11. Prakasvudhisarn, C., Trafalis, T. B., and Raman, S. (2003)

**Support Vector Regression for Determination of Minimum Zone.** *Journal of Manufacturing Science and Engineering*, **125**(4), 736-739. DOI: [10.1115/1.1596572](https://doi.org/10.1115/1.1596572). [Paper abstract](https://www.researchgate.net/publication/238184734_Support_Vector_Regression_for_Determination_of_Minimum_Zone). BibTeX: `prakasvudhisarn2003minimumzone`.

- **Relevance:** subsequent support-vector-regression approach to straightness and flatness, with numerical comparisons to published data.
- **Evidence:** original paper's reproduced abstract and author publication record; full formulation not audited. Do not assume that its SVR loss, normalization, or optimization procedure is identical to [1] or the thesis without reading the full text.

## B. Established exact and approximation algorithms for geometric width

### 12. Houle, M. E., and Toussaint, G. T. (1985)

**Computing the width of a set.** *Proceedings of the First Annual Symposium on Computational Geometry*, 1-7. DOI: [10.1145/323233.323234](https://doi.org/10.1145/323233.323234). BibTeX: `houle1985width`.

- **Relevance:** conference predecessor of [13], establishing exact planar/spatial width algorithms.
- **Evidence:** ACM abstract and bibliographic record. Treat [12] and [13] as versions of the same line of work, not independent discoveries.

### 13. Houle, M. E., and Toussaint, G. T. (1988)

**Computing the Width of a Set.** *IEEE Transactions on Pattern Analysis and Machine Intelligence*, **10**(5), 761-765. DOI: [10.1109/34.6790](https://doi.org/10.1109/34.6790). [Author manuscript](https://cgm.cs.mcgill.ca/~godfried/publications/width.pdf). BibTeX: `houle1988width`.

- **Relevance:** foundational exact algorithms and an explicit connection to minimax line/plane fitting; establishes that the geometric functional and fitting interpretation predate the thesis.
- **Read:** Sections 2.B-2.C and 3.C, especially Theorems 2.2 and 3.3 in the author manuscript.
- **Evidence:** author manuscript and author publication record, with theorem locators also recorded in the earlier project review. This is a geometric algorithm, not projected descent.

### 14. Chazelle, B., Edelsbrunner, H., Guibas, L. J., and Sharir, M. (1993)

**Diameter, Width, Closest Line Pair, and Parametric Searching.** *Discrete & Computational Geometry*, **10**, 183-196. DOI: [10.1007/BF02573973](https://doi.org/10.1007/BF02573973). BibTeX: `chazelle1993diameterwidth`.

- **Relevance:** deterministic exact geometric width computation in three dimensions using parametric searching.
- **Evidence:** publisher abstract. Useful historical exact-computation reference; no claim is made that its bound is currently optimal.

### 15. Agarwal, P. K., and Sharir, M. (1996)

**Efficient Randomized Algorithms for Some Geometric Optimization Problems.** *Discrete & Computational Geometry*, **16**(4), 317-337. DOI: [10.1007/BF02712871](https://doi.org/10.1007/BF02712871). BibTeX: `agarwalsharir1996randomized`.

- **Relevance:** randomized exact width computation in three dimensions; the abstract states expected time \(O(n^{3/2+\varepsilon})\) for any fixed \(\varepsilon>0\).
- **Evidence:** publisher abstract. The expectation refers to running time, not an approximate geometric answer.

### 16. Duncan, C. A., Goodrich, M. T., and Ramos, E. A. (1997)

**Efficient approximation and optimization algorithms for computational metrology.** *Proceedings of the Eighth Annual ACM-SIAM Symposium on Discrete Algorithms*, 121-130. DOI: [10.5555/314161.314196](https://doi.org/10.5555/314161.314196). BibTeX: `duncan1997metrology`.

- **Relevance:** an important earlier source for width approximation. Chan [17], Section 3, explicitly credits and reinterprets its approximation method.
- **Evidence:** this attribution was checked in [Chan's original paper](https://tmc.web.engr.illinois.edu/apx.pdf); the original DGR algorithm was not independently audited in full. The DOI record and proceedings citation identify the original source.

### 17. Chan, T. M. (2002)

**Approximating the Diameter, Width, Smallest Enclosing Cylinder, and Minimum-Width Annulus.** *International Journal of Computational Geometry & Applications*, **12**(1-2), 67-85. DOI: [10.1142/S0218195902000748](https://doi.org/10.1142/S0218195902000748). [Author manuscript](https://tmc.web.engr.illinois.edu/apx.pdf). BibTeX: `chan2002approximating`.

- **Relevance:** central reference for the normalized slab formulation, exact computation in fixed higher dimension, and guaranteed width approximation using direction covers and LPs. It demonstrates that approximating width itself is established.
- **Read:** Section 1 for dimension qualifications and Section 3, "Width," for formulation and algorithms.
- **Evidence:** relevant original manuscript sections inspected. The linked manuscript is dated 2000; the journal publication is 2002. Its fixed-dimension bounds are not dimension-independent guarantees.

### 18. Agarwal, P. K., Har-Peled, S., and Varadarajan, K. R. (2004)

**Approximating Extent Measures of Points.** *Journal of the ACM*, **51**(4), 606-635. DOI: [10.1145/1008731.1008736](https://doi.org/10.1145/1008731.1008736). [Author's project page and abstract](https://valis.cs.illinois.edu/research/papers/01/fitting/). BibTeX: `agarwal2004extent`.

- **Relevance:** small subsets preserving directional extent measures, including width, in arbitrary fixed dimension. Useful for comparing descent to methods that first reduce the point set.
- **Read:** Sections 2-3 and 6.1 of the [author manuscript](https://graphics.stanford.edu/courses/cs468-06-winter/Papers/har-peled-JACM2004%20fitting.pdf), especially Theorems 3.7 and 6.1. The earlier review records the existing thesis bibliography key as `10.1145/1008731.1008736`.
- **Evidence:** author's abstract/project page rechecked; the complete proof was not re-audited for this compilation.
- **Qualification:** kernel construction and globally optimizing width over the kernel are different computational tasks.

### 19. Varadarajan, K. R., Venkatesh, S., Ye, Y., and Zhang, J. (2007)

**Approximating the Radii of Point Sets.** *SIAM Journal on Computing*, **36**(6), 1764-1776. DOI: [10.1137/050627472](https://doi.org/10.1137/050627472). [Author-hosted published paper](https://webhome.cs.uvic.ca/~srinivas/papers/radii.pdf). BibTeX: `varadarajan2007radii`.

- **Relevance:** outer-radius approximation when dimension is part of the input, including hyperplane fitting. Important for evaluating claims about overcoming dimensionality.
- **Read:** introductory definition, abstract, and Theorem 1.
- **Evidence:** original abstract, definition, and theorem context inspected.
- **Qualification:** the hyperplane fitting radius is half the full enclosing-slab width used in the thesis; this factor follows from centering the hyperplane between the slab boundaries.

### 20. Arya, S., da Fonseca, G. D., and Mount, D. M. (2017)

**Near-Optimal epsilon-Kernel Construction and Related Problems.** *SoCG 2017*, LIPIcs **77**, 10:1-10:15. DOI: [10.4230/LIPIcs.SoCG.2017.10](https://doi.org/10.4230/LIPIcs.SoCG.2017.10). BibTeX: `arya2017nearkernel`.

- **Relevance:** improved construction of small subsets preserving directional width; a possible preprocessing comparison or complement to numerical optimization.
- **Read:** Section 1.1 and Theorem 1.1.
- **Evidence:** official proceedings text and theorem statement. Fixed dimension is assumed. This is not itself a claim that the minimum-width optimization step is free.

### 21. Arya, S., da Fonseca, G. D., and Mount, D. M. (2018)

**Approximate Convex Intersection Detection with Applications to Width and Minkowski Sums.** *ESA 2018*, LIPIcs **112**, 3:1-3:14. DOI: [10.4230/LIPIcs.ESA.2018.3](https://doi.org/10.4230/LIPIcs.ESA.2018.3). [Official paper](https://drops.dagstuhl.de/storage/00lipics/lipics-vol112-esa2018/LIPIcs.ESA.2018.3/LIPIcs.ESA.2018.3.pdf). BibTeX: `arya2018widthminkowski`.

- **Relevance:** end-to-end guaranteed width approximation, as distinct from just constructing a kernel. A strong geometric approximation comparator for the discussion of scalability.
- **Read:** Section 1.3, Theorem 3, and the width argument near p. 3:12.
- **Evidence:** official paper and theorem context. Its bounds assume fixed dimension and retain dimension-dependent accuracy costs.

### 22. Gritzmann, P., and Klee, V. (1993)

**Computational Complexity of Inner and Outer j-Radii of Polytopes in Finite-Dimensional Normed Spaces.** *Mathematical Programming*, **59**, 163-213. DOI: [10.1007/BF01581243](https://doi.org/10.1007/BF01581243). BibTeX: `gritzmannklee1993radii`.

- **Relevance:** computational-complexity background for radii, minimax flat fitting, and dimension dependence.
- **Evidence:** publisher abstract. It explicitly distinguishes vertex and halfspace input representations. Do not transfer a hardness statement from one representation, radius, or norm to the thesis problem without checking the applicable theorem.

## C. General optimization framework used by the thesis

### 23. Li, X., Chen, S., Deng, Z., Qu, Q., Zhu, Z., and So, A. M.-C. (2021)

**Weakly Convex Optimization over Stiefel Manifold Using Riemannian Subgradient-Type Methods.** *SIAM Journal on Optimization*, **31**(3), 1605-1634. DOI: [10.1137/20M1321000](https://doi.org/10.1137/20M1321000). [Author preprint](https://arxiv.org/pdf/1911.05047). BibTeX: `li2021weaklyconvexoptimizationstiefel`.

- **Relevance:** the general algorithmic and local-convergence framework underlying the thesis's projected subgradient approach; it is not evidence that the generic method itself is new.
- **Read:** Section 2.1, equation (2.2), for tangent projection; Section 2.2 for algorithms; Section 5, Definition 1 and Theorem 4, for sharpness, initialization, and geometric step-size conditions.
- **Evidence:** relevant original definitions, algorithm discussion, and theorem inspected. Theorem 4 is stated for the incremental method; the single-component specialization connects to the full subgradient method.
- **Qualification:** conditional local convergence does not imply reaching a global minimizer from arbitrary initialization. This source does not establish prior use of the complete AR-width estimator.

## D. Nearby AR/statistical estimation literature

### 24. Horvath, M. (1989)

**AR models with uniformly distributed noise.** *Aplikace matematiky*, **34**(5), 396-401. DOI: [10.21136/AM.1989.104367](https://doi.org/10.21136/AM.1989.104367). [Original journal archive](https://dml.cz/handle/10338.dmlcz/104367). BibTeX: `horvath_ar_1989`.

- **Relevance:** established AR parameter estimation under uniform innovations; an important statistical comparator, distinct from the geometric slab algorithms above.
- **Evidence:** original journal-archive metadata and abstract. The abstract supports maximum-likelihood estimation and an iterative computation method; the specific LP derivation has not been re-audited from the complete paper in this compilation.
- **Qualification:** it assumes a specified uniform innovation law; its presence does not establish precedent for the proposed estimator with an unspecified innovation density. The author's given name is **Michal**, not Lajos.

### 25. Schick, A., and Wefelmeyer, W. (2002)

**Efficient estimation in invertible linear processes.** *Mathematical Methods of Statistics*, **11**, 358-379. [Author paper](https://www.mi.uni-koeln.de/~wefelm/files/m-linear13.pdf); [author publication list](https://people.math.binghamton.edu/anton/publication.html). BibTeX: `schick2002invertible`.

- **Relevance:** finite-dimensional linear-process parameters with an unknown innovation distribution, explicitly including AR(p) models. Useful for positioning the semiparametric aspect separately from the minimum-width construction.
- **Read:** abstract and Section 1.
- **Evidence:** original abstract and introduction, plus author's bibliographic record. No DOI is supplied because one was not verified. The author publication list dates the journal volume to 2002; some records note physical publication in 2003.

## E. Sources for proposed future adaptive optimization

These support the proposed extension, not prior application of these algorithms to minimum width or AR estimation.

### 26. Kingma, D. P., and Ba, J. (2015)

**Adam: A Method for Stochastic Optimization.** *ICLR 2015*. [Original paper](https://arxiv.org/abs/1412.6980). BibTeX: `kingma2015adam`.

- **Relevance:** original reference if Adam is explicitly proposed as a future numerical method.
- **Evidence:** original arXiv record and abstract; initial preprint appeared in 2014.
- **Qualification:** cite it for Adam, not for guarantees on a nonsmooth objective constrained to a sphere.

### 27. Becigneul, G., and Ganea, O.-E. (2019)

**Riemannian Adaptive Optimization Methods.** *ICLR 2019*. [Original paper](https://arxiv.org/abs/1810.00760). BibTeX: `becigneul2019riemannian`.

- **Relevance:** manifold-compatible adaptive methods, including generalizations of Adam, Adagrad, and AMSGrad. Relevant to designing an extension that respects the sphere constraint.
- **Evidence:** original abstract and publication record.
- **Qualification:** the stated theory concerns geodesically convex objectives on products of manifolds; it does not automatically establish guarantees for the nonsmooth minimum-width problem.

## F. Related enclosure variant

### 28. Ahn, T., and Bae, S. W. (2025)

**On k-enclosing slab problems.** *Theoretical Computer Science*, **1058**, 115583. DOI: [10.1016/j.tcs.2025.115583](https://doi.org/10.1016/j.tcs.2025.115583). [Publisher abstract and introduction](https://www.sciencedirect.com/science/article/abs/pii/S0304397525005213). BibTeX: `ahnbae2025kenclosing`.

- **Relevance:** enclosure of a specified number of points; its introduction identifies the all-points minimum-width case as the classical width problem and reports established exact bounds.
- **Evidence:** original abstract and introduction.
- **Qualification:** this is a broader/modified enclosure problem, not evidence of sphere-based projected subgradient descent. Allowing points to be excluded would change the thesis estimator and would require a new statistical analysis.

## Additional bibliographic leads

These are retained for completeness but are not relied on for detailed algorithmic claims.

### L1. Malyscheff, A. M., Trafalis, T. B., and Raman, S. (2000)

**Determination of the Minimum Enclosing Zone using a Sequential Gradient Ascent Approach.** *Proceedings of the 2000 Japan-USA Symposium on Flexible Automation*, Ann Arbor, July 2000. BibTeX: `malyscheff2000sequential`.

The conference publication is listed in [Trafalis's institutional CV, p. 20, item 36](https://www.ou.edu/content/dam/CoE/IE/ISEFaculty/Trafalis/TTrafalis_resume%20revision%202016.pdf). Full text and page range were not located. It may be an earlier version of [1]; that relationship has not been established by comparing both texts.

### L2. Damodarasamy, S., and Anand, S. (1999)

**Evaluation of minimum zone for flatness by normal plane method and simplex search.** *IIE Transactions*, **31**(7), 617-626. DOI: [10.1080/07408179908969863](https://doi.org/10.1080/07408179908969863). BibTeX: `damodarasamy1999flatness`.

The metadata are listed in [Anand's institutional publication record](https://researchdirectory.uc.edu/p/anands) and the paper is cited as reference [20] by [2]. The publisher text could not be inspected here, so no detailed claim about guarantees is made. Useful as a direct-search lead.

## Independent check concerning reference [1]

The following is an original derivation from [1]'s constraints and Table 2, not a result stated in that paper. It was checked with exact rational arithmetic during this conversation.

For \(P=\{(2,1),(2,5),(4,1),(4,5)\}\), its feasible region is
\[
C=\{(w,b):|w_1|+2|w_2|+|3w_1+3w_2+b|\le1\}.
\]
Starting at \((w,b)=((1/10,2/5),-3/2)\), the linearized LP uniquely selects \(((0,1/2),-3/2)\). The next LP selects the same point, triggering the zero-step stopping rule. This returns slab width 4, although \(((1,0),-3)\) is feasible and returns width 2. The latter is globally minimal because \(w_P(u)=2|u_1|+4|u_2|\ge2\) for every unit \(u\). All data coordinates are positive. Thus neither the stopping rule nor positive-coordinate preprocessing supplies a general global-optimality guarantee.

## How to use this bibliography in the thesis

- Cite [1] and [2] when acknowledging prior gradient-based minimum-zone optimization; preserve the difference between sequential LP ascent, finite-difference steepest descent, and the thesis's projected subgradient update.
- Cite [13], [17], [18], and [21] for the established geometric width problem and guaranteed approximation alternatives.
- Cite [23] for the general algorithmic/convergence framework. The width-specific verification of its assumptions is a separate mathematical contribution to assess.
- Cite [24] and [25] when positioning AR estimation and the unspecified innovation distribution. They address different statistical constructions from the proposed geometric estimator.
- Use [26]-[27] only to motivate future adaptive methods, and [28] only when discussing enclosure variants.

The accompanying [BibTeX collection](minimum_width_precedents_2026-10-25.bib) contains 30 distinct entries, matching the 28 main references and the two leads. Several sources already occur in the protected thesis bibliography under different keys. **Do not import the supplement wholesale without checking duplicate DOIs.** The compilation does not change or validate the thesis's mathematical proofs.

## Positioning recommendation and draft paragraph

Consolidated on 28 October 2026 from the research note checked on
18 October 2026. The assessment and draft below are retained verbatim;
no thesis text or bibliography was changed.

These sources establish prior work for computing width and for minimax hyperplane fitting. I would position the thesis around the use of that established functional for AR sliding-window embeddings, the population order-identification result, and empirical consistency; see [the population proposition](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/3_application_to_time_series.tex:288) and [the consistency subsection](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/3_application_to_time_series.tex:315). This search does not establish novelty of those AR-specific results.

For interpreting computation, keep the globally minimized empirical width separate from a value returned by the projected subgradient method. The thesis's stated convergence result requires a suitably close initialization ([discussion at line 452](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/3_application_to_time_series.tex:452)); the cited geometry algorithms address global exact or approximation guarantees.

Suggested paragraph for later insertion, subject to a separate explicit thesis-edit request:

> Computing the minimum width of a finite point set is a classical problem in computational geometry, also formulated as finding a minimum-width enclosing slab. Exact algorithms in two and three dimensions were developed by Houle and Toussaint \cite{houle1988width}, while approximation methods in fixed dimension are discussed by Chan \cite[Section 3]{chan2002approximating} and Agarwal, Har-Peled and Varadarajan \cite{10.1145/1008731.1008736}. The connection with minimax hyperplane fitting is also established in this literature. Here, the minimum-width functional is used to study AR sliding-window point clouds, with emphasis on its population structure and the consistency of its empirical counterpart.

The accompanying [BibTeX supplement](/Users/iljasukhanov/MASTERARBEIT/output/research/minimum_width_literature.bib) contains all five papers. Retain the existing Agarwal entry rather than adding a duplicate to the thesis bibliography.

## Reading details retained from the earlier overview

For detailed running-time comparisons, use the separate
[algorithm survey](minimum_width_algorithms_2026-10-19.md), which retains the
distinction between fixed and variable dimension and between local descent
and global geometric optimization.

- [Houle and Toussaint's author manuscript](https://www-cgrl.cs.mcgill.ca/~godfried/publications/width.pdf), Sections 2.B-2.C and 3.C, Theorems 2.2 and 3.3: the earlier overview records planar width in \(O(n\log n)\), spatial width in \(O(n\log n+I)\), and worst-case quadratic antipodal-edge count \(I\). The 1985 conference predecessor is entry [12].
- [Chan's author manuscript](https://tmc.web.engr.illinois.edu/apx.pdf), Sections 1 and 3: the earlier overview records the normalized slab formulation, direction discretization with multiple LPs, and the fixed-\(k\ge4\) exact bound \(O(n^{\lceil k/2\rceil})\). The attribution to Duncan, Goodrich and Ramos is retained in entries [16]-[17].
- [Varadarajan et al.'s published paper](https://webhome.cs.uvic.ca/~srinivas/papers/radii.pdf), introduction and Theorem 1: the earlier overview records hardness when dimension is part of the input and a polynomial-time \(O(\sqrt{\log n})\) outer-radius approximation. The radius/full-width normalization qualification is retained in entry [19].
- [Chazelle et al.'s institutional publication record](https://collaborate.princeton.edu/en/publications/diameter-width-closest-line-pair-and-parametric-searching-2/): the earlier overview records the abstract's historical deterministic three-dimensional bound \(O(n^{8/5+\delta})\), for each fixed \(\delta>0\). This is not presented as a current fastest bound.

These details preserve the earlier source assessment; the Markdown
consolidation did not perform a new literature review.
