# Choosing starting directions: eight strategies with published precedent

Compiled 17 October 2026 for the AR empirical minimum-width experiments.

Every general strategy below has published precedent. Entries 1, 2 and 8 directly concern initialization on a sphere. Entries 3–7 originate in more general optimization problems; their application to this AR objective requires the explicitly identified adaptations. No cited result establishes their relative performance or global recovery for the AR estimator.

Write the direction dimension as \(d\), and let \(N(v)=v/\|v\|_2\) for nonzero \(v\). Normalization here enforces the unit-length constraint used by the [project optimizer](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/ar_tools/descent.py:68).

| Strategy | Published rule and exact reading location | Application to AR directions |
|---|---|---|
| **1. Normalized Gaussian starts** | Li et al. generate a standard Gaussian initial vector in their nonsmooth manifold optimization experiments: **§7.1 and §7.2, “Experiments.”** [Paper, version v4](https://arxiv.org/html/1911.05047v4). | Draw \(z\sim\mathcal N(0,I_d)\), set \(u_0=N(z)\), and repeat independently for multiple starts. This is also the [current default construction](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/ar_tools/descent.py:58). Their objective differs from empirical width. |
| **2. Normalized rows of a data-derived basis matrix** | Qu, Sun and Wright use each normalized row of an orthonormal basis matrix \(Y\) as a separate initial direction: **§3, “Initialization,” PDF p. 4**. [Paper](https://papers.nips.cc/paper_files/paper/2014/file/419e33cfb3aebab7d7d4ed588f3e61c1-Paper.pdf#page=4). | Their exact rule is \(q_{0,i}=N(y_i)\), where \(y_i\) is a transposed nonzero row of \(Y\). Their coordinates parameterize a sparse-vector subspace problem. Defining an analogous matrix and mapping its directions to the AR variables is additional work; raw sliding-window rows cannot simply be identified with their basis rows. |
| **3. Farthest-first selection of diverse candidates** | Ugray et al. construct a reference set by repeatedly adding the candidate farthest from its nearest selected neighbor: **§3.1.2, author manuscript PDF p. 6**. [Author manuscript](https://www.uv.es/~rmarti/paper/docs/global2.pdf#page=6); [published article](https://pubsonline.informs.org/doi/10.1287/ijoc.1060.0175). | Generate a larger pool of normalized candidates and select the desired number using that rule. Restricting the pool to the sphere and using the sign-invariant distance below are adaptations. This rule selects coverage rather than promising initial objective values. |
| **4. Rank candidates and filter nearby redundant starts (MLSL)** | The multilevel single-linkage approach samples candidates, retains a fraction with better objective values, and excludes candidates near better points or previously found local minima. Accessible account: Ugray et al., **§2, author manuscript PDF p. 3**. [Account](https://www.uv.es/~rmarti/paper/docs/global2.pdf#page=3). Historical source: [Rinnooy Kan and Timmer, Part II (1987)](https://link.springer.com/article/10.1007/BF02592071). | Evaluate empirical width on a sphere candidate pool, then combine objective ranking with proximity filtering. A fixed sphere-specific screening rule would be an MLSL-inspired adaptation; it should not be presented as the original algorithm with its guarantees. |
| **5. Low-discrepancy starts, such as Sobol sequences** | Kucherenko and Sytsko compare random, Sobol and Halton sampling in multistart and MLSL: **§3–§4**, with algorithm variants specified at the start of **§4, pp. 305–306**. [Published record](https://link.springer.com/article/10.1007/s10589-005-4615-1); [author-uploaded full text](https://www.researchgate.net/publication/225961691_Application_of_Deterministic_Low-Discrepancy_Sequences_in_Global_Optimization). | The paper samples bounded Euclidean domains. A rule mapping sequence points to the sphere is needed for directions. The cited comparison does not establish that a particular mapping preserves low discrepancy or improves this AR problem. |
| **6. Warm starts / continuation** | Friedman, Hastie and Tibshirani exploit previous solutions along a decreasing regularization path: **§2.5, “Pathwise coordinate descent.”** [Author manuscript](https://hastie.su.domains/Papers/glmnet.pdf); [published record](https://www.jstatsoft.org/article/view/v033i01). | Reuse the fitted direction from a related problem. For an increase in dimension, \(u_0=(\widehat u_{d-1},0)\) is the [existing project adaptation](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/ar_tools/descent.py:183). The glmnet paper supports warm starts across regularization values, rather than this particular zero-padding construction. |
| **7. Perturb a previous solution and optimize again** | Wales and Doye perform random displacements followed by local minimization in basin-hopping: **§II, “Method,” especially p. 5114 / PDF p. 4**. [Author PDF](https://www-wales.ch.cam.ac.uk/pdf/JPCA.101.5111.1997.pdf#page=4); [published record](https://pubs.acs.org/doi/10.1021/jp970984n). | A spherical restart could use \(u_{0,\mathrm{new}}=N(\widehat u+\delta)\). Normalization and the perturbation scale are adaptations. Repeated perturbed restarts alone do not implement the full basin-hopping method, which also has an acceptance mechanism. |
| **8. Smallest-eigenvalue spectral initialization** | Tsakiris and Vidal explicitly initialize with a smallest-singular-value direction: **§6.1 and Algorithm 1, line 5; §7.1 and Figure 5**. [Paper](https://www.jmlr.org/papers/volume19/17-436/17-436.pdf#page=28). Zhu et al.'s projected subgradient algorithm sets the initial unit vector to minimize the Euclidean norm of its data projections: **Algorithm 1, initialization**. [Paper](https://papers.nips.cc/paper_files/paper/2018/file/af21d0c97db2e27e13572cbf59eb343d-Paper.pdf). | For data columns in \(Y\in\mathbb R^{d\times n}\), choose a unit eigenvector of \(YY^\top\) for its smallest eigenvalue. This supplies an initialization for a different objective; minimizing empirical range still requires the width optimizer. |

The section and page descriptions above were verified from accessible primary material or indexed author-manuscript text. Full access to the Rinnooy Kan and Timmer paper was unavailable; the concrete MLSL rule above is verified in Ugray et al.'s account. The Friedman author-manuscript text for §2.5 was available through search indexing, although direct PDF retrieval failed. Zhu et al.'s Algorithm 1 was available in indexed primary-paper text; direct PDF retrieval failed.

## Spectral initialization: matrix orientation and objective

The following identities explain entry 8 algebraically. For columns \(y_i\) in \(Y\in\mathbb R^{d\times n}\),

\[
\|Y^\top u\|_2^2=u^\top YY^\top u=\sum_{i=1}^n(y_i^\top u)^2.
\]

Writing the symmetric matrix \(YY^\top\) in its orthonormal eigenbasis shows that the unit-vector objective is a weighted average of its eigenvalues, with weights equal to squared coordinates. Its minimum is therefore attained by a unit eigenvector for the smallest eigenvalue. This establishes the equivalence between the squared-projection initializer and the spectral rule. For centered data, dividing the objective by the sample size gives the empirical variance of the projections; uncentered data give the second moment instead. These are direct algebraic consequences, not AR recovery results.

If observations are stored as **rows** in \(X\in\mathbb R^{n\times d}\), as in the [project's sliding-window clouds](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/ar_tools/functions.py:194), use \(X^\top X\). Its eigenvectors have the required length \(d\).

The symbol \(Y\) has a different role in Qu et al. (2014): its columns are orthonormal basis vectors, so \(Y^\top Y=I\); see **§2** of their [paper](https://papers.nips.cc/paper_files/paper/2014/file/419e33cfb3aebab7d7d4ed588f3e61c1-Paper.pdf#page=2). Consequently, \(YY^\top\) is a projector, with eigenvalues zero and one when the subspace is proper. A zero-eigenvalue vector lies in the orthogonal complement of that subspace and has the ambient dimension, rather than the dimension of their optimization variable \(q\). This is a direct consequence of their basis construction and differs from taking a smallest-eigenvalue direction of an ordinary data scatter matrix.

## Sign equivalence for this particular objective

Our direct derivation from the [implemented width formula](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/ar_tools/descent.py:83) gives

\[
\widehat w(-u)=\widehat w(u).
\]

Consequently, when adapting methods 3 and 4, one can regard opposite unit directions as the same orientation and use

\[
d_{\pm}(u,v)=\min\{\|u-v\|_2,\|u+v\|_2\}
             =\sqrt{2-2|u^\top v|}.
\]

The equality is an elementary expansion for unit vectors. This sign-aware distance is our sphere-specific adaptation here, not a formula attributed to the optimization papers above.

## Scope of the list

Multistart is the outer procedure of launching several local optimizations; a starting-point rule specifies which points those optimizations use. Ugray et al., §2, provides precedent for the outer procedure. The eight entries describe sampling, selection, or reuse rules for those points.

The earlier suggestion to obtain an AR coefficient estimate from an L-infinity fit and map it to a unit direction remains an untested project proposal. I have not verified published precedent for that exact initialization, so it is excluded from this catalogue.

My proposed first comparison is normalized Gaussian starts versus farthest-first starts and Sobol-derived starts, all at the same number of local descents and with preparation time included. Warm starts are useful to compare when fitting a sequence of embedding dimensions. This is an experimental priority based on implementation effort, not a performance claim from the cited papers.

## Citation keys

- Normalized Gaussian starts: existing `li2021weaklyconvexoptimizationstiefel` in the [thesis bibliography](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/bibliography.bib:230).
- Basis rows: `qu2014sparsevector`.
- Farthest-first and accessible MLSL account: `ugray2007multistart`.
- Historical MLSL: `rinnooykan1987multilevel`.
- Low-discrepancy starts: `kucherenko2005lowdiscrepancy`.
- Warm starts: `friedman2010glmnet`.
- Perturbed restarts: `wales1997basinhopping`.
- Spectral initialization: `tsakiris2018dpcp`, `zhu2018dpcp`.

The additional keys are in [additional_references.bib](/Users/iljasukhanov/MASTERARBEIT/output/analysis/ar_initialization_literature/additional_references.bib). Thesis sources and optimizer code were not edited for this catalogue.
