# Starting directions and multistart: literature for the AR minimum-width experiments

Research note, 17 October 2026. These sources provide methodological precedent and ideas to test. Their results do not establish global recovery for the AR empirical-width problem. The experiment proposals below are adaptations, not claims proved in these papers.

For a method-by-method list with published precedent and explicit adaptation notes, see the [starting-direction catalogue](/Users/iljasukhanov/MASTERARBEIT/output/analysis/ar_initialization_literature/starting_direction_catalogue.md).

## References and precise reading locations

1. **Ugray, Z., Lasdon, L., Plummer, J., Glover, F., Kelly, J., and Martí, R. (2007). _Scatter Search and Local NLP Solvers: A Multistart Framework for Global Optimization_. INFORMS Journal on Computing, 19(3), 328–340.**

   Best direct citation for running a local optimizer from several starts. In the [author manuscript](https://www.uv.es/~rmarti/paper/docs/global2.pdf), **§2, PDF p. 3**, reviews basic random multistart and filtering redundant searches. **§3.1.2, PDF p. 6**, selects diverse candidates by repeatedly adding the point farthest from its nearest selected neighbor. These locators refer to the manuscript; cite the [published 2007 article](https://pubsonline.informs.org/doi/10.1287/ijoc.1060.0175). Its optimization setting differs from the nonsmooth sphere problem here.

   BibTeX key: `ugray2007multistart`.

2. **Rinnooy Kan, A. H. G., and Timmer, G. T. (1987). _Stochastic global optimization methods part I: Clustering methods_, Mathematical Programming, 39, 27–56; and _part II: Multi level methods_, 39, 57–78.**

   Classical historical references for clustering and multilevel stochastic global search. Publisher abstracts and bibliographic information were verified for [Part I](https://link.springer.com/article/10.1007/BF02592070) and [Part II](https://link.springer.com/article/10.1007/BF02592071). Full texts were not accessible in this search, so no unverified theorem locators are supplied. For an accessible description of multi-level single linkage and its start-selection rules, use Ugray et al., §2, above.

   BibTeX keys: `rinnooykan1987clustering`, `rinnooykan1987multilevel`.

3. **Arthur, D., and Vassilvitskii, S. (2007). _k-means++: The Advantages of Careful Seeding_. Proceedings of SODA, 1027–1035.**

   A direct machine-learning precedent: **§7, PDF p. 8**, explicitly describes repeated k-means runs followed by retaining the best clustering. **§2.2, PDF pp. 2–3**, gives distance-squared weighted initialization, favoring candidates far from existing centers. [Author PDF](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf); [publication record](https://research.google/pubs/k-means-the-advantages-of-careful-seeding/). Cluster-center seeding is different from choosing optimizer starts; transferring the diversity principle to directions is a proposed analogy. Theorem 3.1 concerns the k-means objective only.

   BibTeX key: `arthur2007kmeanspp`.

4. **Qu, Q., Sun, J., and Wright, J. (2014). _Finding a sparse vector in a subspace: Linear sparsity using alternating directions_. Advances in Neural Information Processing Systems, 27.**

   The closest geometric analogy in this selection. Their optimization constrains a direction to the unit sphere. **§3, “Initialization,” PDF p. 4**, launches optimization from every normalized row of an orthonormal basis matrix. **Theorem 4.1, PDF p. 5**, combines these starts with LP rounding and selection under a planted sparse model. This supports data-informed spherical initialization and multiple starts, but the recovery theorem is specific to their sparse-vector problem. [Official paper](https://papers.nips.cc/paper_files/paper/2014/file/419e33cfb3aebab7d7d4ed588f3e61c1-Paper.pdf).

   BibTeX key: `qu2014sparsevector`.

5. **Li, X., Chen, S., Deng, Z., Qu, Q., Zhu, Z., and So, A. M.-C. (2021 arXiv version). _Weakly Convex Optimization over Stiefel Manifold Using Riemannian Subgradient-Type Methods_.**

   **§7.1 and §7.2, experimental subsections**, use Gaussian random initialization. **§5.2, Theorem 4**, provides local linear convergence under sharpness, a sufficiently close initialization, and stepsize conditions. The closeness assumption matters; it is not a justification that twelve arbitrary starts suffice. [Verified version v4](https://arxiv.org/html/1911.05047v4).

   Already present as `li2021weaklyconvexoptimizationstiefel` in the [project bibliography](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/bibliography.bib:230). Omitted from the additional BibTeX file to avoid duplicating that key.

6. **Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., and Talwalkar, A. (2018). _Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization_. Journal of Machine Learning Research, 18(185), 1–52.**

   **§3.1–§3.2; Algorithm 1, PDF p. 8**, allocate small budgets to many configurations, eliminate some, and increase resources for survivors. Hyperband also includes a conservative bracket giving every configuration the full budget. This is a resource-allocation idea rather than a geometric initialization rule. Treating starting directions as configurations would be an adaptation. [Publication record](https://jmlr.org/papers/v18/16-558.html); [paper](https://jmlr.org/papers/volume18/16-558/16-558.pdf).

   BibTeX key: `li2018hyperband`.

## Proposed experiments for this project

These are proposed tests, with no claimed improvement established yet.

1. **Diverse random directions.** Draw, for example, 128 normalized Gaussian candidates and select twelve by the farthest-first rule inspired by Ugray et al., §3.1.2. Account for sign equivalence using

   \[
   d_{\pm}(u,v)=\min\{\|u-v\|_2,\|u+v\|_2\}
                =\sqrt{2-2|u^\top v|}.
   \]

   This formula is our direct algebraic derivation for unit vectors. Its relevance follows directly from the [implemented width objective](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/ar_tools/descent.py:83): exchanging \(u\) for \(-u\) exchanges the extreme projections, giving \(\widehat w(-u)=\widehat w(u)\). Thus opposite directions count as the same candidate orientation. The candidate count is an experimental choice, not a literature recommendation.

2. **One data-informed direction plus random starts.** Obtain a preliminary coefficient estimate from the [existing L-infinity fitting routine](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/ar_tools/classical_selectors.py:204), set

   \[
   u_{\mathrm{seed}}=
     \frac{(1,-\widetilde\phi_1,\ldots,-\widetilde\phi_p)}
          {\sqrt{1+\|\widetilde\phi\|_2^2}},
   \]

   and use it alongside eleven random starts. This is our proposed AR-specific adaptation of data-informed initialization, inspired by Qu et al., §3. The preliminary estimator optimizes a different objective. Include its computation time in comparisons.

3. **Pilot runs followed by selective refinement.** Inspired by Hyperband, §3.2, try many short runs and continue a smaller subset based on best width reached. A pilot could use 48 starts to 200 evaluations, retain twelve to 1,000, then three to 5,000. These numbers are illustrative. Continue the original iteration counter and stepsize schedule when promoting a run. First audit the pilot with patience disabled so its prescribed budgets are actually evaluated; examine the patience rule separately. Include full-budget control runs to detect starts that look poor early but improve later.

For comparison, keep the generated data and optimizer settings fixed within each comparison, repeat across independent data and initialization seeds, and compare best width and coefficient error at matched total computation. Include candidate selection, preliminary fitting, and discarded pilot runs in the cost. These are proposed experimental controls.

An elementary multistart calculation, not a guarantee about this optimizer: conditional on fixed data, if independent starts each have success probability \(q\), the chance that at least one of \(L\) starts succeeds is \(1-(1-q)^L\). This follows by multiplying the independent failure probabilities. The unknown \(q\) prevents using this expression alone to justify \(L=12\).

## Citation files

[additional_references.bib](/Users/iljasukhanov/MASTERARBEIT/output/analysis/ar_initialization_literature/additional_references.bib) contains the additional entries, including the references added for the starting-direction catalogue. This note and that file are separate from the thesis sources.
