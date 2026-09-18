# Preliminary novelty assessment, 18 September 2026

**Assessment:** estimating an unknown bounded-noise support and using the estimate in model identification or order selection is established. The exact combination investigated here—maximum absolute residual from a fixed-order least-squares pilot, minimum feasible AR order, and consistency without an added penalty or slack sequence—was not identified in this targeted search. Its novelty is unconfirmed.

This is a preliminary literature assessment, not an exhaustive review. Failing to find the exact rule is not evidence that no earlier publication contains it. An independently derived argument can coincide with established results.

## Closest verified primary sources

1. **Lauricella and Fagiano (2018), *Guaranteed simulation error bounds for linear time invariant systems identified from data*.** [Author preprint](https://arxiv.org/pdf/1809.01061), Section II-C, Procedures 1 and 2, pp. 4–5. They estimate an unknown disturbance bound and subsequently reduce predictor order to a minimal compatible order, using multi-step error bounds. Their model has a deterministic stable system, external input, and additive measurement noise. It is not the same stochastic AR innovation model, but it is a direct precedent for the broad estimate-bound-then-select-order idea.

2. **Lauricella and Fagiano (2020), *Set Membership identification of linear systems with guaranteed simulation accuracy*.** [Author preprint](https://arxiv.org/pdf/2001.07628), Section III-C, Procedures 1 and 2, pp. 4–5. The updated procedures first estimate the noise bound and then estimate system order using multi-step prediction errors and a convergence tolerance. They do not specify the least-squares pilot maximum-residual rule investigated here. The difference between bounded measurement noise in their model and bounded process innovations in the AR model must be stated when comparing the results.

3. **Bai, Cho, and Tempo (1998), *Convergence Properties of the Membership Set*, Automatica 34(10), 1245–1249.** [Publisher abstract](https://www.sciencedirect.com/science/article/pii/S000510989800065X/pdf?md5=710d04911c4f7aa1fe680e44f3c4b59d&pid=1-s2.0-S000510989800065X-main.pdf). The abstract expressly treats an unknown noise bound estimated from measurements and proves convergence of the estimate under probabilistic assumptions and excitation. Only the abstract was verified here; its full theorem statements were not inspected. The minimax estimator attributed to this paper is explicitly given in the 2024 source below.

4. **Li, Yu, Conger, Kargin, and Wierman (2024), *Learning the Uncertainty Sets of Linear Control Systems via Set Membership: A Non-asymptotic Analysis*, ICML/PMLR 235.** [Proceedings record](https://proceedings.mlr.press/v235/li24ci.html), [author preprint](https://arxiv.org/pdf/2309.14648), Section 3.2, pp. 5–6, equations (4)–(5), Theorems 3.5 and 3.7. They learn an unknown disturbance bound using minimax residuals and add an upper-confidence correction before constructing feasible parameter sets. They explicitly explain that underestimating the bound can exclude the true parameter. Thus the coverage issue discussed in our analysis is established, as is a corrected estimated-bound feasibility construction. The inspected section concerns uncertainty sets at a fixed system dimension, not the exact AR-order selector proposed here.

## The narrow candidate contribution

The proposed AR rule and its derivation are in [estimated_support_findings.md, Sections 1–3](/Users/iljasukhanov/MASTERARBEIT/output/analysis/ar_order_without_penalties/estimated_support_findings.md:5). The potential contribution is the complete combination:

- A fixed upper-order AR least-squares pilot with no known support scale.
- Its uninflated maximum absolute residual as a support-radius estimate.
- Selection of the smallest order admitting residuals within that estimated interval.
- A consistency-in-probability argument under stable minimal AR models with independent uniform innovations, including proof that the pilot radius exceeds the true radius with probability tending to one.

No statement that any one ingredient is individually new is justified by this search. The proof uses established martingale asymptotics and the faster contraction of bounded-noise feasible sets. Whether the precise combination and resulting AR-order theorem are already known needs further checking of older bounded-error, set-membership, and extreme-value AR estimation literature.

## Search scope and justified wording

Searches covered combinations of autoregressive order estimation, bounded or uniform innovations, unknown support/noise bounds, maximum residuals, least-squares pilots, minimum feasible order, set-membership order determination, and bounded-error model selection. References from the 2020 paper and the 2024 set-membership paper led to older bound-estimation work. This was a web-based targeted search; it did not cover every database, dissertation, or reference chain.

The justified present description is **an AR-specific estimated-support feasibility selector with a candidate consistency result, related to established set-membership methods**. A claim of being the first support-based, penalty-free, or estimated-noise-bound model-order method is not justified. A claim that the exact pilot construction and AR-order theorem are new remains open, and also requires independent review of the argument.
