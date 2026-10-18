# Justifying an empirical scope of AR orders 1–5

Research note, 18 October 2026. This note and the accompanying BibTeX file are outside the protected thesis folder. No thesis source or bibliography was edited. This is a literature and study-design assessment, not new experimental evidence about the geometric estimator.

## Recommendation

Describe the numerical scope as **low-order causal AR processes with true order p in {1,2,3,4,5}**. Treat five as a deliberate experimental limit. It is not a mathematical boundary for short-range dependence: fixed, finite-order stable AR models have summable autocorrelations at higher orders too. See [Künsch, §3.1.2, Theorem 4, and §3.3.3](https://people.math.ethz.ch/~hkuensch/teaching/script-time-series.pdf#page=15).

The existing [Numerical Results TODO](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/3_application_to_time_series.tex:519) currently identifies short-range dependence with p=1,...,5. That is the terminology to avoid when the text is explicitly revised. The present research request has not been treated as authorization to edit that TODO.

## Verified precedents and their limits

The distinctions between **true generating order**, **fitted candidate order**, and **embedding dimension** matter. A paper fitting AR(1)–AR(5) does not necessarily generate its data at all five orders.

| Study | Verified use of orders up to five | How closely it matches |
|---|---|---|
| Lucena, Martins, Pinho and Gouveia (2026), *Fast and Interpretable Autoregressive Estimation with Neural Network Backpropagation*, arXiv:2603.19041 | §3.1 generates stationary Gaussian AR data for every p=1,...,5, with 500 parameter sets per order and 50 repetitions. | Direct match for the true-order range in a coefficient-estimation experiment. This reference is the arXiv version discussed in the conversation. Its innovations and estimator differ from the thesis. |
| Bergmeir, Hyndman and Koo (2018), *A note on the validity of cross-validation for evaluating autoregressive time series prediction*, Computational Statistics & Data Analysis 120, 70–83 | §4.4 fits AR(1)–AR(5); §4.3 also states that the maximal fitted order is five. | Exact fitted range, not a true-order range of 1–5. §4.5 includes AR(3), MA(1), and a seasonal AR process with lag 12. |
| Hossain, Moon and Chon (2020), *Estimation of ARMA Model Order via Artificial Neural Network for Modeling Physiological Systems*, IEEE Access 8, 186813–186820 | §III-A, p. 186816, fixes the candidate AR orders at 1–5 and MA orders at 0–5. | Explicit upper bound of five for order identification. The setting is input–output ARMA system identification, not exclusively univariate pure AR coefficient estimation. |
| Le Borgne, Santini and Bontempi (2007), *Adaptive model selection for time series prediction in wireless sensor networks*, Signal Processing 87(12), 3010–3020 | §5 and Tables 3–4 evaluate AR(1)–AR(5), alongside a constant model, on 14 sensor time series. | Exact fitted AR range in an application. This is not a synthetic experiment with known true AR orders. |
| Dufour and Jouini (2006), *Finite-sample simulation-based inference in VAR models with application to Granger causality testing*, Journal of Econometrics 135(1–2), 229–254 | §5, p. 240, varies generating orders p=1,...,5; Table 1(B) shows this range. | Exact generating-order range for vector AR models. It is supporting precedent, not a univariate AR estimation study. |

Primary texts and precise reading locations:

1. **Lucena et al.** [§3.1, Simulation setting](https://arxiv.org/html/2603.19041v1#S3.SS1). BibTeX key: `lucena2026fastar`.
2. **Bergmeir et al.** [Author manuscript, §4.3–§4.5, PDF pp. 9–10](https://robjhyndman.com/papers/cv-wp.pdf#page=9); [publication details](https://robjhyndman.com/publications/cv-time-series/). BibTeX key: `bergmeir2018crossvalidation`.
3. **Hossain et al.** [Author-hosted published PDF, §III-A, p. 186816, PDF p. 4](https://biosignal.uconn.edu/wp-content/uploads/sites/2503/2020/10/12_Hossain_2020_IEEE-Acess.pdf#page=4). The explicit statement is: “in this paper we fixed the maximum AR and MA orders at 5”. Their broader assertion about the adequacy of ARMA(5,5) for real systems is not adopted here as a general justification. BibTeX key: `hossain2020armaorder`.
4. **Le Borgne et al.** [Author manuscript, §5, Tables 3–4, PDF pp. 7–8](https://vs.inf.ethz.ch/publ/papers/santinis_AMS_07.pdf#page=7); [published article](https://doi.org/10.1016/j.sigpro.2007.05.015). BibTeX key: `leborgne2007adaptive`.
5. **Dufour and Jouini.** [Author-hosted published PDF, §5, p. 240, PDF p. 12](https://www2.cirano.qc.ca/~dufourj/Web_Site/Dufour_Jouini_2006_JE_FinSampleInferVARCauality.pdf#page=12). BibTeX key: `dufour2006varinference`.

These precedents establish that restricted ranges up to five have been used. They do not prove that order five is optimal, sufficient for every application, or a universal convention. Of the references above, Lucena et al. is the direct match for a univariate generating-order range covering all five orders.

## A rationale specific to this thesis

The following are our methodological recommendations and deductions, with their premises stated explicitly.

### Focus the empirical claims

The chapter's stated aim is to investigate sliding-window geometry and geometric estimation, as recorded in its [opening paragraphs](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/3_application_to_time_series.tex:3). For that aim, a controlled study of five orders can be a useful first empirical investigation. The reason to stop at five is to obtain adequate depth within a declared scope, not to imply that higher-order models are unimportant.

This recommendation is consistent with Morris, White and Crowther's distinction between proof-of-concept studies and broader attempts to find a method's limits. They also emphasize that simulation findings concern the scenarios studied. See [Morris et al. (2019), §3.1–§3.2](https://onlinelibrary.wiley.com/doi/10.1002/sim.8086). This source provides study-design guidance, not a recommendation of p=5.

### Control geometric dimension and investigate errors separately

The manuscript formulates the target transition at k=p+1 and recovers p coefficients from a direction in R^(p+1); see the [geometric setup](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/3_application_to_time_series.tex:177) and [coefficient-recovery remark](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/3_application_to_time_series.tex:312). Therefore, the proposed restriction puts the transition dimensions in the range 2–6. This is a direct deduction from the proposed setup, not an independent verification of the manuscript's propositions.

Keeping this range modest makes it feasible to allocate computation to repeated optimization starts and to distinguish sample approximation error from optimization error. The existing [error decomposition](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/3_application_to_time_series.tex:561) already treats these separately. Whether the chosen optimization budget is adequate must still be checked empirically.

Related geometric literature supports taking dimension seriously. Brunel (2019) studies convex hulls of i.i.d. points and derives support-approximation bounds depending on dimension and probability mass near supporting hyperplanes. See [§3.1, Theorem 1 and Corollary 1; §3.3, Proposition 1](https://arxiv.org/pdf/1503.01504#page=4). **Those i.i.d. results do not directly give a convergence rate for overlapping time-series windows.** They support the geometric concern, not a proved error bound or a special cutoff at p=5.

### Spend the budget on variation within each order

Our recommendation is to vary sample length, innovation distribution, coefficient structure, stability margin, and initialization within p=1,...,5. Include ordinary and difficult cases, including coefficients close to a lower-order model and roots approaching the unit circle. These are proposed design factors, not a claim that each has already been tested in this project.

The choice of factors and the use of sufficient repetitions should follow the aims and intended performance measures; see [Morris et al., §3.2 and §5.2](https://onlinelibrary.wiley.com/doi/10.1002/sim.8086). Lucena et al. separately examine root proximity in [§4.1](https://arxiv.org/html/2603.19041v1#S4.SS1), illustrating why order alone should not be used as a measure of difficulty.

## Distinguish three upper limits

1. **True order:** Restricting the data-generating processes to p<=5 is the proposed empirical scope.
2. **Candidate order:** If every selector is forced to return at most five, its overestimation probability at true p=5 is zero by construction. This follows directly from the definition of overestimation, P(p_hat>p). If overestimation is a target of evaluation, allow a common candidate maximum above five, or clearly state the consequence of a cap at five. Do not give an estimator the true order in the order-selection experiment.
3. **Embedding dimension:** A transition at k=6 needs observations of the curve beyond six to demonstrate a plateau. For the displayed descent-chain selector, candidates run only to K-1, so making k=6 eligible requires K>=7; see its [definition](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/3_application_to_time_series.tex:514). A larger K may be useful, but is a separate design choice.

All three statements are elementary deductions about the study design. No simulation was run to validate a particular choice of candidate cap or embedding maximum.

## Suggested thesis paragraph

The following is proposed wording for review, not inserted text. The rationale is our methodological argument. The cited studies supply precedent for restricted order ranges, not a proof that five is necessary.

The numerical investigation is restricted to causal autoregressive processes with true orders p in {1,2,3,4,5}. This restriction defines a study of low-order models and permits a detailed assessment of the proposed geometric estimators across sample sizes, innovation distributions, and coefficient configurations. Related evaluations have used the same range of true AR orders (Lucena et al., 2026), fitted AR models (Bergmeir et al., 2018; Le Borgne et al., 2007), or candidate AR orders in ARMA identification (Hossain et al., 2020). For the geometric approach, the restriction bounds the transition dimension p+1 by six and allows computational effort to be devoted to repeated optimization and the assessment of sampling and optimization errors. The cutoff is an experimental design choice; numerical conclusions are confined to the investigated orders, and no performance claim for p>5 is made.

## Sources deliberately not presented as exact matches

- **Moon, Hossain and Chon (2021), AR and ARMA model order selection for time-series modeling with ImageNet classification:** [§2.1, PDF p. 2](https://biosignal.uconn.edu/wp-content/uploads/sites/2503/2021/02/01_Moon_2021_SignalProcessing.pdf#page=2) gives p=1–9 and q=0–9. Do not cite it as an experiment restricted to p<=5.
- No claim is made here that all or most real-world time series require only five lags. The seasonal lag-12 counterexample in [Bergmeir et al., §4.5](https://robjhyndman.com/papers/cv-wp.pdf#page=10) is a concrete reason not to make such a universal statement.

## Citation file

[ar_order_1_to_5_scope.bib](/Users/iljasukhanov/MASTERARBEIT/output/research/ar_order_1_to_5_scope.bib) contains the five precedent references, the two methodological references, and the Künsch lecture notes used for the memory distinction. It is not included in or imported into the thesis bibliography.
