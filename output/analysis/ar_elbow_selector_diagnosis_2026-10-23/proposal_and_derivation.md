# Exploratory diagnosis and a regularized descent-chain score

This note records an exploratory analysis of the frozen 250-series experiment.
The original experiment, its notebook, and its Table 2.2-style outputs are unchanged.
No new simulated series, fitted width curves, or production selector implementation
were introduced. The additional scores use the existing saved fitted widths.

## Observed results

- Original descent chain: 112 correct, 35 underselections, 103 overselections out of 250.
- The last eligible order, 8, was selected 47 times, although every generating order is at most 5.
- Existing first- and second-difference rules gave 75 and 88 correct selections.
- At the true break dimension, 13 fitted widths exceeded the width of the known
  generating-coefficient direction by more than 5%. This certifies some optimization
  failures. Being below this witness does not certify a global minimum.

These counts are reproducible from `per_series.csv` and `summary.json`.
The true coefficients are used only for the diagnostic witness, never for the
candidate elbow scores.

## A late-selection mechanism: direct algebra, not a literature theorem

Write the current score in order notation as

$$S(q)=\frac{\widehat m_q-\widehat m_{q+1}}
                 {\widehat m_{q+1}-\widehat m_K},
\qquad q=1,\ldots,K-2.$$

If the fitted tail is a straight decreasing line, $\widehat m_k=a-bk$ with $b>0$,
then, for candidate locations within that tail,

$$S(q)=\frac{1}{K-q-1}.$$

Thus the score increases toward the last eligible order even though this tail
contains no change in slope. This establishes a failure mechanism for that
particular tail shape. It does not prove that all 103 observed overselections
have this cause.

Source for the implemented unregularized score:
`Code/Code_AR_processes/ar_tools/breakpoint_selectors.py`, function `descent_chain`.
The saved experiment uses common endpoints and one padded warm start per dimension;
its curves are fitted numerical widths rather than certified global minima.

## Proposed modification

Consider the scale-invariant score

$$S_\lambda(q)=
\frac{\widehat m_q-\widehat m_{q+1}}
     {\widehat m_{q+1}-\widehat m_K+\lambda\widehat m_K},
\qquad \widehat p_\lambda=\min\arg\max_{1\le q\le K-2}S_\lambda(q),
\quad\lambda>0.$$

This is a proposed regularized elbow score motivated by the current descent-chain
score. No claim is made that it has the same stationary distribution as the
original chain, that it is new in the literature, or that it has established AR
consistency. The positive term limits denominator collapse on an approximately
flat empirical tail. Its cost is a tuning parameter and increased underselection
when the regularization is too large.

Five exploratory settings were evaluated: 0.001, 0.003, 0.01, 0.03, and 0.1.
All five results are retained in `regularization_sensitivity_exploratory.csv`.
The best setting on these already examined data, 0.003, gave 151 correct,
65 underselections, and 34 overselections. Its correct counts by true order
were 45, 37, 28, 22, and 19 out of 50. BIC remains ahead at 215/250.
The improvement is primarily at orders 1-3.

## Conditional asymptotic argument: original derivation under explicit assumptions

Let K be fixed and 1 <= p <= K-2. Assume positive population widths satisfy
$m_1\ge\cdots\ge m_p>m_{p+1}=\cdots=m_K=c>0$.
Suppose the total fitted-width error obeys

$$E_T=\max_{1\le k\le K}|\widehat m_{k,T}-m_k|=O_p(a_T),$$

and choose a deterministic $\lambda_T$ such that $\lambda_T\to0$ and
$a_T/\lambda_T\to0$. This error includes empirical sampling error and numerical
optimization error. For the score above:

- At q=p, the numerator tends to $m_p-c>0$ and the denominator is
  $\lambda_T c(1+o_p(1))$, so the score diverges at rate $1/\lambda_T$.
- At q<p, the denominator tends to the positive number $m_{q+1}-c$, so the score
  has a finite limit.
- At q>p, the numerator is $O_p(a_T)$ and the denominator is
  $\lambda_T c(1+o_p(1))$, so the score is $o_p(1)$.

There are finitely many candidates, hence the true candidate wins with
probability tending to one under these assumptions. This is a conditional
argument only. No suitable rate $a_T$, finite-sample calibration, or vanishing
optimization-error bound has been established for this experiment. A fixed
finite-sample lambda does not guarantee correctness for every population curve.

## Evaluation recommendation

Treat the original 250 series as development data for this new score. Choose a
calibration rule, freeze it, and evaluate on previously unused coefficient vectors.
There are 300 coefficient vectors absent from the selected sample, with counts
58, 59, 60, 63, and 60 at orders 1 through 5. Their corresponding 1,500 saved
series could supply an evaluation pool. If any of that pool is used for tuning,
reserve a disjoint set of coefficient vectors for final evaluation. Group all
shapes of one coefficient vector together when partitioning models.

The general reason to separate tuning from performance evaluation is selection
bias from optimizing a noisy evaluation criterion; see Cawley and Talbot (2010),
Section 5.1, pp. 2095ff., and Section 6, p. 2103:
https://www.jmlr.org/papers/volume11/cawley10a/cawley10a.pdf

## Workspace sources

- Original results and curve checkpoints:
  `Code/Code_AR_processes/outputs/csv/ar_order_selection_final/030cde6f724e3c30/7ac02863db38b8d0/`.
- Frozen selected-series identities:
  `Code/Code_AR_processes/inputs/ar_order_selection_final/030cde6f724e3c30/series_metadata.csv`.
- Source coefficient bank:
  `Code/Code_AR_processes/inputs/ar_coefficient_estimation_final/1b5bb834ea4e8cc9/models.csv`.
