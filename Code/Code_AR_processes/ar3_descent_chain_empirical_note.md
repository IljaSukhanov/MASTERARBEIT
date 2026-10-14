# Making the original descent chain usable on empirical widths

This note records the new experiment and separates proved properties, numerical
findings, and unresolved assumptions. The selector is unchanged. No thesis source
has been edited, and no statistical denominator stabilizer is used.

## Recommended computation

Follow Algorithm 2 of the supplied preview (p. 9): build all embedding dimensions
from the same response times; run the LP tangent iteration of Eq. (1.7), p. 8;
start from the previous direction padded on the right, padded on the left, and
the first coordinate axis; retain the best feasible iterate. This enforces a
monotone fitted curve by Lemma 1.2.17 and the padding argument on p. 8. It does
not establish global optimality. The general convex-concave procedure is a local
optimization heuristic; see [Lipp and Boyd (2016), abstract](https://web.stanford.edu/~boyd/papers/cvx_ccv.html).

Use the original interior-state score

\[
 S_k=\frac{\widehat m_{k-1}-\widehat m_k}
 {\widehat m_k-\widehat m_K},\qquad 2\leq k\leq K-1,
 \quad \widehat p=\min\arg\max_k S_k-1.
\]

Positive/zero is infinity, and zero/zero is zero. This is Eq. (1.6), p. 6 of
[the preview](/Users/iljasukhanov/Downloads/elbow_point_selection_preview.pdf).
The implementation removes only machine-precision-sized drops when calculating
the chain; the saved fitted curves are unmodified. It does not introduce a
statistical cutoff or smooth the empirical widths.

Applying exactly this score to log-widths is the input transformation explicitly
discussed in Remark 1.2.16, p. 7. It is reported separately from the raw-width
rule. Logarithms change the finite-sample decisions. Continuity and strict
monotonicity preserve population plateau correctness; empirical consistency
still requires the appropriate plateau-ratio condition. For logarithms near a
positive plateau c, the mean-value theorem gives a ratio multiplier tending to
one, so condition (N) for raw widths transfers to log-widths. This last sentence
is a direct derivation, rather than a claim that every continuous increasing
transformation automatically transfers (N).

Choose K before examining the true answers. The K=5 results below are only a
diagnostic for an independently justified upper bound of order 3. Choosing
K=p+2 using the true p is not an order-selection procedure for unknown p.

## Paired experiment: the same 50 AR(3) series

The saved series have T=2000 and uniform innovations. The new fit uses common
times t=10,...,2000 in every dimension, LP tolerance 1e-9, at most 50 steps per
start, and the three starts above. Across 3716 LP solves, no descent start
reached the 50-step limit. All 50 fitted curves were monotone within 1e-12. Configuration
and checks are saved in [configuration.json](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/csv/ar3_descent_chain_lp/configuration.json)
and [verification.json](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/csv/ar3_descent_chain_lp/verification.json).

| Computation/input | K | Correct | Under | Over |
|---|---:|---:|---:|---:|
| Previous subgradient fit with padded start, raw widths | 10 | 19 | 6 | 25 |
| New LP fit, raw widths | 10 | 36 | 13 | 1 |
| New LP fit, log-widths | 10 | 40 | 9 | 1 |
| New LP fit, raw widths | 8 | 39 | 11 | 0 |
| New LP fit, log-widths | 8 | 41 | 9 | 0 |
| Diagnostic: new LP fit, raw widths | 5 | 42 | 8 | 0 |
| Diagnostic: new LP fit, log-widths | 5 | 46 | 4 | 0 |

Sources: [previous summary](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/csv/ar3_order_selection_warm_starts/summary.csv),
[new summary](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/csv/ar3_descent_chain_lp/summary.csv),
and [per-process selections](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/csv/ar3_descent_chain_lp/selections.csv).
The caps 5 and 8 truncate the same common-time K=10 fitted curves, so they
isolate the effect of the terminal dimension. They are not separate Algorithm 2
fits with response times starting at t=5 or t=8. These are descriptive results
on the existing 50 processes, not an independent validation set.

Global floating-point convex-hull audits were performed in dimensions 1–4.
In dimension 4, 49/50 LP fits matched the hull result within 1e-8; the remaining
gap was about 7.2e-5. There were 14 fits above the global audit by more than 1e-8
across dimensions 2–4. Replacing all first four fitted widths by their global
audit values leaves the K=10 selection counts unchanged. See
[width audit](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/csv/ar3_descent_chain_lp/width_curves.csv)
and [replacement decisions](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/csv/ar3_descent_chain_lp/global_prefix_selections.csv).

The hull audit is an original geometric calculation: for C=conv(P), the width
in direction u is the support function of C-C. Its minimum over unit u is the
radius of the largest origin-centered ball contained in C-C, hence the smallest
distance from the origin to a facet. C-C is generated by differences of vertices
of C. The implementation evaluates the facet distances and independently
re-evaluates the projected data range in the winning direction. It uses SciPy's
facet normals and offsets, documented under `equations` in the
[ConvexHull reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html).
This is a global calculation in floating-point arithmetic, not an interval
certificate of numerical precision. Dimensions 5–10 remain local LP fits.

## Why optimization cannot remove every failure

The following deterministic observation is an original derivation. Suppose the
global empirical prefix through the true break is known. For an earlier
candidate j write A=D_j, a=m_j; for the true candidate write B=D_{p+1},
c=m_{p+1}; let b be the terminal minimum. Here a>=c>b. The earlier candidate
wins precisely when

\[
 F(b)=A(c-b)-B(a-b)>0.
\]

An earlier win implies A>B. Thus F has negative derivative B-A, and every
smaller terminal value also gives F(b)>0. A feasible direction gives an upper
bound b_0 on the global terminal minimum. If F(b_0)>0, no improved terminal
optimization can make the true break beat that earlier candidate. The same
argument applies after taking logarithms, using transformed drops and widths.

Using the global prefix audits and feasible LP terminal widths, this diagnosis
holds for 13/50 raw-width cases and 9/50 log-width cases at K=10. Thus all the
observed K=10 underselections persist even if the remaining dimensions are
optimized globally, conditional on the accuracy of the prefix hull audit. The
smallest positive cross-product margin was 8.95e-4 for raw widths and 2.20e-4
for log-widths. These are numerical findings supporting the deterministic
argument; see [earlier-bend diagnostics](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/csv/ar3_descent_chain_lp/earlier_bend_certificates.csv).

For example, replication 5 has global empirical m_3=1.289699 and
m_4=1.262962, while a feasible m_10 is 1.219242. Its true-break raw score is
about 0.612, whereas the score at k=3 (order 2) is 90.776. For log-widths the
scores are about 0.595 and 31.772. There is a small final structural drop and a
substantial empirical tail reduction. These numbers are saved in the audit and
score files; they do not establish a uniform detection threshold for other AR
models.

The single overselected process, replication 38, illustrates the other failure:
its raw k=9 score is about 72.996 versus 40.321 at the true k=4, because the
last empirical drop is tiny. See
[chain scores](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/csv/ar3_descent_chain_lp/descent_chain_scores.csv).
The figures show a successful case, a case rescued by log-widths, and a case
that fails with both inputs.

## What the thesis can prove for the unchanged rule

The preview already supplies population correctness (Proposition 1.2.12, p. 6)
and conditional empirical consistency (Theorem 1.2.14, p. 7). With exact
empirical minima and K=p+2, the theorem gives almost-sure consistency without
a tail condition. For a larger fixed cap it assumes

\[
 \max_{p+2\leq k\leq K-1}
 \frac{\widehat D_k}{\widehat m_k-\widehat m_K}=O_P(1).
 \tag{N}
\]

Do not state that convergence of the widths proves (N). The preview itself
identifies the plateau-fluctuation rate as open in Remark 1.2.15, p. 7.
Local LP convergence and monotonicity also do not prove convergence to exact
empirical minima, so the theorem cannot automatically be applied to the
computed LP curve.

A weaker sufficient condition follows by comparing the unwanted scores directly
with the true score. This is an original deduction from the proof of Theorem
1.2.14, not an established AR fluctuation theorem. Suppose fixed-dimensional
width consistency holds and the final population drop is positive. Write
r_* = mhat_{p+1}-mhat_K and let R_T be the maximum score after the true break.
Assume

\[
 r_* R_T\xrightarrow{P}0,
 \tag{N'}
\]

with the product defined as zero when r_*=0 (then every later drop is zero).
For r_*>0 the ratio of the largest unwanted tail score to the true score is
r_* R_T / Dhat_{p+1}. Its denominator converges to the positive population
drop, so this ratio tends to zero. Earlier scores have finite limits whereas
the true score diverges. Therefore the unchanged selector is consistent.
Condition (N) implies (N'), since r_* tends to zero, but (N') also allows
tail scores that diverge more slowly than the true score. Establishing either
condition for the actual AR empirical minima remains necessary for an
unconditional theorem at a general cap.

To see why width consistency alone is insufficient, consider a constructed
monotone sequence with population m_3=c+d and m_4=m_5=m_6=c, where c,d>0,
and empirical entries

\[
 \widehat m_4=c-\varepsilon,\quad
 \widehat m_5=c-\varepsilon-\varepsilon^2,\quad
 \widehat m_6=c-\varepsilon-\varepsilon^2-\varepsilon^5.
\]

Keep the earlier entries equal to their population values. All entries converge
to the population curve as epsilon tends to zero. Yet the true k=4 score is
asymptotic to d/epsilon^2, while the false k=5 score is 1/epsilon^3 and wins.
This is an algebraic counterexample to an inference from width convergence;
it is not a counterexample proved for actual AR sample minima.

## Defensible completion of the thesis argument

Use the original population theorem, state the empirical theorem with its
explicit fluctuation assumption, and report the new LP experiment with the
remaining finite-sample failures. If an unconditional general-cap consistency
theorem is required, its missing step is a joint plateau-fluctuation argument
for exact empirical minima, plus control of optimization error for the computed
version. None of the current 50-process results proves that step.

The implemented procedure makes the computational experiment substantially
more successful without changing the chain. It does not guarantee correct
order at T=2000 for every coefficient vector. The failure diagnosis explains
why a stronger numerical optimizer alone cannot provide that guarantee.
