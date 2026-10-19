# Stationarity as a restart diagnostic for the failed AR(3) run

This is our same-path exploratory experiment. It does not replace the shared optimizer or establish a generally calibrated restart rule.

## What failed, and does changing the start help?

The original failed estimate has spectral radius 26.97911610. All 5000 visited current estimates in that run violate the causal stationarity condition. The zero-start control ends at radius 0.50251823, and all its visited estimates are stationary. Sources: [failed trajectory](failed_stationarity_history.csv), [comparison](restart_results.csv). With data and step settings fixed, this is direct evidence of initialization sensitivity. It does not prove initialization is the only cause or certify a local minimum.

## Model-based meaning of a bad estimate

For a causal stationary AR(p), the zeros of 1-phi_1 z-...-phi_p z^p lie outside the unit circle. Equivalently the companion eigenvalues have modulus below one. Primary documentation: [statsmodels ArmaProcess.isstationary](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_process.ArmaProcess.isstationary.html); [companion and reciprocal-root criterion, Notes](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.tools.is_invertible.html#notes). Workspace implementation: [functions.py](../../../Code/Code_AR_processes/ar_tools/functions.py), is_stationary. This check uses estimated coefficients and the model assumption; it does not require knowing the true vector. Passing it does not establish coefficient accuracy or global optimality.

## A necessary AR(3) coefficient and direction bound: our derivation

Write the AR(3) characteristic polynomial as lambda^3-phi_1 lambda^2-phi_2 lambda-phi_3 = product_i(lambda-lambda_i), with |lambda_i|<1. Matching coefficients and applying the triangle inequality gives |phi_1|<3, |phi_2|<3 and |phi_3|<1. Consequently ||phi||_2^2<19, and for a unit direction that recovers phi, |u_0|=1/sqrt(1+||phi||_2^2)>1/sqrt(20)=0.2236067977. This is a necessary condition, not a sufficient stationarity test. It provides a model-based rejection threshold without choosing an arbitrary coefficient-size cutoff. The full root test is stronger. The derivation is ours from the root factorization documented in the Notes cited above.

Individual coefficients larger than one can be stationary: (2.7,-2.43,0.729) has characteristic polynomial (lambda-0.9)^3 and numerically measured radius 0.90000676. Thus a rule rejecting every coefficient outside [-1,1] would be incorrect. This example follows by expansion of that polynomial; numerical check: [script](run_analysis.py).

## Tested restart, same total evaluation budget

At the checkpoint after 100 evaluated states, inspect the best recovered estimate. If undefined/nonfinite or nonstationary, restart from e_0=(1,0,0,0), corresponding to phi=0, and reset the geometric step schedule. Spend the remaining 4900 states on that run. Keep the smallest original width among visited stationary estimates, retaining the unconstrained winner separately. This explicitly restricts acceptable outputs to stationary models; in this example the stationary and unrestricted winners coincide. The true coefficients enter diagnostics only. The checkpoint 100 is an exploratory budget choice, not a theoretically justified threshold.

| Method | Evaluated states | Width | Spectral radius | Coefficient L2 error |
|---|---:|---:|---:|---:|
| Original failed start | 5000 | 1.8629524395 | 26.97911610 | 30.57810722 |
| Stationarity restart after 100 states | 5000 | 1.7476115691 | 0.50251837 | 0.00402705 |
| Zero-coefficient start from the beginning | 5000 | 1.7476115640 | 0.50251823 | 0.00402689 |


Source: [restart results](restart_results.csv), [restart trajectory](restart_history.csv), [settings and checks](configuration_and_results.json). The restart solves this observed failure to approximately the same empirical optimum as the zero-start control, within the same evaluation budget. It is not a guarantee for other data or starts.

## Why not reject at the first unstable estimate?

On this same path, random initialization seed 3 begins with radius 1.26422917, becomes stationary at evaluated state 1, and finishes with error 0.00402695. An immediate rejection would discard that successful trajectory. Sources: [recovering initialization history](recovering_initialization_history.csv), [12 fixed-seed initialization audit](same_path_initialization_audit.csv). These 12 starts share a single sample path and do not establish a population failure rate.

## Suggested first implementation

Always include the known stationary phi=0 start. Run additional starts if budget allows. For each run retain a best stationary candidate separately from the unrestricted best. At a run endpoint, or a chosen checkpoint after persistent lack of valid progress, restart if the best estimate is nonstationary or undefined. Reset the local geometric schedule and cap the restart count or total budget. Do not interpret stationarity as an accuracy certificate, and do not use a positive stability margin without stating the additional assumption it imposes near the unit-root boundary. This is our proposed optimization heuristic based on the model check and the experiment, not a published theorem.

A stronger alternative is to parameterize only stationary coefficients and optimize the original normalized width there. A documented stationary parameter transform is [statsmodels constrain_stationary_univariate](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.tools.constrain_stationary_univariate.html), which cites Monahan (1984), Biometrika 71(2), pp. 403-404. Adapting that parameterization to this width objective would change the update geometry and has not been tested in this audit.
