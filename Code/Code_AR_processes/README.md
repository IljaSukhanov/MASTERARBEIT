# AR simulation and optimization tools

## Notebook folders

- [Estimation](xx_notebooks_estimation/README.md): coefficient recovery, estimator comparisons, and sample-size experiments.
- [Figures](xx_notebooks_figures/README.md): geometric illustrations and thesis figures.
- [Order selection](xx_notebooks_order_selection/README.md): order selectors and related diagnostics.
- [Tuning](xx_notebooks_tuning/README.md): starting directions and step-schedule comparisons.

## Coefficient accuracy: MAE only

Every active coefficient comparison uses
[`coefficient_mae`](ar_tools/coefficient_errors.py):

\[
\operatorname{MAE}(\widehat\phi,\phi)
=\frac1p\sum_{j=1}^{p}|\widehat\phi_j-\phi_j|.
\]

Monte Carlo summaries average per-fit MAEs across replications. Per-coefficient
summaries use mean absolute error as well. A fit with undefined or nonfinite
coefficients has unavailable MAE; unconditional summaries retain that failure
instead of silently dropping it. The implementation follows the uniform output
average in [scikit-learn's `mean_absolute_error`, `multioutput`
parameter](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html).

Coefficient rankings, co-winner counts, thresholds, bootstrap comparisons, plots,
and generated reports use this same convention. Squared residuals in OLS/QML,
the Horváth residual objective, width minimization, and stationarity checks keep
their estimator-specific meanings.

The updated notebooks clear stale saved displays. MAE experiment outputs use
new `mae` or `mae_vs_T_100_processes` paths (or explicit `_mae` filenames),
and replication caches identify the metric as `mae_per_coefficient_v1`.
Tuning figures use configuration hashes that include the MAE implementation.
Existing coefficient-error output files outside these paths
are historical results under their original metrics; they are not relabelled
or reused as MAE. Rerun the notebooks to generate the current results.

## Shared empirical-width descent

All empirical projected-subgradient updates use
[`projected_subgradient_empirical_width`](ar_tools/descent.py).
Initial steps are absolute values. Notebook and experiment callers use
`initial_step`; no point-cloud diameter or Lipschitz bound scales the step.

Enable checkpoint-triggered restarts with the boolean `stationarity_check`:

```python
from Code.Code_AR_processes.ar_tools.descent import projected_subgradient_empirical_width

result = projected_subgradient_empirical_width(
    point_cloud,
    iterations=2000,
    patience=200,
    initial_step=0.8,
    step_schedule="geometric",
    geometric_decay=2**(-1/50),
    stationarity_check=True,
    rng=20261019,
)
```

The option defaults to `False`. When enabled, each attempt checks its
best-width direction after exactly `patience` evaluated states, before the
next update. Coefficients are recovered as `-u[1:]/u[0]`, following the
[thesis coefficient-recovery remark](../../Abschlussarbeit_2021/LaTeX/chapters/3_application_to_time_series.tex).
The shared [companion-matrix check](ar_tools/functions.py) passes only when
all eigenvalue moduli are below one. Undefined or nonfinite coefficients fail.

A failed checkpoint discards that attempt's incumbent and draws a fresh
normalized Gaussian direction from `rng`. The next attempt resets the step
schedule, patience counter and full `iterations` budget. After passing the
checkpoint, an attempt continues under ordinary width-based stopping; there
is no further checkpoint or final stationarity filter.

`max_restarts` defaults to 100. If that limit is exhausted, the result is
explicitly marked `termination_reason="stationarity_abort"`,
`restart_limit_reached=True` and `stationarity_check_passed=False`.
Callers must handle that failure rather than treating the returned direction
as an accepted fit. `max_restarts=0` checks a single attempt without drawing
another direction, as used by the fixed twelve-start early-abort comparison.

Top-level histories describe the final attempt. `attempts` retains each
attempt's result and checkpoint; `total_evaluations` includes rejected
attempts, and `restart_count` records fresh draws. Optional direction and
coefficient histories retain their existing recording switches.

The old local checkpoint update loop and the initialization audit's local
vectorized update loop have been replaced by calls to the shared optimizer.
Studies whose step sizes changed have cleared notebook outputs. Restart-study
data and the large coefficient-experiment cache use `absolute_steps/` directories
so historical fitted results cannot be silently reused under the new policy.

## Normalized Beta innovations in the shared AR generators

Shared AR innovation generation uses `ar_tools.innovations.sample_beta_innovations`.
Every supported shape has population mean zero and population variance `1/3`.
The default `alpha=beta=1` is exactly uniform on `[-1,1]`.

The [AR(1) Beta figure notebook](xx_notebooks_figures/ar_beta_sliding_window_2d.ipynb)
is a figure-specific exception requested for the thesis TODO: it draws
`2*Beta(alpha,beta)-1` locally so all five innovation distributions have fixed
support `[-1,1]`. Their means and variances vary with shape. This does not change
the shared samplers or the other experiments.

```python
from Code.Code_AR_processes.ar_tools import (
    beta_innovation_bounds,
    sample_beta_innovations,
)
from Code.Code_AR_processes.ar_tools.functions import simulate_beta_ar

z = sample_beta_innovations(1_000, rng=42)  # Uniform[-1,1]
z_asymmetric = sample_beta_innovations(1_000, alpha=2, beta=5, rng=42)
a, b = beta_innovation_bounds(alpha=2, beta=5)
L = b - a
# a is about -1.0328, b about 2.5820, L about 3.6148.
# Both z distributions have theoretical mean 0 and variance 1/3.

x = simulate_beta_ar(
    [0.7], sample_size=1_000, alpha=2, beta=5, rng=42,
)
```

## Normalization

For positive, finite `alpha,beta`, let `U ~ Beta(alpha,beta)`. The sampler uses

\[
Z=a+(b-a)U
 =L\left(U-\frac{\alpha}{\alpha+\beta}\right),\qquad
L=(\alpha+\beta)\sqrt{\frac{\alpha+\beta+1}{3\alpha\beta}},
\]

\[
a=-\sqrt{\frac{\alpha(\alpha+\beta+1)}{3\beta}},\qquad
b=\sqrt{\frac{\beta(\alpha+\beta+1)}{3\alpha}}.
\]

These are the unique ordered endpoints satisfying `alpha*b + beta*a = 0`
and `Var(Z)=1/3`. The derivation uses the beta mean and variance in
[NIST/SEMATECH, section 1.3.6.6.17, Common Statistics](https://itl.nist.gov/div898/handbook/eda/section3/eda366h.htm).
The value `1/3` matches the uniform baseline. For precedents for standardizing
innovation distributions to a common mean and variance, see
[Vávra (2018), Appendix B, p. 19](https://nbs.sk/_img/documents/publik/wp_3_2018_vavra_forecast_errors_en.pdf#page=19)
and [Herwartz and Wang (2024), section 3.1](https://onlinelibrary.wiley.com/doi/10.1002/jae.3044).

The endpoints and width are **derived values**, not independent controls.
There is no fixed-support or freely scaled innovation mode. Optional legacy
`a,b,L` arguments are accepted only as consistency checks against the derived
values; a mismatch raises `ValueError` before drawing. For `simulate_beta_ar`,
`innovation_bounds=(a,b)` is likewise a consistency check. Do not combine it
with explicit `a,b`.

Centering and variance normalization use the distribution's theoretical moments.
The finite sample is not recentered or rescaled, so its observed mean and variance
will fluctuate. The NumPy primitives are
[Generator.beta](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.beta.html)
and, when `alpha=beta=1`,
[Generator.uniform](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.uniform.html).
The uniform special case preserves existing seeded values and generator state.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `alpha`, `beta` | `1.0`, `1.0` | Positive, finite shape parameters. |
| `size` | `None` | One scalar, an integer count, or a tuple of array dimensions. |
| `rng` | `None` | Random seed or an existing NumPy generator. |
| `a`, `b`, `L` | `None` | Optional consistency checks, not support or scale overrides. |

## Integration and migration

| Entry point | Generation path |
| --- | --- |
| `ar_tools.functions.simulate_beta_ar` | Uses `sample_beta_innovations`, with zero innovation mean in AR history initialization. |
| `ar_tools.functions.simulate_uniform_ar` | Compatibility wrapper selecting `alpha=beta=1` in the same simulator. Its legacy `half_width` must be `1`; other values conflict with variance `1/3`. |
| Shared experiment runners, tuning and order-selection notebooks | Their uniform wrapper calls reach the same normalized sampler. Existing default experiments remain uniform. |
| [ar1_sliding_window_embedding.ipynb](xx_notebooks_figures/ar1_sliding_window_embedding.ipynb) | Calls the sampler directly; its uniform radius is derived from `beta_innovation_bounds()`. Its teaching-example AR recurrence and seeded path are preserved. |
| [ar_beta_sliding_window_2d.ipynb](xx_notebooks_figures/ar_beta_sliding_window_2d.ipynb) | Figure-specific fixed-support Beta draws on [-1,1], following the thesis TODO. Exports separate time-series panels, clouds with a common support boundary, and a colored overlay; does not call the normalized shared sampler. |
| Analysis and research scripts | Use the same shared simulators. Some restart studies load already-saved uniform paths instead of generating new ones. |

The previous default for nonuniform shapes kept support width 2. It is now
replaced by the paragraph's common-variance normalization. For example,
`sample_beta_innovations(..., alpha=2, beta=5, L=2)` now raises an error; omit
`L` and let it be derived. An arbitrary explicit interval can no longer bypass
centering or variance normalization. All-default uniform experiments retain their
original distribution and seeded paths. Saved experiment inputs are not regenerated.

Beta draws in `jones_ar_coefficients` sample partial autocorrelations, not
innovations, and remain separate. Gaussian optimizer-direction draws are also
unrelated to AR innovation generation. Geometry helpers may still analyze
arbitrary intervals; they do not generate processes.

Restart an existing notebook kernel after updating the shared Python modules.
The Beta figure notebook uses its own fixed-support sampling cell. No thesis files
are edited by these code changes.

## Historical normalization verification (20 October 2026)

`tests/test_beta_innovations.py` checks the theoretical endpoint conditions,
full transformed beta distributions, sampled means and variances across shapes,
AR residuals and stationary means, all sampler entry points, rejection of overrides,
legacy consistency checks, scalar/array sampling, and notebook support overlays.
It also checks that the uniform generator state and AR(1) draws are preserved.

Run the AR tests from the repository root:

```sh
MPLBACKEND=Agg python3 -m unittest discover -s Code/Code_AR_processes/tests -v
```

At the time of this verification, the pre-change suite had one unrelated failure in
`test_notebooks_have_no_local_function_definitions`: the now-removed notebook
`ar3_failed_coefficient_tracking.ipynb` defined a local `save_figure`.
It was not part of the innovation generation implementation.

All 13 normalization tests pass. The full AR suite has 60 passes and that same
pre-existing failure (61 tests total). Twenty-four uniform AR paths match the
previous implementation bit for bit. The audit found no innovation sampling
bypass among the current generators; all 19 AR notebooks parse. Hashes confirm
that all 24 protected thesis files are unchanged.

At the time of that verification, an additional comparison with historical saved
paths was unavailable because `inputs/ar3_coefficient_recovery_50.npz` was absent.
No historical data were regenerated to replace it. The saved input is now
[available](inputs/ar3_coefficient_recovery_50.npz); this does not change the
historical verification result.

See the [verification record](../../output/analysis/ar_beta_variance_normalization_2026-10-20/verification.json),
[test log](../../output/analysis/ar_beta_variance_normalization_2026-10-20/test_results.txt),
and the current [fixed-support Beta notebook figure](outputs/pdfs/ar_beta_sliding_window_2d/beta_sliding_window_clouds_2d.pdf).
