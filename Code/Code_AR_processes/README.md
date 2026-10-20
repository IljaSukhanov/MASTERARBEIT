# Normalized Beta innovations for every AR generator

All AR innovation generation uses `ar_tools.innovations.sample_beta_innovations`.
Every supported shape has population mean zero and population variance `1/3`.
The default `alpha=beta=1` is exactly uniform on `[-1,1]`.

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
| `ar1_sliding_window_embedding.ipynb` | Calls the sampler directly; its uniform radius is derived from `beta_innovation_bounds()`. Its teaching-example AR recurrence and seeded path are preserved. |
| `ar_beta_sliding_window_2d.ipynb` | Every displayed shape is normalized; each support overlay uses that shape's computed endpoints. Fixed-support and free-width settings have been removed. |
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
The beta comparison notebook also refreshes its shared sampling module when its
plotting cell is rerun. Its stale embedded outputs are cleared. No thesis files
are edited by this change.

## Verification

`tests/test_beta_innovations.py` checks the theoretical endpoint conditions,
full transformed beta distributions, sampled means and variances across shapes,
AR residuals and stationary means, all sampler entry points, rejection of overrides,
legacy consistency checks, scalar/array sampling, and notebook support overlays.
It also checks that the uniform generator state and AR(1) draws are preserved.

Run the AR tests from the repository root:

```sh
MPLBACKEND=Agg python3 -m unittest discover -s Code/Code_AR_processes/tests -v
```

The pre-change suite has one unrelated failure in
`test_notebooks_have_no_local_function_definitions`: the notebook
`ar3_failed_coefficient_tracking.ipynb` already defines a local `save_figure`.
It is not part of the innovation generation implementation.

All 13 normalization tests pass. The full AR suite has 60 passes and that same
pre-existing failure (61 tests total). Twenty-four uniform AR paths match the
previous implementation bit for bit. The audit found no innovation sampling
bypass among the current generators; all 19 AR notebooks parse. Hashes confirm
that all 24 protected thesis files are unchanged.

An additional comparison with historical saved paths was unavailable because
`inputs/ar3_coefficient_recovery_50.npz` is absent in the current workspace.
No historical data were regenerated to replace it.

See the [verification record](../../output/analysis/ar_beta_variance_normalization_2026-10-20/verification.json),
[test log](../../output/analysis/ar_beta_variance_normalization_2026-10-20/test_results.txt),
and [normalized Beta notebook preview](../../output/analysis/ar_beta_variance_normalization_2026-10-20/beta_sliding_window_clouds_2d.png).
