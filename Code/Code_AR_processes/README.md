# AR innovations

All AR innovation draws use `ar_tools.innovations.sample_beta_innovations`.
The sampler is also available from `ar_tools` and `ar_tools.functions`.
The default is uniform on `[-1, 1]`.

```python
from Code.Code_AR_processes.ar_tools import (
    beta_innovation_bounds,
    sample_beta_innovations,
)
from Code.Code_AR_processes.ar_tools.functions import simulate_beta_ar

z = sample_beta_innovations(1_000, rng=42)  # Uniform[-1, 1]

# Centered, asymmetric Beta innovations with interval width L=2.
z = sample_beta_innovations(1_000, alpha=2, beta=5, L=2, rng=42)
a, b = beta_innovation_bounds(alpha=2, beta=5, L=2)
# a=-4/7, b=10/7; the population mean is zero.

# Explicit endpoints request fixed support, without automatic centering.
z = sample_beta_innovations(1_000, alpha=2, beta=5, a=-1, b=1, rng=42)
# The population mean here is -3/7.

# Use exactly the same parameterization in an AR simulation.
x = simulate_beta_ar(
    [0.7], sample_size=1_000, alpha=2, beta=5, L=2, rng=42,
)
x_fixed = simulate_beta_ar(
    [0.7], sample_size=1_000, alpha=2, beta=5, a=-1, b=1, rng=42,
)
```

## Parameter rules

| Parameter | Default | Meaning |
| --- | --- | --- |
| `alpha`, `beta` | `1.0`, `1.0` | Positive, finite beta shape parameters. |
| `a`, `b` | Both omitted (`None`) | Omit both for mean-zero innovations; supply both for fixed support. |
| `L` | `None`, resolved to `2.0` without endpoints | Support width. With explicit endpoints it is inferred as `b-a`. |
| `rng` | `None` | Seed or an existing NumPy generator. |
| `size` | `None` | One scalar, an integer sample count, or a tuple of array dimensions. |

With no endpoints, the implementation uses the following derived construction:

\[
U\sim\operatorname{Beta}(\alpha,\beta),\qquad
Z=L\left(U-\frac{\alpha}{\alpha+\beta}\right),\qquad
[a,b]=\left[-L\frac{\alpha}{\alpha+\beta},\ L\frac{\beta}{\alpha+\beta}\right].
\]

With endpoints, it uses `Z=a+(b-a)*U`, with mean
`a+(b-a)*alpha/(alpha+beta)`. These formulas follow from the standard beta mean
and affine transformation; see [NIST/SEMATECH, section 1.3.6.6.17,
Probability Density Function and Mean](https://itl.nist.gov/div898/handbook/eda/section3/eda366h.htm).
An explicit interval does not enforce mean zero. The parameters `a,b,L` are not
independent: when all are supplied, inconsistent `L != b-a` is rejected. Supplying
just one endpoint is also rejected.

Centering uses the theoretical mean, not the observed sample mean. A finite
sample therefore need not average to exactly zero. The sampling primitives are
NumPy's [Generator.beta](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.beta.html)
and, for `alpha=beta=1`, [Generator.uniform](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.uniform.html).
The uniform special case preserves the existing seeded draws and generator state.

## Integration and duplicate-code review

The review covered all Python sources and all 19 notebooks under this directory,
including `xx_notebooks_tuning/` and `xx_notebooks_order_selection/`. The sampling
paths are:

| Entry point | Shared sampling path |
| --- | --- |
| `ar_tools.functions.simulate_beta_ar` | Calls `sample_beta_innovations`; contains the shared AR recurrence. |
| `ar_tools.functions.simulate_uniform_ar` | Compatibility wrapper for `simulate_beta_ar` with `alpha=beta=1`. |
| `ar_tools.experiments` and the other AR notebooks | Already call the two shared simulators and inherit the new sampler. |
| `ar1_sliding_window_embedding.ipynb` | Its former direct uniform draw now calls `sample_beta_innovations`; its explicit teaching-example recurrence and seeded path are preserved. |
| `ar_beta_sliding_window_2d.ipynb` | Defaults to centered innovations; can select fixed endpoints. Each panel's theoretical support uses its actual interval. |

This unifies the standalone AR(1) innovation draw with the shared beta and uniform
sampling implementation. The uniform simulator was already a wrapper, so it was
not a duplicate implementation. That wrapper remains to preserve existing call
sites. Beta draws in `jones_ar_coefficients` sample partial autocorrelations,
not innovations, and are intentionally separate. Random optimizer directions are
also unrelated to innovation sampling.

**Behavior change:** `simulate_beta_ar(alpha=..., beta=...)` with no endpoint
arguments now keeps the population mean zero and lets the endpoints move.
For the previous fixed-support behavior, pass `a=-1, b=1` or the supported older
argument `innovation_bounds=(-1, 1)`. Do not combine `innovation_bounds` with `a,b`.
All-default calls and `simulate_uniform_ar` retain uniform innovations on
`[-1,1]`. Explicit fixed-support calls retain their interval and possible
nonzero mean. AR history initialization accounts for that mean.

No thesis sources or historical experiment data were edited. The beta notebook's
old embedded outputs were cleared because its default mode has changed.

## Verification

`tests/test_beta_innovations.py` covers the uniform random stream, centered and
fixed-support beta distributions, support and moments, the full transformed CDF,
AR residuals and means, generator handling, parameter validation, and both beta
notebook modes. It also checks that the AR(1) notebook retains its previous draws.

Run the AR tests from the repository root:

```sh
MPLBACKEND=Agg python3 -m unittest discover -s Code/Code_AR_processes/tests -v
```

The pre-change suite had one unrelated failure:
`test_notebooks_have_no_local_function_definitions`, because
`ar3_failed_coefficient_tracking.ipynb` already defines a local `save_figure`.
That notebook is outside the sampling changes.

Final validation: all 17 innovation tests pass; the full AR suite has 64 passes
and that same pre-existing failure (65 tests total). Twenty-four seeded uniform
AR paths match the pre-change implementation bit for bit. All 19 notebooks parse,
and hashes confirm that the 11 protected thesis source files are unchanged.
See the [verification record](../../output/analysis/ar_beta_innovation_unification_2026-10-20/verification.json)
and [full test log](../../output/analysis/ar_beta_innovation_unification_2026-10-20/test_results.txt).
