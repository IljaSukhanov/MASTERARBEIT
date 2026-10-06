# KDE code guide

The supported experiment code uses analytic Gaussian formulas whenever they
are available and composite trapezoidal integration for TV, Hellinger, and JS.
Numerical work lives in `kde_tools/`:

- `density.py`: direct KDE evaluation, quadrature grids, and exact
  Gaussian/Gaussian-mixture integrals.
- `quadrature.py`: shared spatial grids and composite trapezoidal integration.
  `quadrature_js_divergence_matrix(density_values, x_grid)` returns raw JS
  in natural-log units from cached density values. The `js_similarity` option
  returns `1 - JS/log(2)`; use the raw divergence for `exp(-lambda * JS)`.
  TV, Hellinger, and JS use the same finite spatial interval and can reuse
  KDE evaluations. Increase spatial resolution and interval padding to check
  convergence; these routines do not return a total error certificate.
  The integration rule is described in the
  [SciPy composite trapezoidal documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.trapezoid.html).
- `bandwidths.py`: fixed bandwidth grids plus the Silverman, LSCV, and plug-in
  rules.
- `similarities.py`: beta-2 and KL divergence matrices; exponential
  similarities parameterized as `exp(-lambda_value * D)`, where the thesis
  notation is $\lambda=1/\tau$; Perron vectors; and spectral selection.
- `experiments.py`: one-repetition simulation runners.  Notebooks control the
  settings, repetition loops, summaries, and plots.

The optional Fourier implementation is isolated in `kde_tools/fourier/` and is
never imported by the main package or experiment runners. Import it explicitly
with, for example:

```python
from kde_tools.fourier import kde_family_fft
```

The main exploratory workflow is `spectral_scale_selection_kde.ipynb`. Its
beta-2 experiments use the optional Fourier implementation explicitly, while
the reusable numerical APIs remain in `kde_tools/`.

- `100_kde_test.ipynb`: early KDE and similarity experiments.
- `gaussian_symmetric_divergence_bandwidth_distributions.ipynb`: repeated
  Gaussian-sample bandwidth distributions for `exp(-lambda * D)` with
  L², TV, H², JS, and the sample-specific ISE choice. Includes preview and
  Figure 3.5 parameter profiles, resumable checkpoints, CSV results, and
  automatic PDF exports.
- `distance_matrix_perron_vector_tests.ipynb`: distance-versus-similarity and
  Perron/Doob investigations.

Run the lightweight checks from this directory with:

```bash
python3 -m unittest discover -s tests
```
