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
- `similarities.py`: beta-2 divergence matrices; exponential
  similarities parameterized as `exp(-lambda_value * D)`, where the thesis
  notation is $\lambda=1/\tau$; compatibility entry points and spectral selection.
- `perron.py`: the single symmetric Perron implementation used by every
  active KDE experiment. It constructs shifted off-diagonal exponents for
  exponential similarities, uses SciPy's symmetric Lanczos solver, and checks
  the eigenpair residual and spectral gap. The scalar and 2-by-2 cases are
  handled in the same module because ARPACK requires fewer requested
  eigenpairs than matrix rows. Algorithm reference:
  [SciPy eigsh documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html).
- `experiments.py`: one-repetition simulation runners.  Notebooks control the
  settings, repetition loops, summaries, and plots.

The optional Fourier implementation is isolated in `kde_tools/fourier/` and is
never imported by the main package. FFT experiment runners import it explicitly;
notebooks can import it with, for example:

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
- `experiment_1_joint_N_lambda_limit.ipynb`: joint grid/localization limits,
  mesh diagnostics, and repeated-sampling experiments.
- `fast_l2_bandwidth_distributions_vs_n_lambda.ipynb`: FFT-based L²
  distributions with shared divergence-based spectral selection.
- `spectral_lambda_jump_study.ipynb`: 1,000-sample Figure 3.3 lambda sweeps,
  adjacent-pair histograms of the largest upward index changes, jump-size
  diagnostics, and resumable raw-sample/trajectory checkpoints.
- `gaussian_hellinger_bandwidth_distributions_vs_n.ipynb`: H² sample-size
  experiments with versioned checkpoints.

## Central Perron API

For `S_ij = exp(-lambda * D_ij)`, pass the divergence itself, before
exponentiation can round the similarity to the identity:

```python
from kde_tools.perron import perron_eigenpair_from_divergence

result = perron_eigenpair_from_divergence(divergence, lambda_value)
vector = result.vector
selected_index = result.selected_index
```

The shared transformation is
`B_ii = 0`, `B_ij = exp(-lambda * (D_ij - d_min))` for `i != j`.
Subtracting the identity and multiplying by a positive scalar preserve
the eigenvectors. See Section 3.5.1 in
[`5_bandwidth_selection.tex`](../../Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex).
For diagnostics, call `perron_eigenpair_from_divergence`; its `PerronResult`
includes the vector, two leading eigenvalues of B, spectral gap, relative
residual, and Lanczos matrix-vector count. `which="LA"` selects the largest
algebraic eigenvalues even when the closest-pair limit has a negative
eigenvalue of the same magnitude.

Peak selection uses a shared numerical tie convention: components within
64 machine epsilons times the largest absolute weight count as tied, and
the smallest grid index wins. This makes the two equal closest-pair weights
consistent across runs without changing the computed eigenvector.
Vector-only code uses `perron_peak_index(vector)` for the same convention.

Already assembled symmetric affinities and distance matrices use
`perron_eigenpair` or the compatible `perron_vector` entry point. They share
the same backend; an already underflowed exponential matrix cannot recover
the divergence information. Symmetric Doob masses use
`doob_mass_from_perron(vector)`, avoiding separate left/right solves.
`first_order_perron_vector` remains an explicitly labeled mathematical
approximation for comparison plots, not an alternative numerical solver.

Restart existing notebook kernels after this refactor and rerun setup.
Cached notebook displays have been cleared because they belong to the old
solvers. Existing exported PDFs and CSVs are retained as historical results;
the expensive full Monte Carlo runs have not been regenerated by the refactor.
Resumable experiment configurations include `PERRON_SOLVER_VERSION`, and the
joint notebook writes new sampling results into a solver-version directory.
Old checkpoints therefore cannot be silently mixed with new solver results.

Run the lightweight checks from this directory with:

```bash
python3 -m unittest discover -s tests
```

For reduced numerical runs through all six notebooks, including checkpoint
resuming and the original 101-point Figure 3.3 sweep, run:

```bash
MPLBACKEND=Agg python3 tests/validate_notebook_experiments.py
```

The validation runner overrides expensive sampling settings only in memory,
keeps historical figures and data intact, and writes its report under
`outputs/analysis/perron_refactor_validation/`.
