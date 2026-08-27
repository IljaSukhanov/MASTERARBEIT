# KDE code guide

The supported experiment code uses analytic Gaussian formulas whenever they
are available and ordinary spatial quadrature otherwise. Numerical work lives
in `kde_tools/`:

- `density.py`: direct KDE evaluation, quadrature grids, and exact
  Gaussian/Gaussian-mixture integrals.
- `bandwidths.py`: fixed and sample-scaled bandwidth grids plus the Silverman,
  LSCV, and plug-in rules.
- `similarities.py`: beta-2 and KL divergence matrices; exponential
  similarities; Perron vectors; and spectral selection.
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
- `distance_matrix_perron_vector_tests.ipynb`: distance-versus-similarity and
  Perron/Doob investigations.

Run the lightweight checks from this directory with:

```bash
python3 -m unittest discover -s tests
```
