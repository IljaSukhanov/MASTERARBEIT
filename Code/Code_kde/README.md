# KDE code guide

The supported experiment code uses analytic Gaussian formulas whenever they
are available and ordinary spatial quadrature otherwise. Numerical work lives
in `kde_tools/`:

- `density.py`: direct KDE evaluation, quadrature grids, and exact
  Gaussian/Gaussian-mixture integrals.
- `bandwidths.py`: bandwidth grids and the Silverman, LSCV, and plug-in rules.
- `similarities.py`: divergence/similarity matrices, Perron vectors, Toeplitz
  corrections, and spectral selection.
- `experiments.py`: one-repetition simulation runners.  Notebooks control the
  settings, repetition loops, summaries, and plots.

The optional Fourier implementation is isolated in `kde_tools/fourier/` and is
never imported by the main package or experiment runners. Import it explicitly
with, for example:

```python
from kde_tools.fourier import kde_family_fft
```

The two `run_*.py` programs use direct Gaussian KDE evaluation followed by
composite trapezoidal quadrature. The beta-2/L2 selector uses the exact analytic
Gaussian formula by default.

The notebooks are exploratory records. In particular,
`spectral_scale_selection_kde.ipynb` contains historical and optional Fourier
experiments; it is not the implementation used by the default runners.

- `100_kde_test.ipynb`: early KDE and similarity experiments.
- `distance_matrix_perron_vector_tests.ipynb`: distance-versus-similarity and
  Perron/Doob investigations.
- `renyi_divergence_doob_kde.ipynb`: Rényi and Doob-transform experiments.

The two `run_*.py` files are reproducible command-line Monte Carlo experiments.
They reuse the exact/quadrature helpers in `kde_tools` rather than maintaining
separate numerical implementations.

Run the lightweight checks from this directory with:

```bash
python3 -m unittest discover -s tests
```
