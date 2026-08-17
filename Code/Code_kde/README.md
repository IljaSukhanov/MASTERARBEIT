# KDE code guide

Start with `spectral_scale_selection_kde.ipynb`.  It is the main, explanatory
notebook.  Numerical work used by that notebook lives in `kde_tools/`:

- `density.py`: direct KDE evaluation, quadrature grids, and exact
  Gaussian/Gaussian-mixture integrals.
- `fft.py`: optional FFT evaluation for the large periodic-grid experiments.
- `bandwidths.py`: bandwidth grids and the Silverman, LSCV, and plug-in rules.
- `similarities.py`: divergence/similarity matrices, Perron vectors, Toeplitz
  corrections, and spectral selection.
- `experiments.py`: one-repetition simulation runners.  Notebooks control the
  settings, repetition loops, summaries, and plots.

The other notebooks are exploratory records:

- `100_kde_test.ipynb`: early KDE and similarity experiments.
- `distance_matrix_perron_vector_tests.ipynb`: distance-versus-similarity and
  Perron/Doob investigations.
- `renyi_divergence_doob_kde.ipynb`: Rényi and Doob-transform experiments.

The two `run_*.py` files are reproducible command-line Monte Carlo experiments.
They reuse `kde_tools` rather than maintaining separate numerical implementations.

Run the lightweight checks from this directory with:

```bash
python3 -m unittest discover -s tests
```
