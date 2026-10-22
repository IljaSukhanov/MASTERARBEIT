# Direct frequency quadrature versus the existing inverse-FFT route

Date: 22 October 2026.

For the half-squared L2 distance matrices studied here, I recommend implementing the thesis's frequency integral directly as a weighted real Gram product. In the local comparison below this was faster than reconstructing KDEs first, while both routes closely matched the exact Gaussian pair formula in the small accuracy checks. This recommendation is limited to the tested settings and this L2 calculation.

The prototype is isolated in [benchmark.py](/Users/iljasukhanov/MASTERARBEIT/output/analysis/kde_fourier_efficiency_2026-10-22/benchmark.py). No production KDE helper, notebook, figure, protected thesis file or existing experiment result was changed. Full measurements and settings are in [results.json](/Users/iljasukhanov/MASTERARBEIT/output/analysis/kde_fourier_efficiency_2026-10-22/results.json).

## Mathematical implementation

This factorization is our algebraic derivation from the [thesis's frequency sum](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:331). Let

\[
 w_k=\frac{\Delta\omega}{2\pi}c_k,
 \qquad c_0=c_K=1,\quad c_k=2\ (0<k<K),
\]

and define the real feature matrix

\[
 B_{rk}=\sqrt{w_k}\,|P_n(\omega_k)|
          \exp\!\left(-\frac12h_r^2\omega_k^2\right).
\]

Then

\[
 G^{\rm quad}=BB^T,
 \qquad
 D_{rs}=\frac12G^{\rm quad}_{rr}
       +\frac12G^{\rm quad}_{ss}-G^{\rm quad}_{rs}.
\]

This is precisely the chapter's composite trapezoidal approximation with the stated endpoint weights. The data spectrum is computed once per sample, and the Gaussian vectors once per bandwidth. Matrix multiplication then assembles the complete Gram matrix. NumPy's matrix product uses an optimized BLAS library when possible; see the [official Notes](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html#notes).

## Operation counts

The counts below are our analysis of the two algorithms, using ordinary matrix multiplication and the usual O(m log m) FFT cost; the latter is described, for example, in the [FFTW introduction](https://www.fftw.org/fftw3_doc/Introduction.html). They do not assert that NumPy uses FFTW.

Let n denote observations, N bandwidths, m spatial points, and q frequency nodes. The matched benchmark grids have q=m/2+1.

| Route | Work |
|---|---|
| Existing reconstruction and spatial Gram | O(nq + Nm log m + N²m) |
| Direct frequency Gram | O(nq + Nq + N²q) |

Both routes directly evaluate the empirical characteristic function in O(nq). The direct route omits the inverse FFTs, density clipping/renormalization and the N-by-m density array, and the Gram product uses q frequency columns rather than m spatial columns. The current route is documented by [reconstruction.py](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/kde_tools/fourier/reconstruction.py:35) and [similarities.py](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/kde_tools/fourier/similarities.py:18).

## Measured runtime

Both routes used H=12, m=8192, q=4097, and a geometric bandwidth grid from 0.01 to 2. Each timing includes empirical-characteristic-function evaluation and full distance-matrix construction. Medians of seven repetitions after two warmups are reported; method order alternated. No eigensolver, plotting or checkpoint I/O was timed. The observations were normal samples with a recorded seed. This is a local benchmark, not a hardware-independent speed guarantee. [Full measurements](/Users/iljasukhanov/MASTERARBEIT/output/analysis/kde_fourier_efficiency_2026-10-22/results.json).

| n | N | Existing route | Direct frequency route | Speedup |
|---|---|---|---|---|
| 200 | 201 | 55.2 ms | 35.6 ms | 1.55× |
| 1000 | 201 | 160.2 ms | 139.5 ms | 1.15× |
| 200 | 1000 | 205.9 ms | 88.8 ms | 2.32× |

BLAS thread-limit environment variables were set to one before NumPy import for both routes. The backend did not expose its runtime thread count to threadpoolctl, so the requested limit is recorded without claiming independent confirmation.

## Accuracy and remaining scope

Three separate Gaussian samples with n=100 and 20 bandwidths were compared with the shared exact Gaussian formula. The largest absolute distance-matrix errors were 2.24e-14 for the existing route and 1.86e-14 for direct frequency quadrature. In the larger timed cases, the largest difference between the two routes was 1.14e-14. [Validation entries and timings](/Users/iljasukhanov/MASTERARBEIT/output/analysis/kde_fourier_efficiency_2026-10-22/results.json).

The direct prototype follows the continuous-integrand trapezoidal formula literally. It does not reproduce the existing helper's density clipping/renormalization or its real Nyquist coefficient convention. Their effects were small in these comparisons. The algebraic relationship, including the endpoint difference, is derived in [the research note](/Users/iljasukhanov/MASTERARBEIT/output/research/kde_l2_fourier_sources_2026-10-22.md:65).

Frequency cutoff and spacing still control numerical accuracy. These checks do not establish agreement of every selected bandwidth or validate all previously saved experiments. Before changing the production experiment route, selected-bandwidth agreement and convergence should be checked for the settings being replaced. That migration was not performed for this question.
