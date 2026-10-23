# L2 migration to the thesis frequency quadrature

Implemented on 23 October 2026. Protected thesis files and historical exported
figures/results were not modified. Full Monte Carlo experiments were not rerun.

## Change

The shared L2 implementation now forms the real weighted feature matrix

\[
B_{rk}=\sqrt{w_k}\,|P_n(\omega_k)|\exp(-h_r^2\omega_k^2/2),
\qquad G=BB^T,
\]

with composite trapezoidal weights including the factor 1/pi for integration
on the positive frequency axis. Half squared L2 distances are assembled from G.
This is an algebraic factorization of the existing thesis approximation, after
`eq:beta2_exact_formula02`; the mathematical basis is
[Parseval's identity, DLMF 1.14.7₅](https://dlmf.nist.gov/1.14.E7_5).
See [the implementation](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/kde_tools/fourier/similarities.py:13).

There are no inverse FFTs, spatial-density clipping or renormalization in this
L2 helper. `kde_family_fft` remains available for diagnostics requiring spatial
density values. The combined sample-size ISE/Q-hat diagnostic retains those
spatial values for its other criteria, while its L2 Gram uses direct quadrature.

Public names are `gaussian_kde_l2_gram_fourier`,
`beta2_divergence_matrix_fourier`, and `beta2_similarity_matrix_fourier`.
Old `_fft` L2 entry points are compatibility wrappers for the new calculation.
Existing half-width and point-count parameters preserve the previous frequency
spacing and cutoff. The Gram helper also accepts an explicit cutoff and node count.

Eight active notebooks now call the Fourier-quadrature API and identify the
method as `frequency_trapezoid_v1`. New checkpoint fingerprints/directories
separate these calculations from historical FFT runs. Cached notebook cell
outputs were cleared so old displays are not presented as new calculations.

## Validation

- All eight migrated notebook setup cells execute, and all code cells parse.
  [Setup report](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/outputs/analysis/fourier_quadrature_migration/setup_validation_report.json).
- Three small exact-formula checks give a maximum absolute L2-distance error
  of 1.86e-14 for the new implementation. All 40 selected-bandwidth comparisons
  agree between the new method, the previous reconstruction method, and a grid
  with halved frequency spacing. These cover eight sample/grid cases and five
  lambda values per case, including n=1000 and N=1000 cases.
  [Numerical report](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/outputs/analysis/fourier_quadrature_migration/migration_report.json).
- Local median distance-matrix construction speedups were 1.54x for
  n=200/N=201, 1.14x for n=1000/N=201, and 2.40x for n=200/N=1000.
  These timings exclude eigensolvers and notebook I/O and are specific to the
  recorded runtime; they are not whole-notebook speed guarantees.
- 69 unit tests pass, including the direct trapezoidal endpoint formula,
  narrow-bandwidth cutoff convergence, translation invariance, compatibility,
  and input checks. Four reduced notebook workflows pass, including the
  101-point Figure 3.3 selection path, checkpoint resuming, and rejection of
  legacy solver/L2-method checkpoints.
  [Notebook validation report](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/outputs/analysis/fourier_quadrature_migration/notebook_validation_report.json).

The unfiltered suite runs 71 test methods and still reports four pre-existing
subtest failures in two solver-routing test methods: two historical scripts and
two calls in the eigenvalue notebook use `eigh` for reference checks. An AST
comparison against HEAD confirms that the flagged calls are unchanged and no
new forbidden calls were introduced.
[Baseline audit](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/outputs/analysis/fourier_quadrature_migration/routing_baseline_audit.json).
The four-workflow report excludes those two known-failing routing methods only
in its invocation harness; the source routing tests remain unchanged.

## Use

Restart notebook kernels and run from setup. New sampling runs use the new
method identity; historical outputs have not been relabeled or automatically
regenerated. The migration checks do not certify every saved experiment or
arbitrary choices of frequency cutoff and spacing.
