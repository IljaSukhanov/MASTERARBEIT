# KDE code guide

The active notebooks focus on L² (beta-2) spectral bandwidth selection,
continuous sample-energy comparisons, and ISE benchmarks, using exact
Gaussian formulas or the optional Fourier implementation. Experiments with
other divergences have been removed from
the notebooks and the notebook validation runner. Shared numerical helpers
and historical outputs are retained for optional future work.
Numerical work lives in `kde_tools/`:

- `energy_minimization.py`: the single continuous search used by all Q-r
  experiments, including sample energies, Gaussian expected energies and the
  continuous ISE comparison. It refines derivative roots, classifies detected
  interior extrema, compares every detected minimum with both endpoints, and
  checks agreement after scan refinement. `EnergySearchConfig` holds the central
  defaults for scan resolution, exact/Fourier candidate discovery, root and
  comparison tolerances, and refinement limits.
- `density.py`: direct KDE evaluation, quadrature grids, and exact
  Gaussian/Gaussian-mixture integrals.
- `quadrature.py`: shared spatial grids and composite trapezoidal integration,
  including helpers retained for optional divergence experiments.
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
- `perron.py`: the single symmetric Perron implementation used by active
  spectral KDE experiments. It constructs shifted off-diagonal exponents for
  exponential similarities, uses SciPy's symmetric `eigh` solver, and checks
  the eigenpair residual and spectral gap. For matrices with at least two rows,
  `subset_by_index=[N-2, N-1]` requests the two largest algebraic eigenpairs.
  Algorithm reference:
  [SciPy eigh documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html).
- `distribution_runs.py`: shared checkpointed Figure 3.5, localized, and exact-Gaussian
  histogram experiments, with exact comparison selectors and eigenpair diagnostics.
- `spectral_sweeps.py`: shared, resumable spectral sample-size sweeps for
  the main, fast, and joint-limit notebooks. Matching numerical inputs reuse one cache;
  the joint-limit view also stores eigenpair diagnostics.
- `continuous_runs.py`: shared Q-hat/Q₀ and continuous ISE/Q-r sampling runner,
  with a cache under `outputs/data/shared_continuous_minimizers/`. It verifies
  actual sample prefixes and search settings and records the producer version
  for imported results. Both continuous notebook views retain their CSVs,
  numerical refinement checks, and figures.
- `experiments.py`: one-repetition simulation runners.  Notebooks control the
  settings, repetition loops, summaries, and plots.
- `qhat.py`: analytic Gaussian sample energies and `minimize_qhat`, the public
  Q-r entry point to the shared search. The historical `continuous_qhat_minimum`
  name is a compatibility view, not a separate algorithm. The exact Gaussian
  expected-energy objective uses the same one-half normalization as the sample
  criterion; older joint-limit notebook formulas had an extra constant factor
  of two, which did not change their minimizer locations.
- `continuous_minimizers.py`: continuous ISE and Q-r minimizers in the thesis's
  r notation, sharing the analytic pair calculations and comparing both
  interval endpoints.

## Control Q-r minimization

Start with
[`qhat_minimization_control.ipynb`](xx_notebooks_continuous_minimizers/qhat_minimization_control.ipynb).
It runs the shared function on the saved fixed sample and displays interval
and interior minima together, exports the full reports as JSON and CSV, and
automatically saves its plot as PDF.

```python
from kde_tools import EnergySearchConfig, minimize_qhat

controls = EnergySearchConfig(scan_points=513, scan_backend="exact")
report = minimize_qhat(samples, h_min=0.01, h_max=2.0, r=13/14, config=controls)
report["global_minimum"]       # Lowest energy among detected minima and endpoints.
report["local_minima"]         # Each includes is_global_minimum.
report["best_local_minimum"]   # None when no interior local minimum was detected.
report["endpoints"]
report["diagnostics"]          # Actual controls, resolution agreement and evaluations.
```

Change the defaults in `EnergySearchConfig` in `kde_tools/energy_minimization.py`
to change shared controls across experiments, then restart notebook kernels.
Explicit notebook/function arguments override individual defaults and are
recorded in each report. `scan_backend="exact"` scans analytic pair derivatives;
`"fourier"` uses an approximation only to propose candidate brackets. Every
reported energy, root and endpoint comparison uses the exact objective.
Refinement uses [SciPy brentq](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html).
`comparison_rtol` and `comparison_atol` define when candidates count as tied.

The interval result and all interior minima are computed in one analysis.
Experiments displaying local-minimum distributions select `best_local_minimum`
and retain missing outcomes when it is `None`; interval comparisons select
`global_minimum`. A finite scan does not certify exhaustive root isolation:
`diagnostics["certified_global"]` is always false, and
`diagnostics["resolution_stable"]` records the numerical resolution check.

New continuous caches include the shared search version and controls. The local
experiment runners write to `outputs/data/<experiment>/unified_<settings-id>/`
and the pre-unification energy results have been removed. Raw inputs are kept
in `inputs/`, independently of generated results. Shared sampling runs also export
`n<N>_energy_analysis.json`, containing the endpoint/local/global reports for
each newly computed sample and method. Selection-only historical imports do
not contain those additional reports. The exact-grid Q-hat benchmark still
selects on its prescribed grid using the shared Gaussian energy formula. The
overview notebook's Fourier-based grid controls also retain their original
discrete definition. These grid comparisons are separate from continuous
interval minimization.

The optional Fourier implementation is isolated in `kde_tools/fourier/` and is
imported explicitly by Fourier experiment runners. L² inner products now follow
the thesis's direct frequency integral, using composite trapezoidal quadrature
and a real weighted Gram product `G = B @ B.T`. There is no inverse FFT, density
clipping or density renormalization in this L² route. The mathematical reference
is the numerical approximation after `eq:beta2_exact_formula02` in
[`5_bandwidth_selection.tex`](../../Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex);
see also [Parseval's formula, DLMF 1.14.7₅](https://dlmf.nist.gov/1.14.E7_5).

```python
from kde_tools.fourier import (
    FOURIER_L2_METHOD_VERSION,
    beta2_divergence_matrix_fourier,
    gaussian_kde_l2_gram_fourier,
)

divergence = beta2_divergence_matrix_fourier(
    samples, h_grid  # Omega=450, K=1536 intervals (1537 frequency nodes).
)
```

The existing `half_width=H` and `spatial_points=m` parameters are retained as
frequency-grid controls: spacing `pi/H`, cutoff `pi*m/(2*H)`, and `m/2+1` nodes.
All active L² experiments use the shared defaults `DEFAULT_FREQUENCY_CUTOFF=450`
and `DEFAULT_FREQUENCY_INTERVALS=1536`, defined in `kde_tools/fourier/similarities.py`.
Their compatibility controls are `H=pi*1536/450` and `m=3072`; notebooks import
these defaults rather than repeating numerical grid settings.
For an independently chosen cutoff and node count, call
`gaussian_kde_l2_gram_fourier(samples, h_grid, frequency_cutoff=..., frequency_points=...)`.
This helper also defaults to cutoff 450 and 1537 nodes. The cutoff/K diagnostic
notebooks retain their explicit comparison grids. Separate spatial density,
continuous ISE, and Q-hat search grids keep their own controls.
The old `beta2_divergence_matrix_fft` and `beta2_similarity_matrix_fft` names
remain compatibility wrappers for direct quadrature. `kde_family_fft` still
performs periodic reconstruction when density values are needed for other
diagnostics; it is not used by the L² helper.

`FOURIER_L2_METHOD_VERSION` identifies the new calculation as
`frequency_trapezoid_v1`. Migrated notebooks include it in checkpoint settings
or use method-specific output directories. Existing exports and saved FFT
checkpoints remain historical; new sampling runs do not silently resume them.
Restart notebook kernels and rerun setup after updating the helpers. Cached cell
outputs in the migrated notebooks have been cleared rather than relabeled.
The Omega=450/K=1536 update changes configuration identities and adds a grid
namespace where outputs previously depended only on the numerical method.
Existing results and checkpoints from other frequency grids remain historical.

## Notebook themes

The notebooks are grouped into the following folders. The `xx_notebooks_*`
prefix follows the neighboring AR code's folder convention. Notebook filenames
are preserved, so automatic PDFs remain under `outputs/pdfs/<notebook-name>/`
and saved data remain under their existing output directories.

Open a notebook in its theme folder and run its cells in order. Setup locates
`kde_tools/` from the project, the KDE directory, or the notebook's own folder.

### [Overview](xx_notebooks_overview/README.md)

Start here for the canonical spectral KDE experiments and comparisons.

- [spectral_scale_selection_kde.ipynb](xx_notebooks_overview/spectral_scale_selection_kde.ipynb): Main L² spectral KDE workflow, thesis figure experiments, and selector comparisons.
- [g_rs_approximation_vs_omega.ipynb](xx_notebooks_overview/g_rs_approximation_vs_omega.ipynb): Exact G_rs and its Fourier approximation versus the cutoff Omega, for one fixed sample and bandwidth pair and a fixed number K of trapezoidal intervals.
- [g_rs_k_validation.ipynb](xx_notebooks_overview/g_rs_k_validation.ipynb): Repeated-sample K validation at Omega = 450, with all Gram pairs on 50 bandwidths, calibration and independent verification, accuracy summaries, and timings.
- [g_rs_runtime_comparison.ipynb](xx_notebooks_overview/g_rs_runtime_comparison.ipynb): Exact versus Fourier computation times over 100 standard-normal samples of size 200, with 50 bandwidths and the chosen Omega = 450, K = 1536; exports mean timings and maximum error as CSV, Markdown, and LaTeX tables.
- [g_rs_error_vs_k_single_sample.ipynb](xx_notebooks_overview/g_rs_error_vs_k_single_sample.ipynb): E_K versus K at Omega = 450 for one fixed normal sample of size 200, comparing all Gram pairs on 50 bandwidths with the analytic reference.
- [g_rs_error_vs_k_repeated_samples.ipynb](xx_notebooks_overview/g_rs_error_vs_k_repeated_samples.ipynb): Retained error experiment: maximum over 200 independent normal samples of size 200 and all pairs of 50 bandwidths, versus K at Omega = 450. Reuses verified saved calculations when available.
- [g_rs_error_vs_k_sample_sizes.ipynb](xx_notebooks_overview/g_rs_error_vs_k_sample_sizes.ipynb): Historical comparison of five sample sizes; the retained experiment is the n = 200 maximum-error notebook above.

The main notebook's section 2 contains three histogram profiles:
`beta2_profile = "figure_3_5"` uses lambda = 1,
`beta2_profile = "localized"` uses lambda = N^(3/2), and
`beta2_profile = "exact_gaussian"` uses the former Gaussian L²/ISE benchmark.
The exact profile preserves independent seeds `910000 + repetition`, exact pair
divergence, selected ISE exports, and its three distribution plots. Its
`beta2_gaussian_run_profile` selects preview (M=20, N=21) or thesis (M=1000, N=201)
settings, with n=200 and lambda=1. The separate Gaussian benchmark notebook
has been removed; its saved outputs remain under their original name.
After changing the profile, run the settings, selections, and histogram cells. All three profiles retain
checkpointing, CSV exports, and eigenpair diagnostics. The localized profile
replaces its former standalone notebook.

New checkpoints live under
`outputs/data/spectral_scale_selection_kde/<profile>/<configuration>/`;
PNGs use the corresponding path under `outputs/images/`. Every displayed
figure is still automatically saved under `outputs/pdfs/spectral_scale_selection_kde/`.
Earlier outputs stay in their original directories. Compatible old checkpoints
are imported only after validating method versions, settings, grid, and sample
hashes; historical solver results are kept separately.
The [localized launcher](outputs/analysis/spectral_bandwidth_distribution_n1000_lambda_n3_2_m1000/run_notebook.py)
now runs these main-notebook cells with the localized profile.

Section 3 of the main notebook, the fast L² notebook, and the repeated-sample
and sample-size sections of the joint-limit notebook now use the same
`spectral_sweeps.run_l2_point` runner and cache under
`outputs/data/shared_spectral_sweeps/`. Cache identities include the actual
sample prefixes, bandwidth grid, quadrature settings, and solver/method versions.
Matching spectral choices can be reused across notebooks and lambda subsets;
new lambdas share one divergence calculation per sample. Compatible earlier
fast-notebook checkpoints are imported without modifying their files.
The joint-limit notebook retains its original sample seeds, summaries,
mesh diagnostics, and figures. Older selection-only caches gain checked
eigenpair diagnostics when that notebook requests them.

The main notebook retains its spatial-grid ISE and grid Q-hat comparisons,
three-selector histograms, overlays, thesis PDFs, and rate plots. The fast
notebook retains its spectral-only distributions, multi-lambda CSV exports,
and separate continuous Q-hat comparison. Its new run identifiers include the
shared runner version; earlier saved outputs remain historical.

### [Asymptotics](xx_notebooks_asymptotics/README.md)

Study bandwidth-grid size, localization scaling, and similarity eigenvalues.

- [experiment_1_joint_N_lambda_limit.ipynb](xx_notebooks_asymptotics/experiment_1_joint_N_lambda_limit.ipynb): Joint N and lambda limits, mesh diagnostics, and fixed-sample scaling experiments.
- [similarity_max_eigenvalue_vs_n.ipynb](xx_notebooks_asymptotics/similarity_max_eigenvalue_vs_n.ipynb): Largest similarity eigenvalue versus bandwidth-grid size, with numerical reference checks.

### [Bandwidth distributions](xx_notebooks_bandwidth_distributions/README.md)

Compare the sampling distributions of spectral and benchmark bandwidth choices.

- [fast_l2_bandwidth_distributions_vs_n_lambda.ipynb](xx_notebooks_bandwidth_distributions/fast_l2_bandwidth_distributions_vs_n_lambda.ipynb): Spectral bandwidth distributions across sample sizes and lambda values, plus saved-sample Q-hat comparisons.

### [Continuous minimizers](xx_notebooks_continuous_minimizers/README.md)

Compare bandwidths obtained by continuously minimizing ISE and Q criteria.

- [continuous_ise_qhat_distributions_vs_n.ipynb](xx_notebooks_continuous_minimizers/continuous_ise_qhat_distributions_vs_n.ipynb): Paired sampling distributions of continuous ISE, Q₀, Q₁, and Q₁₃/₁₄ minimizers across sample sizes.

### [Lambda sensitivity](xx_notebooks_lambda_sensitivity/README.md)

Inspect changes in the selected bandwidth as lambda varies.

- [spectral_lambda_jump_study.ipynb](xx_notebooks_lambda_sensitivity/spectral_lambda_jump_study.ipynb): Lambda-sweep jumps, competing Perron peaks, and refined dominance crossings.
- [spectral_lambda_switch_distribution.ipynb](xx_notebooks_lambda_sensitivity/spectral_lambda_switch_distribution.ipynb): Sampling distributions of switch locations and severity, using coarse and fine lambda scans.

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
includes the vector, two leading eigenvalues of B, spectral gap, and relative
residual. `eigh` selects the largest algebraic eigenvalues with
`subset_by_index=[N-2, N-1]`, including when the closest-pair limit has a
negative eigenvalue of the same magnitude; see the
[`subset_by_index` and return-value documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html).
The compatibility field `matvecs` is zero for this direct solver.
`tolerance` controls the residual check; `max_iterations` is accepted and
validated for compatibility but has no effect on `eigh`.

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

The current solver version is `stabilized_eigh_v1`. Restart existing notebook
kernels after changing the solver and rerun setup.
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

For reduced numerical runs through four notebook workflows (all three main
histogram profiles and both continuous views included), with checkpoint
resuming and the original 101-point Figure 3.3 sweep, run:

```bash
MPLBACKEND=Agg python3 tests/validate_notebook_experiments.py
```

The validation runner overrides expensive sampling settings only in memory,
keeps historical figures and data intact, and writes its report under
`outputs/analysis/perron_refactor_validation/`.
