# Overview

Start here for the canonical spectral KDE experiments and comparisons.

| Notebook | Purpose |
| --- | --- |
| [spectral_scale_selection_kde.ipynb](spectral_scale_selection_kde.ipynb) | Main L² spectral KDE workflow, Figure 3.5, localized, and exact-Gaussian histogram profiles, and selector comparisons. |
| [g_rs_approximation_vs_omega.ipynb](g_rs_approximation_vs_omega.ipynb) | One fixed sample and bandwidth pair: exact G_rs versus its Fourier approximation, sweeping Omega at fixed K and sweeping K for several fixed cutoffs Omega. |
| [g_rs_k_validation.ipynb](g_rs_k_validation.ipynb) | Choose K at Omega = 450 using 200 normal samples of size 100 and all pairs of 50 bandwidths; calibration, independent verification, accuracy, and timing. |
| [g_rs_error_vs_k_single_sample.ipynb](g_rs_error_vs_k_single_sample.ipynb) | Plot E_K versus K at Omega = 450 for one fixed normal sample of size 200, taking the maximum absolute Gram error over all pairs of 50 bandwidths. |
| [g_rs_error_vs_k_repeated_samples.ipynb](g_rs_error_vs_k_repeated_samples.ipynb) | Retained experiment: maximum error over 200 normal samples of size 200 and all pairs of 50 bandwidths, versus K at Omega = 450. Reuses verified saved calculations. |
| [g_rs_error_vs_k_sample_sizes.ipynb](g_rs_error_vs_k_sample_sizes.ipynb) | Historical sample-size comparison; the retained experiment is the n = 200 maximum-error notebook above. |

Section 2 supports `beta2_profile = "figure_3_5"` (lambda = 1) and
`beta2_profile = "localized"` (lambda = N^(3/2)), and
`beta2_profile = "exact_gaussian"` (exact L²/ISE benchmark). All retain checkpointing
and eigenpair diagnostics; rerun the three histogram cells after switching.

The exact-Gaussian profile retains seeds `910000 + repetition`, selected ISE
exports, and its three plots. Choose `beta2_gaussian_run_profile = "preview"`
or `"thesis"` for the former standalone notebook's original settings.

The main sample-size sweep, the joint-limit notebook, and the fast L² notebook share their spectral
runner and `../outputs/data/shared_spectral_sweeps/` cache. Their additional
comparison methods and plots retain their existing roles.

Run cells in order. Setup locates the shared helpers in `../kde_tools/` and
enables automatic PDF exports under `../outputs/pdfs/<notebook-name>/`.
Saved data, images, and checkpoints keep their existing locations in
`../outputs/`.

[All KDE notebooks and code guide](../README.md)
