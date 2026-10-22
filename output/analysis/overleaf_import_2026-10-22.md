# Overleaf import diagnosis and upload package

The Git checkout at fdee304 contains 5,327 tracked files. Overleaf's documented
limit is 2,000 files per project, so importing the full repository would exceed
that limit. Most tracked files are experimental results and archives.
[Official plan limits](https://docs.overleaf.com/getting-started/free-and-premium-plans/plan-limits).

The reported PDF has the basename
`02_random_per_process_absolute_width_iterations_and_mean_mae_eps_1e-08.pdf`
(74 characters), and a 155-character repository-relative path. The excerpt
contains no forbidden characters under the public Overleaf filename validator.
The exact reason the GitHub importer classified this name as invalid could not
be verified from the truncated error. No unsupported 150-character filename
limit is assumed. The public validator uses a 1,024-character path limit.
[Public validation implementation](https://github.com/overleaf/overleaf/blob/main/libraries/overleaf-editor-core/lib/safe_pathname.js).

## Prepared package

[ZIP](/Users/iljasukhanov/MASTERARBEIT/output/overleaf_thesis_2026-10-22.zip) contains 41 files and 1,373,784 compressed bytes.
It includes the original thesis sources, required external figure files, a
root main.tex entry point, bibliography copies and upload instructions.
The notebooks and bulk simulation outputs are excluded from this package.
The original repository and original thesis files were not renamed or edited.

All 20 active figure references resolve to included files. All copied source
bytes were verified against their originals and against the ZIP contents.
The ZIP passes an integrity check and the public filename/path rules.
[File manifest and validation](/Users/iljasukhanov/MASTERARBEIT/output/overleaf_import_2026-10-22_manifest.json).

A local pdfLaTeX/latexmk build succeeded and produced a 106-page PDF.
[Build log](/Users/iljasukhanov/MASTERARBEIT/output/overleaf_import_2026-10-22_build.log),
[compiled preview](/Users/iljasukhanov/MASTERARBEIT/output/overleaf_import_2026-10-22_preview.pdf).
The pre-existing unresolved reference `eq:hat_Q_n_beta` and duplicate label
`fig:distribution_of_chosen_bandwidths` remain unchanged. The app's XeTeX-based
compiler could not process the original pdfTeX-specific `\pdfinfo` primitive;
pdfLaTeX was therefore used for the successful build.

Use New Project > Upload Project, upload the ZIP, select main.tex and pdfLaTeX.
[Official upload instructions](https://www.overleaf.com/learn/latex/Kb/Uploading_a_project),
[compiler settings](https://docs.overleaf.com/getting-started/recompiling-your-project/selecting-a-tex-live-version-and-latex-compiler).
For ongoing GitHub synchronization, a dedicated repository containing these
project files avoids importing the full research repository. No new GitHub
repository was created, and nothing was uploaded to Overleaf.
