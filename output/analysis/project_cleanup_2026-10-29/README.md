# Project file cleanup — 29 October 2026

The filtered [manifest](removal_manifest.json) covers **24 removed files (245,452 bytes; 239.7 KiB)** and **59 empty generated folders**. Relocated **5 report sources and verification files** from temporary storage. Entries for subsequently removed assets are omitted from this historical record.

## Removed files

- Three OS folder metadata files.
- Twenty Python/Numba cache files.
- The superseded mean-and-maximum Fourier-error preview. The current maximum-error PDF and original numerical results remain available.

## Relocated sources

| Previous location | Retained location |
| --- | --- |
| `tmp/pdfs/build_minimum_width_note.py` | [build_note.py](../../../output/analysis/minimum_width_linf_beta_experiment/build_note.py) |
| `tmp/pdfs/minimum_width_linf_beta_experiment.tex` | [minimum_width_linf_beta_experiment.tex](../../../output/pdf/minimum_width_linf_beta_experiment.tex) |
| `tmp/pdfs/ar_harmonics_v6/check_formulas.py` | [check_formulas.py](../../../output/analysis/ar_support_harmonics/check_formulas.py) |
| `tmp/pdfs/novelty_review/render.swift` | [render_pdf.swift](../../../output/analysis/report_tools/render_pdf.swift) |
| `tmp/pdfs/width_harmonics/verification.json` | [verification.json](../../../output/analysis/width_harmonics/verification.json) |

The harmonics checker now reads the retained report PDF instead of an absent temporary PDF. The width-harmonics verification file is valid JSON: its trailing PDF note is stored as a `pdf` field, with every original numerical field preserved. The other three moves preserve file contents byte for byte.

## Verification

- All **27 protected thesis files** retain their original hashes.
- At the time of this cleanup, all **25 thesis graphic references retained in the manifest** resolved, and their assets matched the recorded hashes.
- All **691 existing files outside the removal/relocation list** retain their original paths and hashes.
- Parsed **101 Python files**, **136 code cells across 15 notebooks**, and **244 JSON files** outside the protected thesis directory.
- The relocated [harmonics checker](../ar_support_harmonics/check_formulas.py) passes its existing assertions and ten-page PDF content checks using the bundled workspace interpreter. The Anaconda interpreter lacks its `pypdf` dependency; no packages were installed.

Distinct numerical runs, raw inputs, required automatic/delivery PDFs, research reports, recovery evidence, and Git history were retained. Existing experiment results were not regenerated. Historical documentation links to previously removed assets were left as historical records.
