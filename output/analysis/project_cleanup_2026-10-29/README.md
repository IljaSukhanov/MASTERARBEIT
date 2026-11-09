# Project file cleanup — 29 October 2026

The filtered [manifest](removal_manifest.json) covers **24 removed files (245,452 bytes; 239.7 KiB)** and **59 empty generated folders**. Relocated **5 report sources and verification files** from temporary storage. Entries for subsequently removed assets are omitted from this historical record.

## Removed files

- Three OS folder metadata files.
- Twenty Python/Numba cache files.
- The superseded mean-and-maximum Fourier-error preview. The current maximum-error PDF and original numerical results remain available.

## Relocated sources

| Previous location | Retained location |
| --- | --- |
| `tmp/pdfs/novelty_review/render.swift` | [render_pdf.swift](../../../output/analysis/report_tools/render_pdf.swift) |


## Verification

- All **27 protected thesis files** retain their original hashes.
- At the time of this cleanup, all **25 thesis graphic references retained in the manifest** resolved, and their assets matched the recorded hashes.
- All **691 existing files outside the removal/relocation list** retain their original paths and hashes.
- Parsed **101 Python files**, **136 code cells across 15 notebooks**, and **244 JSON files** outside the protected thesis directory.

Distinct numerical runs, raw inputs, required automatic/delivery PDFs, research reports, recovery evidence, and Git history were retained. Existing experiment results were not regenerated. Historical documentation links to previously removed assets were left as historical records.
