# Proposed synchronization repair

This proposal keeps all current main files and incorporates only the two newer thesis files from the latest Overleaf export, without rewriting their contents. The full exact changes are in [proposed_thesis_sync.diff](proposed_thesis_sync.diff).

- Main snapshot: `9bd03c0d5012aa9f2b21a8e495b6af999726469f`.
- Latest Overleaf snapshot: `15e0ae519ea41d52cb9fae991283188d0e167bc5` (`overleaf-2026-10-19-1431`).
- Chapter 3: imports the newer expanded literature paragraph, removes the original four-item citation list, keeps the newer commented two-item list, moves the minimum-width TODO to the literature paragraph and removes the radial-projection TODO. All of these are existing Overleaf edits, not newly written text.
- Bibliography: imports the 55 added lines containing the newer literature entries, exactly from Overleaf.
- All code, experiment outputs, recovery files, and other thesis files remain as in main.

The proposed commit will have both main and the latest Overleaf export as parents, so both histories are recorded as reconciled. This avoids repeatedly replaying the same divergent changes.

No protected thesis file has been changed by this proposal. The exact two-file import needs explicit authorization under AGENTS.md. Once authorized, it can be prepared as a reviewable branch and merged without removing local work. Overleaf must then pull the reconciled main version before another export; browser access to Firefox was not approved, so this final Overleaf action is not currently accessible to this chat.
