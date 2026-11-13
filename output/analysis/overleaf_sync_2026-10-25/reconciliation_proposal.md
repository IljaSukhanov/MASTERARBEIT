# Overleaf investigation and thesis reconciliation — 25 October 2026

This is a historical investigation and completion record, consolidated on
28 October 2026. Counts, branch state, service observations, and dependency
checks below describe the checks on 25 October, not a new live verification.
The filename is retained for existing references.

**Applied after explicit user approval.** The four exact ZIP versions below were published to GitHub in [175265f332acea34318ad069dd6804bea7c3480b](https://github.com/IljaSukhanov/MASTERARBEIT/commit/175265f332acea34318ad069dd6804bea7c3480b). At publication verification, local `main` matched that commit. The recorded GitHub comparison confirmed exactly four changed files and matching blob IDs. See [preservation_verification.json](preservation_verification.json). Resolution of Overleaf's original service error was not verified.

## Initial synchronization investigation

Reported error: “Sorry, there was a problem checking our GitHub service. Please try again in a few moments.”

The initial 1,057-file comparison was historical evidence, not a measurement
of the supplied ZIP or Overleaf’s live synchronization cursor.

- At the initial check, local `main` and the live GitHub `main` branch both pointed to `e5b3cd04ec03f4d56944abd4f56f614efc541c7d` (`update 25.10`). The local working tree was clean before this report was added. [GitHub commit](https://github.com/IljaSukhanov/MASTERARBEIT/commit/e5b3cd04ec03f4d56944abd4f56f614efc541c7d).
- That committed snapshot contained 556 files totaling 37,281,411 bytes (37.28 decimal MB), measured with `git ls-tree -rlz HEAD`. All files other than PDF, PNG, and NPZ total 5,297,252 bytes; this is an upper bound for the text content in that snapshot, not a measurement of Overleaf's live editable content.
- The most recent commit then labelled `Updates from Overleaf` in the inspected local history is `4e5e2ce1db9bf96507d5db0c319e57783124c00e`, dated 23 October 2026 at 15:28:13 Europe/Berlin. Its snapshot contains 1,392 files totaling 169,654,334 bytes. This historical snapshot does not prove Overleaf's current sync cursor or current content.
- `git diff --name-status -M 4e5e2ce e5b3cd0` reports 1,057 changed paths: 919 deletions, 51 modifications, 83 additions, and 4 renames. Even comparison with merge commit `cccd74e` reports 221 changed paths.
- A live call to [GitHub's comparison API](https://api.github.com/repos/IljaSukhanov/MASTERARBEIT/compare/4e5e2ce...e5b3cd0) succeeded and returned `status=ahead`, 12 commits ahead, zero behind, and exactly 300 file entries. The 300-file comparison cap is documented in [GitHub REST API, Compare two commits](https://docs.github.com/en/rest/commits/commits#compare-two-commits).
- [Overleaf's GitHub synchronization documentation, GitHub synchronization file limits](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/github-synchronization#github-synchronization-file-limits) recommends fewer than 100 changed files per commit and a project below 100 MB. It also states that synchronization fails when editable content exceeds 7 MB.
- The [Overleaf status page](https://status.overleaf.com/) reported fully operational during this investigation. This does not rule out a project-specific problem.
- `git diff --quiet 4e5e2ce HEAD -- Abschlussarbeit_2021` succeeded: all protected thesis files in that inspected commit matched that recorded Overleaf commit. No protected thesis file was modified during this investigation.


### Interpretation and limits of the initial diagnosis

An oversized pending comparison is a plausible explanation for the continuing failure: reducing the current snapshot does not make the difference from an older Overleaf snapshot small. The live comparison result demonstrates truncated file enumeration, but does not establish that Overleaf uses that endpoint for this operation or that truncation caused the reported error.

The exact cause remains unconfirmed. Browser access to Firefox was not approved, so Overleaf's live sync state and failed request could not be inspected. Account authorization, integration state, and server-side failures have not been ruled out.


## Applied thesis reconciliation

The supplied Overleaf ZIP contained four protected thesis files differing from the inspected GitHub snapshot. All four GitHub versions matched recorded Overleaf commit `4e5e2ce1db9bf96507d5db0c319e57783124c00e`; their newer ZIP versions could therefore be retained without choosing between competing GitHub thesis edits. The other 23 protected files are identical.

The applied operation replaced **only these four repository files with their exact bytes from the supplied ZIP**. It retained every existing Overleaf edit in those files, including equations, captions, labels, references, formatting, and TODOs. It added no editorial or mathematical corrections.

| Repository file | What the ZIP changes | Exact comparison |
| --- | --- | --- |
| `Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex` | Rewrites the introduction; adds the truncation-error and node-selection discussion; expands the Perron-vector stability discussion and Markov-chain comparison; moves literature-positioning TODOs; retains the user's other TODO and notation edits. | [Chapter 5 diff](diffs/5_bandwidth_selection.tex.diff) |
| `Abschlussarbeit_2021/LaTeX/chapters/9999_appendix.tex` | Replaces the red continuity placeholder with the supplied proof discussion. | [Appendix diff](diffs/9999_appendix.tex.diff) |
| `Abschlussarbeit_2021/LaTeX/bibliography.bib` | Adds the citation keys `de2009intermediate` and `hershey_approximating_2007`. | [Bibliography diff](diffs/bibliography.bib.diff) |

These descriptions summarize the author's changes; this investigation does not validate their mathematical claims or the cited literature. The retained non-AR source changes are in the diffs; the original changes are in the supplied ZIP (recorded input path: `/Users/iljasukhanov/Downloads/MASTERARBEIT.zip`; checksum in [comparison.json](comparison.json)).


## Recorded figure and result dependencies

- All **16 figure references in the main thesis and its included chapters** resolve to an existing asset in both the ZIP and GitHub. Every corresponding asset has identical content. A seventeenth reference in the separate, inactive `Application_to_point_processes.tex` file also resolves identically. No figure-path edit or figure replacement is needed for this reconciliation.
- The static scan checked explicit `input`, `include`, and `includegraphics` arguments using the repository root and main-document directory as candidate bases. It verifies file availability and content identity, not compilation or all possible TeX macro expansion. The main file mixes repository-relative and main-directory-relative paths; the reconciliation does not alter them.
- Both newly used bibliography keys are supplied by the ZIP's bibliography. No duplicate bibliography entry keys were found there.
- The newly written chapter 5 still has literal `Figure~[]` placeholders and an explicit TODO to add a figure. These are preserved exactly. Available PDFs for the quadrature discussion include [error over repeated samples](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/outputs/pdfs/g_rs_error_vs_k_repeated_samples/001_maximum_error_over_200_samples_at_omega_450_n_200_n_50_h_in_0_01_2_geometric_ban.pdf) and [error across sample sizes](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/outputs/pdfs/g_rs_error_vs_k_sample_sizes/001_maximum_error_for_five_sample_sizes_at_omega_450_200_samples_per_size_n_50_h_in_.pdf). Those two PDFs are in GitHub and absent from the ZIP. The continuous-minimizer figure (recorded filename `001_continuous_minimizers_n_100_1000_full_interval_gaussian_samples_m_1000_per_n_h_i.pdf`; identity in [comparison.json](comparison.json)) is already present in both. Choosing or inserting any of these is separate from preserving the supplied text.

The full dependency inventory, hashes, source locations, and checks are recorded in [comparison.json](comparison.json).


## Why only four ZIP files were imported

The archive has 1,112 files. Against GitHub commit `e5b3cd04ec03f4d56944abd4f56f614efc541c7d`, the comparison finds 422 identical paths, 48 paths with different contents, 642 ZIP-only paths, and 86 GitHub-only paths. This is a path-by-path inventory, without attempting to pair renamed files.

Every ZIP-only file's exact content is already present among objects reachable from local Git refs. Of the 48 files with different contents, the four thesis files are the only ZIP contents absent from that inspected history. The remaining ZIP content is historical content, including older code, generated outputs, metadata, and inputs. This supports retaining the current GitHub code and outputs while bringing across the four thesis files. It is not a claim about every external or unreachable Git commit.

For example, the ZIP's `inputs/README.md` lacks the current final order-selection sample section, and its `.gitignore` lacks the current generated-diagnostic-table exclusion. A whole-project replacement would undo those current repository changes. The supplied ZIP remains intact as the complete Overleaf snapshot.


## Completed operation

After explicit approval of the exact four-file patch and publication, the
recorded operation rechecked the input versions, copied only the four exact
ZIP files, reviewed that four-file change, and committed and published it.
All current non-thesis content was retained. Byte-preservation evidence is
recorded in [preservation_verification.json](preservation_verification.json); the
[repository instructions](../../../AGENTS.md) require a separate explicit
request for other protected thesis edits.

## Patch, validation, and input provenance

- [overleaf_thesis_changes.patch](overleaf_thesis_changes.patch) retains the non-AR portions of the historical review diff; AR portions were removed later.
- At proposal time, `git apply --check` succeeded against the unchanged working files. After approval, the four files were copied directly from the ZIP and verified byte for byte before and after publication.
- An independent in-memory application of every patch hunk reconstructed all four ZIP files byte for byte, including line endings. The original ZIP checksum was also rechecked.
- All 27 protected repository files retained their original SHA-256 checksums throughout the initial inspection. After approval, only the four listed thesis files changed; the other 23 protected files and all non-thesis tracked content stayed identical.
- No source ZIP contents were executed. The preservation commit contains only the four approved thesis files. Investigation artifacts remain local and were not published.
- No compilation or numerical experiment was run. The investigation checks synchronization content, source differences, and dependencies.

Input provenance: `/Users/iljasukhanov/Downloads/MASTERARBEIT.zip` (supplied ZIP; checksum recorded in [comparison.json](comparison.json)), GitHub/local `main` at [e5b3cd04ec03f4d56944abd4f56f614efc541c7d](https://github.com/IljaSukhanov/MASTERARBEIT/commit/e5b3cd04ec03f4d56944abd4f56f614efc541c7d), and recorded Overleaf snapshot [4e5e2ce1db9bf96507d5db0c319e57783124c00e](https://github.com/IljaSukhanov/MASTERARBEIT/commit/4e5e2ce1db9bf96507d5db0c319e57783124c00e). Exact input hashes are in [comparison.json](comparison.json).


## Unverified synchronization follow-up

The four-file preservation was completed; resolution of the original
service error was not verified by this investigation. The remaining
diagnostic below was recorded as a possible follow-up, not as an operation
performed during the reconciliation or this Markdown consolidation.

Import `IljaSukhanov/MASTERARBEIT` through **New project → GitHub repo** into a separate Overleaf project, retaining the original project. This is a diagnostic test of importing the current repository snapshot, not a guaranteed repair of the existing project's synchronization. Compare any unsynchronized edits in the original before adopting a replacement. Overleaf documents this import route under [Creating a new Overleaf project from a GitHub repository](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/github-synchronization#creating-a-new-overleaf-project-from-a-github-repository).

If import succeeds, that is evidence favoring a problem specific to the existing project or its pending synchronization. If it fails, capture the exact response and investigate account-level access or import limits. Overleaf support can be contacted through [the official contact page](https://www.overleaf.com/contact) if server-side investigation is needed.
