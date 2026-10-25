# Preserve the Overleaf thesis and retain the current GitHub project

**Applied after explicit user approval.** The four exact ZIP versions below were published to GitHub in [175265f332acea34318ad069dd6804bea7c3480b](https://github.com/IljaSukhanov/MASTERARBEIT/commit/175265f332acea34318ad069dd6804bea7c3480b). Local `main` matches that commit. GitHub's live comparison confirms exactly four changed files and matching blob IDs. See [preservation_verification.json](preservation_verification.json). Resolution of Overleaf's original service error remains unverified.

## Result

The downloaded Overleaf ZIP contains four protected thesis files that differ from GitHub. All four GitHub versions still match recorded Overleaf commit `4e5e2ce1db9bf96507d5db0c319e57783124c00e`; their newer ZIP versions can therefore be retained without choosing between competing GitHub thesis edits. The other 23 protected files are identical.

The proposed operation is to replace **only these four repository files with their exact bytes from the supplied ZIP**. It includes every existing Overleaf edit in those files, including equations, captions, labels, references, formatting, and TODOs. It adds no editorial or mathematical corrections.

| Repository file | What the ZIP changes | Exact comparison |
| --- | --- | --- |
| `Abschlussarbeit_2021/LaTeX/chapters/3_application_to_time_series.tex` | Adds a plotting TODO; replaces the AR(1)–AR(3) order-selection table with the AR(1)–AR(5) descent-chain/BIC/PACF table; updates the table reference, caption, label, and TODO dates. | [Chapter 3 diff](diffs/3_application_to_time_series.tex.diff) |
| `Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex` | Rewrites the introduction; adds the truncation-error and node-selection discussion; expands the Perron-vector stability discussion and Markov-chain comparison; moves literature-positioning TODOs; retains the user's other TODO and notation edits. | [Chapter 5 diff](diffs/5_bandwidth_selection.tex.diff) |
| `Abschlussarbeit_2021/LaTeX/chapters/9999_appendix.tex` | Removes the Beta-extension TODO and replaces the red continuity placeholder with the supplied proof discussion. | [Appendix diff](diffs/9999_appendix.tex.diff) |
| `Abschlussarbeit_2021/LaTeX/bibliography.bib` | Adds the citation keys `de2009intermediate` and `hershey_approximating_2007`. | [Bibliography diff](diffs/bibliography.bib.diff) |

These descriptions summarize the author's changes; this investigation does not validate their mathematical claims or the cited literature. The exact source text is in the diffs and the original [Overleaf ZIP](/Users/iljasukhanov/Downloads/MASTERARBEIT.zip).

## Figure and result dependencies

- All **16 figure references in the main thesis and its included chapters** resolve to an existing asset in both the ZIP and GitHub. Every corresponding asset has identical content. A seventeenth reference in the separate, inactive `Application_to_point_processes.tex` file also resolves identically. No figure-path edit or figure replacement is needed for this reconciliation.
- The static scan checked explicit `input`, `include`, and `includegraphics` arguments using the repository root and main-document directory as candidate bases. It verifies file availability and content identity, not compilation or all possible TeX macro expansion. The main file mixes repository-relative and main-directory-relative paths; the reconciliation does not alter them.
- The 45 selection-count and under/over-selection values in the ZIP's updated AR-order table match the saved [generated table](/Users/iljasukhanov/MASTERARBEIT/Code/Code_AR_processes/outputs/csv/ar_order_selection_final/030cde6f724e3c30/7ac02863db38b8d0/table_2_2_style.tex). This is a comparison with saved results; the experiments were not rerun.
- Both newly used bibliography keys are supplied by the ZIP's bibliography. No duplicate bibliography entry keys were found there.
- The newly written chapter 5 still has literal `Figure~[]` placeholders and an explicit TODO to add a figure. These are preserved exactly. Available PDFs for the quadrature discussion include [error over repeated samples](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/outputs/pdfs/g_rs_error_vs_k_repeated_samples/001_maximum_error_over_200_samples_at_omega_450_n_200_n_50_h_in_0_01_2_geometric_ban.pdf) and [error across sample sizes](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/outputs/pdfs/g_rs_error_vs_k_sample_sizes/001_maximum_error_for_five_sample_sizes_at_omega_450_200_samples_per_size_n_50_h_in_.pdf). Those two PDFs are in GitHub and absent from the ZIP. The [continuous-minimizer figure](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/outputs/pdfs/continuous_ise_qhat_distributions_vs_n/001_continuous_minimizers_n_100_1000_full_interval_gaussian_samples_m_1000_per_n_h_i.pdf) is already present in both. Choosing or inserting any of these is separate from preserving the supplied text.

The full dependency inventory, hashes, source locations, and checks are recorded in [comparison.json](comparison.json).

## Why the entire ZIP should not replace the current repository

The archive has 1,112 files. Against GitHub commit `e5b3cd04ec03f4d56944abd4f56f614efc541c7d`, the comparison finds 422 identical paths, 48 paths with different contents, 642 ZIP-only paths, and 86 GitHub-only paths. This is a path-by-path inventory, without attempting to pair renamed files.

Every ZIP-only file's exact content is already present among objects reachable from local Git refs. Of the 48 files with different contents, the four thesis files are the only ZIP contents absent from that inspected history. The remaining ZIP content is historical content, including older code, generated outputs, metadata, and inputs. This supports retaining the current GitHub code and outputs while bringing across the four thesis files. It is not a claim about every external or unreachable Git commit.

For example, the ZIP's `inputs/README.md` lacks the current final order-selection sample section, and its `.gitignore` lacks the current generated-diagnostic-table exclusion. A whole-project replacement would undo those current repository changes. The supplied ZIP remains intact as the complete Overleaf snapshot.

## Proposed next operation

1. Recheck that the repository, GitHub branch, and four local files still match the recorded versions. The ZIP is a snapshot; any subsequent Overleaf edits must be accounted for before application.
2. Copy only the four exact ZIP versions listed above into their corresponding protected repository paths. Their intended SHA-256 checksums are recorded in [comparison.json](comparison.json).
3. Review that the resulting change contains exactly those four files, then commit and push the thesis-preservation change if authorized. Retain all current non-thesis repository content.
4. Retry synchronization. If the old project's integration still fails, a separate GitHub import made after this preservation commit would include the supplied thesis edits. Its successful import would still need to be verified. No repair of the old project's integration is guaranteed by this comparison.

The first three operations were applied after the user explicitly approved the exact four-file patch and its publication. [The repository instructions](/Users/iljasukhanov/MASTERARBEIT/AGENTS.md) require: “Any other thesis-source modification requires a separate, explicit request describing that exact text change.” That approval covered exact replacement of these four files from the supplied ZIP, with no other thesis changes.

## Prepared patch and validation

- [overleaf_thesis_changes.patch](overleaf_thesis_changes.patch) contains exactly the four proposed replacements as a reviewable diff. It adds 149 lines and removes 59 lines in total.
- At proposal time, `git apply --check` succeeded against the unchanged working files. After approval, the four files were copied directly from the ZIP and verified byte for byte before and after publication.
- An independent in-memory application of every patch hunk reconstructed all four ZIP files byte for byte, including line endings. The original ZIP checksum was also rechecked.
- All 27 protected repository files retained their original SHA-256 checksums throughout the initial inspection. After approval, only the four listed thesis files changed; the other 23 protected files and all non-thesis tracked content stayed identical.
- No source ZIP contents were executed. The preservation commit contains only the four approved thesis files. Investigation artifacts remain local and were not published.
- No compilation or numerical experiment was run. The investigation checks synchronization content, source differences, and dependencies.

Input provenance: [MASTERARBEIT.zip](/Users/iljasukhanov/Downloads/MASTERARBEIT.zip), GitHub/local `main` at [e5b3cd04ec03f4d56944abd4f56f614efc541c7d](https://github.com/IljaSukhanov/MASTERARBEIT/commit/e5b3cd04ec03f4d56944abd4f56f614efc541c7d), and recorded Overleaf snapshot [4e5e2ce1db9bf96507d5db0c319e57783124c00e](https://github.com/IljaSukhanov/MASTERARBEIT/commit/4e5e2ce1db9bf96507d5db0c319e57783124c00e). Exact input hashes are in [comparison.json](comparison.json).
