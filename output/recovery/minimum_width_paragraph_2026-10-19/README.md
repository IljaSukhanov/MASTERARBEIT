# Recovered minimum-width literature paragraph

Investigation date: 19 October 2026. All times below are Europe/Berlin (CEST, UTC+02:00).

## Verified sequence

1. The local checkout was clean at the start and pointed to `f8b618f25dc0557375ea5ac27580f899cd5ee3c8` (19 October, 12:44). Its chapter 3 does not contain the missing paragraph.
2. Overleaf commit [`dde9702f73466ef4aff491157eccf012c579104b`](https://github.com/IljaSukhanov/MASTERARBEIT/commit/dde9702f73466ef4aff491157eccf012c579104b), 19 October, 12:46:12, contains the paragraph, four-item citation list, and bibliography additions.
3. The next Overleaf commit [`29576993a57db362bbb5317301aa0218c0513499`](https://github.com/IljaSukhanov/MASTERARBEIT/commit/29576993a57db362bbb5317301aa0218c0513499), 13:23:45, is one commit ahead of `dde9702`. Its chapter 3 has blob ID `656ea18c509eab8b4b466e45a0e9f67bfe8e79fc`, exactly matching the older local chapter at `f8b618f`. The paragraph and citation list are absent. See the [comparison](https://github.com/IljaSukhanov/MASTERARBEIT/compare/dde9702f73466ef4aff491157eccf012c579104b...29576993a57db362bbb5317301aa0218c0513499).
4. Commit [`0945cb0d06b1d473238ce6341fc0145adf8c9509`](https://github.com/IljaSukhanov/MASTERARBEIT/commit/0945cb0d06b1d473238ce6341fc0145adf8c9509), 13:23:47, merged `overleaf-2026-10-19-1121` into main. Its chapter is also blob `656ea18...`.
5. Live GitHub main was [`078a1e835cff6786307a1def15522a0fe70e3b99`](https://github.com/IljaSukhanov/MASTERARBEIT/commit/078a1e835cff6786307a1def15522a0fe70e3b99), 13:26:35. Its chapter remains the exact older blob. The local checkout has not been updated or reset during this investigation.

## Diagnosis and limits

The confirmed event is a replacement of newer chapter text by the exact older local-main version in an Overleaf update. A stale synchronization or a conflict resolution choosing older text is a reasonable inference. Git history alone does not establish which button, user, or client caused it, or whether an integration defect was involved.

The repeated dated Overleaf branches are consistent with unresolved synchronization conflicts: [Overleaf's documentation, “Resolving merge conflicts”](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/github-synchronization#resolving-merge-conflicts), says it creates dated branches when it cannot merge both sides. As additional local evidence, commits `0b6cfae`, `dc7c153`, `d7225ea`, `75e66f8`, `6f5f174`, and `9a75afe` from 18 October all share the old parent `222304372497ed14374aef5860ebe5ee325b193f` from 17 October.

[GitHub's official status page](https://www.githubstatus.com/) reported all systems operational and no incidents on 18 or 19 October when checked. This does not rule out an unreported or project-specific issue.

## Recovery artifacts

- `recovered_minimum_width_literature.tex`: exact saved TODO, citation list, and paragraph from `dde9702`; no rewriting or scientific fact-checking was performed.
- `chapter_3_at_dde9702.tex`: complete original chapter snapshot.
- `bibliography_at_dde9702.bib`: complete original bibliography snapshot.
- `proposed_paragraph_restoration.diff`: review-only proposal to append the saved citation list and paragraph to the current local chapter. It does not duplicate the existing TODO or change any other text. It has not been applied. The required bibliography snapshot already matches live GitHub main (blob `c378ee87717c96a998432b9f1e9f0500001432b4`); the local bibliography is older, so this proposal should be used after reconciling the local checkout with live main.

No file under `Abschlussarbeit_2021/` was modified. No remote branches, merges, commits, or messages were created.

Restoration into the protected thesis needs an explicit request under [AGENTS.md](/Users/iljasukhanov/MASTERARBEIT/AGENTS.md): “Any other thesis-source modification requires a separate, explicit request describing that exact text change.”
