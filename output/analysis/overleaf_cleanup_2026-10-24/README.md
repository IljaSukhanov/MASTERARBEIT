# Overleaf synchronization cleanup — 24 October 2026

The scan covered all project files outside Git internals, including ignored outputs. Tracked files were reduced from approximately 72.38 MB to 57.03 MB. The detailed manifest records each affected path, its original checksum, and the reason.

- Removed 33 unreferenced PNG previews while retaining their PDF counterparts.
- Removed 42 cache and operating-system metadata files outside the protected thesis folder.
- Cleared saved displays in 9 notebooks; all source cells and notebook metadata remain unchanged.
- Kept 29 generated diagnostic CSV files on this computer and excluded them from GitHub/Overleaf. They can still be read by the existing analysis scripts.
- Deduplicated existing ignore rules and added the diagnostic-table exclusion.

Every protected thesis file, Python source, input file, and exported PDF retained its original checksum. All thesis graphic dependencies retain the same availability and checksums as before. No source-code or mathematical changes were made; numerical experiments were not rerun.

Original changed/deleted files are also saved in `/private/tmp/MASTERARBEIT-cleanup-2026-10-24-d9_89fp6/original-files.tar.gz`. Tracked originals remain recoverable from commit `7e919a1e0835e122be008b2e997a973e831e5a14`.

This reduces the current project snapshot; it does not remove old commits or establish the cause of Overleaf’s GitHub service error.

References: [Git ignore behavior](https://git-scm.com/docs/gitignore), [Overleaf GitHub sync limits](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/github-synchronization#github-synchronization-file-limits).
