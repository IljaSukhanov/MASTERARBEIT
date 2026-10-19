# Overleaf / GitHub synchronization diagnosis

Investigated 19 October 2026. Times in Europe/Berlin. No thesis edits or sync configuration changes were made in this investigation.

## Confirmed repeated stale exports

Six exports on 18 October have the exact same complete file tree (`9a37717d3b30c89c1ec01206ee90eac987c8ead8`) and the same parent from 17 October (`222304372497ed14374aef5860ebe5ee325b193f`). Main had advanced after that parent.

| Overleaf commit | Export time |
|---|---|
| [197f9c5](https://github.com/IljaSukhanov/MASTERARBEIT/commit/197f9c5c8be06792858b721f7b832e85b4cb75a5) | 2026-10-18 15:33:53 +0200 |
| [1583f4b](https://github.com/IljaSukhanov/MASTERARBEIT/commit/1583f4bfe5050a6d317d99f6a62b28cb0f3a651c) | 2026-10-18 15:44:15 +0200 |
| [dc7c153](https://github.com/IljaSukhanov/MASTERARBEIT/commit/dc7c153f838d16e39ef4b437bfcc4dad3ec76e63) | 2026-10-18 16:36:04 +0200 |
| [d7225ea](https://github.com/IljaSukhanov/MASTERARBEIT/commit/d7225eabd98e8b6106822fdc88fb92cbfd7e4918) | 2026-10-18 16:49:50 +0200 |
| [75e66f8](https://github.com/IljaSukhanov/MASTERARBEIT/commit/75e66f84e075308be866fbeeeecfb28582b67895) | 2026-10-18 16:53:38 +0200 |
| [6f5f174](https://github.com/IljaSukhanov/MASTERARBEIT/commit/6f5f17447bc2b5a513f65f251cb5c9a73ba52a7e) | 2026-10-18 16:57:21 +0200 |

The first new local commit on 18 October, [`7263de6`](https://github.com/IljaSukhanov/MASTERARBEIT/commit/7263de61477fe5f18a8a2c43e6020acc238c8277), changed 65 files. Subsequent commits moved and deleted experiment outputs; repeated Overleaf snapshots continued presenting earlier versions of those paths. For example, merge `b085930` reintroduced four paths, including a configuration file, CSV, PNG, and PDF, under the former `ar3_single_path_geometric_descent` directory after its outputs had been moved locally.

Interpretation: Overleaf was repeatedly exporting a stale baseline instead of consistently incorporating the reconciled main version. Git history proves the stale snapshots; it cannot identify whether an omitted pull, a failed synchronization, or an integration defect caused them.

Overleaf documents creating dated branches when both sides cannot be synchronized automatically: [GitHub synchronization, “Resolving merge conflicts”](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/github-synchronization#resolving-merge-conflicts).

## Project size and delay

| Snapshot | Tracked files | File bytes, decimal MB |
|---|---:|---:|
| 2223043 | 1653 | 224.64 |
| 7263de6 | 1677 | 224.31 |
| 9bd03c0 | 1687 | 215.33 |

These sizes sum the uncompressed file contents in each committed snapshot. They exclude Git history and do not imply that every sync transfers the entire total.

Overleaf recommends a project below 100 MB and individual commits changing fewer than 100 files. The current project exceeds the total-size recommendation by more than a factor of two. It contains large raw datasets, CSV histories, notebooks, and generated images in addition to thesis material. [Official file-limit guidance](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/github-synchronization#github-synchronization-file-limits).

The large project is a plausible contributor to long sync waits, but it was already large on 17 October. It does not, by itself, explain why the problem began on 18 October. The new repeated stale exports and file churn provide a more specific explanation for the change in behavior.

Five read-only local content-merge checks took 0.0233–0.0262 seconds per check and produced no conflict markers. These are local simulations with Git merge-tree, not measurements of Overleaf requests, uploads, or actual end-to-end synchronization. They provide no evidence of a slow local content-merge calculation. Exact service-side latency remains unmeasured.

GitHub reported no incidents on 18 or 19 October when checked: [GitHub Status](https://www.githubstatus.com/). This does not exclude an unreported or project-specific problem.

## Lost paragraph

The newer paragraph exists in `dde9702` (19 October, 12:46). The next Overleaf update, `2957699` (13:23), replaced the chapter with exactly the earlier local-main chapter blob. See the [version comparison](https://github.com/IljaSukhanov/MASTERARBEIT/compare/dde9702f73466ef4aff491157eccf012c579104b...29576993a57db362bbb5317301aa0218c0513499) and the [recovery record](/Users/iljasukhanov/MASTERARBEIT/output/recovery/minimum_width_paragraph_2026-10-19/README.md). The exact paragraph and citation list were restored in the previous authorized turn.

## Suggested workflow

Reconcile the latest versions once, bring Overleaf and local main to that reconciled version, and verify the restored paragraph in both places before resuming parallel edits. A longer-term improvement would be an Overleaf project limited to thesis sources and the figures/data it uses, with notebooks, raw datasets, and experiment histories kept separately. These are recommendations; no restructuring or remote operations were performed.
