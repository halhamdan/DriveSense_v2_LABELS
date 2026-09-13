# Release snapshot, labels version 2 (read-only mirror)

Copies of the descriptive files that ship at the top level of the **version-2 label release**
(`Published_Dataset_Final_v2_LABELS/`), a label-only delta over the v1 release
(`Published_Dataset_Final/`, mirrored in `../release_snapshot/`). The v2 tree contains only the 79
`Preprocessed_Dataset/**/{tag}_fused.csv` files (with `label` recomputed by
`01_annotation/label_harsh_events_v2.py` and the previous labels kept as `label_v1_peak`; every other
column byte-identical to v1) plus these four files. `Raw_Dataset/`, `Front_emotions.csv` and
`Side_pose.csv` are unchanged and are taken from v1.

**Canonical copies are the ones in the data deposit. If they ever differ from these, the deposit
wins.** Built by `02_dataset_construction/regen_v2/regenerate_labels_v2.py --apply`.

Snapshot taken 2026-09-13. `data_schema.json` `schema_version`: 2.0, `generated`: 2026-09-13.
`dataset_metadata.json` `date`: 2026-09-13.

| File | Bytes | SHA-256 |
|---|---|---|
| `data_schema.json` | 29,489 | `cf54c0727aabae9c4ad5432f4e4e0c1b003fc09e17a17c9c8dcb5600e01d22fc` |
| `dataset_metadata.json` | 122,363 | `fe3cbf9c9e852b8d13569bf153429e528a34c363df552aa83f9097a07815788e` |
| `manifest.csv` | 62,800 | `8087161250f8ddb8254ae6511fbcfeaab1a3a5aa7ab2b933b400bac0507365f0` |
| `README_v2_LABELS.md` | 713 | `cbba4840bb26d4620745842a53d011253b23f61e4c51ea5f3cba0f4d78d7ed94` |

- `data_schema.json` -- as v1, with the `label` column redefined (version-2 rule and calibration in
  its description), the new `label_v1_peak` column, and changelog entry 8 (2026-09-13) giving the
  reason for the redefinition and the v1 -> v2 event totals.
- `dataset_metadata.json` -- as v1 plus a top-level `label_version` block (rule parameters, code
  path, event totals) and `sessions[*].events_v2` per-session counts.
- `manifest.csv` -- 465 rows as v1; the 79 fused-file rows carry the v2 sizes, row counts and
  SHA-256s, all other rows refer to the unchanged v1 files.
- `README_v2_LABELS.md` -- the note shipped at the root of the v2 tree.

Event totals v1 -> v2: Harsh Acceleration 1,429 -> 156; Harsh Braking 6,320 -> 37; Harsh Turning
2,645 -> 92 (10,394 -> 285 events; non-Normal rows 542,451 -> 6,470 of 3,504,071).

None of these files contains participant data.
