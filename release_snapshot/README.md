# Release snapshot (read-only mirror)

Copies of the three descriptive files that ship at the top level of the released dataset
(`Published_Dataset_Final/`), so that the schema, per-session metadata and file manifest can be
read **before** data access is granted (the dataset is released under controlled access).

**Canonical copies are the ones in the data deposit. If they ever differ from these, the deposit
wins.** This snapshot is refreshed automatically as the last step of the deployment script
(`02_dataset_construction/regen_v2/deploy_regen_v3.py`) and manually at each tagged release.

Snapshot taken 2026-09-09. `data_schema.json` `generated`: 2026-09-09. `dataset_metadata.json` `date`: 2026-09-09T13:52:58.331568+00:00.

| File | Bytes | SHA-256 |
|---|---|---|
| `data_schema.json` | 27,824 | `468bbebf73206c82d719b2660301b1d2db5d8abf133a71a02001197266fc9fbb` |
| `dataset_metadata.json` | 112,067 | `47f00e29ba06a764fcf2d00d913b51317b15b65d54fcc44aa03693d41cbe9c61` |
| `manifest.csv` | 62,794 | `957b284323c35ff4e36e08f187cf1274716fdaaaa5228b7ac2c7e49419e931a8` |

- `data_schema.json` -- every column of every released file (dtype, unit, plausible range,
  description) and the changelog of all corrections applied to the release.
- `dataset_metadata.json` -- per-session availability and quality flags, known mid-session
  stops, per-session synchronisation (`sync`) and manual-exception tags (`exceptions`).
- `manifest.csv` -- tier, relative path, size, row count and SHA-256 of all 466 released files;
  use it to verify a downloaded copy of the deposit.

None of these files contains participant data.
