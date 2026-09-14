# Release snapshot, version 3 (read-only mirror)

Descriptive files of the deposited **version-3** release (`Published_Dataset_Final_v3_VIDEO_ALIGNED`,
`dataset_metadata.json` -> `release_version: 3`, `data_schema.json` -> `schema_version` 3.0,
generated 2026-09-13), copied here so that the schema, per-session metadata (including the per-session
video lead-in under `sessions[*].video`), the manifest of all 1502 released files, the harsh-event table, the native
SCR event table and the calibration drives can be inspected before data access is granted.

**Canonical copies are the ones in the data deposit; if they ever differ from these, the deposit wins.**
Built by `02_dataset_construction/regen_v3/build_v3_video_realign.py` (per-file frame counts in
`build_v3_report.csv`). Version 3 differs from version 2 only in the video-derived files (realigned to the
telemetry timeline), the truncation of D17_S2 at its logging pause, and per-session timeline notes; see
`README_v3_VIDEO_ALIGNED.md` and `MANUAL_EXCEPTIONS_v3.md` section 9.

Second v3 pass (2026-09-14): videos restored to the full telemetry window (`build_v3_tails_report.csv`); manifest rebuilt (1502 files).

Third v3 pass (2026-09-14): `hands_on_wheel_proxy` recomputed in all 75 `Side_pose.csv` with a wheel region measured from the frames (v1 values kept as `hands_on_wheel_proxy_v1`); see `audit_tables/hands_on_wheel_recalibration.csv`.

| File | Bytes | SHA-256 |
|---|---|---|
| `audit_tables/calibration_drives.csv` | 961 | `27460723da0f83bd77e801a538f859a872c05cf943711f115b23d6df597483c4` |
| `audit_tables/calibration_manoeuvres.csv` | 3,711 | `3b886df4663c51645707b3362370556a8cc5fc13699616572995bc0860de5525` |
| `audit_tables/hands_on_wheel_box.json` | 191 | `08e80e1f09e43561b10659545863cfe6353aeebf05c7281633f820e39195a545` |
| `audit_tables/hands_on_wheel_recalibration.csv` | 4,797 | `45992da9cf51dd023f602fb7e441a47dd2869e3bf3bb12a34591c44bd7f35057` |
| `audit_tables/hands_on_wheel_recomputed_per_session.csv` | 6,413 | `63b12c3a9005c45e3999608cb4704cf9f1a9d9059ccfa7e431752748c8db0871` |
| `audit_tables/released_video_stop_check.csv` | 1,174 | `f3d82a99c8bc4e8d9d712b3eb0313f58bc3693f7ed40aee7435dd6e40a1b202a` |
| `audit_tables/si_physio_quality_table.csv` | 6,097 | `a763f11afe43f33ef2dd2b66a82ef98ca88c10e09d4426bbb46c6315413e0f2c` |
| `audit_tables/v3_video_stop_check.csv` | 1,174 | `f3d82a99c8bc4e8d9d712b3eb0313f58bc3693f7ed40aee7435dd6e40a1b202a` |
| `audit_tables/video_feature_counts.csv` | 5,662 | `492a6a07c674cde0ed63b8778444c96490f58826e348ba5ddda17134e5cbd08d` |
| `build_v3_report.csv` | 6,479 | `1c13dae868ce8d937934f91fe06d30683d34768647f6aa6b52c9937e27875827` |
| `build_v3_tails_report.csv` | 8,808 | `4901e88532ed70cb6ee60280dd093e63c115a7da5378c39f8887a925798bf31f` |
| `Calibration_Drives/calibration_drive1_VBOX_native.csv` | 3,033,537 | `4f4ebcb6105e93a01abfd5efefb79a0f1100011bf35dc3e9e3d8a2488a4333ff` |
| `Calibration_Drives/calibration_drive2_VBOX_native.csv` | 5,287,751 | `2cc9fca05404bdf892279c055f30a4b8520e4c8c5a66e8ff638ecdaafe606acb` |
| `Calibration_Drives/calibration_drive3_VBOX_native.csv` | 712,191 | `112730120022c646fe9a86c9ae344f9083710fa59094eb4c2ac20faeab0b8e26` |
| `Calibration_Drives/calibration_drives.csv` | 961 | `27460723da0f83bd77e801a538f859a872c05cf943711f115b23d6df597483c4` |
| `Calibration_Drives/calibration_manoeuvres.csv` | 3,711 | `3b886df4663c51645707b3362370556a8cc5fc13699616572995bc0860de5525` |
| `data_schema.json` | 39,248 | `90777d5fc4415dbce8d2c2a7ed47b6f1d0707133aadc475bf61e6416a8f62cbf` |
| `dataset_metadata.json` | 167,118 | `1bd0d5d468c87cc7c6d6803dabd8565769962af32df1b40d08fdd595d52976f7` |
| `harsh_events_v2.csv` | 25,213 | `f3f6ffae1c3824db4f2b87749126848363de27900083175a01ae11bf67375701` |
| `manifest.csv` | 207,369 | `bf70c8bc7b2870345123602a5c570e06e0261435a2c8f8ff5663d3c1bb0e81a6` |
| `README_v3_VIDEO_ALIGNED.md` | 1,855 | `b3c4c5c1747e74cff2f2e335ddec5b85035fbc836726bfa1cf1f18366f592d0a` |
| `scr_events_native.csv` | 1,322,937 | `a826f03c3ad641519b8346b7e15d5fcdb17ef0029cb6b463d5a563c62a33066e` |

None of these files contains participant data.
