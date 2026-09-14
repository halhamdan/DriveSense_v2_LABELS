# Published_Dataset_Final_v3_VIDEO_ALIGNED

Complete release tree, version 3, built 2026-09-13 from v1 (`Published_Dataset_Final`) and v2
(`Published_Dataset_Final_v2_LABELS`) by `02_dataset_construction/regen_v3/build_v3_video_realign.py`.

* **Videos realigned.** `{tag}_Front_blurred.mp4` / `{tag}_Side_blurred.mp4`: frame k <-> `elapsed_s` k/30.
  The VBOX HD2 video starts 10.4-11.3 s before the first telemetry sample; v1/v2 had not applied that
  lead-in, so their videos, `Front_emotions.csv` and `Side_pose.csv` were late by that amount. v3 drops
  the first round(lead_in*30) frames of the released, de-identified files (re-encoded, NVENC cq 26) and
  shifts the two CSVs identically. The frames the earlier cut had left off the END of each video were restored (front: full-length output of the same blurring run; side: raw frames blurred with the same detector, carry-forward rule and researcher mask, visually reviewed), so the videos span the full telemetry window. `Front_emotions.csv` and `Side_pose.csv` still end lead_in seconds before the telemetry. D1_S1's side video, which v1 had cut twice (51.5 s late, 41 s short), was restored the same way.
  Per-session lead-in: `dataset_metadata.json` -> `sessions[*].video`.
* **D17_S2 truncated at 654.52 s** (VBOX logging pause of 123.4 s, satellites 0; unix_time after it was
  123.4 s too early in v1/v2). Labels recomputed on the truncated file.
* **D8_S1, D18_S1, D4_S1, D19_S1**: UTC-vs-elapsed divergence of 1.44 / 0.92 / 0.64 / 0.60 s, disclosed
  in `sessions[*].exceptions`, not corrected.
* Unchanged: `fused.csv` (v2 labels, `label_v1_peak`), raw VBOX/EmotiBit CSVs (v1), `harsh_events_v2.csv`,
  `scr_events_native.csv`, `Calibration_Drives/` (D17_S2 rows beyond the cut removed from the two tables).
* `manifest.csv` rebuilt by hashing every file in this tree.
