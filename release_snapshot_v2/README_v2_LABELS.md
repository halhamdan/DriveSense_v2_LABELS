# Published_Dataset_Final_v2_LABELS

Label-only delta over the v1 release (`Published_Dataset_Final`), created 2026-09-13.

* `Preprocessed_Dataset/**/{tag}_fused.csv` -- the 79 v1 files with `label` recomputed by
  `01_annotation/label_harsh_events_v2.py` and the previous labels kept as `label_v1_peak`.
  Every other column is identical to v1.
* `data_schema.json`, `dataset_metadata.json`, `manifest.csv` -- v2 versions (manifest rows for
  files not in this tree refer to the unchanged v1 files).
* NOT duplicated here (unchanged, take from v1): `Raw_Dataset/`, `Front_emotions.csv`, `Side_pose.csv`.

Event totals v1 -> v2: {"Harsh Acceleration": 134, "Harsh Braking": 30, "Harsh Turning": 12}.
