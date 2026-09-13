# DriveSense

Code for constructing, de-identifying and validating the dataset released with the Data
Descriptor *"A Multimodal On-Road Driving Dataset with Vehicle, Physiological, Facial, and
In-Cabin Data"* (Alhamdan, Alosaimi, Atapour-Abarghouei & Arvin), submitted to Nature
*Scientific Data*. "DriveSense" is this project's internal/working name and is not part of
the paper's published title. This repository contains the code referenced in the paper's
**Code Availability** section: it covers the full path from raw sensor exports to the
released dataset — harsh-event annotation, multimodal fusion, face/video de-identification,
session trimming, and the technical-validation checks and figures reported in the
manuscript.

> **Labels version 2 (2026-09-13).** This repository line carries the redefined harsh-event
> labels described in manuscript v6: `01_annotation/label_harsh_events_v2.py` (the rule),
> `02_dataset_construction/regen_v2/regenerate_labels_v2.py` (builds the separately named
> `Published_Dataset_Final_v2_LABELS` release, a label-only delta over v1 in which `label` is
> recomputed and the previous labels are kept as `label_v1_peak`), `release_snapshot_v2/`
> (its schema/metadata/manifest) and `MANUAL_EXCEPTIONS_v2.md` (§7: acceleration provenance,
> why the labels were redefined, calibration on the deliberate-manoeuvre drives). The v1 code
> (`01_annotation/label_harsh_events.py`) and `release_snapshot/` are retained unchanged. See
> "Labels: version 1 and version 2" below.

**This repository does not contain the code for the separate companion research paper**
(classification benchmarking, multimodal fusion-model comparisons, precursor/stress-index
analysis). That work lives in a different repository and is intentionally out of scope here.

## Scope and honesty note

Stages `01`–`04` were run once by the authors against a pre-publication staging directory
tree (raw camera/sensor exports as they came off the VBOX HD2 and EmotiBit, before the final
`Raw_Dataset`/`Preprocessed_Dataset` release layout existed). They are included here for
**methodological transparency** — so a reader can see exactly how the released files were
produced and audit the logic — not as a single-command pipeline that regenerates the dataset
from scratch on another machine (that would additionally require the original video/sensor
recordings, which are not part of this code repository).

Stages `05`–`06` (technical validation, figures) **are** written to run directly against the
**released dataset** (`Raw_Dataset/` + `Preprocessed_Dataset/`) and should reproduce the
corresponding manuscript numbers/figures for anyone who has downloaded the public release.

## Repository structure

```
01_annotation/              Harsh-event labelling (Methods: "Harsh-event annotation")
02_dataset_construction/    Sensor fusion to 25 Hz, in-cabin pose extraction (Methods: "Data-acquisition
                             equipment", "Derived video and in-cabin features")
03_video_deidentification/  Face blurring for both cameras + an automated post-hoc audit script
                             (Methods: "Video de-identification")
04_session_trimming/        Start/end trim detection, trim application, final Raw_Dataset /
                             Preprocessed_Dataset reorganisation (Methods: "Session trimming and
                             temporal alignment")
05_technical_validation/    Vehicle-telemetry consistency, frame-drop and video/telemetry-sync
                             checks, data-availability table (Technical Validation; Supplementary
                             Table S1)
06_figures/                 Class-distribution, biometric-quality, route-map and route-harsh-
                             density figures, and the equipment diagram (Data Overview; Technical
                             Validation)
_paths.py                   Shared staging-directory path constants (env-var based, see below)
tutorial_load_and_split.py  Worked usage example over the RELEASED dataset (not the construction
                             pipeline above): load a session, apply documented QC masks, align the
                             facial/pose modalities, build a leave-one-driver-out split with a
                             leakage check. Run directly: `python tutorial_load_and_split.py
                             --dataset-root /path/to/Published_Dataset_Final`.
MODEL_CARD.md                Facial-expression and pose model documentation: versions, known
                             detection-failure behaviour, and documented general limitations of
                             these model families (demographic/skin-tone accuracy disparities in
                             FER-style models, occlusion/loose-clothing sensitivity in pose models),
                             with the caveats this dataset's own stratified checks did and didn't
                             establish.
```

Each script's own docstring documents what it does and which manuscript section it supports.

## Environment variables

No path is hardcoded to the authors' machine. Set whichever variables the stage you're running
needs:

| Variable | What it points to | Used by |
|---|---|---|
| `DATASET_ROOT` | Root of the **released** dataset (containing `Raw_Dataset/` and `Preprocessed_Dataset/`) | `05_technical_validation/*`, `06_figures/*`, `04_session_trimming/regenerate_metadata.py`, `03_video_deidentification/verify_deidentification.py` |
| `DATASET_RAW_DIR` | Full, **untrimmed** native per-session CSVs (`D{driver}_S{session}.csv`, pre-annotation, pre-fusion) -- distinct from `DATASET_ROOT`'s already-trimmed `Raw_Dataset/` tier | `01_annotation/label_harsh_events.py`, `05_technical_validation/make_priority_overlap_check.py` |
| `STAGING_LABELED_DIR` | Pre-publication dir of `D{driver}_S{session}_LABELED.csv` (raw telemetry + GPS + label, output of stage `01`) | `01_annotation`, `04_session_trimming/*` |
| `STAGING_FUSED_DIR` | Pre-publication dir of fused per-session output (output of stage `02`, before trimming) | `02_dataset_construction/*`, `04_session_trimming/*` |
| `STAGING_DATA_ROOT` | Pre-publication root containing each driver's raw EmotiBit + VBOX video folders | `02_dataset_construction/prepare_dataset.py`, `04_session_trimming/*` |
| `STAGING_TRIMMED_ROOT` | Output root for trimmed CSV/video (stage `04`, before final reorganisation) | `04_session_trimming/apply_trim.py`, `build_raw_tier.py` |
| `STAGING_TRIMMED_VIDEO_ROOT` | Output root for trimmed, blurred Side video specifically | `03_video_deidentification/blur_side_video.py` |
| `STAGING_FRONT_V2_ROOT` | Output root for the corrected two-pass Front-camera blurring (full untrimmed pass, then trimmed to match the released window) | `03_video_deidentification/blur_front_video.py` |
| `FFMPEG`, `FFPROBE` | ffmpeg/ffprobe binaries, if not already on `PATH` | video-touching scripts throughout |

## Pipeline stages

1. **`01_annotation/label_harsh_events.py`** — rule-based harsh-event labelling
   (deceleration/acceleration/lateral-g thresholds, 1 s rolling window, temporal
   closing/dilation/minimum-duration filtering, fixed class-priority resolution). Implements
   exactly the thresholds and logic reported in Methods, Eqs. 1–3.

2. **`02_dataset_construction/`** — `prepare_dataset.py` fuses VBOX telemetry with EmotiBit
   physiology onto a common 25 Hz timeline and runs DeepFace/MTCNN facial-expression
   extraction. It also contains the *original* single-pass Front-camera blurring step, which
   `blur_front_video.py` (below) supersedes — the original write path is left as-is for
   historical/audit purposes, but Front-camera de-identification should be produced by
   `blur_front_video.py`, not by running this script's blur step. `extract_side_pose.py` runs
   MediaPipe Pose on the Side camera for the in-cabin features. `data_checks.py`,
   `emotibit_sync.py`, `fuse_stage1.py`, `parse_labeled.py` are supporting modules used by the
   above.

3. **`03_video_deidentification/`** — three scripts, all new or substantially rewritten
   relative to the original pipeline:
   - `blur_front_video.py` and `blur_side_video.py` share the same two-pass detect-then-fill
     design (full-video MTCNN detection pass, forward/backward gap-fill with extra padding on
     carried-forward boxes, then a blur+write pass). `blur_side_video.py` came first; testing
     found `prepare_dataset.py`'s original single-pass Front-camera approach genuinely missed
     faces outright on some frames (not just low-confidence misses) and wrote them through
     unblurred with no gap-filling, so `blur_front_video.py` was added to bring the Front
     camera up to the same standard — confirmed on a session with known failures to eliminate
     them (raw per-frame detection 97-99%, gap-filled to 0% still-missing).
   - `verify_deidentification.py` re-runs face detection on every frame of the released,
     already-blurred videos and flags residual detections. It gates each detection on local
     image sharpness (Laplacian variance) inside the box, not just detector confidence —
     an earlier confidence-only version produced enormous false-positive counts because a
     detector will still fire on a blurred face-shaped region (and on hands, and on sharp
     background visible at a box edge); only a box that is both detected *and* still sharp
     (i.e., not actually blurred) counts as a genuine miss. See "De-identification
     verification" below.

4. **`04_session_trimming/`** — the sequence used to determine and apply start/end trim points
   (`detect_garage_trim.py` → manual review via `build_full_trim_screening.py` /
   `extract_trim_review_frames*.py` → `finalize_trim_points.py` → `apply_trim.py` +
   `build_raw_tier.py`), and finally `reorganize_final.py` + `regenerate_metadata.py`, which
   assemble the actual `Raw_Dataset`/`Preprocessed_Dataset` release tree and
   `dataset_metadata.json`.

5. **`05_technical_validation/`** — `check_video_telemetry_sync.py` and
   `make_framedrop_check.py` are the checks behind the Technical Validation section's video
   and cross-modal timing claims. `make_full_video_decode_audit.py` extends the fast
   container-metadata frame-completeness check to a full per-frame decode of all 153
   released video files, confirming zero decode errors, non-monotonic timestamps or
   duplicate frames dataset-wide ("Video and pose completeness"). `make_multimodal_completeness_check.py` is behind
   "Multimodal completeness around harsh events": for each released discrete harsh event, the
   fraction with valid (non-dropout/non-missing) physiology, facial and cabin-pose data
   available during the event window. `make_route_and_channel_checks.py` is behind "Route
   consistency" (per-session GPS bounding box and total driven distance, verifying every
   session followed the same fixed route) and the `engine_rpm`/thermopile-temperature
   plausible-range checks in the telemetry-anomaly table and physiological-signal section.
   `make_priority_overlap_check.py` reproduces the harsh-event mask logic from
   `01_annotation/label_harsh_events.py`, run on the full untrimmed native per-session CSVs
   (requires `DATASET_RAW_DIR`, not just `DATASET_ROOT` -- see below; running it on the
   trimmed `fused.csv` instead gives wrong, internally-contradictory results, since clipping
   first starves the rolling-window calculations of context at the boundary), to quantify,
   before the fixed Turning > Braking > Acceleration priority order is applied, how many
   samples each higher-priority class relabels away from a lower-priority one (Methods,
   "harsh-event annotation").
   `make_ppg_saturation_check.py` is behind "Physiological signal characterisation"'s PPG
   photodetector-saturation check. `make_availability_table.py` generates the released
   Supplementary Table S1 (per-driver, per-session data availability).

6. **`06_figures/`** — `make_technical_validation_figures.py` produces the class-distribution
   and biometric-quality/by-driver figures directly from `Preprocessed_Dataset/*/fused.csv`.
   `make_route_map_figure.py` and `make_route_harsh_density_figure.py` do the same for the
   route-map and spatial harsh-density figures, reading GPS from `Raw_Dataset` and merging the
   label from `Preprocessed_Dataset` by nearest timestamp. `make_equipment_diagram.py` draws
   the sensor-installation schematic; the manuscript's Figure 2 now uses three static 3D renders
   of the installation (side, top and front views, `figures/fig_installation_{side,top,front}.png`
   in the paper folder), produced outside this pipeline, with the schematic kept as a
   script-generated alternative. `make_alignment_figure.py` produces the Data Overview's
   multimodal-alignment figure from one session's `fused.csv`.

   Note: an earlier workflow diagram (dropped from the manuscript in v4) was built directly as a static
   image/diagram rather than from a script in this pipeline, and its source file is not part
   of this repository.

## De-identification verification

Recommended order: run `blur_front_video.py` (Front) and `blur_side_video.py` (Side) to
produce de-identified video with the two-pass gap-filling design, then verify the result:

```bash
python 03_video_deidentification/verify_deidentification.py --dataset-root /path/to/Published_Dataset_Final
```

This re-runs face detection (MTCNN, falling back to DeepFace/RetinaFace) on every frame of
every released `*_Front_blurred.mp4` / `*_Side_blurred.mp4` file, keeping only detections that
are also locally sharp (see the sharpness-gating note above), and writes
`verification_output/deidentification_summary.csv` (one row per session/camera) and
`deidentification_audit.csv` (one row per flagged frame, with confidence and sharpness). Pass
`--dump-flagged-frames` to also save an annotated image of every flagged frame — recommended
for at least a first pass, since sharpness-gating reduces false positives but doesn't
eliminate them (background scenery and hands can still register as "sharp"), so flagged
frames still warrant a quick human glance rather than trusting the raw count. It exits with a
non-zero status if any residual detection survives the sharpness gate.

Use `--drivers`/`--sessions` to check a subset first. Prefer full frame-by-frame checking
(`--sample-every 1`, the default) over sparse sampling for a release-gating run: the Front
camera's original failure mode was isolated single unblurred frames, which sparse sampling
can easily miss entirely. On this project's hardware, single-frame MTCNN detection ran at
~9 fps (multi-day for the full dataset); batching frames per GPU call (32 at a time) reached
~50+ fps, which is what makes a full-dataset, full-frame audit tractable in roughly a day
rather than a week — if you fork this for a larger dataset, check batching is still in effect
before assuming a runtime estimate.

## Requirements

See `requirements.txt`. Additionally:
- `ffmpeg`/`ffprobe` must be installed and on `PATH` (or pointed to via `FFMPEG`/`FFPROBE`).
- `extract_side_pose.py` requires the MediaPipe `pose_landmarker_lite.task` model file,
  downloaded from Google's MediaPipe model zoo, placed at
  `02_dataset_construction/models/pose_landmarker_lite.task` (not redistributed in this
  repository due to its size).
- A CUDA-capable GPU is recommended but not required for `torch`/`facenet-pytorch`/`deepface`
  steps; they fall back to CPU.

## Reproducing and interpreting the release

- **`MANUAL_EXCEPTIONS.md`** lists, in one place, every session- or channel-level decision that
  cannot be inferred from the files: excluded/partial sessions, the eight face exposures found
  and corrected, researcher masking, removed/special-cased channels, timing and
  synchronisation decisions, and known outliers left as-is. The same facts are carried per
  session in the released `dataset_metadata.json` (`sessions[*].exceptions`, `sessions[*].sync`),
  written by `05_technical_validation/add_session_provenance.py`.
- **`02_dataset_construction/regen_v2/`** is the pipeline that produced the released biometric
  columns as of 2026-09-09 (`regenerate_fused_v2.py` with `SYNC_MODEL=anchor_only`, writing to a
  separate tree; `deploy_regen_v3.py` verifies non-biometric columns are byte-identical, backs
  up, copies, re-hashes `manifest.csv` and appends the `data_schema.json` changelog).
  `timesync_calibration.py` documents both the released anchor-only model and the superseded
  fitted-slope model, and why.
- **`05_technical_validation/audit_*.py`** are the pre-submission audit scripts (label
  reproducibility, acceleration artefacts, synchronisation evidence, physiological quality);
  `make_physio_quality_table.py` builds Supplementary Table S2 from the physiological-quality
  audit.
- **`tutorial_load_and_split.py`** is the one-session worked example: it loads a session, applies
  the documented QC masks and the Supplementary-Table-S2 session screens, aligns the facial and
  pose files onto the 25 Hz grid, and builds a leakage-free leave-one-driver-out split.
- `requirements.txt` pins the versions used to produce the released files and the manuscript's
  Technical Validation numbers.

## Labels: version 1 and version 2

The released `lat_acc_g` / `lon_acc_g` are the VBOX HD2's GNSS-derived channels -- the one-sample
backward difference of the 25 Hz Doppler speed and speed x heading rate, unsmoothed
(`05_technical_validation/audit_acceleration_provenance.py`; no IMU channel was logged). The
version-1 rule (`01_annotation/label_harsh_events.py`) thresholded rolling extrema of that raw
derivative over a trailing 1 s window, so a single 40 ms spike could sustain a condition for 25
samples; on the release 78-87 % of v1 events contained at most one consecutive sample above
threshold (`audit_label_sensitivity_gnss.py`, `event_exceedance_structure.csv`).

Version 2 (`01_annotation/label_harsh_events_v2.py`) is computed from the released `fused.csv`
columns alone: 0.5 s centred moving average of `lon_acc_g` / |`lat_acc_g`|, exceedance sustained
>= 0.5 s (gaps <= 0.2 s closed, no dilation), thresholds 0.20 g acceleration (throttle > 25 %,
speed > 15 km/h), 0.35 g braking (speed > 15 km/h and CAN-speed-derived deceleration
<= -0.3 m/s^2), 0.45 g |lateral| (speed > 30 km/h), priority Turning > Braking > Acceleration.
Thresholds were calibrated on the three deliberate-manoeuvre drives in the study vehicle
(`05_technical_validation/estimate_redefined_labels.py`; 10/10 brakes, 11/11 turns, accelerations
peak at the vehicle's own 0.21-0.29 g ceiling). Totals: 156 / 37 / 92 = 285 events (v1: 10,394).

```
DATASET_ROOT=/path/to/Published_Dataset_Final \
python 02_dataset_construction/regen_v2/regenerate_labels_v2.py            # dry run: counts
python 02_dataset_construction/regen_v2/regenerate_labels_v2.py --apply    # writes ../Published_Dataset_Final_v2_LABELS
LABELS_V2=1 python 05_technical_validation/audit_manuscript_numbers.py     # checks manuscript v6 numbers (0 FAIL / 78)
PREPROCESSED_ROOT=/path/to/Published_Dataset_Final_v2_LABELS python 06_figures/make_route_harsh_density_figure.py
python 06_figures/make_alignment_figure.py --dataset-root /path/to/Published_Dataset_Final_v2_LABELS \
       --driver 6 --session 4 --start 160.2 --out-name fig_multimodal_alignment_v2
```

## Known limitations of this release

- Stages `01`–`04` are transparency/audit code against the authors' original staging tree, not
  a turnkey reproduction pipeline (see "Scope and honesty note" above).
- `05_technical_validation/make_availability_table.py` (Supplementary Table S1) and
  `make_physio_quality_table.py` (Table S2) build their tables directly from the released
  `dataset_metadata.json`, the release tree and the physiological-quality audit output, so both
  tables can be regenerated by anyone holding the release; the main text's session counts
  (76 side-camera recordings, 74 released side videos, 75 released pose files) are taken from
  S1's own totals.

## License

MIT — see `LICENSE`.

## Citation

See `CITATION.cff`. If you use this code, please cite the Data Descriptor once published
(details pending final journal assignment).
