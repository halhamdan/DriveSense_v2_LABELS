# Manual exceptions and per-session deviations

Everything in the released dataset was produced by the scripts in this repository, but a
number of sessions or channels received a decision or a correction that a reader cannot infer
from the files alone. They are all listed here, in one place, with the reason and where the
evidence lives. The same information is carried per session, machine-readably, in
`dataset_metadata.json` (`sessions[*].exceptions` and `sessions[*].sync`).

Conventions: `D{driver}_S{session}` tags; 79 valid sessions out of 80 planned; all times are
seconds from the trimmed session start unless stated.

## 1. Sessions excluded or partially released

| Session | What is missing | Reason | Where |
|---|---|---|---|
| D6_S2 | Entire session | Usable VBOX–EmotiBit temporal overlap < 30 s | `dataset_metadata.json` (status `skipped`) |
| D6_S3, D10_S3, D16_S1 | `Side_blurred.mp4`, `Side_pose.csv` | Side camera disconnected during recording | Data Records; `dataset_metadata.json` |
| D2_S3 | `Side_blurred.mp4`, `Side_pose.csv` | Side camera became unmounted mid-session (pose detection 14.5 % vs 99.2 % mean); footage judged unusable and removed rather than released with a caveat | `Removed_D2_S3_Side_CAMERA_UNMOUNTED/` (kept locally, not released) |
| D11_S3 | `Side_blurred.mp4` only (`Side_pose.csv` retained) | De-identification audit found the driver's face exposed in a subset of frames (stale carried-forward blur box during a 17.4 % raw-detection stretch); withdrawn rather than reissued | `Removed_D11_S3_Side_DEIDENTIFICATION_FAILURE/` (kept locally, not released) |

Net: 79 sessions with vehicle, physiology and front-camera data; 74 with side-camera video;
75 with a pose file.

## 2. Video de-identification corrections (face exposures found and fixed)

Three verification passes followed the initial two-pass blur (README, "De-identification
verification"): (a) a 60-frame visual review of every one of the 153 released videos
(2026-08-19/20); (b) a frame-by-frame automated audit of all 153 files
(`03_video_deidentification/verify_deidentification.py`, MTCNN re-detection gated on local
sharpness; 66 files flagged, 462 candidate frames — `verification_output/deidentification_summary.csv`,
`deidentification_audit.csv`); (c) visual review of every flagged cluster plus dense review of
the highest-risk backlit intervals (2026-08-24 → 09-01). Eight residual exposures were found in
five files. All arose under severe backlight / lens flare, where the detector failed for many
consecutive frames and the carried-forward blur box drifted onto the chest or hands.

| File | Corrected interval (released frame index, 30 fps) | Approx. time |
|---|---|---|
| D18_S4_Side | 710.6–724.2 s (first pass, static mask, 2026-08-20) | ~14 s |
| D18_S4_Side | 17500–17900 | 583–597 s |
| D4_S2_Front | 7300–7600; 11500–11700; 28000–28600 | 243–253 s; 383–390 s; 933–953 s |
| D10_S1_Front | 88700–89200 | 2957–2973 s |
| D15_S1_Front | 76605–78255 | 2554–2609 s |
| D9_S3_Side | 45000–45300 | 1500–1510 s |

Fix method (identical for all): the affected interval was re-processed from the unblurred
original with a manually verified static safety region (drawn to cover the driver's full
head-position range across the window) unioned with per-frame MTCNN detections, re-encoded, and
re-verified with a fresh dense scan of the whole file before deployment. Intervals above include
a safety margin; the exposures themselves are shorter. Pre-fix files are retained locally
(`exposure_fix_backups/`, `Raw_Dataset_D18_S4_Side_PRE_FIX_EXPOSED_FACE_BACKUP/`), not released.

Residual risk: the automated detector is demonstrably unreliable under exactly these lighting
conditions, so an exposure missed by it *and* falling between visual samples cannot be
excluded. Users with strict de-identification requirements should verify their subset.

## 3. Researcher masking (side camera)

The researcher present in the rear seat is intermittently visible in side-camera video. Every
side-camera video was screened and affected frames were covered with a fixed black-fill mask
over the researcher's position (a crop was not used, because `Side_pose.csv` coordinates are
normalised to the full frame and a crop would silently invalidate them). Pre-mask originals:
`Raw_Dataset_SIDE_BACKUP_PRE_BACKSEAT_MASK/` (local only).

## 4. Channels removed or treated specially

| Channel | Treatment | Reason |
|---|---|---|
| `brake_pct` (CAN brake-pedal %) and the discrete brake-trigger channel | Removed from `Preprocessed_Dataset`; left unmodified in `Raw_Dataset/*_VBOX_raw.csv` | Non-functional on this vehicle: median 100 % in every session regardless of driving state; trigger never changes |
| `heading_deg` | Resampled via unit-vector (cos, sin) components | Linear interpolation across the 0°/360° wrap produced false long-way-round values (0.11 % of samples before the fix) |
| `lat_acc_g`, `lon_acc_g` | Hampel despiking applied only to statistical outliers that also exceed 3 g (magnitude gate) | Un-gated despiking attenuated genuine short harsh-event peaks (label reproducibility 93.8 % → 98.0 % after gating) |
| `HR`, `IBI`, `SCR_AMPLITUDE`, `SCR_RISE_TIME` | Never despiked; not interpolated across native gaps > 3 × the nominal period; companion `*_dropout_flag` columns | Their outliers/gaps are the dropout signal itself; interpolation fabricated values (including negative intervals) |
| `gradient_pct` | Retained, but the optional gradient correction in the braking rule was removed | 3.97 % of Harsh Braking events depended on an implausible gradient tail |
| `height_m` | Retained with a route-specific plausible range (550–800 m) | Smooth GNSS altitude excursion to 1060 m in D10_S1 (not a spike; not removable by despiking) |

## 5. Timing and synchronisation

* **Trim points**: automatic rule (speed ≥ 15 km/h and ≥ 120 m from base for ≥ 80 % of an
  8 s window) with every start/end reviewed against source video; reviewer overrides are the
  `manual_start_s` / `manual_end_s` columns of
  `04_session_trimming/trim_review_output/trim_points_final.csv` and take precedence over the
  `auto_*` columns. Three automatic end cuts were rejected as GPS/speed-noise false positives.
* **Mid-session stops** are not trimmed; the ones identified are listed in
  `dataset_metadata.json` → `known_midsession_stops` (D7_S1, D8_S3, D20_S3).
* **EmotiBit-to-VBOX anchor** (all 79 sessions, 2026-09-09): session-median offset over all
  logged NTP-style sync pings, slope fixed at 1 (`02_dataset_construction/regen_v2/`). Twelve
  sessions' ping logs contain a discrete offset step of 0.15–0.75 s (most plausibly host-clock
  adjustments); the largest step per session is released as `sync.max_step_s` in
  `dataset_metadata.json` and is the bound on residual cross-device alignment error. Previous
  versions of the fused files: `Preprocessed_Dataset_PRE_EMOTIBIT_SYNC_FIX_BACKUP/` (original
  whole-second filename anchor) and `Preprocessed_Dataset_PRE_ANCHOR_ONLY_FIX_BACKUP/` (fitted-slope
  intermediate). Local only.
* **Sessions with two raw EmotiBit recordings** (D1_S1, D3_S2, D4_S1, D16_S3): a brief aborted
  connection attempt precedes the real recording; the larger file is used and paired with its
  own sync log by filename stem.
* **D9_S3** was regenerated through the standard pipeline on 2026-08-22 after an earlier pass
  had used an incorrect time reference for its telemetry (source folder found in a secondary
  location); row count and labels unchanged.
* **VBOX-side anchor anomaly**: D1_S4 and D12_S2 show a 0.12–0.16 s gap between the first two
  GPS timestamps (intermittent GPS time-of-day stall); not corrected, disclosed.

## 6. Known data-quality outliers left as-is (disclosed, not altered)

* D17_S2: noisier telemetry generally (speed-agreement SD 3.6 km/h vs ~1.0) and an isolated
  28 g lateral-acceleration glitch (removed by despiking).
* D12_S1: EDA out of the sensor's 0–50 µS range for the whole session; fails the EDA screen.
* D2_S3: covers only ~6 km of the 17 km route (heavy congestion); telemetry and labels valid.
* Ten sessions exceed 30 % EDA flatline and twelve exceed 30 % HR/IBI dropout — see
  Supplementary Table S2 and `05_technical_validation/validation_output/physiological_quality_per_session.csv`.

## 7. Label provenance

Released labels reproduce exactly (10,394 of 10,394 events) from native-resolution telemetry
with `01_annotation/label_harsh_events.py`
(`05_technical_validation/audit_figure4_and_labels.py`). Two annotation bugs were fixed before
this first public release (throttle corroboration silently disabled by a column-name mismatch;
a brake-trigger gate keeping the weak-braking rule dormant); the archived
`*_LABELED.csv` staging files and the pre-2026-09-07 route-density figure predate those fixes
and must not be used as references.
