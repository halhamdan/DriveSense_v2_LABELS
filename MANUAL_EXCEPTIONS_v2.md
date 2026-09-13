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

* **Clock sources.** VBOX HD2: GNSS-disciplined UTC on telemetry, GNSS and both videos (one
  clock; exact to the sample/frame). EmotiBit: free-running counter, no absolute clock; anchored
  to UTC through the dedicated logging laptop (macOS Sonoma 14.6.1, EmotiBit Oscilloscope
  application, CP210x VCP driver v6 for USB configuration) via NTP-style sync pings over Wi-Fi.
  The laptop clock was not referenced to GNSS time; macOS auto-synchronises with Apple time
  servers when networked, and the 0.15–0.75 s offset steps in 12/79 ping logs are consistent
  with such corrections; the laptop's own offset from UTC during each drive was not measured
  (`dataset_metadata.json` → `timing_model`, `sessions[*].sync`, Supplementary Table S3).

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

## 7. Label provenance (version 2, 2026-09-13)

* **Acceleration provenance.** `lat_acc_g` / `lon_acc_g` are Racelogic's LatAcc / LongAcc
  channels of the VBOX HD2 GNSS solution: the one-sample backward difference of the 25 Hz
  Doppler speed (`lon`, r = 1.000 against the released speed, slope 1.00) and speed x heading
  rate (`lat`, r ≈ 0.98, slope 0.97), unsmoothed. The .vbo headers contain no IMU channel; the
  VBIMU05 was connected but never logged. (`05_technical_validation/audit_acceleration_provenance.py`,
  `validation_output/acceleration_provenance.csv`.) Manuscript v5 wrongly said the IMU supplied
  acceleration; corrected in v5 (text) and carried into v6.
* **Why version 2.** The v1 rule (`01_annotation/label_harsh_events.py`: 0.38 / -0.35 / 0.55 g on
  rolling extrema of the raw derivative over a trailing 1 s window, 0.4 s min duration, dilation)
  let a single 40 ms spike sustain a condition for 25 samples. On the release, 86 % / 87 % / 78 %
  of v1 braking / acceleration / turning events had at most one consecutive sample above
  threshold; CAN-derived deceleration inside v1 braking events was typically -0.08 g; on the
  deliberate-braking validation drive v1 produced 47 "harsh braking" events for 10 brakes.
  (`validation_output/event_exceedance_structure.csv`, `label_sensitivity_gnss.csv`.)
* **Version-2 rule** (`01_annotation/label_harsh_events_v2.py`, computed from the released
  fused.csv columns only): 0.5 s centred moving average of `lon_acc_g` / |`lat_acc_g`|;
  exceedance sustained ≥ 0.5 s, gaps ≤ 0.2 s closed, no dilation; thresholds 0.20 g accel
  (throttle > 25 %, GNSS speed > 15 km/h), 0.35 g brake (speed > 15 km/h and 1 s min of CAN-speed
  derivative ≤ -0.3 m/s²), 0.45 g |lateral| (speed > 30 km/h); priority Turning > Braking >
  Acceleration (only overlap: T over A, 57 samples).
* **Calibration** on the three deliberate-manoeuvre drives in the study vehicle
  (`05_technical_validation/estimate_redefined_labels.py`, `validation_output/redefinition_grid.csv`):
  S1 (10 brakes / 10 full-throttle accels / 10 attempted turns) -> 10/10 brakes (smoothed peaks
  0.41-0.61 g), 0 turns (attempts peaked 0.32 g), 15 accels (10 deliberate + 5 natural at the same
  0.21-0.29 g ceiling); S2 (further full-throttle accels + attempted turns, which failed again,
  lateral peak 0.42 g; deliberate counts not recorded) -> 7 accels, 0 turns, and 3 non-deliberate
  firm brakes at 0.38-0.44 g; S3 (11 turns counted aloud) -> 12 turns (11 + one genuine uncounted
  turn, peaks 0.50-0.62 g). Native validation exports have no CAN speed / throttle: GPS speed
  substituted, throttle gate disabled. All three drives in the study Renault Koleos.
* **Release.** `Published_Dataset_Final_v2_LABELS` (label-only delta over v1; built by
  `02_dataset_construction/regen_v2/regenerate_labels_v2.py --apply`; report
  `regenerated_v3/labels_v2_report.csv`): `label` = v2, `label_v1_peak` = v1, all other columns
  byte-identical; schema 2.0; `dataset_metadata.json` -> `label_version`, `sessions[*].events_v2`.
  Totals v1 -> v2: HA 1,429 -> 156, HB 6,320 -> 37, HT 2,645 -> 92 (10,394 -> 285 events;
  non-Normal rows 542,451 -> 6,470). Sensitivity ±10 %: 650 / 285 / 158. Clipping raw |acc| to
  1.2 g changes 288 rows and 14 events (11 T, 2 B, 1 A). v1 release untouched.
* **Figures regenerated on v2:** `figures/fig_route_harsh_density_v2.{pdf,png}` (1,699 cells
  visited by ≥ 10 drivers) and `figures/fig_multimodal_alignment_v2.png` (D6_S4, 160.2-282.2 s).
* **v1 history (kept for the record).** v1 labels reproduced exactly (10,394 of 10,394 events)
  from native-resolution telemetry (`audit_figure4_and_labels.py`). Two v1 annotation bugs were
  fixed before the first release (throttle corroboration disabled by a column-name mismatch;
  a brake-trigger gate keeping the weak-braking rule dormant); the archived `*_LABELED.csv`
  staging files and the pre-2026-09-07 route-density figure predate those fixes and must not
  be used as references. The Hampel/3 g despiking row in §4 describes the released raw
  channels (unchanged in v2); the "label reproducibility 93.8 % -> 98.0 %" figure there refers
  to v1.
