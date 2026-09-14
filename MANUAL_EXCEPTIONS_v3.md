# Manual exceptions and per-session deviations (release version 3)

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
  `dataset_metadata.json`. `sync.bound_s` = max(step, 2×MAD, 0.10 s) is an *observed
  timing-variability indicator*, NOT a bound on absolute cross-device alignment error (the
  laptop's offset from UTC was never measured; renamed 2026-09-13, CSV/JSON key kept). Pings
  arrive in 5.1 s bursts with long silent gaps (longest gap median 392 s, 47 sessions > 5 min;
  `make_ping_cadence_table.py`), so "coverage" means bracketing, not continuous sampling. Previous
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
* **RESOLVED IN v3 — video lead-in not applied in v1/v2 (found 2026-09-13; see §9).** The VBOX HD2
  starts its video 10.4–11.3 s (median 11.1 s; `video_lead_in_per_session.csv`, from Racelogic's
  `Avi sync time` at row 0) before the first logged telemetry sample. `apply_trim.py` /
  `build_raw_tier.py` mapped frames as round((t − t0_raw)·fps), i.e. assumed frame 0 = row 0, so
  every released video, `Front_emotions.csv` and `Side_pose.csv` shows, at elapsed τ, the scene at
  telemetry time τ − lead_in. Verified on D1_S1 against the raw video's burned-in speedometer
  (`validation_output/video_offset_evidence/`). Fix = re-trim from full-length staging files with
  frame = round((t − t0_raw + lead_in)·fps) (39 sessions have >1 raw video segment — map via
  `Avi sync time` per row), then re-apply the §2 exposure fixes and re-review the new footage.
  Nothing re-trimmed yet; manuscript v6 carries `[PENDING: VIDEO OFFSET]` markers.

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
* **Version-2 rule, final form** (`01_annotation/label_harsh_events_v2.py` →
  `label_rule_variants.RuleParams`; computed from the released fused.csv columns only; all
  windows are explicit integer sample counts at 25 Hz): 5-sample centred running median of the
  SIGNED `lon_acc_g` / `lat_acc_g`, then 13-sample centred mean (k = −6..+6); lateral test on
  |mean| (not mean|.|); exceedance sustained ≥ 13 samples (0.52 s), gaps ≤ 5 samples closed, no
  dilation; thresholds 0.20 g accel (throttle 13-sample max > 25 %, GNSS speed > 15 km/h), 0.35 g
  brake (speed > 15 km/h and 25-sample min of CAN-speed derivative ≤ −0.3 m/s²), 0.45 g |lateral|
  (speed > 30 km/h, AND net GNSS heading change ≥ 5° from 3 samples before to 3 after the run);
  priority Turning > Braking > Acceleration (only overlap: T over A, 21 samples).
* **Robustness audit that fixed the draft** (`05_technical_validation/audit_label_robustness_v2.py`,
  2026-09-13 evening, after the critique on rectified noise / single-spike duration): the first
  v2 draft (mean of |lat|, window = round(0.5×25) = **12** samples, min 12) labelled a single
  6.06 g sample (12 × 0.505 g ≥ 0.45 g) and alternating ±0.6 g noise as turning; on the release
  80 of its 92 turning events vanished when the signed mean replaced the rectified one. Final rule
  on synthetic signals: single sample of any size, 2-sample burst of any size, 3-sample burst
  ≤ 2.6 g, alternating ±1.0 g → no event; sustained 0.55 g × 1 s → event. Draft tree preserved as
  `Published_Dataset_Final_v2_LABELS_DRAFT1_20260913` (local only; never distributed).
  Variant table: `validation_output/label_robustness_{synthetic,drives,release_totals}.csv`.
* **Per-event table** `harsh_events_v2.csv` (release root; from `audit_label_robustness_v2.py events`):
  every event with duration, peak smoothed g, speed, raw samples above threshold and > 1.2 g,
  signed/rectified ratio, net heading change (observed vs expected), outcome under ±1.2 g clipping
  (176/176 survive; 2 turning boundaries move 1–2 samples), presence at ±10 % thresholds, v1 overlap.
* **Calibration** on the three deliberate-manoeuvre drives in the study vehicle
  (`05_technical_validation/estimate_redefined_labels.py`, `validation_output/redefinition_grid.csv`):
  S1 (10 brakes / 10 full-throttle accels / 10 attempted turns) -> 10/10 brakes (smoothed peaks
  0.41-0.61 g), 0 turns (attempts peaked 0.32 g), 16 accels (10 deliberate + 6 natural at the same
  0.21-0.29 g ceiling); S2 (further full-throttle accels + attempted turns, which failed again,
  lateral peak 0.42 g; deliberate counts not recorded) -> 5 accels, 0 turns, and 3 non-deliberate
  firm brakes at 0.38-0.44 g; S3 (11 turns counted aloud) -> 12 turns (11 + one genuine uncounted
  turn, peaks 0.50-0.62 g, all with heading change > 5°). Native validation exports have no CAN speed / throttle: GPS speed
  substituted, throttle gate disabled. All three drives in the study Renault Koleos.
* **Release.** `Published_Dataset_Final_v2_LABELS` (label-only delta over v1; built by
  `02_dataset_construction/regen_v2/regenerate_labels_v2.py --apply`; report
  `regenerated_v3/labels_v2_report.csv`): `label` = v2, `label_v1_peak` = v1, all other columns
  byte-identical; schema 2.0; `dataset_metadata.json` -> `label_version`, `sessions[*].events_v2`.
  Totals v1 -> v2 (final): HA 1,429 -> 134, HB 6,320 -> 30, HT 2,645 -> 12 (10,394 -> 176 events;
  non-Normal rows 542,451 -> 4,080; 26 sessions and one driver with no event). Sensitivity ±10 %:
  454 / 176 / 72. Clipping raw |acc| to 1.2 g: no event lost or reclassified, 2 turning boundaries
  move by 1-2 samples. 23/176 events contain a > 1.2 g raw sample (10/12 turning). v1 release untouched.
* **Figures regenerated on v2:** `figures/fig_route_harsh_density_v2.{pdf,png}` and
  `figures/fig_multimodal_alignment_v2.png` (D13_S2 from 133.3 s: one acceleration + two turns).
* **Calibration drives released** (`Calibration_Drives/` at the v2 release root; `make_calibration_table.py`):
  native VBOX exports of the three drives (2026-08-05 18:25 UTC, 11.2 min; 2026-08-17 16:41 UTC,
  19.7 min; 2026-08-17 19:15 UTC, 2.7 min) + `calibration_manoeuvres.csv` (46 labelled intervals) +
  `calibration_drives.csv`. These are CALIBRATION, not independent validation: same drives chose the
  thresholds; GNSS speed substituted for CAN; throttle gate disabled.
* **v1 history (kept for the record).** v1 labels reproduced exactly (10,394 of 10,394 events)
  from native-resolution telemetry (`audit_figure4_and_labels.py`). Two v1 annotation bugs were
  fixed before the first release (throttle corroboration disabled by a column-name mismatch;
  a brake-trigger gate keeping the weak-braking rule dormant); the archived `*_LABELED.csv`
  staging files and the pre-2026-09-07 route-density figure predate those fixes and must not
  be used as references. The Hampel/3 g despiking row in §4 describes the released raw
  channels (unchanged in v2); the "label reproducibility 93.8 % -> 98.0 %" figure there refers
  to v1.

## 8. Physiological channels: what the screens and flags mean (2026-09-13)

* **Screens are descriptive, named by the quantity tested** (Supplementary Table S2,
  `make_physio_quality_table.py`): EDA constant-value fraction < 30 %, `SCR_FREQ` constant-value
  fraction < 30 %, HR/IBI dropout fraction < 30 %. Passing certifies only that quantity
  (D11_S3 passes the dropout screen with ~40 % IBI availability). S2 now also gives **IBI
  availability** = present-and-in-range rows as % and minutes (median 67 %, range 22–94 %;
  20 sessions < 50 %).
* **Constant-value runs** (≥ 10 identical 25 Hz samples) are a descriptive flag; contact loss,
  quantisation and the device's value-reporting behaviour were not separated.
* **IBI**: the 25 Hz `IBI` column is an interpolation; ~62 % of rows present-and-in-range is its
  *availability*, not HRV suitability. The native per-beat file `Raw_Dataset/.../{tag}_EmotiBit_IBI.csv`
  (~1,500 rows/session) IS beat-to-beat: successive row spacing equals the reported interval within
  50 ms for 100 % of rows in every session (`make_scr_native_event_table.py`, `ibi_per_beat_frac`).
* **SCR_AMPLITUDE / SCR_RISE_TIME**: native device events released as `scr_events_native.csv`
  (26,292 events, 78 sessions; **D1_S4's export has no SA/SR channel** → both columns flagged
  absent for the whole session; D1_S1 and D1_S2 have no event inside the released window).
  `*_dropout_flag` on these two columns means *no event value at this row*, not acquisition loss.
  Internal-consistency assessment against EDA (same device, same modality — not validation):
  events ≥ 0.01 µS show an EDA rise > 0.01 µS over the rise time in 95.4 % (control 18.8 %);
  per-session Spearman(amplitude, rise) median 0.96. Native events/min ≈ session-mean `SCR_FREQ`
  (ratio 0.99; ρ = 0.90) — the "1.7×" discrepancy in manuscript v5 came from an event-recovery
  procedure on the interpolated column that over-counted (61,949 vs 26,292 native); withdrawn.
* **Device validation citation**: Costantini et al. 2023 (`costantini2023`) is an Empatica E4 study;
  the EmotiBit evidence is `emotibitworkload` (good HR agreement, weaker HRV/EDA). v6 no longer
  claims a driving-specific validation of the EmotiBit.


## 9. Release version 3 (2026-09-13): video realignment, D17_S2 truncation

Built by `02_dataset_construction/regen_v3/build_v3_video_realign.py` into
`Published_Dataset_Final_v3_VIDEO_ALIGNED` (complete tree; v1 and v2 untouched).

* **Video lead-in.** The VBOX HD2 starts its video 10.4–11.3 s (median 11.1 s) before the first logged
  telemetry sample (`Avi sync time` at row 0 of every native export;
  `validation_output/video_lead_in_per_session.csv`). `apply_trim.py` / `build_raw_tier.py` mapped
  telemetry time to frames as round((t − t0_raw)·30), i.e. assumed frame 0 = row 0, so every v1/v2 video,
  `Front_emotions.csv` and `Side_pose.csv` showed, at elapsed τ, the scene at τ − lead_in.
  Verification: raw-video speedometer overlay (D1_S1: telemetry departs 21.5 s, overlay 000 km/h at
  23.5 s, moving at 30 s; a telemetry stop maps to a frame showing 039 km/h = speed 10.4 s earlier;
  `validation_output/video_offset_evidence/`), and the frame-pair stop test on released files
  (`audit_released_video_stop_frames.py`: 49 sessions with a qualifying stop, 54 of 68 tests OFFSET,
  13 inconclusive, none genuinely aligned).
* **Fix applied in v3.** The first L = round(lead_in·30) frames of each released, de-identified video were
  dropped (re-encoded h264_nvenc cq 26) and the two frame-indexed CSVs shifted by −L (frame, timestamp_s
  re-indexed). No new footage: every blur, researcher mask and exposure fix stays valid; each video ends
  lead_in s before the telemetry. Frame counts verified per file (`regen_v3/build_v3_report.csv`).
  Trim instants (telemetry times) unchanged → fused.csv identical to v2 except D17_S2.
  Contact sheets used for the trim review had been extracted with the same wrong mapping (frames ~11 s
  earlier than labelled); the trims are nonetheless sound: 78/79 sessions start with the vehicle moving,
  longest stationary tail 7.9 s (`audit_trim_endpoints.py`); D13_S3 starts with 218 s at standstill
  (deliberate manual choice).
* **§2 exposure-fix intervals in v3 frame indices** (v1 index − L): D18_S4_Side (L=336) 17164–17564
  (+ the first-pass static mask window 710.6−11.2 → 699.4–713.0 s); D4_S2_Front (L=333) 6967–7267,
  11167–11367, 27667–28267; D10_S1_Front (L=332) 88368–88868; D15_S1_Front (L=336) 76269–77919;
  D9_S3_Side (L=333) 44667–44967.
* **D17_S2 logging pause.** `audit_vbox_time_continuity.py`: at released elapsed 654.52 s the receiver's
  UTC jumps by 123.36 s while `Elapsed time` and the video continue (satellites 0 → GPS loss). Because
  unix_time = first UTC + elapsed, v1/v2 rows after that instant are 123.4 s too early (EmotiBit
  columns misaligned there; video offset not constant). v3 truncates D17_S2 at 654.52 s in all tiers
  (fused 24,284 → 16,364 rows; raw VBOX/EmotiBit CSVs by unix_time; videos/CSVs by frame) and recomputes
  its v2 labels on the truncated file (D17_S2 had no v2 event either way). Alternative (re-fuse the
  post-pause segment on a UTC-based timeline) left for the supervisor's decision.
* **UTC-vs-elapsed divergence ≤ 1.5 s** in D8_S1 (1.44 s), D18_S1 (0.92), D4_S1 (0.64), D19_S1 (0.60):
  gradual, inside the first video segment; disclosed in `sessions[*].exceptions`, not corrected.
* `dataset_metadata.json` → `release_version: 3`, `video_alignment`, `sessions[*].video`
  (lead_in_s, frames_dropped_at_start, front/side frame counts); `data_schema.json` 3.0 with changelog
  entry; `manifest.csv` rebuilt by hashing every file in the tree.

## 10. Video-derived feature audit (2026-09-14, v3 frames)

`05_technical_validation/make_visual_feature_audit_sheet.py` (seed 20260913; 30 front + 30 side frames,
≤ 1 per session per camera, rows with a detection). Notes: `validation_output/visual_feature_audit/reviewer_notes.md`.
* Face box (Front_emotions.csv): on the face in 30/30, incl. 6 backlit frames.
* Wrists: adequate in ~26/30; near-side shoulder ~28/30; **far-side (occluded) shoulder misplaced on the
  centre console in ~10/30** → `posture_deviation` biased in those frames.
* **`hands_on_wheel_proxy` is unreliable as released**: agrees with the visible hand placement in 9/30,
  under-reports in 20/30 — the fixed rectangle (x 0.25–0.75, y 0.55–1.0) lies mostly below the wheel
  (≈ x 0.45–0.65, y 0.35–0.55 in this framing). Column kept (documented), Usage Notes tell users to
  re-derive from the wrist coordinates. Fixing the rectangle would change a released column → left for
  a later version if the supervisor prefers.
* Denominators (`make_video_feature_counts.py`, v3): 4,170,627 front frames; 4,170,203 emotion rows;
  4,127,040 detections (98.96 %); 100 % of detections sum to 100 ± 1; 3,872,036 side frames; 654,562 pose
  rows / 648,792 detected (99.1 %).

### 9b. Video coverage restored (2026-09-14, second v3 pass)

* A full-length blurred FRONT video exists for every session
  (`Documents/Front_Blur_V2_Fix/D*/Session_*/{tag}_Front_blurred_v2_full.mp4`, the output of the
  same blurring run the release came from; released frame k = its frame fs_old + k, verified by
  pixel matching). No full-length blurred SIDE video exists (blurred and trimmed in place).
* `regen_v3/restore_video_tails_v3.py` rebuilt every v3 video as released[L:] + the L frames the
  earlier cut had left off the end: front from `_v2_full`; side by blurring the raw frames
  (`Published_Dataset/.../{tag}_Side.mp4`) with the release pipeline's detector and rules
  (MTCNN keep_all p ≥ 0.7, carry-forward with extra padding, Gaussian k = 51) **plus** a static
  blur floor over the driver-head region (x 0.08–0.42, y 0.05–0.55) and the session's researcher
  mask (black left strip measured from the released side video: 300 px for D1_S1, ~200 px otherwise).
  Every restored segment has a review sheet in `05_technical_validation/validation_output/video_tail_review/`
  (not in the public repo — controlled-access frames). Per-file counts/detections:
  `regen_v3/build_v3_tails_report.csv`.
* **D1_S1 side video had been cut twice in v1** (frame k = raw frame 2·fs_old + k → a further 51.5 s
  late and 1,546 frames short; the only side file whose frame count differed from its front). Its
  missing raw frames 1859–3091 (41 s) were blurred and restored the same way (1,217/1,233 detections).
* `Front_emotions.csv` / `Side_pose.csv` are NOT extended (no DeepFace / MediaPipe environment on this
  machine): they end lead_in s before the telemetry; documented in Methods, Usage Notes and metadata.
* fs_old per session = round((released first `utc_tod_s` − native row-0 UTC) · 30), refined by matching
  released front frame 100 to the full-length front (match diff ≈ 0.3 grey levels).

### 10b. `hands_on_wheel_proxy` recomputed (v3, 2026-09-14)

* Exact scoring of the audited 30 side frames: v1 column agrees 7/30, under-reports 19, over-reports 4
  (the first tally of 9/20/1 was approximate). `validation_output/hands_on_wheel_recalibration.csv`.
* New definition (`regen_v3/recompute_hands_on_wheel_v3.py`, applied to all 75 v3 `Side_pose.csv`):
  wheel region **x 0.40–0.70, y 0.28–0.62** measured from the frames (not fitted; a 0.05-step grid search
  finds no better rectangle), released wrist coordinates only (no visibility gate is possible — not
  released), missing wrist = not on wheel. Agreement 23/30 (5 under, 2 over, all half-step).
  Dataset-wide mean 0.33 → 0.62; both hands in region on 41 % of detected rows, none on 17 %.
* Old values kept as `hands_on_wheel_proxy_v1` (13 in-cabin columns now); schema/metadata/manifest updated.
* Per-session alternatives tried and rejected (`regen_v3/wheel_region_per_session.py`): circle fit to the
  wrist cloud (wrists do not trace the rim) and GMM grip clusters in union with the box (21/30, over-reports
  hands resting near the wheel). Fixed measured box kept. A blind 100-frame second-rater pack is prepared in
  `05_technical_validation/validation_output/hands_on_wheel_rating/` (`make_hands_on_wheel_rating_pack.py`).
* **Independent blind check, 100 frames** (`make_hands_on_wheel_rating_pack.py`, `score_hands_on_wheel_ratings.py`,
  rater CL): recomputed proxy exact 59 %, within one hand 93 %, quadratic κ 0.55 (stopped 74 %, moving 44 %);
  v1 25 %. This — not 23/30 — is what v6 quotes. Second rater (Hanadi) pending: `hands_on_wheel_rating/rating_template.csv`.
