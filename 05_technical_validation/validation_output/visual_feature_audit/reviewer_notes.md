# 60-frame visual audit of the released video-derived features (v3 release)

Selection: `make_visual_feature_audit_sheet.py`, seed 20260913, 30 front + 30 side frames, at most one
frame per session per camera, drawn from rows with a detection (`selection.csv`). Reviewed 2026-09-14
on the annotated contact sheets (`sheet_front_*.png`, `sheet_side_*.png`; not committed to the public
code repository because they are frames of controlled-access video). Reviewer: the assistant, on
Hanadi Alhamdan's instruction; judgements are qualitative, frame by frame.

## Front camera — released face box (`Front_emotions.csv`: face_x/y/w/h), 30 frames
* 30/30: the box lies on the (blurred) face; none has drifted onto the chest, hands or headrest.
* Includes 6 strongly backlit / flared frames (D20_S2, D6_S3, D8_S3, D14_S3, D2_S4, D19_S1): box still
  on the face. Box size is consistent (~90-110 px on the 960-px frame).
* Not assessable from de-identified frames: whether the expression label is right (the face is blurred
  in the release; the scores were computed on the unblurred frames).

## Side camera — released keypoints and `hands_on_wheel_proxy` (`Side_pose.csv`), 30 frames
* Wrists (red): on or within a hand's width of the wrist in ~26/30; the remaining cases have one wrist
  placed on the forearm or on the wheel rim rather than the hand (D15_S1, D17_S1 dark frame, D19_S4
  raised arm, D3_S3).
* Near-side (left, camera-side) shoulder (cyan): correct in ~28/30.
* Far-side (right, occluded) shoulder: placed on the centre console / gear lever region rather than on
  the body in ~10/30 frames — the inferred position of an occluded landmark and should not be used as a
  measured position. This affects `posture_deviation` (shoulder midpoint) by pulling the midpoint down
  and forward in those frames.
* `hands_on_wheel_proxy` vs. what the frame shows (0 / 0.5 / 1 hands on the wheel):
  agrees in 7/30 (exact scoring of the CSV values against the per-frame judgement, `hands_on_wheel_recalibration.csv`); under-reports in 19/30 (both hands visibly on the wheel with proxy 0.0 in D17_S4,
  D4_S1, D4_S4, D5_S2, D7_S1, D16_S3, D12_S4, D12_S1, D7_S3, D3_S3; one hand on the wheel with proxy
  0.0 in D19_S4, D14_S4, D13_S1, D5_S4, D9_S2; both hands on the wheel with proxy 0.5 in D3_S4, D17_S3,
  D5_S1, D18_S2); over-reports in 4/30 (D17_S1 hands on lap, proxy 1.0, dark frame; D9_S1, D3_S1 one hand on the wheel, proxy 1.0; D15_S1 hands on lap, proxy 0.5).
  Cause: in this camera framing the steering wheel occupies roughly x 0.45-0.65, y 0.35-0.55 of the
  normalised frame, i.e. mostly ABOVE the fixed rectangle x 0.25-0.75, y 0.55-1.0 used by the proxy,
  so a wrist on the wheel usually falls outside the region. The released wrist coordinates themselves
  are adequate; the rectangle is not. Users should re-derive a hands-on-wheel indicator from the
  released wrist coordinates with a wheel region fitted to this framing (or per session), and should
  not use the released `hands_on_wheel_proxy` as it stands.

## Summary statement used in the manuscript
Face boxes: 30/30 on the face. Wrist keypoints: ~26/30 adequate; near shoulder ~28/30; far shoulder
misplaced in ~10/30 (occluded). `hands_on_wheel_proxy` v1: agrees with visual reading in 7/30, under-reports in 19/30 because the wheel lies above the fixed region.

## Recalibration (v3, 2026-09-14)
The column was recomputed from the released wrist coordinates with the wheel region measured from the
frames, x 0.40-0.70, y 0.28-0.62 (`regen_v3/recompute_hands_on_wheel_v3.py`; the old column is kept as
`hands_on_wheel_proxy_v1`). On the same 30 frames the recomputed value agrees with the judgement in 23/30
(a coarse grid search over rectangles reaches the same 23/30, so the measured box was kept rather than a
fitted one). Remaining disagreements are half-step errors (one hand scored as on/off the wheel) in frames
where a hand rests on the rim's lower edge or near the wheel; no frame is off by a full step. The
validation is 30 frames judged by one reviewer: it supports descriptive use, not a claimed accuracy.

## Per-session region: negative result (2026-09-14)
Two data-driven per-session alternatives were tried (`regen_v3/wheel_region_per_session.py`): (a) a RANSAC
circle fit to the wrist positions at > 30 km/h -- the wrists do not trace the rim (drivers hold fixed grips;
the cloud is a compact blob), so the circle lands on the grip cluster, not the wheel; (b) a Gaussian-mixture
model of the driving-speed wrist positions, used in union with the fixed box -- agreement fell to 21/30
(8 over-reports), because some drivers rest a hand on the lap or gear lever even at speed, so the
"driving position" cluster is not always the wheel. The fixed measured box (23/30) is therefore kept.
Review images for every session: `validation_output/wheel_fit_review/`.

## Independent 100-frame blind check (2026-09-14, `hands_on_wheel_rating/`)
Because the 30 frames above also served to diagnose the region, a fresh sample was rated blind
(`make_hands_on_wheel_rating_pack.py`, seed 20260914: 50 frames > 30 km/h, 50 < 5 km/h, <= 2 per
session, no overlays; ratings in `ratings_CL.csv`, scoring `score_hands_on_wheel_ratings.py`).
Recomputed column (fixed measured region): exact 59/100, within one hand 93/100, kappa 0.39
(quadratic 0.55), 25 over / 16 under; stopped 74 %, moving 44 % (hands hovering near or holding the
rim outside the region). v1 column: 25/100, kappa < 0. The 23/30 figure was therefore optimistic;
59 % / 93 % is the number to quote. Second rater pending (rating_template.csv).

## Second rater (Hanadi Alhamdan, 2026-09-15, `ratings_HA.csv`)
87 frames judged, 13 marked u (hand hidden). Recomputed proxy vs HA: exact 65/87 = 75 %, within one hand 98 %,
kappa 0.62 (quadratic 0.74), 16 under / 6 over; moving 77 %, stopped 72 %. v1 vs HA: 37 %. Inter-rater CL vs
HA: exact 57 %, kappa 0.38 (quadratic 0.53) -- CL (thumbnails) scored many more frames as 0 hands (41 vs 19).
The proxy agrees with the better-informed rater about as well as the two raters agree with each other.
Manuscript v6 quotes both raters and the inter-rater agreement.
