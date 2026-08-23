# Model card: facial-expression and in-cabin pose models

This card documents the two pretrained model pipelines used to derive the
released `Front_emotions.csv` (facial-expression confidence scores) and
`Side_pose.csv` (in-cabin pose keypoints) files. Both are third-party models,
used as-is, not trained or fine-tuned for this dataset. Their outputs are
**model predictions, not ground truth** (Data Descriptor, Methods and Usage
Notes) — this card exists to help users judge fitness for their own purpose,
not to imply the outputs have been validated as accurate labels.

## Facial-expression pipeline

- **Face detection**: MTCNN (`mtcnn` 1.0.0), within a DeepFace-based workflow
  (`deepface` 0.0.100, TensorFlow 2.21.0 backend).
- **Expression classification**: DeepFace's default emotion model, which
  outputs confidence scores (percentage scale) for seven categories (angry,
  disgust, fear, happy, neutral, sad, surprise) plus a dominant-expression
  label. DeepFace's emotion classifier is trained on FER-style facial
  expression datasets; the exact training composition is set by the
  `deepface` package maintainers, not by us, and is not independently
  verified here.
- **Detection-failure behaviour**: verified directly in this project (Data
  Descriptor, Technical Validation) — a frame with no detected face has all
  seven expression columns, the dominant-expression label, and the four
  face-box coordinates set to a missing value, with no separate
  detection-flag column.

### Known/documented limitations relevant to this release

Facial-expression-recognition systems, including those built on FER-style
training data, have been repeatedly documented in the literature to show
accuracy disparities by skin tone and demographic group -- e.g. Xu et al.
(2022), *"Investigating Bias in Deep Face Analysis"* and related work on
racial bias in facial expression recognition report systematically higher
error rates for darker skin tones and for specific demographic subgroups
across several FER systems. This project has not independently measured
whether this pipeline's accuracy varies across our own participant pool (20
drivers, one geographic/demographic population); the general finding is
cited here as a documented risk for this *class* of model, not a measured
property of this release. Researchers requiring calibrated, bias-audited
expression estimates should treat the released confidence scores as raw
model output requiring their own validation, not as ground truth (Data
Descriptor, Usage Notes).

A stratified visual accuracy check performed for this release (10 of 20
drivers, one weekday and one weekend session each) found the released face
bounding box correctly localised the driver's face in all 10 sampled frames
(Data Descriptor, Technical Validation); this checks box *localisation*
only, not expression-label correctness, for which no ground truth exists in
this release.

## In-cabin pose pipeline

- **Pose estimation**: MediaPipe Pose Landmarker task (`mediapipe` 0.10.35,
  `pose_landmarker_lite.task` model), the smallest/fastest of MediaPipe's
  three published pose-model tiers (lite/full/heavy), chosen for
  batch-processing throughput across the full dataset. The `_lite` tier is
  expected to trade some accuracy for speed relative to `_full`/`_heavy`;
  this release does not include a comparison against the larger tiers.
- **Released outputs**: wrist and shoulder keypoint coordinates
  (image-normalised, Methods) and three derived summary features
  (`hands_on_wheel_proxy`, `posture_deviation`, `motion_energy`).
- **Detection-failure behaviour**: a frame with no detected pose has
  `pose_detected = False` and all keypoint/summary columns set to a missing
  value (explicit, machine-readable, unlike the facial pipeline above).

### Known/documented limitations relevant to this release

Pose-estimation models in general are well documented to lose accuracy under
partial occlusion and loose/baggy clothing, which obscures the silhouette
and joint-adjacent visual cues these models rely on. This project's own
direct checks (Data Descriptor, Technical Validation) found:

- The side camera's fixed profile framing gives consistently better-observed
  landmarks for whichever arm is nearer the camera than for the far,
  partially self-occluded arm, and *which* side (left or right) is the
  better-observed one is not fixed and cannot be predicted from the released
  columns alone -- it must be checked per frame against the source image.
- A stratified visual check (10 of 20 drivers, weekday and weekend sessions)
  found the wrist keypoint nearest a visible hand consistently well-localised
  in every sampled frame, including one case (driver 5) where the keypoint
  correctly tracked a hand raised away from the wheel rather than defaulting
  to a fixed position.
- Shoulder keypoints land in the correct general region (neck/collar area)
  but are not always cleanly separated into a correct left/right assignment
  (Technical Validation).

We have not independently measured whether pose accuracy in this release
varies with participants' clothing style specifically; several drivers in
this collection wear loose traditional garments (e.g. thobe, abaya), which
is a plausible risk factor given the general, well-documented
loose-clothing sensitivity of pose-estimation models, but this is a
reasoned caveat, not a measured finding, and should not be read as a
verified claim about this release's actual accuracy by clothing type.

## Recommended use

Both pipelines' outputs should be treated as noisy, unaudited model
predictions for exploratory or method-development use, not as validated
ground truth for driver state, identity, or demographic inference. Users
building on either modality for a downstream claim about accuracy or
fairness across subgroups should conduct their own stratified validation
rather than assume uniform performance from the completeness/localisation
checks reported in the Data Descriptor.
