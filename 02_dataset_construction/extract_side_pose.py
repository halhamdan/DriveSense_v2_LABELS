"""
In-Cabin Behaviour (Side-Camera) Pose Extraction
=================================================
Extracts lightweight posture/hand-position features from each session's
`{tag}_Side.mp4` using MediaPipe Pose, for the new "In-Cabin Behaviour Data"
category of the Nature Scientific Data descriptor.

Per sampled frame, exports:
  frame, timestamp_s, pose_detected,
  wrist_l_x, wrist_l_y, wrist_r_x, wrist_r_y,
  shoulder_l_x, shoulder_l_y, shoulder_r_x, shoulder_r_y,
  hands_on_wheel_proxy   -- mean visibility-weighted wrist detection near the
                            lower-frame steering-wheel region (0-1)
  posture_deviation      -- shoulder-midpoint displacement (in normalized
                            image coords) from this session's rolling median
                            posture, as an L2 distance
  motion_energy          -- sum of frame-to-frame landmark displacement
                            across all tracked landmarks (proxy for gross
                            body motion)

Frames are subsampled (default: every 6th frame, ~5 fps of a 30 fps source)
since posture/motion features do not require full frame-rate resolution;
this keeps the batch runtime tractable across ~76 valid sessions. Output is
later resampled/aggregated onto the 25 Hz fused timeline exactly like
Front_emotions.csv is (mean/aggregate per fused-row window).

Sessions excluded (per dataset_metadata.json audit, 2026-07-05):
  - D6_S2:  entire session excluded upstream (no fused/Side data at all)
  - D6_S3, D10_S3, D16_S1: Side camera disconnected during recording
    (Side.mp4 contains only the VBOX green chromakey background)

Usage:
  python extract_side_pose.py [--drivers 1 2 3] [--sessions 1 2] [--smoke]
                               [--frame-skip 6] [--out-root <path>]
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker, PoseLandmarkerOptions, RunningMode, PoseLandmark
)

# ── Configuration ──────────────────────────────────────────────────────────────
OUT_ROOT  = Path(os.environ.get("STAGING_FUSED_DIR", ""))   # same root as prepare_dataset.py
MODEL_PATH = Path(__file__).parent / "models" / "pose_landmarker_lite.task"

ALL_DRIVERS  = list(range(1, 21))
ALL_SESSIONS = [1, 2, 3, 4]

# (driver, session) pairs to skip entirely — Side.mp4 invalid or session missing
INVALID_SIDE = {(6, 2), (6, 3), (10, 3), (16, 1)}

DEFAULT_FRAME_SKIP = 6  # ~5 fps effective from a 30 fps source

# Steering-wheel region proxy: lower-center portion of the frame, normalized
# image coordinates (x_min, x_max, y_min, y_max). Tuned for the fixed VBOX
# side-camera framing used across all sessions; verify visually on spot-checks.
WHEEL_REGION = (0.25, 0.75, 0.55, 1.0)

POSE_LANDMARKS = PoseLandmark


def _make_landmarker() -> PoseLandmarker:
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return PoseLandmarker.create_from_options(options)


def _session_dir(driver_n: int, session_n: int) -> Path:
    return OUT_ROOT / f"D{driver_n}" / f"Session_{session_n}"


def _in_region(x: float, y: float, region: tuple[float, float, float, float]) -> bool:
    x_min, x_max, y_min, y_max = region
    return x_min <= x <= x_max and y_min <= y <= y_max


def process_session(driver_n: int, session_n: int, frame_skip: int, overwrite: bool = False) -> dict:
    tag = f"D{driver_n}_S{session_n}"
    result = {"driver": driver_n, "session": session_n, "tag": tag}

    if (driver_n, session_n) in INVALID_SIDE:
        result["status"] = "skipped_invalid_side"
        return result

    sess_dir = _session_dir(driver_n, session_n)
    side_path = sess_dir / f"{tag}_Side.mp4"
    out_csv = sess_dir / f"{tag}_Side_pose.csv"

    if not side_path.exists():
        result["status"] = "missing_side_video"
        return result
    if out_csv.exists() and not overwrite:
        result["status"] = "already_done"
        return result

    cap = cv2.VideoCapture(str(side_path))
    if not cap.isOpened():
        result["status"] = "open_failed"
        return result

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    rows = []
    prev_landmarks = None
    posture_history = []  # rolling shoulder-midpoint history for this session
    landmarker = _make_landmarker()

    try:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue

            timestamp_s = round(frame_idx / fps, 4)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect_for_video(mp_image, int(timestamp_s * 1000))

            row = {"frame": frame_idx, "timestamp_s": timestamp_s, "pose_detected": False,
                   "wrist_l_x": np.nan, "wrist_l_y": np.nan,
                   "wrist_r_x": np.nan, "wrist_r_y": np.nan,
                   "shoulder_l_x": np.nan, "shoulder_l_y": np.nan,
                   "shoulder_r_x": np.nan, "shoulder_r_y": np.nan,
                   "hands_on_wheel_proxy": np.nan,
                   "posture_deviation": np.nan,
                   "motion_energy": np.nan}

            if res.pose_landmarks:
                lm = res.pose_landmarks[0]
                wl = lm[POSE_LANDMARKS.LEFT_WRIST]
                wr = lm[POSE_LANDMARKS.RIGHT_WRIST]
                sl = lm[POSE_LANDMARKS.LEFT_SHOULDER]
                sr = lm[POSE_LANDMARKS.RIGHT_SHOULDER]

                row.update({
                    "pose_detected": True,
                    "wrist_l_x": wl.x, "wrist_l_y": wl.y,
                    "wrist_r_x": wr.x, "wrist_r_y": wr.y,
                    "shoulder_l_x": sl.x, "shoulder_l_y": sl.y,
                    "shoulder_r_x": sr.x, "shoulder_r_y": sr.y,
                })

                hands_in_region = [
                    _in_region(wl.x, wl.y, WHEEL_REGION) and wl.visibility > 0.5,
                    _in_region(wr.x, wr.y, WHEEL_REGION) and wr.visibility > 0.5,
                ]
                row["hands_on_wheel_proxy"] = sum(hands_in_region) / 2.0

                shoulder_mid = np.array([(sl.x + sr.x) / 2.0, (sl.y + sr.y) / 2.0])
                posture_history.append(shoulder_mid)
                if len(posture_history) >= 5:
                    baseline = np.median(np.stack(posture_history), axis=0)
                    row["posture_deviation"] = float(np.linalg.norm(shoulder_mid - baseline))

                cur_landmarks = np.array([[p.x, p.y] for p in lm])
                if prev_landmarks is not None:
                    row["motion_energy"] = float(np.linalg.norm(cur_landmarks - prev_landmarks, axis=1).sum())
                prev_landmarks = cur_landmarks

            rows.append(row)
            frame_idx += 1
    finally:
        landmarker.close()
        cap.release()

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    result.update({
        "status": "ok",
        "fps": fps,
        "total_frames": total_frames,
        "sampled_frames": len(df),
        "pose_detection_rate": float(df["pose_detected"].mean()) if len(df) else 0.0,
    })
    return result


def main():
    parser = argparse.ArgumentParser(description="In-Cabin Behaviour (Side-Camera) Pose Extraction")
    parser.add_argument("--drivers", type=int, nargs="+", default=None)
    parser.add_argument("--sessions", type=int, nargs="+", default=None)
    parser.add_argument("--smoke", action="store_true", help="Process only D1 Session 1 (sanity check)")
    parser.add_argument("--frame-skip", type=int, default=DEFAULT_FRAME_SKIP)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    drivers = [1] if args.smoke else (args.drivers or ALL_DRIVERS)
    sessions = [1] if args.smoke else (args.sessions or ALL_SESSIONS)

    print(f"In-Cabin Pose Extraction | frame_skip={args.frame_skip} | drivers={drivers} sessions={sessions}")

    summaries = []
    for d in drivers:
        for s in sessions:
            res = process_session(d, s, frame_skip=args.frame_skip, overwrite=args.overwrite)
            summaries.append(res)
            print(f"  {res['tag']:<8} {res['status']:<20} "
                  f"{res.get('sampled_frames', ''):>8} sampled  "
                  f"det_rate={res.get('pose_detection_rate', ''):.3f}" if res.get("status") == "ok"
                  else f"  {res['tag']:<8} {res['status']}")

    n_ok = sum(1 for r in summaries if r["status"] == "ok")
    print(f"\nDone: {n_ok} ok / {len(summaries)} total")


if __name__ == "__main__":
    main()
