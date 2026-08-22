"""
Regenerates dataset_metadata.json for the final, trimmed, reorganized
dataset (Published_Dataset_Final), replacing the pre-trim version built by
prepare_dataset.py on 2026-07-05. Same schema, values recomputed from the
actual trimmed files so the manuscript's Data Records totals are accurate.

Usage:
  python regenerate_metadata.py
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

FINAL_ROOT = Path(os.environ.get("DATASET_ROOT", ""))
RAW_DIR = FINAL_ROOT / "Raw_Dataset"
PRE_DIR = FINAL_ROOT / "Preprocessed_Dataset"

INVALID_SIDE = {(6, 3), (10, 3), (16, 1)}
MISSING_SESSIONS = {(6, 2)}

# Mid-session non-driving stops identified by visual review of the 10 longest
# (>60s, speed<3km/h) stationary segments per session across the dataset --
# NOT an exhaustive review of all such segments (100+ exist, most are
# ordinary traffic/congestion in the high-traffic sessions and were left
# untouched). Data is NOT modified/cut for these -- flagged here only so
# users can exclude them if desired. Identified via Front/Side camera visual
# inspection (2026-07-29).
KNOWN_MIDSESSION_STOPS = [
    {"tag": "D7_S1", "start_s": 1219.4, "end_s": 1434.9, "duration_s": 215.6,
     "confidence": "confirmed", "description": "Gas station stop (fuel pump, pump-number signage, another vehicle fueling adjacent)"},
    {"tag": "D8_S3", "start_s": 144.9, "end_s": 411.7, "duration_s": 266.8,
     "confidence": "confirmed", "description": "Gas station stop, confirmed by driver; longest stationary segment in the dataset"},
    {"tag": "D20_S3", "start_s": 2745.5, "end_s": 3004.4, "duration_s": 258.9,
     "confidence": "reported_by_driver", "description": "Driver-reported minor incident during this stop; driver's seat is empty from approx. t=2900s to t=2930s "
     "(stepped out of vehicle), returning around t=2940-2950s carrying an object. Both cameras are interior-facing "
     "(driver-facing / cabin-facing) so neither can show the vehicle exterior or confirm visible damage from footage alone."},
]


def process_session(driver: int, session: int) -> dict | None:
    tag = f"D{driver}_S{session}"
    fused_path = PRE_DIR / f"D{driver}" / f"Session_{session}" / f"{tag}_fused.csv"
    if not fused_path.exists():
        return None

    fused = pd.read_csv(fused_path, usecols=["elapsed_s"])
    drive_duration_s = float(fused["elapsed_s"].iloc[-1])
    rows = len(fused)

    emo_path = PRE_DIR / f"D{driver}" / f"Session_{session}" / f"{tag}_Front_emotions.csv"
    face_detection_rate = None
    emotion_frames = None
    if emo_path.exists():
        emo = pd.read_csv(emo_path, usecols=["dominant_emotion"])
        emotion_frames = len(emo)
        face_detection_rate = round(float(emo["dominant_emotion"].notna().mean()), 4)

    has_road_camera = (driver, session) not in INVALID_SIDE
    pose_path = PRE_DIR / f"D{driver}" / f"Session_{session}" / f"{tag}_Side_pose.csv"
    pose_detection_rate = None
    if pose_path.exists():
        pose = pd.read_csv(pose_path, usecols=["pose_detected"])
        pose_detection_rate = round(float(pose["pose_detected"].mean()), 4) if len(pose) else None

    return {
        "driver": driver,
        "session": session,
        "tag": tag,
        "status": "ok",
        "has_physiology": True,
        "has_facial": True,
        "has_road_camera": has_road_camera,
        "drive_duration_s": round(drive_duration_s, 1),
        "rows": rows,
        "face_detection_rate": face_detection_rate,
        "emotion_frames": emotion_frames,
        "pose_detection_rate": pose_detection_rate,
    }


def main():
    sessions = []
    for d in range(1, 21):
        for s in range(1, 5):
            if (d, s) in MISSING_SESSIONS:
                continue
            row = process_session(d, s)
            if row is not None:
                sessions.append(row)

    ok = len(sessions)
    total_rows = sum(r["rows"] for r in sessions)
    total_duration_s = sum(r["drive_duration_s"] for r in sessions)
    sessions_full_cameras = sum(1 for r in sessions if r["has_road_camera"])
    sessions_front_only = ok - sessions_full_cameras

    metadata = {
        "pipeline": "prepare_dataset.py + apply_trim.py + blur_side_video.py + build_raw_tier.py + reorganize_final.py",
        "date": datetime.now(timezone.utc).isoformat(),
        "fuse_hz": 25.0,
        "trim_method": "sustained speed (>=15 km/h) + distance (>=120 m from session base) for >=80% of an 8 s window; "
                        "every session's cut point visually verified against source video (see trim_points_final.csv)",
        "sessions": sessions,
        "known_midsession_stops": KNOWN_MIDSESSION_STOPS,
        "totals": {
            "ok": ok,
            "skipped": len(MISSING_SESSIONS),
            "errors": 0,
            "total_rows": total_rows,
            "total_duration_s": round(total_duration_s, 1),
            "with_physiology": ok,
            "with_facial": ok,
            "sessions_full_cameras": sessions_full_cameras,
            "sessions_front_camera_only": sessions_front_only,
            "sessions_skipped": len(MISSING_SESSIONS),
            "sessions_total_valid": ok,
            "total_duration_h": round(total_duration_s / 3600.0, 2),
        },
        "notes": [
            "D6_S2 excluded: insufficient VBOX/EmotiBit temporal overlap (< 30 s).",
            "D6_S3, D10_S3, D16_S1: road-facing (side) camera not operational during recording. "
            "Side_blurred.mp4/Side_pose.csv are absent for these sessions. All other modalities "
            "(fused.csv, Front_blurred.mp4, Front_emotions.csv, VBOX_raw.csv, EmotiBit_*.csv) are valid.",
            "All files (both Raw_Dataset and Preprocessed_Dataset tiers) are trimmed to the driving-only "
            "window per session -- see trim_method above. No untrimmed files are published.",
            "Both camera streams (front and side) are face-blurred in the public release; unblurred "
            "originals are retained locally only and are not part of the public release.",
            "Sessions may include brief mid-drive non-driving stops (e.g. gas station) as part of "
            "naturalistic driving; these are NOT trimmed out (only session start/end are trimmed). "
            "See known_midsession_stops for the stops identified so far via a non-exhaustive visual "
            "review of the longest stationary segments -- other sessions may contain similar untagged "
            "stops among the many ordinary traffic-congestion pauses that were left unreviewed.",
        ],
    }

    out_path = FINAL_ROOT / "dataset_metadata.json"
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved {out_path}")
    print(f"  {ok} valid sessions, {total_rows:,} total rows, {total_duration_s/3600:.2f} total driving hours")
    print(f"  {sessions_full_cameras} sessions with both cameras, {sessions_front_only} front-camera-only")


if __name__ == "__main__":
    main()
