"""
Separate denominators for the video-derived features (Table 7 / Supplementary
Table S1): per session and pooled --
  front_frames        frames in the released Front video (ffprobe)
  facial_rows         rows in Front_emotions.csv (one per video frame)
  facial_detections   rows with a face box (face_x not missing)
  facial_sum_ok       detections whose seven confidences sum to 100 +/- 1
  side_frames         frames in the released Side video (if any)
  pose_rows           rows in Side_pose.csv (every 6th frame, 5 Hz)
  pose_detected       rows with pose_detected = 1
Usage: DATASET_ROOT=/path/to/release python make_video_feature_counts.py
Output: validation_output/video_feature_counts.csv + pooled summary
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("DATASET_ROOT", "") or r"C:\Users\halha\OneDrive - Durham University\Documents\Published_Dataset_Final_v3_VIDEO_ALIGNED")
OUT = Path(__file__).resolve().parent / "validation_output" / "video_feature_counts.csv"
EMO = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def frames(p: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames", "-of", "csv=p=0", str(p)], capture_output=True, text=True).stdout.strip()
    return int(out.split(",")[0])


rows = []
for f in sorted((ROOT / "Preprocessed_Dataset").rglob("*_fused.csv")):
    tag = f.stem.replace("_fused", ""); d, s = tag[1:].split("_S"); pre = f.parent; raw = ROOT / "Raw_Dataset" / f"D{d}" / f"Session_{s}"
    r = {"tag": tag, "fused_rows": sum(1 for _ in open(f, "rb")) - 1}
    fv = raw / f"{tag}_Front_blurred.mp4"; sv = raw / f"{tag}_Side_blurred.mp4"
    r["front_frames"] = frames(fv) if fv.exists() else 0; r["side_frames"] = frames(sv) if sv.exists() else 0
    e = pd.read_csv(pre / f"{tag}_Front_emotions.csv")
    cols = [c for c in e.columns if c.lower() in EMO]
    det = e["face_x"].notna() if "face_x" in e else e[cols].notna().all(axis=1)
    ssum = e.loc[det, cols].sum(axis=1)
    r.update(facial_rows=len(e), facial_detections=int(det.sum()), facial_sum_ok=int(((ssum - 100).abs() <= 1).sum()),
             facial_frame_min=int(e.frame.min()), facial_frame_max=int(e.frame.max()))
    pf = pre / f"{tag}_Side_pose.csv"
    if pf.exists():
        p = pd.read_csv(pf); r.update(pose_rows=len(p), pose_detected=int(p["pose_detected"].astype(bool).sum()), pose_frame_max=int(p.frame.max()))
    else:
        r.update(pose_rows=0, pose_detected=0, pose_frame_max=np.nan)
    rows.append(r); print(r, flush=True)
T = pd.DataFrame(rows); T.to_csv(OUT, index=False)
tot = T[["fused_rows", "front_frames", "side_frames", "facial_rows", "facial_detections", "facial_sum_ok", "pose_rows", "pose_detected"]].sum()
print("\nPOOLED:", tot.to_dict())
print(f"front videos {int((T.front_frames > 0).sum())}, side videos {int((T.side_frames > 0).sum())}, pose files {int((T.pose_rows > 0).sum())}")
print(f"facial rows == front frames in {(T.facial_rows == T.front_frames).sum()} sessions; detections {tot.facial_detections / tot.facial_rows * 100:.2f}% of rows; "
      f"sum within +/-1: {tot.facial_sum_ok / tot.facial_detections * 100:.2f}% of detections = {tot.facial_sum_ok / tot.facial_rows * 100:.2f}% of rows; "
      f"pose detected {tot.pose_detected / tot.pose_rows * 100:.1f}% of pose rows (per-session mean {(T.pose_detected / T.pose_rows.replace(0, np.nan) * 100).mean():.1f}%)")
print(f"front frames / 30 vs fused rows / 25 (s): median difference {(T.front_frames / 30 - T.fused_rows / 25).median():.1f} s (expected ~ -lead_in)")
