"""
Multimodal completeness around harsh events (Technical Validation:
"Multimodal completeness around harsh events").

For each released discrete harsh event, checks whether the other three
modalities (physiology, facial, cabin pose) have valid data available
during that same event window. Purely descriptive (data availability), not
a test of whether physiology or facial/cabin behaviour respond to harsh
events -- that question is out of scope for this Data Descriptor.

Physiology validity per fused.csv timestep: NOT (IBI_dropout_flag OR
HR_dropout_flag). An event is physiology-majority-valid if this holds for
at least half its duration.
Facial/cabin validity: at least one frame within the event window with a
non-missing detection (Front_emotions.csv expression columns /
Side_pose.csv pose_detected), reflecting their lower, non-25 Hz native
frame rates relative to fused.csv.

Usage:
  python make_multimodal_completeness_check.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import DATASET_ROOT

PREPROCESSED = DATASET_ROOT / "Preprocessed_Dataset"
OUT_DIR = Path(__file__).parent / "validation_output"
OUT_DIR.mkdir(exist_ok=True)

# Sessions with no side-camera file at all (camera disconnected).
NO_SIDE = {("D6", 3), ("D10", 3), ("D16", 1)}
# Side camera present but unreliable (11% pose-detection rate); excluded
# from the cabin-specific completeness stat (Video and pose completeness).
EXCLUDE_CABIN_UNRELIABLE = {("D2", 3)}

EMOTION_COLS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def main():
    rows = []
    for sess_dir in sorted(PREPROCESSED.glob("D*/Session_*")):
        driver = sess_dir.parent.name
        session_n = int(sess_dir.name.replace("Session_", ""))
        tag = f"{driver}_S{session_n}"
        fused_fp = sess_dir / f"{tag}_fused.csv"
        if not fused_fp.exists():
            continue
        df = pd.read_csv(fused_fp, usecols=["elapsed_s", "label", "IBI_dropout_flag", "HR_dropout_flag"])

        # Discrete events: same contiguous-run grouping as
        # 01_annotation/label_harsh_events.py:count_distinct_events.
        grp = (df["label"] != df["label"].shift()).cumsum()
        df = df.assign(grp=grp)
        events = df[df["label"] != "Normal"].groupby(["label", "grp"])

        front_fp = sess_dir / f"{tag}_Front_emotions.csv"
        side_fp = sess_dir / f"{tag}_Side_pose.csv"
        front = pd.read_csv(front_fp, usecols=["timestamp_s"] + EMOTION_COLS) if front_fp.exists() else None
        has_side_file = (driver, session_n) not in NO_SIDE and side_fp.exists()
        side = pd.read_csv(side_fp, usecols=["timestamp_s", "pose_detected"]) if has_side_file else None
        cabin_reliable = (driver, session_n) not in EXCLUDE_CABIN_UNRELIABLE

        if front is not None:
            front_valid = front[EMOTION_COLS].notna().any(axis=1).to_numpy()
            front_t = front["timestamp_s"].to_numpy()
        if side is not None:
            side_valid = side["pose_detected"].fillna(False).to_numpy().astype(bool)
            side_t = side["timestamp_s"].to_numpy()

        for (label, _grp_id), sub in events:
            t0, t1 = sub["elapsed_s"].iloc[0], sub["elapsed_s"].iloc[-1]

            phys_valid_frac = float(
                (~(sub["IBI_dropout_flag"].astype(bool) | sub["HR_dropout_flag"].astype(bool))).mean()
            )

            if front is not None:
                in_win = (front_t >= t0) & (front_t <= t1)
                facial_any_valid = bool(front_valid[in_win].any()) if in_win.any() else False
            else:
                facial_any_valid = False

            if side is not None and cabin_reliable:
                in_win = (side_t >= t0) & (side_t <= t1)
                cabin_any_valid = bool(side_valid[in_win].any()) if in_win.any() else False
            else:
                cabin_any_valid = False

            rows.append({
                "tag": tag, "driver": driver, "session": session_n, "label": label,
                "duration_s": t1 - t0,
                "phys_valid_frac": phys_valid_frac,
                "phys_majority_valid": phys_valid_frac >= 0.5,
                "has_side_file": side is not None,
                "cabin_reliable": cabin_reliable,
                "facial_any_valid": facial_any_valid,
                "cabin_any_valid": cabin_any_valid,
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "multimodal_completeness_events.csv", index=False)

    print(f"Total events: {len(out)}")
    cabin_eligible = out[out["has_side_file"] & out["cabin_reliable"]]
    print(f"Cabin-eligible events: {len(cabin_eligible)} / {len(out)}")
    print("\nBy class:")
    for lbl, g in out.groupby("label"):
        ce = g[g["has_side_file"] & g["cabin_reliable"]]
        all3 = ce["phys_majority_valid"] & ce["facial_any_valid"] & ce["cabin_any_valid"]
        print(f"  {lbl:20s} n={len(g):5d}  phys={g['phys_majority_valid'].mean()*100:5.1f}%  "
              f"facial={g['facial_any_valid'].mean()*100:5.1f}%  "
              f"cabin(n={len(ce)})={ce['cabin_any_valid'].mean()*100:5.1f}%  all3={all3.mean()*100:5.1f}%")
    all3_overall = (cabin_eligible["phys_majority_valid"] & cabin_eligible["facial_any_valid"]
                     & cabin_eligible["cabin_any_valid"])
    print(f"\n  {'All classes':20s} n={len(out):5d}  phys={out['phys_majority_valid'].mean()*100:5.1f}%  "
          f"facial={out['facial_any_valid'].mean()*100:5.1f}%  "
          f"cabin(n={len(cabin_eligible)})={cabin_eligible['cabin_any_valid'].mean()*100:5.1f}%  "
          f"all3={all3_overall.mean()*100:5.1f}%")


if __name__ == "__main__":
    main()
