"""
Builds the "raw" tier for each session -- native per-sensor sampling rate,
unresampled, no engineered features, no label column -- trimmed to the exact
same real-world window already applied to the published fused.csv:

  {tag}_VBOX_raw.csv           -- native VBOX rate, decimal-degree GPS, label dropped
  {tag}_EmotiBit_{SIGNAL}.csv  -- one file per physiological channel, native rate

Also trims {tag}_Side_pose.csv (already extracted from the raw, untrimmed
Side.mp4 by Data_Fusion/extract_side_pose.py) using the same raw-video frame
range already computed for Side_blurred.mp4, since its `frame` column indexes
the raw Side.mp4 identically to Front_emotions.csv.

Writes into Published_Dataset_Trimmed (same tree as apply_trim.py's output),
alongside the already-trimmed fused.csv/Front_blurred.mp4/Front_emotions.csv/
Side_blurred.mp4 -- final Raw_Dataset/Preprocessed_Dataset split happens in a
separate reorganization step once this is done.

Usage:
  python build_raw_tier.py --drivers 1 --sessions 1   # single-session test
  python build_raw_tier.py --drivers 1 2 3 4 5 6
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from apply_trim import raw_vbox_t0, get_fps, OUT_ROOT, DATA_ROOT, LABELED_DIR, CODE_DIR, PUBLISHED

sys.path.insert(0, str(CODE_DIR))
import parse_labeled as pl  # noqa: E402
import emotibit_sync as es  # noqa: E402

EMOTIBIT_SIGNALS = [
    "EDA", "EDA_LEVEL", "SCR_FREQ", "HR", "IBI",
    "TEMP_CONTACT", "TEMP_THERMOPILE",
    "PPG_IR", "PPG_RED", "PPG_GREEN",
    "ACC", "GYRO", "MAG",
]


def build_vbox_raw(driver: int, session: int, tag: str, out_dir: Path,
                    w_start: float, w_end: float) -> None:
    csv_path = LABELED_DIR / f"D{driver}_S{session}_LABELED.csv"
    session_dir = DATA_ROOT / f"D{driver}" / f"Session {session}"
    df = pl.load_labeled_csv(csv_path, session_dir=session_dir if session_dir.exists() else None)
    keep = df[(df["unix_time"] >= w_start) & (df["unix_time"] <= w_end)].copy()
    if keep.empty:
        print(f"  [WARN] {tag}: VBOX raw window produced 0 rows")
        return
    keep["elapsed_s"] = keep["unix_time"] - keep["unix_time"].iloc[0]
    if "label" in keep.columns:
        keep = keep.drop(columns=["label"])
    out_path = out_dir / f"{tag}_VBOX_raw.csv"
    keep.to_csv(out_path, index=False, float_format="%.6f")
    print(f"  VBOX_raw.csv written: {len(keep)} rows")


def build_emotibit_raw(driver: int, session: int, tag: str, out_dir: Path,
                        w_start: float, w_end: float) -> None:
    session_dir = DATA_ROOT / f"D{driver}" / f"Session {session}"
    if not session_dir.exists():
        print(f"  [WARN] {tag}: session_dir not found for EmotiBit -- skipping")
        return
    try:
        signals, meta = es.load_emotibit(session_dir)
    except Exception as exc:
        print(f"  [WARN] {tag}: EmotiBit load failed ({exc}) -- skipping")
        return

    n_written = 0
    for sig_name in EMOTIBIT_SIGNALS:
        df = signals.get(sig_name)
        if df is None or df.empty:
            continue
        keep = df[(df["unix_time"] >= w_start) & (df["unix_time"] <= w_end)].copy()
        if keep.empty:
            continue
        keep["elapsed_s"] = keep["unix_time"] - w_start
        cols = ["unix_time", "elapsed_s"] + [c for c in keep.columns if c not in ("unix_time", "elapsed_s")]
        keep = keep[cols]
        out_path = out_dir / f"{tag}_EmotiBit_{sig_name}.csv"
        keep.to_csv(out_path, index=False, float_format="%.6f")
        n_written += 1
    print(f"  EmotiBit raw signals written: {n_written}/{len(EMOTIBIT_SIGNALS)}")


def trim_side_pose(tag: str, in_dir: Path, out_dir: Path, frame_start: int, frame_end: int, fps: float) -> None:
    pose_in = in_dir / f"{tag}_Side_pose.csv"
    if not pose_in.exists():
        print(f"  [INFO] {tag}: no Side_pose.csv found -- skipping")
        return
    df = pd.read_csv(pose_in)
    keep = df[(df["frame"] >= frame_start) & (df["frame"] <= frame_end)].copy()
    keep["frame"] = keep["frame"] - frame_start
    keep["timestamp_s"] = (keep["frame"] / fps).round(4)
    out_path = out_dir / f"{tag}_Side_pose.csv"
    keep.to_csv(out_path, index=False)
    print(f"  Side_pose.csv trimmed: {len(keep)} rows")


def process_session(driver: int, session: int) -> None:
    tag = f"D{driver}_S{session}"
    trimmed_dir = OUT_ROOT / f"D{driver}" / f"Session_{session}"
    orig_dir = PUBLISHED / f"D{driver}" / f"Session_{session}"

    fused_path = trimmed_dir / f"{tag}_fused.csv"
    if not fused_path.exists():
        print(f"{tag}: [SKIP] no trimmed fused.csv found")
        return

    fused = pd.read_csv(fused_path, usecols=["unix_time"])
    w_start, w_end = float(fused["unix_time"].iloc[0]), float(fused["unix_time"].iloc[-1])

    print(f"\n{tag}: window unix[{w_start:.3f}, {w_end:.3f}]")
    build_vbox_raw(driver, session, tag, trimmed_dir, w_start, w_end)
    build_emotibit_raw(driver, session, tag, trimmed_dir, w_start, w_end)

    t0_raw = raw_vbox_t0(driver, session)
    front_video = orig_dir / f"{tag}_Front_blurred.mp4"
    if t0_raw is not None and front_video.exists():
        fps = get_fps(front_video)
        frame_start = max(0, round((w_start - t0_raw) * fps))
        frame_end = round((w_end - t0_raw) * fps)
        trim_side_pose(tag, orig_dir, trimmed_dir, frame_start, frame_end, fps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drivers", type=int, nargs="+", required=True)
    ap.add_argument("--sessions", type=int, nargs="+", default=[1, 2, 3, 4])
    args = ap.parse_args()

    for d in args.drivers:
        for s in args.sessions:
            process_session(d, s)


if __name__ == "__main__":
    main()
