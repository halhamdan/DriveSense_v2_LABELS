"""
Applies the finalized start/end trim points (trim_review_output/trim_points_final.csv,
manual_start_s / manual_end_s columns) to a session's published files, writing
results to a NEW output root (never touches Published_Dataset).

For each session, trims:
  - {tag}_fused.csv           -- cut directly on its own elapsed_s
  - {tag}_Front_blurred.mp4   -- cut on the corresponding video frame range,
                                  computed via the exact Unix-time mapping
                                  between fused.csv and the raw VBOX video
                                  (see build_full_trim_screening.py for why
                                  video frame 0 != fused elapsed_s 0)
  - {tag}_Front_emotions.csv  -- cut on the same video frame range (its
                                  `frame` column already indexes raw video
                                  frames 1:1)

Side.mp4 is handled separately (needs face-blurring first).

Usage:
  python apply_trim.py --drivers 1 2 3 4 5 6
  python apply_trim.py --drivers 1 --sessions 1   # single-session test
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from _paths import PUBLISHED, STAGING_DATA_ROOT as DATA_ROOT, STAGING_LABELED_DIR as LABELED_DIR

CODE_DIR = REPO_ROOT / "02_dataset_construction"
sys.path.insert(0, str(CODE_DIR))
import parse_labeled as pl  # noqa: E402

OUT_ROOT = Path(os.environ.get("STAGING_TRIMMED_ROOT", ""))
TRIM_CSV = Path(__file__).parent / "trim_review_output" / "trim_points_final.csv"

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")  # assumed on PATH unless overridden
FFPROBE = os.environ.get("FFPROBE", "ffprobe")


def raw_vbox_t0(driver: int, session: int) -> float | None:
    csv_path = LABELED_DIR / f"D{driver}_S{session}_LABELED.csv"
    session_dir = DATA_ROOT / f"D{driver}" / f"Session {session}"
    if not csv_path.exists():
        return None
    df = pl.load_labeled_csv(csv_path, session_dir=session_dir if session_dir.exists() else None)
    if df.empty:
        return None
    return float(df["unix_time"].iloc[0])


def get_fps(video_path: Path) -> float:
    cmd = [FFPROBE, "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(video_path)]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    num, den = out.split("/")
    return float(num) / float(den)


def trim_fused_csv(fused_path: Path, out_path: Path, start_s: float, end_s: float) -> tuple[float, float]:
    """Returns (unix_time_at_new_start, unix_time_at_new_end) for video-frame mapping."""
    df = pd.read_csv(fused_path)
    orig_end = df["elapsed_s"].iloc[-1]
    keep = df[(df["elapsed_s"] >= start_s) & (df["elapsed_s"] <= orig_end - end_s)].copy()
    unix_start = float(keep["unix_time"].iloc[0])
    unix_end = float(keep["unix_time"].iloc[-1])
    keep["elapsed_s"] = keep["elapsed_s"] - keep["elapsed_s"].iloc[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keep.to_csv(out_path, index=False, float_format="%.6f")
    return unix_start, unix_end


def trim_video(video_path: Path, out_path: Path, frame_start: int, frame_end: int, fps: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vf = f"select='between(n\\,{frame_start}\\,{frame_end})',setpts=PTS-STARTPTS"
    cmd = [FFMPEG, "-y", "-i", str(video_path), "-vf", vf,
           "-c:v", "libx264", "-crf", "23", "-preset", "medium",
           "-an", str(out_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {video_path.name}: {result.stderr[-2000:]}")


def trim_emotions_csv(emotions_path: Path, out_path: Path, frame_start: int, frame_end: int, fps: float) -> None:
    df = pd.read_csv(emotions_path)
    keep = df[(df["frame"] >= frame_start) & (df["frame"] <= frame_end)].copy()
    keep["frame"] = keep["frame"] - frame_start
    keep["timestamp_s"] = (keep["frame"] / fps).round(4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keep.to_csv(out_path, index=False)


def process_session(driver: int, session: int, row: pd.Series) -> None:
    tag = f"D{driver}_S{session}"
    in_dir = PUBLISHED / f"D{driver}" / f"Session_{session}"
    out_dir = OUT_ROOT / f"D{driver}" / f"Session_{session}"

    start_s = float(row["manual_start_s"])
    end_s = float(row["manual_end_s"])

    print(f"\n{tag}: trim_start={start_s:.2f}s  trim_end={end_s:.2f}s")

    # 1. fused.csv
    fused_in = in_dir / f"{tag}_fused.csv"
    fused_out = out_dir / f"{tag}_fused.csv"
    if not fused_in.exists():
        print(f"  [SKIP] {fused_in} not found")
        return
    unix_start, unix_end = trim_fused_csv(fused_in, fused_out, start_s, end_s)
    print(f"  fused.csv written: {fused_out.name}")

    # 2. video frame mapping
    t0_raw = raw_vbox_t0(driver, session)
    if t0_raw is None:
        print(f"  [WARN] no raw VBOX unix_time -- skipping video/emotions trim")
        return

    front_in = in_dir / f"{tag}_Front_blurred.mp4"
    if not front_in.exists():
        print(f"  [WARN] {front_in} not found -- skipping video/emotions trim")
        return
    fps = get_fps(front_in)
    frame_start = max(0, round((unix_start - t0_raw) * fps))
    frame_end = round((unix_end - t0_raw) * fps)
    print(f"  video frame range: [{frame_start}, {frame_end}]  (fps={fps:.3f})")

    # 3. trim video
    front_out = out_dir / f"{tag}_Front_blurred.mp4"
    trim_video(front_in, front_out, frame_start, frame_end, fps)
    print(f"  video written: {front_out.name}")

    # 4. trim emotions csv
    emo_in = in_dir / f"{tag}_Front_emotions.csv"
    if emo_in.exists():
        emo_out = out_dir / f"{tag}_Front_emotions.csv"
        trim_emotions_csv(emo_in, emo_out, frame_start, frame_end, fps)
        print(f"  emotions.csv written: {emo_out.name}")

    # 5. trim Side_blurred.mp4 -- this one already lives in OUT_ROOT (written by
    # blur_side_video.py, blurred but not yet trimmed), not in the original
    # Published_Dataset tree, so it's trimmed in place via a temp file.
    side_blurred = out_dir / f"{tag}_Side_blurred.mp4"
    if side_blurred.exists():
        side_fps = get_fps(side_blurred)
        side_frame_start = max(0, round((unix_start - t0_raw) * side_fps))
        side_frame_end = round((unix_end - t0_raw) * side_fps)
        tmp_out = out_dir / f"{tag}_Side_blurred_TMP.mp4"
        trim_video(side_blurred, tmp_out, side_frame_start, side_frame_end, side_fps)
        side_blurred.unlink()
        tmp_out.rename(side_blurred)
        print(f"  Side_blurred.mp4 trimmed in place: [{side_frame_start}, {side_frame_end}] (fps={side_fps:.3f})")
    else:
        print(f"  [INFO] no Side_blurred.mp4 found yet for {tag} -- skipping")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drivers", type=int, nargs="+", required=True)
    ap.add_argument("--sessions", type=int, nargs="+", default=[1, 2, 3, 4])
    args = ap.parse_args()

    trims = pd.read_csv(TRIM_CSV)
    for d in args.drivers:
        for s in args.sessions:
            row = trims[(trims["driver"] == d) & (trims["session"] == s)]
            if row.empty:
                continue
            process_session(d, s, row.iloc[0])


if __name__ == "__main__":
    main()
