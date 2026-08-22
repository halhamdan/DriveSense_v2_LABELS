"""
Full 79-session screening pass for garage-trim cutoffs, using an EXACT
(non-heuristic) mapping from fused.csv elapsed-time to video frame number,
instead of the earlier approximation that assumed video frame 0 == fused.csv
elapsed_s 0.

Why this is needed: fused.csv is already clipped to the VBOX/EmotiBit overlap
window and passed through the pipeline's own motion filter (>5 km/h, single
sample, no sustain) before being published -- so its elapsed_s=0 is NOT the
same instant as video frame 0. Both the raw VBOX CSV and the published
fused.csv carry real Unix timestamps, so the correspondence can be computed
exactly:

    video_frame_for(fused_elapsed_s) =
        round((fused_unix_time_at(fused_elapsed_s) - raw_vbox_unix_time[0]) * fps)

This script re-uses garage_trim_detection.csv's auto-detected trim_start_s /
trim_end_s (computed on fused.csv's own timeline) purely as WHERE to look --
it maps those cutoffs to the correct video frame, grabs a single frame there,
and tiles all 79 sessions into a small number of screening composite sheets
(for start cutoffs and end cutoffs separately) for fast visual review.

Analysis only -- does not touch any files.

Usage:
  python build_full_trim_screening.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from _paths import PUBLISHED, STAGING_DATA_ROOT as DATA_ROOT, STAGING_LABELED_DIR as LABELED_DIR

sys.path.insert(0, str(REPO_ROOT / "02_dataset_construction"))
import parse_labeled as pl  # noqa: E402

OUT_DIR = Path(__file__).parent / "trim_review_output"
SHEET_DIR = OUT_DIR / "full_screening"
SHEET_DIR.mkdir(parents=True, exist_ok=True)

THUMB_W, THUMB_H = 220, 124
COLS = 5
SESSIONS_PER_SHEET = 20


def raw_vbox_t0(driver: int, session: int) -> float | None:
    csv_path = LABELED_DIR / f"D{driver}_S{session}_LABELED.csv"
    session_dir = DATA_ROOT / f"D{driver}" / f"Session {session}"
    if not csv_path.exists():
        return None
    df = pl.load_labeled_csv(csv_path, session_dir=session_dir if session_dir.exists() else None)
    if df.empty:
        return None
    return float(df["unix_time"].iloc[0])


def get_frame_at(cap, fps, t_s):
    idx = max(0, int(round(t_s * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def main():
    trims = pd.read_csv(Path(__file__).parent / "trim_review_output" / "garage_trim_detection.csv")
    trims = trims[trims["status"] == "ok"].reset_index(drop=True)

    start_thumbs = []  # (label, image or None)
    end_thumbs = []

    for _, row in trims.iterrows():
        d, s = int(row["driver"]), int(row["session"])
        tag = f"D{d}_S{s}"
        video_path = PUBLISHED / f"D{d}" / f"Session_{s}" / f"{tag}_Front_blurred.mp4"
        fused_path = PUBLISHED / f"D{d}" / f"Session_{s}" / f"{tag}_fused.csv"
        if not video_path.exists() or not fused_path.exists():
            print(f"  [SKIP] {tag}: missing files")
            start_thumbs.append((tag, None))
            end_thumbs.append((tag, None))
            continue

        t0_raw = raw_vbox_t0(d, s)
        if t0_raw is None:
            print(f"  [SKIP] {tag}: no raw VBOX unix_time")
            start_thumbs.append((tag, None))
            end_thumbs.append((tag, None))
            continue

        fused = pd.read_csv(fused_path, usecols=["elapsed_s", "unix_time"])

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # --- start cutoff ---
        start_elapsed = float(row["trim_start_s"])
        idx = (fused["elapsed_s"] - start_elapsed).abs().idxmin()
        target_unix = float(fused["unix_time"].iloc[idx])
        video_t_start = target_unix - t0_raw
        rgb = get_frame_at(cap, fps, video_t_start)
        img = Image.fromarray(rgb).resize((THUMB_W, THUMB_H)) if rgb is not None else None
        start_thumbs.append((f"{tag} @{video_t_start:.0f}s", img))

        # --- end cutoff ---
        end_elapsed = float(fused["elapsed_s"].iloc[-1]) - float(row["trim_end_s"])
        idx2 = (fused["elapsed_s"] - end_elapsed).abs().idxmin()
        target_unix2 = float(fused["unix_time"].iloc[idx2])
        video_t_end = target_unix2 - t0_raw
        rgb2 = get_frame_at(cap, fps, video_t_end)
        img2 = Image.fromarray(rgb2).resize((THUMB_W, THUMB_H)) if rgb2 is not None else None
        end_thumbs.append((f"{tag} @{video_t_end:.0f}s", img2))

        cap.release()
        print(f"  {tag}: start video_t={video_t_start:.1f}s (fused {start_elapsed:.1f}s)   "
              f"end video_t={video_t_end:.1f}s (fused {end_elapsed:.1f}s)")

    save_screening_sheets(start_thumbs, "start")
    save_screening_sheets(end_thumbs, "end")

    # persist the exact video-time cutoffs for later use
    out_rows = []
    for (slabel, _), (elabel, _) in zip(start_thumbs, end_thumbs):
        out_rows.append({"start_label": slabel, "end_label": elabel})
    pd.DataFrame(out_rows).to_csv(OUT_DIR / "full_screening_labels.csv", index=False)


def save_screening_sheets(thumbs, which):
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    n_sheets = -(-len(thumbs) // SESSIONS_PER_SHEET)
    for sheet_i in range(n_sheets):
        chunk = thumbs[sheet_i * SESSIONS_PER_SHEET:(sheet_i + 1) * SESSIONS_PER_SHEET]
        rows = -(-len(chunk) // COLS)
        sheet = Image.new("RGB", (COLS * THUMB_W, rows * (THUMB_H + 18)), "white")
        draw = ImageDraw.Draw(sheet)
        for i, (label, img) in enumerate(chunk):
            r, c = divmod(i, COLS)
            x, y = c * THUMB_W, r * (THUMB_H + 18)
            if img is not None:
                sheet.paste(img, (x, y + 18))
            else:
                draw.rectangle([x, y + 18, x + THUMB_W, y + 18 + THUMB_H], fill="#333")
            draw.text((x + 2, y + 1), label, fill="black", font=font)
        out_path = SHEET_DIR / f"screen_{which}_{sheet_i + 1:02d}.png"
        sheet.save(out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
