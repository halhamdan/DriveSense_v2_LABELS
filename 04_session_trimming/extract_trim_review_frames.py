"""
Extracts a labeled contact-sheet (grid of timestamped stills) from
Front_blurred.mp4 for each flagged session, covering the time range around
its auto-detected start and/or end trim point, so the actual cutoff can be
picked visually without opening the video files directly.

Usage:
  python extract_trim_review_frames.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import PUBLISHED

OUT_DIR = Path(__file__).parent / "trim_review_output" / "trim_review_frames"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (driver, session, which_point, auto_value_s, orig_duration_s)
REVIEW_TARGETS = [
    (13, 3, "start", 1033.24, 3186.32),
    (16, 3, "start", 460.04, 2604.52),
    (2, 1, "start", 239.56, 2118.96),
    (5, 1, "start", 219.52, 3304.04),
    (17, 1, "start", 212.04, 1910.08),
    (20, 3, "start", 205.56, 3747.56),
    (12, 3, "start", 199.76, 2314.64),
    (13, 2, "start", 171.84, 1221.84),
    (4, 1, "start", 141.04, 2427.40),
    (4, 1, "end", 51.48, 2427.40),
    (9, 3, "end", 63.48, 2098.16),
    (13, 4, "end", 50.08, 1218.76),
]

N_FRAMES = 9
THUMB_W, THUMB_H = 320, 180
COLS = 3


def get_frame_at(cap, fps, t_s):
    idx = int(round(t_s * fps))
    idx = max(0, idx)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def make_contact_sheet(driver, session, which_point, auto_val, orig_dur):
    tag = f"D{driver}_S{session}"
    video_path = PUBLISHED / f"D{driver}" / f"Session_{session}" / f"{tag}_Front_blurred.mp4"
    if not video_path.exists():
        print(f"  [SKIP] {video_path} not found")
        return None

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if which_point == "start":
        t0, t1 = 0.0, min(auto_val + 40.0, orig_dur)
    else:
        t0, t1 = max(orig_dur - auto_val - 40.0, 0.0), orig_dur - 0.5

    timestamps = np.linspace(t0, t1, N_FRAMES)
    thumbs = []
    for t in timestamps:
        rgb = get_frame_at(cap, fps, t)
        if rgb is None:
            thumbs.append((t, None))
            continue
        img = Image.fromarray(rgb).resize((THUMB_W, THUMB_H))
        thumbs.append((t, img))
    cap.release()

    rows = -(-len(thumbs) // COLS)
    sheet = Image.new("RGB", (COLS * THUMB_W, rows * THUMB_H + 40), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
        font_small = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = font_small = ImageFont.load_default()

    title = f"{tag} -- auto-detected {which_point} trim = {auto_val:.1f}s (session duration {orig_dur:.0f}s)"
    draw.text((6, 6), title, fill="black", font=font)

    for i, (t, img) in enumerate(thumbs):
        r, c = divmod(i, COLS)
        x, y = c * THUMB_W, 40 + r * THUMB_H
        if img is not None:
            sheet.paste(img, (x, y))
        else:
            draw.rectangle([x, y, x + THUMB_W, y + THUMB_H], fill="#333")
        label = f"t={t:.1f}s"
        draw.rectangle([x, y, x + 90, y + 22], fill="black")
        draw.text((x + 4, y + 2), label, fill="white", font=font_small)

    out_path = OUT_DIR / f"{tag}_{which_point}_review.png"
    sheet.save(out_path)
    print(f"  Saved {out_path}")
    return out_path


def main():
    for driver, session, which_point, auto_val, orig_dur in REVIEW_TARGETS:
        print(f"D{driver}_S{session} ({which_point})...")
        make_contact_sheet(driver, session, which_point, auto_val, orig_dur)


if __name__ == "__main__":
    main()
