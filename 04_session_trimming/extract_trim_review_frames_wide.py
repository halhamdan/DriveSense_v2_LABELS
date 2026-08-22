"""
Extended end-point review for the 3 sessions (D4_S1, D9_S3, D13_S4) where the
original +-40s contact sheet showed continuous active driving with no visible
arrival/parking moment. Samples a much longer window before the very end of
the file (last ~300s, 15 frames, denser near the tail) to hunt for a real
arrival frame, or confirm the recording simply ends mid-drive.

Usage:
  python extract_trim_review_frames_wide.py
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

# (driver, session, orig_duration_s)
TARGETS = [
    (4, 1, 2427.40),
    (9, 3, 2098.16),
    (13, 4, 1218.76),
]

WINDOW_S = 300.0   # look back this far from the literal end of the file
N_FRAMES = 15
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


def make_contact_sheet(driver, session, orig_dur):
    tag = f"D{driver}_S{session}"
    video_path = PUBLISHED / f"D{driver}" / f"Session_{session}" / f"{tag}_Front_blurred.mp4"
    if not video_path.exists():
        print(f"  [SKIP] {video_path} not found")
        return None

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    true_frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    true_dur = true_frame_count / fps

    t0 = max(true_dur - WINDOW_S, 0.0)
    t1 = true_dur - (1.0 / fps)  # the literal last decodable frame
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

    title = f"{tag} -- WIDE end review: last {WINDOW_S:.0f}s (true video duration {true_dur:.1f}s vs fused {orig_dur:.1f}s)"
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

    out_path = OUT_DIR / f"{tag}_end_review_WIDE.png"
    sheet.save(out_path)
    print(f"  Saved {out_path} (true_dur={true_dur:.1f}s, fused_dur={orig_dur:.1f}s)")
    return out_path


def main():
    for driver, session, orig_dur in TARGETS:
        print(f"D{driver}_S{session} (wide end review)...")
        make_contact_sheet(driver, session, orig_dur)


if __name__ == "__main__":
    main()
