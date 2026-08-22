"""
Two-pass face-blurring for {tag}_Front.mp4, replacing prepare_dataset.py's
original single-pass Front-camera blurring.

Why this exists: verify_deidentification.py's audit found genuine, fully
identifiable unblurred frames in the currently-released Front_blurred.mp4
files (confirmed visually, e.g. D1_S1 frames 8114 and 15705 -- conf=1.00,
clearly a sharp, front-on face with no blur at all). This is exactly the
Side-camera pipeline's own v1-vs-v2 lesson (see blur_side_video.py's
docstring) applied to the Front camera: single-pass per-frame detection
genuinely misses faces outright on some frames (not just low-confidence
misses), and writes those frames through completely unblurred with no
gap-filling. This script ports blur_side_video.py's two-pass detect-then-
fill design onto the Front camera:

  Pass 1: run MTCNN (keep_all=True -- a passenger's face can also appear
          briefly in the Front frame, not just the driver's) over every
          frame of the ORIGINAL, UNTRIMMED, UNBLURRED {tag}_Front.mp4
          (retained locally, never released), storing the detected box
          list per frame.
  Fill:   temporal carry-forward (forward-fill then backward-fill for any
          leading gap), with extra safety padding on carried-forward boxes.
  Pass 2: re-read and blur using the (filled) box list, write a full,
          untrimmed, corrected {tag}_Front_blurred_v2.mp4.
  Trim:   cut the corrected video to the EXACT same frame range as the
          currently-released Front_blurred.mp4, using the identical
          unix-time-based frame-range computation apply_trim.py already
          uses (raw_vbox_t0 + fused.csv's first/last unix_time), so the
          output is a drop-in replacement with matching duration/frame
          count.

Writes into STAGING_FRONT_V2_ROOT -- never overwrites the original
Published_Dataset or Published_Dataset_Final trees. Once spot-checked and
confirmed correct, the operator should manually replace the corresponding
file(s) in the release tree and re-run verify_deidentification.py on the
replaced file to confirm zero genuine residual detections before publishing.

Usage:
  python blur_front_video.py --drivers 1 --sessions 1   # single-session test
  python blur_front_video.py --drivers 1 2 3 4 5 6
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import PUBLISHED

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "04_session_trimming"))
from apply_trim import raw_vbox_t0, get_fps, TRIM_CSV  # noqa: E402

OUT_ROOT = Path(os.environ.get("STAGING_FRONT_V2_ROOT", ""))

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
BLUR_K = 51
BATCH = 32  # facenet_pytorch batches a list of frames as one GPU call -- ~50+ fps vs ~9 fps single-frame
FACE_PROB_THRESH = 0.7
MAX_CARRY_GAP_FRAMES = 300  # ~10s at 30fps

_mtcnn = None
_device = None


def _init_face_detector():
    global _mtcnn, _device
    if _mtcnn is not None:
        return
    import torch
    from facenet_pytorch import MTCNN
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _mtcnn = MTCNN(
        image_size=160, margin=20, min_face_size=30,
        thresholds=[0.6, 0.7, 0.7], factor=0.709,
        keep_all=True, device=_device,
    )
    print(f"  Face detector: MTCNN on {_device.upper()} (keep_all=True)")


def _check_nvenc() -> bool:
    cmd = [FFMPEG, "-y", "-f", "lavfi", "-i", "color=black:s=64x64:d=0.1",
           "-c:v", "h264_nvenc", "-frames:v", "1", "-f", "null", "-"]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=15)
        return r.returncode == 0
    except Exception:
        return False


def _open_writer(out_path: Path, fps: float, width: int, height: int, use_nvenc: bool):
    codec_args = (
        ["-c:v", "h264_nvenc", "-rc", "vbr", "-cq", "28", "-preset", "p4", "-tune", "hq"]
        if use_nvenc else
        ["-c:v", "libx264", "-crf", "23", "-preset", "medium"]
    )
    cmd = [FFMPEG, "-y", "-f", "rawvideo", "-pixel_format", "bgr24",
           "-video_size", f"{width}x{height}", "-framerate", str(int(round(fps))),
           "-i", "pipe:0"] + codec_args + ["-pix_fmt", "yuv420p", str(out_path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def detect_all_frames(video_path: Path) -> tuple[list, float, int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    per_frame_boxes: list = [None] * total
    idx = 0
    batch = []

    def flush(batch, start_idx):
        rgb_list = [f[:, :, ::-1] for f in batch]
        try:
            all_boxes, all_probs = _mtcnn.detect(rgb_list)
        except Exception as exc:
            print(f"    [WARN] detect error @frame {start_idx}: {exc}")
            return
        for i, (boxes, probs) in enumerate(zip(all_boxes, all_probs)):
            if boxes is None:
                continue
            kept = [b for b, p in zip(boxes, probs) if p is not None and p >= FACE_PROB_THRESH]
            if kept:
                per_frame_boxes[start_idx + i] = kept

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        batch.append(frame)
        if len(batch) == BATCH:
            flush(batch, idx)
            idx += BATCH
            batch = []
            if idx % 10000 < BATCH:
                print(f"    detect pass: {idx}/{total} ({idx/total*100:.0f}%)", flush=True)
    if batch:
        flush(batch, idx)
    cap.release()
    return per_frame_boxes, fps, w, h, total


def fill_gaps(per_frame_boxes: list) -> tuple[list, int, int]:
    n = len(per_frame_boxes)
    filled = list(per_frame_boxes)

    last_valid = None
    last_valid_idx = -1
    n_carried = 0
    max_gap = 0
    for i in range(n):
        if filled[i] is not None:
            last_valid = filled[i]
            last_valid_idx = i
        elif last_valid is not None:
            filled[i] = last_valid
            n_carried += 1
            max_gap = max(max_gap, i - last_valid_idx)

    first_valid = next((b for b in per_frame_boxes if b is not None), None)
    if first_valid is not None:
        for i in range(n):
            if per_frame_boxes[i] is not None:
                break
            if filled[i] is None:
                filled[i] = first_valid
                n_carried += 1

    return filled, n_carried, max_gap


def blur_frame(frame_bgr: np.ndarray, boxes, extra_pad: bool) -> np.ndarray:
    if boxes is None:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    out = frame_bgr.copy()
    pad_x_frac = 0.22 if extra_pad else 0.1
    pad_y_frac = 0.35 if extra_pad else 0.2
    for box in boxes:
        x1, y1, x2, y2 = (int(v) for v in box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        fw, fh = x2 - x1, y2 - y1
        if fw <= 10 or fh <= 10:
            continue
        pad_x, pad_y = int(fw * pad_x_frac), int(fh * pad_y_frac)
        bx, by = max(0, x1 - pad_x), max(0, y1 - pad_y)
        bw, bh = min(fw + 2 * pad_x, w - bx), min(fh + 2 * pad_y, h - by)
        roi = out[by:by + bh, bx:bx + bw]
        out[by:by + bh, bx:bx + bw] = cv2.GaussianBlur(roi, (BLUR_K, BLUR_K), 0)
    return out


def process_session(driver: int, session: int, trims: pd.DataFrame) -> None:
    tag = f"D{driver}_S{session}"
    front_in = PUBLISHED / f"D{driver}" / f"Session_{session}" / f"{tag}_Front.mp4"
    if not front_in.exists():
        print(f"{tag}: [SKIP] {front_in} not found")
        return

    final_out_check = OUT_ROOT / f"D{driver}" / f"Session_{session}" / f"{tag}_Front_blurred.mp4"
    if final_out_check.exists():
        print(f"{tag}: [SKIP] already completed ({final_out_check.name} exists) -- resuming past it")
        return

    _init_face_detector()

    out_dir = OUT_ROOT / f"D{driver}" / f"Session_{session}"
    out_dir.mkdir(parents=True, exist_ok=True)
    full_out = out_dir / f"{tag}_Front_blurred_v2_full.mp4"
    final_out = out_dir / f"{tag}_Front_blurred.mp4"

    print(f"{tag}: pass 1 -- detecting faces (full untrimmed video)...")
    per_frame_boxes, fps, w, h, total = detect_all_frames(front_in)
    n_raw_detected = sum(1 for b in per_frame_boxes if b is not None)

    filled, n_carried, max_gap = fill_gaps(per_frame_boxes)
    n_still_missing = sum(1 for b in filled if b is None)

    print(f"  raw detections: {n_raw_detected}/{total} ({n_raw_detected/total*100:.1f}%)  "
          f"carried-forward: {n_carried} (max gap {max_gap} frames = {max_gap/fps:.1f}s)  "
          f"still missing: {n_still_missing}")
    if max_gap > MAX_CARRY_GAP_FRAMES:
        print(f"  [NOTE] longest carry-forward gap ({max_gap/fps:.1f}s) exceeds "
              f"{MAX_CARRY_GAP_FRAMES/fps:.0f}s -- worth a manual spot-check on this session")

    print(f"{tag}: pass 2 -- blurring + writing full video...")
    use_nvenc = _check_nvenc()
    writer = _open_writer(full_out, fps, w, h, use_nvenc)
    time.sleep(0.3)

    cap = cv2.VideoCapture(str(front_in))
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        was_carried = per_frame_boxes[idx] is None and filled[idx] is not None
        blurred = blur_frame(frame, filled[idx], extra_pad=was_carried)
        writer.stdin.write(blurred.tobytes())
        idx += 1
        if idx % 10000 == 0:
            print(f"    write pass: {idx}/{total} ({idx/total*100:.0f}%)", flush=True)
    cap.release()
    writer.stdin.close()
    writer.wait()
    print(f"  Full corrected video written: {full_out.name}")

    # Trim to match the currently-released file's exact window, using the
    # same unix-time-based frame-range computation as apply_trim.py.
    row = trims[(trims["driver"] == driver) & (trims["session"] == session)]
    if row.empty:
        print(f"  [WARN] no trim_points_final.csv row for {tag} -- leaving {full_out.name} untrimmed")
        return
    start_s = float(row.iloc[0]["manual_start_s"])
    end_s = float(row.iloc[0]["manual_end_s"])
    fused_in = PUBLISHED / f"D{driver}" / f"Session_{session}" / f"{tag}_fused.csv"
    if not fused_in.exists():
        print(f"  [WARN] {fused_in} not found -- leaving {full_out.name} untrimmed")
        return
    df = pd.read_csv(fused_in, usecols=["elapsed_s", "unix_time"])
    orig_end = df["elapsed_s"].iloc[-1]
    keep = df[(df["elapsed_s"] >= start_s) & (df["elapsed_s"] <= orig_end - end_s)]
    unix_start, unix_end = float(keep["unix_time"].iloc[0]), float(keep["unix_time"].iloc[-1])

    t0_raw = raw_vbox_t0(driver, session)
    if t0_raw is None:
        print(f"  [WARN] no raw VBOX unix_time -- leaving {full_out.name} untrimmed")
        return
    out_fps = get_fps(full_out)
    frame_start = max(0, round((unix_start - t0_raw) * out_fps))
    frame_end = round((unix_end - t0_raw) * out_fps)
    print(f"  Trimming to released window: [{frame_start}, {frame_end}] (fps={out_fps:.3f})")

    vf = f"select='between(n\\,{frame_start}\\,{frame_end})',setpts=PTS-STARTPTS"
    cmd = [FFMPEG, "-y", "-i", str(full_out), "-vf", vf,
           "-c:v", "libx264", "-crf", "23", "-preset", "medium", "-an", str(final_out)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg trim failed on {full_out.name}: {result.stderr[-2000:]}")
    print(f"  Final, release-window-matched video written: {final_out.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drivers", type=int, nargs="+", required=True)
    ap.add_argument("--sessions", type=int, nargs="+", default=[1, 2, 3, 4])
    args = ap.parse_args()

    if not OUT_ROOT or str(OUT_ROOT) == ".":
        ap.error("STAGING_FRONT_V2_ROOT env var must be set")

    trims = pd.read_csv(TRIM_CSV)
    for d in args.drivers:
        for s in args.sessions:
            process_session(d, s, trims)


if __name__ == "__main__":
    main()
