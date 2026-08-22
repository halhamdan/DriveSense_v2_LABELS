"""
Face-blurring for {tag}_Side.mp4, extending the existing Front.mp4 pipeline
(Data_Fusion/prepare_dataset.py) which only ever blurred the driver-facing
Front camera. Side.mp4 is a profile/in-cabin view that can also show the
driver's face (side-on) and occasionally passengers or people outside the
car through the window while parked -- all need blurring before publication.

Two-pass design (v2, after v1's single-pass per-frame MTCNN detection was
found to genuinely miss faces, not just score them low -- verified by
querying raw MTCNN output on several frames from long no-detection runs and
getting boxes=None outright, in both daytime and nighttime footage, in
frames where the driver's profile is clearly visible and unobstructed):

  Pass 1: run MTCNN (keep_all=True, since Side can show more than one
          person) over every frame, storing the detected box list per frame
          (or None if nothing found above threshold).
  Fill:   temporal carry-forward -- for any frame with no detection, reuse
          the nearest frame's box list in time (forward-fill then
          backward-fill for any leading gap before the first detection),
          with extra safety padding since a carried-forward box may not
          perfectly track head position during the gap.
  Pass 2: re-read the video and blur using the (possibly carried-forward)
          box list per frame, write output.

Writes {tag}_Side_blurred.mp4 into Published_Dataset_Trimmed (never touches
Published_Dataset).

Usage:
  python blur_side_video.py --drivers 1 --sessions 1        # single-session test
  python blur_side_video.py --drivers 1 2 3 4 5 6
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import PUBLISHED

OUT_ROOT = Path(os.environ.get("STAGING_TRIMMED_VIDEO_ROOT", ""))

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")  # assumed on PATH unless overridden
BLUR_K = 51
BATCH = 8
FACE_PROB_THRESH = 0.7          # MTCNN's own final-cascade stage threshold
MAX_CARRY_GAP_FRAMES = 300      # ~10s at 30fps -- beyond this, still carry
                                 # forward (better over-blur than miss), but
                                 # flagged in the summary for review

# Sessions with an invalid/disconnected Side camera (chromakey-only) --
# confirmed during earlier dataset audit, skip these.
INVALID_SIDE = {(6, 3), (10, 3), (16, 1)}

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
    """Pass 1: returns (per_frame_boxes, fps, width, height, total_frames)."""
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
    """Forward-fill then backward-fill. Returns (filled, n_carried, max_gap)."""
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

    # backward-fill any leading gap before the first detection
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


def process_session(driver: int, session: int) -> None:
    if (driver, session) in INVALID_SIDE:
        print(f"D{driver}_S{session}: [SKIP] known-invalid Side camera")
        return

    tag = f"D{driver}_S{session}"
    side_in = PUBLISHED / f"D{driver}" / f"Session_{session}" / f"{tag}_Side.mp4"
    if not side_in.exists():
        print(f"{tag}: [SKIP] {side_in} not found")
        return

    _init_face_detector()

    out_dir = OUT_ROOT / f"D{driver}" / f"Session_{session}"
    out_dir.mkdir(parents=True, exist_ok=True)
    side_out = out_dir / f"{tag}_Side_blurred.mp4"

    print(f"{tag}: pass 1 -- detecting faces...")
    per_frame_boxes, fps, w, h, total = detect_all_frames(side_in)
    n_raw_detected = sum(1 for b in per_frame_boxes if b is not None)

    filled, n_carried, max_gap = fill_gaps(per_frame_boxes)
    n_still_missing = sum(1 for b in filled if b is None)

    print(f"  raw detections: {n_raw_detected}/{total} ({n_raw_detected/total*100:.1f}%)  "
          f"carried-forward: {n_carried} (max gap {max_gap} frames = {max_gap/fps:.1f}s)  "
          f"still missing: {n_still_missing}")
    if max_gap > MAX_CARRY_GAP_FRAMES:
        print(f"  [NOTE] longest carry-forward gap ({max_gap/fps:.1f}s) exceeds "
              f"{MAX_CARRY_GAP_FRAMES/fps:.0f}s -- worth a manual spot-check on this session")

    print(f"{tag}: pass 2 -- blurring + writing...")
    use_nvenc = _check_nvenc()
    writer = _open_writer(side_out, fps, w, h, use_nvenc)
    time.sleep(0.3)

    cap = cv2.VideoCapture(str(side_in))
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

    print(f"  Done: {side_out.name}")


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
