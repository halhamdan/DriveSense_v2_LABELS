"""
Dataset Publication Pipeline
=============================
Processes all 20 drivers x 4 sessions into a clean, publication-ready dataset.

Steps per session:
  1. Load VBOX labeled CSV (telemetry + driving event labels)
  2. Load + synchronize EmotiBit/MotiBit physiological data
  3. Fuse and resample to 25 Hz
  4. Apply motion filter — trim leading/trailing stationary periods (speed < 5 km/h)
  5. Export fused telemetry+physiology CSV
  6. Split VBOX split-screen MP4s (1920x1080) into Front (top-left) + Side (top-right)
  7. Run DeepFace emotion analysis on Front frames, export Front_emotions.csv
  8. Blur detected faces in Front video, export Front_blurred.mp4
  9. Export dataset_metadata.json summary

VBOX frame layout (1920x1080):
  Top-left  (0:540, 0:960)   = Front camera (driver-facing)
  Top-right (0:540, 960:)    = Side camera (road-facing)
  Bottom half                = Unused / overlay

Multiple MP4 files per session (_0001, _0002, ...) are sequential recording
segments and are processed as one continuous stream.

Input paths (configure below):
  DATA_ROOT   : D1-D20 session folders (EmotiBit + VBOX videos)
  LABELED_DIR : D{n}_S{s}_LABELED.csv files
  OUT_ROOT    : output folder (will be created)

Usage:
  python prepare_dataset.py [--drivers 1 2 3] [--smoke] [--no-video] [--out-root <path>]

  --smoke      Process only D1 Session 1 (sanity check)
  --no-video   Skip video processing (export CSVs only)
  --drivers    Process specific driver numbers only
  --out-root   Override output directory
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import traceback
import warnings
from pathlib import Path

# Force UTF-8 stdout/stderr so DeepFace emoji log lines don't crash on Windows cp1252
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Configuration ──────────────────────────────────────────────────────────────
# All paths below are the authors' pre-publication staging tree; set via env
# vars rather than hardcoded to a specific machine. See README.md.
DATA_ROOT   = Path(os.environ.get("STAGING_DATA_ROOT", ""))     # D1-D20 session folders
LABELED_DIR = Path(os.environ.get("STAGING_LABELED_DIR", ""))
CODE_DIR    = Path(__file__).parent                              # data_checks.py, emotibit_sync.py, fuse_stage1.py, parse_labeled.py are co-located
OUT_ROOT    = Path(os.environ.get("STAGING_FUSED_DIR", ""))

FUSE_FS     = 25.0        # Hz — output sample rate
SPEED_COL   = "speed_kph" # column used for motion filter (fallback: gps_speed_kmh)
PARK_THRESH = 5.0         # km/h — below this = stationary
MIN_DRIVE_S = 30.0        # minimum session duration after motion filter (seconds)

ALL_DRIVERS   = list(range(1, 21))
ALL_SESSIONS  = [1, 2, 3, 4]

# ── Import existing D2 fusion modules ─────────────────────────────────────────
sys.path.insert(0, str(CODE_DIR))
try:
    import parse_labeled as pl
    import emotibit_sync as es
except ImportError as e:
    print(f"[ERROR] Cannot import D2 fusion modules from {CODE_DIR}: {e}")
    print("  Check that CODE_DIR points to D2_fusion/code/")
    sys.exit(1)

# EMOTIBIT signal sample rates (Hz) — from data_checks.py
EMOTIBIT_FS = {
    "PPG_IR": 25.0, "PPG_RED": 25.0, "PPG_GREEN": 25.0,
    "ACC": 25.0, "GYRO": 25.0, "MAG": 25.0,
    "EDA": 15.0, "EDA_LEVEL": 15.0, "SCR_FREQ": 3.0,
    "TEMP_CONTACT": 7.5, "TEMP_THERMOPILE": 7.5,
    "HR": 1.0, "IBI": 1.0,
}


# ── Signal resampling (kept in sync with fuse_stage1.py — do not let these
# drift apart again; fuse_stage1.py is the reference implementation) ─────────
def _lowpass(data: np.ndarray, fs_in: float, cutoff: float = 1.9, order: int = 3) -> np.ndarray:
    nyq = fs_in / 2.0
    if cutoff >= nyq or len(data) < 3 * (order + 1):
        return data.copy()
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, data)


def _hampel(x: np.ndarray, win: int = 7, n_sigma: float = 3.0, abs_threshold: float | None = None) -> np.ndarray:
    """Replaces samples deviating from the local median by more than n_sigma
    robust MADs with that median; leaves genuine sustained signal changes
    (e.g. a real harsh-event ramp) untouched. If abs_threshold is set, a
    statistical outlier is only despiked when it ALSO exceeds that absolute
    value — a genuine event's sharp onset is statistically indistinguishable
    from an isolated glitch, so lat_acc_g/lon_acc_g are additionally gated
    at 3g (documented plausible range +-1.2g; the glitch that motivated
    despiking was ~28g). See fuse_stage1.py for the full rationale."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2 * win + 1:
        return x.copy()
    out = x.copy()
    for i in range(n):
        lo, hi = max(0, i - win), min(n, i + win + 1)
        w = x[lo:hi]
        med = np.median(w)
        mad = np.median(np.abs(w - med)) * 1.4826
        if mad < 1e-6:
            mad = 1e-6
        is_stat_outlier = abs(x[i] - med) > n_sigma * mad
        is_gated_in = abs_threshold is None or abs(x[i]) > abs_threshold
        if is_stat_outlier and is_gated_in:
            out[i] = med
    return out


# HR/IBI "outliers" are a real PPG beat-detection dropout characteristic, not
# glitches — must not be despiked (see fuse_stage1.py).
_NO_DESPIKE_SIGNALS = {"HR", "IBI"}

# Magnitude-gated during despiking (see _hampel docstring) — the channels
# driving the harsh-event g-force thresholds.
_MAGNITUDE_GATED_SIGNALS = {"lat_acc_g": 3.0, "lon_acc_g": 3.0}


def _resample_series(t_in: np.ndarray, v_in: np.ndarray, t_out: np.ndarray, fs_in: float,
                      despike: bool = True, abs_threshold: float | None = None) -> np.ndarray:
    """Despike then linearly interpolate. Anti-alias lowpass only applied when
    genuinely downsampling (fs_in > FUSE_FS) — no native rate in this dataset
    exceeds FUSE_FS, so this never fires, but is kept for correctness."""
    mask = np.isfinite(v_in)
    t_in, v_in = t_in[mask], v_in[mask]
    if len(t_in) < 4:
        return np.full(len(t_out), np.nan)
    v_despiked = _hampel(v_in, win=7, n_sigma=3.0, abs_threshold=abs_threshold) if despike else v_in
    if fs_in > FUSE_FS:
        v_use = _lowpass(v_despiked, fs_in, cutoff=min(1.9, FUSE_FS / 2.0 * 0.9))
    else:
        v_use = v_despiked
    f = interp1d(t_in, v_use, kind="linear", bounds_error=False,
                 fill_value=(v_use[0], v_use[-1]))
    return f(t_out)


def _gap_mask(t_in: np.ndarray, t_out: np.ndarray, max_gap_s: float) -> np.ndarray:
    """True at every t_out position falling inside a native dropout > max_gap_s."""
    idx = np.searchsorted(t_in, t_out)
    out = np.zeros(len(t_out), dtype=bool)
    for i, j in enumerate(idx):
        if j <= 0 or j >= len(t_in):
            out[i] = True
            continue
        out[i] = (t_in[j] - t_in[j - 1]) > max_gap_s
    return out


def _resample_series_gap_aware(t_in: np.ndarray, v_in: np.ndarray, t_out: np.ndarray,
                                native_dt_s: float, gap_threshold_x: float = 3.0) -> np.ndarray:
    """Like _resample_series (no despiking), but NaNs out t_out samples inside
    a native dropout longer than gap_threshold_x * native_dt_s, instead of
    linearly interpolating (and thus fabricating) across it."""
    mask = np.isfinite(v_in)
    t_in, v_in = t_in[mask], v_in[mask]
    if len(t_in) < 4:
        return np.full(len(t_out), np.nan)
    v = _resample_series(t_in, v_in, t_out, fs_in=1.0 / native_dt_s, despike=False)
    gap = _gap_mask(t_in, t_out, max_gap_s=gap_threshold_x * native_dt_s)
    v = v.copy()
    v[gap] = np.nan
    return v


def _resample_emotibit(df: pd.DataFrame, t_out: np.ndarray, signal_name: str) -> dict[str, np.ndarray]:
    fs_in = EMOTIBIT_FS.get(signal_name, 25.0)
    despike = signal_name not in _NO_DESPIKE_SIGNALS
    if "value" in df.columns:
        t_in = df["unix_time"].to_numpy()
        v_in = df["value"].to_numpy()
        if signal_name in _NO_DESPIKE_SIGNALS:
            v = _resample_series_gap_aware(t_in, v_in, t_out, native_dt_s=1.0 / fs_in)
            return {signal_name: v, f"{signal_name}_dropout_flag": np.isnan(v)}
        v = _resample_series(t_in, v_in, t_out, fs_in, despike=despike)
        return {signal_name: v}
    out = {}
    mag_parts = []
    for ax in ["x", "y", "z"]:
        if ax not in df.columns:
            continue
        v = _resample_series(df["unix_time"].to_numpy(), df[ax].to_numpy(), t_out, fs_in)
        out[f"{signal_name}_{ax}"] = v
        mag_parts.append(v)
    if mag_parts:
        out[f"{signal_name}_mag"] = np.sqrt(sum(a**2 for a in mag_parts))
    return out


def _resample_vbox(vbox_df: pd.DataFrame, t_out: np.ndarray) -> dict[str, np.ndarray]:
    fs_in = 25.0
    num_cols = [
        "gps_speed_kmh", "speed_kph", "engine_rpm", "throttle_pct", "brake_pct",
        "lat_acc_g", "lon_acc_g", "gradient_pct", "heading_deg", "height_m",
        "wheel_fr_kph", "wheel_fl_kph",
    ]
    t_v = vbox_df["unix_time"].to_numpy()
    out = {}
    for col in num_cols:
        if col not in vbox_df.columns:
            continue
        v = vbox_df[col].to_numpy(dtype=float)
        # heading_deg is circular (0 == 360); Hampel despiking's raw numeric
        # differences don't know about wraparound, so it must be excluded.
        despike = col != "heading_deg"
        abs_threshold = _MAGNITUDE_GATED_SIGNALS.get(col)
        out[col] = _resample_series(t_v, v, t_out, fs_in, despike=despike, abs_threshold=abs_threshold)
    if "label" in vbox_df.columns:
        labels = vbox_df["label"].to_numpy()
        idx = np.clip(np.searchsorted(t_v, t_out, side="left"), 0, len(t_v) - 1)
        out["label"] = labels[idx]
    return out


# ── Motion filter ─────────────────────────────────────────────────────────────
def apply_motion_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trim leading and trailing stationary segments.
    Finds the first and last sample where speed > PARK_THRESH,
    then keeps only that window.  Intermediate stops (traffic lights)
    are preserved because they are bounded by moving periods.
    """
    if SPEED_COL in df.columns:
        speed = df[SPEED_COL].values
    elif "gps_speed_kmh" in df.columns:
        speed = df["gps_speed_kmh"].values
    else:
        return df

    moving = speed > PARK_THRESH

    # Find first moving sample
    start_idx = 0
    for i, m in enumerate(moving):
        if m:
            start_idx = i
            break

    # Find last moving sample
    end_idx = len(df) - 1
    for i in range(len(df) - 1, -1, -1):
        if moving[i]:
            end_idx = i
            break

    filtered = df.iloc[start_idx:end_idx + 1].copy().reset_index(drop=True)
    # Reset elapsed_s to start from 0
    if "elapsed_s" in filtered.columns:
        filtered["elapsed_s"] = filtered["elapsed_s"] - filtered["elapsed_s"].iloc[0]
    return filtered


# ── Path helpers ──────────────────────────────────────────────────────────────
def find_labeled_csv(driver_n: int, session_n: int) -> Path | None:
    p = LABELED_DIR / f"D{driver_n}_S{session_n}_LABELED.csv"
    return p if p.exists() else None


def find_session_dir(driver_n: int, session_n: int) -> Path | None:
    p = DATA_ROOT / f"D{driver_n}" / f"Session {session_n}"
    return p if p.exists() else None


def find_vbox_videos(session_dir: Path) -> list[Path]:
    vbox_dir = session_dir / "VBOX"
    if not vbox_dir.exists():
        return []
    return sorted(vbox_dir.glob("*.mp4"))


# ── Video split + DeepFace emotion extraction + face blurring ─────────────────
FRONT_ROW   = slice(0, 540)    # top half of 1080 frame
FRONT_COL   = slice(0, 960)    # left half
SIDE_COL    = slice(960, 1920) # right half
BLUR_K      = 51               # Gaussian blur kernel size (must be odd)
EMOTION_COLS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# ffmpeg path: $FFMPEG env var if set, otherwise assumed to be on PATH
_FFMPEG_CANDIDATES = [
    Path(os.environ["FFMPEG"]) if os.environ.get("FFMPEG") else None,
    Path("ffmpeg"),  # if in PATH
]
FFMPEG_PATH: Path | None = next(
    (p for p in _FFMPEG_CANDIDATES if p is not None and (p.exists() or p.name == "ffmpeg")), None
)


_NVENC_AVAILABLE: bool | None = None  # cached after first test


def _open_ffmpeg_writer(
    out_path: Path, fps: float, width: int = 960, height: int = 540,
    force_cpu: bool = False,
) -> "subprocess.Popen | None":
    """
    Open an ffmpeg subprocess that accepts raw BGR24 frames on stdin and
    writes H.264 MP4 to out_path.

    Tries NVENC (GPU) first via a one-frame probe; if the probe fails or
    force_cpu=True, uses libx264 (CPU).  Returns None if ffmpeg unavailable.
    """
    global _NVENC_AVAILABLE
    if FFMPEG_PATH is None:
        return None

    use_nvenc = (not force_cpu) and _check_nvenc()

    codec_args = (
        ["-c:v", "h264_nvenc", "-rc", "vbr", "-cq", "28", "-preset", "p4", "-tune", "hq"]
        if use_nvenc else
        ["-c:v", "libx264",    "-crf", "23",  "-preset", "medium"]
    )
    cmd = [
        str(FFMPEG_PATH), "-y",
        "-f", "rawvideo", "-pixel_format", "bgr24",
        "-video_size", f"{width}x{height}", "-framerate", str(int(round(fps))),
        "-i", "pipe:0",
    ] + codec_args + ["-pix_fmt", "yuv420p", str(out_path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _check_nvenc() -> bool:
    """Return True if h264_nvenc is available and working (cached after first call)."""
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is not None:
        return _NVENC_AVAILABLE
    if FFMPEG_PATH is None:
        _NVENC_AVAILABLE = False
        return False
    import tempfile, os, numpy as np
    test_out = Path(tempfile.mktemp(suffix="_nvenc_probe.mp4"))
    cmd = [
        str(FFMPEG_PATH), "-y",
        "-f", "rawvideo", "-pixel_format", "bgr24",
        "-video_size", "64x64", "-framerate", "30",
        "-i", "pipe:0",
        "-c:v", "h264_nvenc", "-frames:v", "1",
        "-pix_fmt", "yuv420p", str(test_out),
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        proc.stdin.write(np.zeros((64, 64, 3), dtype=np.uint8).tobytes())
        proc.stdin.close()
        rc = proc.wait()
        _NVENC_AVAILABLE = (rc == 0 and test_out.exists() and test_out.stat().st_size > 0)
    except Exception:
        _NVENC_AVAILABLE = False
    finally:
        if test_out.exists():
            test_out.unlink(missing_ok=True)
    print(f"    NVENC available: {_NVENC_AVAILABLE}", flush=True)
    return _NVENC_AVAILABLE


def _init_face_detector():
    """
    Initialize GPU-accelerated MTCNN face detector.
    Falls back to CPU if CUDA unavailable.
    Returns (detector, device_str) or (None, None) if facenet_pytorch missing.
    """
    try:
        import torch
        from facenet_pytorch import MTCNN
        device = "cuda" if torch.cuda.is_available() else "cpu"
        detector = MTCNN(
            image_size=160, margin=20, min_face_size=30,
            thresholds=[0.6, 0.7, 0.7], factor=0.709,
            keep_all=False, device=device,
        )
        return detector, device
    except ImportError:
        return None, None


def process_video_segments(
    video_paths: list[Path],
    out_dir: Path,
    tag: str,
) -> bool:
    """
    Process all VBOX MP4 segments for one session.

    Each 1920x1080 frame is split:
      - Front (top-left  0:540, 0:960)   — driver-facing
      - Side  (top-right 0:540, 960:)    — road-facing

    Face detection uses GPU-accelerated MTCNN (facenet_pytorch).
    Emotion classification uses DeepFace on the cropped face region.
    Falls back to DeepFace retinaface on CPU if MTCNN unavailable.

    Multiple segment files (_0001, _0002, ...) are treated as one continuous
    recording and written to a single set of output files.

    Outputs:
      {tag}_Front.mp4          — driver-facing (unblurred)
      {tag}_Side.mp4           — road-facing
      {tag}_Front_blurred.mp4  — driver-facing with face Gaussian-blurred
      {tag}_Front_emotions.csv — DeepFace emotions + bounding box per frame

    Returns True if outputs were written successfully.
    """
    try:
        import cv2
        from deepface import DeepFace
    except ImportError as e:
        print(f"    [WARN] Missing dependency ({e}) — skipping video processing")
        return False

    out_front         = out_dir / f"{tag}_Front.mp4"
    out_side          = out_dir / f"{tag}_Side.mp4"
    out_front_blurred = out_dir / f"{tag}_Front_blurred.mp4"
    out_emotions_csv  = out_dir / f"{tag}_Front_emotions.csv"

    if out_front_blurred.exists() and out_emotions_csv.exists():
        print(f"    [SKIP] Video already processed ({out_front_blurred.name})")
        return True

    # Initialize GPU face detector
    mtcnn, device = _init_face_detector()
    if mtcnn is not None:
        print(f"    Face detector: MTCNN on {device.upper()}")
    else:
        print(f"    Face detector: DeepFace retinaface (CPU fallback)")

    # Probe first segment for FPS
    cap0 = cv2.VideoCapture(str(video_paths[0]))
    fps = cap0.get(cv2.CAP_PROP_FPS) or 30.0
    cap0.release()

    # Open video writers — prefer ffmpeg H.264 (NVENC/libx264), fall back to cv2 mp4v.
    # Open sequentially: NVENC has a per-session resource limit; opening all three
    # at once can cause the second or third to fail silently.  We open Front first,
    # wait 0.5 s, then Side and Blurred (libx264 CPU) to avoid contention.
    import time as _time
    ff_front = _open_ffmpeg_writer(out_front, fps)
    _time.sleep(0.5)
    # Use libx264 (CPU) for Side and Blurred to avoid hitting NVENC session limits
    ff_side    = _open_ffmpeg_writer(out_side,          fps, force_cpu=True)
    ff_blurred = _open_ffmpeg_writer(out_front_blurred, fps, force_cpu=True)
    if ff_front is not None:
        encoder_name = "NVENC" if _check_nvenc() else "libx264"
        print(f"    Video encoder: ffmpeg {encoder_name}+libx264")
        wr_front = wr_side = wr_blurred = None  # not used
    else:
        print(f"    Video encoder: OpenCV mp4v (ffmpeg unavailable)")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        wr_front   = cv2.VideoWriter(str(out_front),         fourcc, fps, (960, 540))
        wr_side    = cv2.VideoWriter(str(out_side),          fourcc, fps, (960, 540))
        wr_blurred = cv2.VideoWriter(str(out_front_blurred), fourcc, fps, (960, 540))

    def _write_frame(front: np.ndarray, side: np.ndarray, blurred: np.ndarray) -> None:
        if ff_front is not None:
            ff_front.stdin.write(front.tobytes())
            ff_side.stdin.write(side.tobytes())
            ff_blurred.stdin.write(blurred.tobytes())
        else:
            wr_front.write(front)
            wr_side.write(side)
            wr_blurred.write(blurred)

    def _close_writers() -> None:
        if ff_front is not None:
            for p in (ff_front, ff_side, ff_blurred):
                p.stdin.close()
                p.wait()
        else:
            wr_front.release()
            wr_side.release()
            wr_blurred.release()

    rows = []
    global_frame = 0
    n_detected = 0
    n_errors = 0
    BATCH = 8  # MTCNN GPU batch size
    total_frames = sum(
        int(cv2.VideoCapture(str(p)).get(cv2.CAP_PROP_FRAME_COUNT))
        for p in video_paths
    )

    print(f"    Splitting + analysing {len(video_paths)} segment(s), "
          f"{total_frames} frames total at {fps:.0f}fps...")

    def _process_batch(batch_fronts, batch_sides, batch_start_idx):
        """Run MTCNN on a batch of front frames, emit (emotion_row, blurred_frame, side) tuples."""
        nonlocal n_detected, n_errors
        out = []

        if mtcnn is not None:
            # Batch GPU detection: convert all frames BGR→RGB
            rgb_list = [f[:, :, ::-1] for f in batch_fronts]
            try:
                all_boxes, all_probs = mtcnn.detect(rgb_list)
            except Exception as exc:
                # Detection batch failed — fall back to per-frame
                all_boxes = [None] * len(batch_fronts)
                all_probs = [None] * len(batch_fronts)
                n_errors += 1
                if n_errors <= 3:
                    print(f"      [batch @{batch_start_idx}] detect error: {exc}")
        else:
            all_boxes = [None] * len(batch_fronts)
            all_probs = [None] * len(batch_fronts)

        for i, (front_frame, side_frame) in enumerate(zip(batch_fronts, batch_sides)):
            gf = batch_start_idx + i
            emotion_row: dict = {
                "frame": gf,
                "timestamp_s": round(gf / fps, 4),
                **{e: np.nan for e in EMOTION_COLS},
                "dominant_emotion": np.nan,
                "face_x": np.nan, "face_y": np.nan,
                "face_w": np.nan, "face_h": np.nan,
            }
            blurred_frame = front_frame.copy()

            try:
                if mtcnn is not None:
                    boxes = all_boxes[i] if all_boxes is not None else None
                    probs = all_probs[i] if all_probs is not None else None
                    if (boxes is not None and len(boxes) > 0
                            and probs is not None and probs[0] is not None
                            and probs[0] > 0.85):
                        x1, y1, x2, y2 = (int(v) for v in boxes[0])
                        x1 = max(0, x1); y1 = max(0, y1)
                        x2 = min(960, x2); y2 = min(540, y2)
                        fw, fh = x2 - x1, y2 - y1
                        if fw > 10 and fh > 10:
                            # Blur with padding
                            pad_x = int(fw * 0.1); pad_y = int(fh * 0.2)
                            bx = max(0, x1 - pad_x); by = max(0, y1 - pad_y)
                            bw = min(fw + 2 * pad_x, 960 - bx)
                            bh = min(fh + 2 * pad_y, 540 - by)
                            roi = blurred_frame[by:by + bh, bx:bx + bw]
                            blurred_frame[by:by + bh, bx:bx + bw] = cv2.GaussianBlur(
                                roi, (BLUR_K, BLUR_K), 0)
                            # Emotion on cropped face (skip re-detection)
                            face_crop = front_frame[y1:y2, x1:x2]
                            emo = DeepFace.analyze(
                                img_path=face_crop, actions=["emotion"],
                                enforce_detection=False, detector_backend="skip",
                                silent=True,
                            )[0]
                            for e in EMOTION_COLS:
                                emotion_row[e] = round(emo["emotion"].get(e, np.nan), 4)
                            emotion_row["dominant_emotion"] = emo.get("dominant_emotion", np.nan)
                            emotion_row.update({"face_x": x1, "face_y": y1,
                                                "face_w": fw, "face_h": fh})
                            n_detected += 1
                else:
                    # CPU fallback: full DeepFace retinaface
                    result = DeepFace.analyze(
                        img_path=front_frame, actions=["emotion"],
                        enforce_detection=True, detector_backend="retinaface",
                        silent=True,
                    )
                    face = result[0]
                    for e in EMOTION_COLS:
                        emotion_row[e] = round(face["emotion"].get(e, np.nan), 4)
                    emotion_row["dominant_emotion"] = face.get("dominant_emotion", np.nan)
                    reg = face.get("region", {})
                    fx, fy = reg.get("x", 0), reg.get("y", 0)
                    fw, fh = reg.get("w", 0), reg.get("h", 0)
                    emotion_row.update({"face_x": fx, "face_y": fy,
                                        "face_w": fw, "face_h": fh})
                    if fw > 0 and fh > 0:
                        pad_x = int(fw * 0.1); pad_y = int(fh * 0.2)
                        bx = max(0, fx - pad_x); by = max(0, fy - pad_y)
                        bw = min(fw + 2 * pad_x, 960 - bx)
                        bh = min(fh + 2 * pad_y, 540 - by)
                        roi = blurred_frame[by:by + bh, bx:bx + bw]
                        blurred_frame[by:by + bh, bx:bx + bw] = cv2.GaussianBlur(
                            roi, (BLUR_K, BLUR_K), 0)
                    n_detected += 1
            except Exception as exc:
                n_errors += 1
                if n_errors <= 3:
                    print(f"      [frame {gf}] {type(exc).__name__}: {str(exc)[:120]}")

            out.append((emotion_row, blurred_frame, front_frame, side_frame))
        return out

    for seg_path in video_paths:
        cap = cv2.VideoCapture(str(seg_path))
        batch_fronts: list = []
        batch_sides:  list = []
        batch_start = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            batch_fronts.append(frame[FRONT_ROW, FRONT_COL].copy())
            batch_sides.append(frame[FRONT_ROW, SIDE_COL].copy())

            if len(batch_fronts) == BATCH:
                for emotion_row, blurred, front, side in _process_batch(
                        batch_fronts, batch_sides, batch_start):
                    rows.append(emotion_row)
                    _write_frame(front, side, blurred)
                    global_frame += 1
                    if global_frame % 1000 == 0:
                        pct = global_frame / total_frames * 100 if total_frames > 0 else 0
                        print(f"      {global_frame} / {total_frames} frames ({pct:.0f}%)  "
                              f"detected: {n_detected}", flush=True)
                batch_start = global_frame
                batch_fronts.clear()
                batch_sides.clear()

        # Flush remaining frames
        if batch_fronts:
            for emotion_row, blurred, front, side in _process_batch(
                    batch_fronts, batch_sides, batch_start):
                rows.append(emotion_row)
                _write_frame(front, side, blurred)
                global_frame += 1
        cap.release()

    _close_writers()

    df = pd.DataFrame(rows)
    df.to_csv(out_emotions_csv, index=False)

    print(f"    Front.mp4 + Side.mp4 + Front_blurred.mp4 written ({global_frame} frames)")
    print(f"    Front_emotions.csv: {len(df)} rows, "
          f"face detected in {n_detected} ({n_detected/max(len(df),1)*100:.0f}%) frames")
    return True


# ── Fuse one session ──────────────────────────────────────────────────────────
def fuse_session(
    driver_n: int,
    session_n: int,
    session_dir: Path,
    labeled_csv: Path,
) -> pd.DataFrame | None:
    """
    Load, synchronize, fuse, and motion-filter one session.
    Returns the fused DataFrame at 25 Hz, or None if fusion failed.
    """
    tag = f"D{driver_n}_S{session_n}"

    # 1. Load VBOX labeled CSV
    vbox_df = pl.load_labeled_csv(labeled_csv, session_dir=session_dir)
    if vbox_df is None or len(vbox_df) == 0:
        print(f"    [SKIP] Empty VBOX CSV")
        return None
    print(f"    VBOX: {len(vbox_df)} rows, {vbox_df['elapsed_s'].max():.1f}s")

    # 2. Load EmotiBit
    emotibit_signals, emb_meta = es.load_emotibit(session_dir)
    if not emotibit_signals:
        print(f"    [WARN] No EmotiBit signals found — exporting telemetry only")

    # 3. Find overlap window
    t_vbox_start = vbox_df["unix_time"].min()
    t_vbox_end   = vbox_df["unix_time"].max()

    if emotibit_signals:
        emb_starts = [df["unix_time"].min() for df in emotibit_signals.values() if len(df) > 0]
        emb_ends   = [df["unix_time"].max() for df in emotibit_signals.values() if len(df) > 0]
        t_start = max(t_vbox_start, min(emb_starts))
        t_end   = min(t_vbox_end, max(emb_ends))
    else:
        t_start = t_vbox_start
        t_end   = t_vbox_end

    overlap_s = t_end - t_start
    if overlap_s < 30.0:
        print(f"    [SKIP] Overlap only {overlap_s:.1f}s — insufficient for publication")
        return None
    print(f"    Overlap: {overlap_s:.1f}s")

    # 4. Clip to overlap
    vbox_clip = vbox_df[(vbox_df["unix_time"] >= t_start) & (vbox_df["unix_time"] <= t_end)].copy()
    emb_clip = {
        name: df[(df["unix_time"] >= t_start) & (df["unix_time"] <= t_end)].reset_index(drop=True)
        for name, df in emotibit_signals.items()
    } if emotibit_signals else {}

    # 5. Resample to 25 Hz
    dt = 1.0 / FUSE_FS
    t_out = np.arange(t_start, t_end, dt)

    fused: dict[str, np.ndarray] = {
        "unix_time": t_out,
        "elapsed_s": t_out - t_start,
    }
    fused.update(_resample_vbox(vbox_clip, t_out))
    for sig_name, df in emb_clip.items():
        if not df.empty:
            fused.update(_resample_emotibit(df, t_out, sig_name))

    df_out = pd.DataFrame(fused)
    df_out.insert(0, "session",  session_n)
    df_out.insert(0, "driver",   driver_n)

    # 6. Motion filter
    before_rows = len(df_out)
    df_out = apply_motion_filter(df_out)
    after_rows = len(df_out)
    drive_s = after_rows / FUSE_FS
    trimmed_s = (before_rows - after_rows) / FUSE_FS
    print(f"    Motion filter: trimmed {trimmed_s:.1f}s stationary -> {drive_s:.1f}s driving")

    if drive_s < MIN_DRIVE_S:
        print(f"    [SKIP] Only {drive_s:.1f}s of driving after filter")
        return None

    return df_out


# ── Process one session (fusion + video) ─────────────────────────────────────
def process_session(
    driver_n: int,
    session_n: int,
    out_root: Path,
    process_video: bool = True,
) -> dict | None:
    tag = f"D{driver_n}_S{session_n}"
    print(f"\n{'='*60}")
    print(f"  {tag}")
    print(f"{'='*60}")

    labeled_csv = find_labeled_csv(driver_n, session_n)
    if labeled_csv is None:
        print(f"  [SKIP] No labeled CSV: D{driver_n}_S{session_n}_LABELED.csv")
        return None

    session_dir = find_session_dir(driver_n, session_n)
    if session_dir is None:
        print(f"  [SKIP] Session directory not found: D{driver_n}/Session {session_n}")
        return None

    out_dir = out_root / f"D{driver_n}" / f"Session_{session_n}"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "driver": driver_n,
        "session": session_n,
        "tag": tag,
        "status": "ok",
        "has_physiology": False,
        "has_facial": False,
        "drive_duration_s": 0.0,
        "rows": 0,
    }

    # ── Fusion ────────────────────────────────────────────────────────────────
    try:
        fused_df = fuse_session(driver_n, session_n, session_dir, labeled_csv)
    except Exception as exc:
        print(f"  [ERROR] Fusion failed: {exc}")
        traceback.print_exc()
        result["status"] = f"fusion_error: {exc}"
        return result

    if fused_df is None:
        result["status"] = "skipped"
        return result

    # Check whether physiological data was fused
    phys_cols = [c for c in fused_df.columns if c in
                 ["HR", "EDA", "TEMP_CONTACT", "PPG_IR", "ACC_mag", "GYRO_mag"]]
    result["has_physiology"] = len(phys_cols) > 0
    result["rows"] = len(fused_df)
    result["drive_duration_s"] = float(fused_df["elapsed_s"].max())

    # Export fused CSV
    fused_path = out_dir / f"{tag}_fused.csv"
    fused_df.to_csv(fused_path, index=False, float_format="%.6f")
    print(f"  Fused CSV: {fused_path.name}  ({len(fused_df)} rows × {len(fused_df.columns)} cols)")

    # ── Video processing ──────────────────────────────────────────────────────
    if not process_video:
        return result

    videos = find_vbox_videos(session_dir)
    if not videos:
        print(f"  [INFO] No MP4 files found")
        return result

    try:
        ok = process_video_segments(videos, out_dir, tag)
        result["has_facial"] = ok
    except Exception as exc:
        print(f"  [ERROR] Video processing failed: {exc}")
        traceback.print_exc()
        result["status"] = f"video_error: {exc}"

    return result


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Dataset Publication Pipeline")
    parser.add_argument("--drivers",  type=int, nargs="+", default=None,
                        help="Driver numbers to process (default: all 1-20)")
    parser.add_argument("--sessions", type=int, nargs="+", default=None,
                        help="Session numbers to process (default: 1 2 3 4)")
    parser.add_argument("--smoke",    action="store_true",
                        help="Process only D1 Session 1 (quick sanity check)")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip video processing (export CSVs only)")
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT,
                        help=f"Output directory (default: {OUT_ROOT})")
    args = parser.parse_args()

    drivers  = [1] if args.smoke else (args.drivers or ALL_DRIVERS)
    sessions = [1] if args.smoke else (args.sessions or ALL_SESSIONS)
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"\nDataset Publication Pipeline")
    print(f"  DATA_ROOT   : {DATA_ROOT}")
    print(f"  LABELED_DIR : {LABELED_DIR}")
    print(f"  OUT_ROOT    : {out_root}")
    print(f"  Drivers     : {drivers}")
    print(f"  Sessions    : {sessions}")
    print(f"  Video       : {'disabled' if args.no_video else 'enabled'}")

    summaries = []
    n_ok = n_skip = n_err = 0

    for d in drivers:
        for s in sessions:
            res = process_session(
                driver_n=d,
                session_n=s,
                out_root=out_root,
                process_video=not args.no_video,
            )
            if res is None:
                res = {"driver": d, "session": s, "tag": f"D{d}_S{s}", "status": "skipped"}
            summaries.append(res)

            status = res.get("status", "ok")
            if status == "ok":
                n_ok += 1
            elif status == "skipped":
                n_skip += 1
            else:
                n_err += 1

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Session':<12} {'Status':<18} {'Dur(s)':>8} {'Rows':>8} {'Phys':>6} {'Facial':>7}")
    print(f"  {'-'*58}")
    for r in summaries:
        print(
            f"  {r['tag']:<12} {r.get('status',''):<18} "
            f"{r.get('drive_duration_s', 0):>8.1f} "
            f"{r.get('rows', 0):>8} "
            f"{'yes' if r.get('has_physiology') else 'no':>6} "
            f"{'yes' if r.get('has_facial') else 'no':>7}"
        )

    print(f"\n  Processed: {n_ok} ok | {n_skip} skipped | {n_err} errors")
    print(f"  Output   : {out_root}")

    # Save metadata JSON
    meta_path = out_root / "dataset_metadata.json"
    meta = {
        "pipeline": "prepare_dataset.py",
        "date": __import__("datetime").datetime.now().isoformat(),
        "fuse_hz": FUSE_FS,
        "motion_filter_kmh": PARK_THRESH,
        "sessions": summaries,
        "totals": {
            "ok": n_ok,
            "skipped": n_skip,
            "errors": n_err,
            "with_physiology": sum(1 for r in summaries if r.get("has_physiology")),
            "with_facial": sum(1 for r in summaries if r.get("has_facial")),
        }
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"  Metadata : {meta_path.name}")


if __name__ == "__main__":
    main()
