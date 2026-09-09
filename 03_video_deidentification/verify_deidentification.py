"""
Automated de-identification audit: re-runs face detection on every frame of
the RELEASED (blurred) Front and Side videos and flags any frame where an
UNBLURRED face is still detectable.

This exists because neither prepare_dataset.py (Front-camera blurring) nor
blur_side_video.py (Side-camera blurring) includes a verification step of its
own -- the manuscript's Usage Notes currently discloses that blurring was
spot-checked on detection-gap frames from a single test session rather than
verified frame-by-frame across the full release. Run this before public
release to replace that spot check with actual evidence across all sessions.

IMPORTANT -- why this checks sharpness, not just detection: an early version
of this script flagged any MTCNN detection above a confidence threshold as a
"residual face," which produced enormous false-positive counts (tens of
thousands per session). Visual inspection showed why: MTCNN readily fires on
a driver's HAND on the steering wheel, small reflective objects (e.g. a side-
mirror sticker), and background clutter through windows, while correctly NOT
firing on the actually-blurred face right next to them.

A second calibration round (see manuscript Technical Validation) found the
first fix -- raw Laplacian variance inside the box, threshold 120 -- was
itself unreliable: released video is H.264-encoded, and macroblock
compression artefacts inside smooth (Gaussian-blurred) regions inflate raw
Laplacian variance enough to misflag genuinely, visibly blurred faces as
"sharp" (observed 125-245 on faces with no recoverable detail). The metric
now pre-smooths each candidate box with a small Gaussian blur before
computing Laplacian variance, which suppresses compression-block edges while
still responding to genuine fine facial detail (eyes/nose/mouth) if a face
were actually unblurred. Recalibrated across three sessions spanning both
camera views (D1_S1_Front, D15_S1_Front, D18_S4_Side -- the last being the
session with a previously-confirmed real exposure, since corrected), every
visually-inspected flagged detection was still a false positive (hand,
mirror decal, window clutter) with the highest observed score 245;
--sharpness-threshold defaults to 300 accordingly, comfortably above every
false positive observed so far.

This calibration is still based on a limited sample without a confirmed
true-positive example to validate sensitivity against. Before trusting a
full run, spot-check a handful of --dump-flagged-frames output images (see
below) to confirm they show genuine unblurred faces, not another
false-positive category -- adjust --sharpness-threshold if needed.

Output is written incrementally (one row appended per session-camera file
processed), since a full run is many hours -- an interruption partway
through does not lose prior sessions' results.

Usage:
  python verify_deidentification.py --dataset-root /path/to/Published_Dataset_Final
  python verify_deidentification.py --dataset-root ... --drivers 1 2 3         # spot check a subset first
  python verify_deidentification.py --dataset-root ... --dump-flagged-frames   # save annotated images of flagged frames for visual review

Output:
  verification_output/deidentification_audit.csv    -- one row per flagged frame (with sharpness)
  verification_output/deidentification_summary.csv   -- one row per session-camera file
  verification_output/flagged_frames/*.png            -- annotated images, if --dump-flagged-frames
  Non-zero exit code if any session has a residual detection, so this can be
  wired into a CI/pre-release check.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np

_HAVE_MTCNN = False
try:
    from facenet_pytorch import MTCNN
    import torch
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _detector = MTCNN(keep_all=True, device=_DEVICE)
    _HAVE_MTCNN = True

    def detect_faces_batch(frames_rgb: list[np.ndarray], min_conf: float):
        # facenet_pytorch batches a list of same-size frames as one GPU call --
        # single-frame-at-a-time calls only reach ~9 fps on this hardware due to
        # per-call Python/CUDA-launch overhead; batching reaches ~50+ fps, the
        # difference between a multi-day and a same-day full-dataset audit.
        boxes_list, probs_list = _detector.detect(frames_rgb)
        if boxes_list is None:
            boxes_list = [None] * len(frames_rgb)
            probs_list = [None] * len(frames_rgb)
        out = []
        for boxes, probs in zip(boxes_list, probs_list):
            if boxes is None:
                out.append([])
            else:
                out.append([(box, float(p)) for box, p in zip(boxes, probs)
                            if p is not None and p >= min_conf])
        return out

except ImportError:
    from deepface import DeepFace

    def detect_faces_batch(frames_rgb: list[np.ndarray], min_conf: float):
        out = []
        for rgb in frames_rgb:
            try:
                faces = DeepFace.extract_faces(rgb, detector_backend="retinaface",
                                                enforce_detection=False, align=False)
            except Exception:
                faces = []
            frame_out = []
            for f in faces:
                conf = f.get("confidence", 0)
                area = f.get("facial_area", {})
                w = area.get("w", 0)
                if conf >= min_conf and w > 10:
                    box = (area["x"], area["y"], area["x"] + area["w"], area["y"] + area["h"])
                    frame_out.append((box, float(conf)))
            out.append(frame_out)
        return out


DEFAULT_MIN_CONF = 0.90        # deliberately stricter than the blurring pass's own 0.7-0.85 thresholds
DEFAULT_SHARPNESS_THRESHOLD = 300.0  # see calibration note in the module docstring
PRESMOOTH_KSIZE = 5  # Gaussian pre-smoothing kernel, suppresses H.264 macroblock artefacts
BATCH_SIZE = 32  # only used for the facenet_pytorch/MTCNN path


def box_sharpness(gray: np.ndarray, box) -> float:
    h, w = gray.shape
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    patch = gray[y1:y2, x1:x2]
    if patch.size == 0 or patch.shape[0] < PRESMOOTH_KSIZE or patch.shape[1] < PRESMOOTH_KSIZE:
        # too small for the pre-smoothing kernel -- fall back to raw Laplacian
        # directly rather than risk cv2.GaussianBlur on a degenerate patch.
        return float(cv2.Laplacian(patch, cv2.CV_64F).var()) if patch.size else 0.0
    patch = np.ascontiguousarray(patch)
    smoothed = cv2.GaussianBlur(patch, (PRESMOOTH_KSIZE, PRESMOOTH_KSIZE), 0)
    return float(cv2.Laplacian(smoothed, cv2.CV_64F).var())


def audit_video(path: Path, sample_every: int, min_conf: float, sharpness_threshold: float,
                 dump_dir: Path | None) -> tuple[list[dict], int, int]:
    """Returns (flagged_frame_rows, n_frames_checked, n_frames_total)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return [], 0, 0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    flagged = []
    checked = 0
    frame_idx = 0
    batch_frames: list[np.ndarray] = []
    batch_indices: list[int] = []
    batch_bgr: list[np.ndarray] = []

    def flush():
        nonlocal flagged, checked
        if not batch_frames:
            return
        for idx, bgr, dets in zip(batch_indices, batch_bgr, detect_faces_batch(batch_frames, min_conf)):
            checked += 1
            if not dets:
                continue
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            for box, conf in dets:
                try:
                    sharp = box_sharpness(gray, box)
                except Exception as e:
                    # A privacy audit should fail open (flag for manual review),
                    # not silently drop a detection it couldn't score -- and must
                    # not crash a multi-hour run over one malformed box.
                    print(f"  [WARN] sharpness computation failed on frame {idx}, "
                          f"box {box}: {type(e).__name__}: {e} -- flagging for manual review", flush=True)
                    flagged.append({"frame": idx, "confidence": round(conf, 3), "sharpness": "ERROR"})
                    if dump_dir is not None:
                        cv2.imwrite(str(dump_dir / f"{path.stem}_frame{idx}_ERROR.png"), bgr)
                    continue
                if sharp < sharpness_threshold:
                    continue  # detected region is blurred -- not a genuine residual face
                flagged.append({"frame": idx, "confidence": round(conf, 3), "sharpness": round(sharp, 1)})
                if dump_dir is not None:
                    annotated = bgr.copy()
                    x1, y1, x2, y2 = [int(v) for v in box]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(annotated, f"conf={conf:.2f} sharp={sharp:.0f}", (x1, max(0, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.imwrite(str(dump_dir / f"{path.stem}_frame{idx}.png"), annotated)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % sample_every == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch_frames.append(rgb)
            batch_bgr.append(frame)
            batch_indices.append(frame_idx)
            if (_HAVE_MTCNN and len(batch_frames) >= BATCH_SIZE) or not _HAVE_MTCNN:
                flush()
                batch_frames, batch_bgr, batch_indices = [], [], []
        frame_idx += 1
    flush()
    cap.release()
    return flagged, checked, n_total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", default=os.environ.get("DATASET_ROOT", ""),
                         help="Path to the released dataset root (containing Raw_Dataset/); default: $DATASET_ROOT")
    parser.add_argument("--drivers", type=int, nargs="*", default=list(range(1, 21)),
                         help="Driver numbers to check (default: all 20)")
    parser.add_argument("--sessions", type=int, nargs="*", default=list(range(1, 5)),
                         help="Session numbers to check (default: all 4)")
    parser.add_argument("--sample-every", type=int, default=1,
                         help="Check every Nth frame instead of every frame (default: 1 = every frame). "
                              "Use e.g. 5 for a fast first pass; a full pre-release audit should use 1.")
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONF,
                         help=f"Minimum detector confidence to consider (default: {DEFAULT_MIN_CONF})")
    parser.add_argument("--sharpness-threshold", type=float, default=DEFAULT_SHARPNESS_THRESHOLD,
                         help=f"Minimum Laplacian-variance sharpness inside a detected box to count it as a "
                              f"genuine (unblurred) residual face, rather than a blurred region the detector "
                              f"still fired on (default: {DEFAULT_SHARPNESS_THRESHOLD}; see module docstring)")
    parser.add_argument("--dump-flagged-frames", action="store_true",
                         help="Save an annotated PNG of every flagged frame to verification_output/flagged_frames/ "
                              "for visual review -- recommended for at least the first run.")
    args = parser.parse_args()
    if not args.dataset_root:
        parser.error("--dataset-root must be set (or DATASET_ROOT env var)")
    dataset_root = Path(args.dataset_root)

    out_dir = Path(__file__).parent / "verification_output"
    out_dir.mkdir(exist_ok=True)
    dump_dir = None
    if args.dump_flagged_frames:
        dump_dir = out_dir / "flagged_frames"
        dump_dir.mkdir(exist_ok=True)

    summary_path = out_dir / "deidentification_summary.csv"
    audit_path = out_dir / "deidentification_audit.csv"
    summary_f = open(summary_path, "w", newline="")
    audit_f = open(audit_path, "w", newline="")
    summary_w = csv.DictWriter(summary_f, fieldnames=[
        "tag", "driver", "session", "camera", "frames_checked", "frames_total",
        "residual_detections", "pct_flagged"])
    audit_w = csv.DictWriter(audit_f, fieldnames=["tag", "camera", "frame", "confidence", "sharpness"])
    summary_w.writeheader()
    audit_w.writeheader()

    any_residual = False
    n_sessions_checked = 0
    n_sessions_flagged = 0

    for d in args.drivers:
        for s in args.sessions:
            for cam in ("Front", "Side"):
                video_path = (dataset_root / "Raw_Dataset" / f"D{d}" / f"Session_{s}"
                              / f"D{d}_S{s}_{cam}_blurred.mp4")
                if not video_path.exists():
                    continue
                tag = f"D{d}_S{s}_{cam}"
                print(f"Checking {tag} ...", flush=True)
                flagged, checked, total = audit_video(video_path, args.sample_every, args.min_confidence,
                                                        args.sharpness_threshold, dump_dir)
                for row in flagged:
                    audit_w.writerow({"tag": tag, "camera": cam, **row})
                audit_f.flush()
                n_flagged = len(flagged)
                if n_flagged:
                    any_residual = True
                    n_sessions_flagged += 1
                    print(f"  ** {n_flagged} residual (sharp, unblurred) detection(s) in {checked} checked frames **")
                summary_w.writerow({
                    "tag": tag, "driver": d, "session": s, "camera": cam,
                    "frames_checked": checked, "frames_total": total,
                    "residual_detections": n_flagged,
                    "pct_flagged": round(100 * n_flagged / checked, 4) if checked else "",
                })
                summary_f.flush()
                n_sessions_checked += 1

    summary_f.close()
    audit_f.close()

    print(f"\n{'=' * 60}")
    print(f"Checked {n_sessions_checked} session-camera files.")
    print(f"{n_sessions_flagged} file(s) had at least one residual (sharp, unblurred) face detection.")
    print(f"Full results: {summary_path}, {audit_path}")
    print(f"{'=' * 60}")

    if any_residual:
        sys.exit(1)


if __name__ == "__main__":
    main()
