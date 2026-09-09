"""
Technical Validation: full per-frame video decode audit ("Video and pose
completeness").

Decodes every frame of every released video file (both camera views, 153
files) and checks presentation-timestamp (PTS) monotonicity, decode errors,
and actual vs container-metadata frame count. This confirms, directly on the
full release rather than only a spot-checked subset, that the fast
container-metadata frame-completeness check (make_framedrop_check.py) is
reliable everywhere.

Runs incrementally: writes one row to the output CSV per file as soon as it's
processed, so partial progress can be inspected while running, and can be
safely re-run/resumed (already-processed files are skipped).

Usage:
  python make_full_video_decode_audit.py
"""
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import DATASET_ROOT

import os
FFPROBE = os.environ.get("FFPROBE", "ffprobe")  # assumed on PATH unless overridden
RAW_ROOT = DATASET_ROOT / "Raw_Dataset"
OUT_CSV = Path(__file__).parent / "validation_output" / "full_video_decode_audit.csv"
OUT_CSV.parent.mkdir(exist_ok=True)


def main():
    video_files = sorted(RAW_ROOT.glob("D*/Session_*/D*_S*_Front_blurred.mp4")) + \
                  sorted(RAW_ROOT.glob("D*/Session_*/D*_S*_Side_blurred.mp4"))
    print(f"{len(video_files)} released video files to audit", flush=True)

    rows = []
    if OUT_CSV.exists():
        prev = pd.read_csv(OUT_CSV)
        done_tags = set(prev["file"])
        rows = prev.to_dict("records")
        print(f"Resuming: {len(done_tags)} already done", flush=True)
    else:
        done_tags = set()

    t_start = time.time()
    for fp in video_files:
        tag = fp.name
        if tag in done_tags:
            continue

        duration_fp = subprocess.run(
            [FFPROBE, "-v", "error", "-show_entries", "format=duration,r_frame_rate",
             "-select_streams", "v:0", "-of", "csv=p=0", str(fp)],
            capture_output=True, text=True, timeout=120,
        )
        dur_line = duration_fp.stdout.strip().split(",")
        nominal_fps = None
        duration_s = None
        for tok in dur_line:
            if "/" in tok:
                num, den = tok.split("/")
                nominal_fps = float(num) / float(den) if float(den) != 0 else None
            else:
                try:
                    duration_s = float(tok)
                except ValueError:
                    pass

        proc = subprocess.run(
            [FFPROBE, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "frame=pts_time", "-of", "csv=p=0", str(fp)],
            capture_output=True, text=True, timeout=600,
        )
        stderr = proc.stderr.strip()
        n_decode_errors = len([l for l in stderr.splitlines() if l.strip()]) if stderr else 0

        pts_lines = [l for l in proc.stdout.strip().splitlines() if l.strip()]
        # ffprobe's csv=p=0 output occasionally has a trailing-comma artefact
        # on some rows; parse per-line so one malformed row doesn't zero out
        # the whole file's PTS array.
        pts_vals = []
        n_unparseable = 0
        for l in pts_lines:
            try:
                pts_vals.append(float(l.rstrip(",").strip()))
            except ValueError:
                n_unparseable += 1
        pts = np.array(pts_vals)

        n_frames = len(pts)
        if n_frames >= 2:
            diffs = np.diff(pts)
            n_nonmonotonic = int((diffs < 0).sum())
            n_duplicate = int((diffs == 0).sum())
            mean_dt = float(diffs[diffs > 0].mean()) if (diffs > 0).any() else float("nan")
            actual_fps = 1.0 / mean_dt if mean_dt and mean_dt > 0 else float("nan")
            fps_std = float(diffs[diffs > 0].std())
        else:
            n_nonmonotonic = n_duplicate = 0
            actual_fps = fps_std = float("nan")

        expected_frames = duration_s * nominal_fps if (duration_s and nominal_fps) else float("nan")

        row = {
            "file": tag, "duration_s": duration_s, "nominal_fps": nominal_fps,
            "n_frames_decoded": n_frames, "expected_frames": expected_frames,
            "frame_count_diff": (n_frames - expected_frames) if not np.isnan(expected_frames) else float("nan"),
            "n_decode_errors": n_decode_errors, "n_nonmonotonic_pts": n_nonmonotonic,
            "n_duplicate_pts": n_duplicate, "actual_fps_mean": actual_fps, "actual_fps_std": fps_std,
            "n_unparseable_pts_lines": n_unparseable,
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

        elapsed = time.time() - t_start
        done_now = len(rows) - len(done_tags)
        rate = elapsed / done_now if done_now else 0
        remaining = (len(video_files) - len(rows)) * rate
        print(f"[{len(rows)}/{len(video_files)}] {tag}: frames={n_frames} decode_err={n_decode_errors} "
              f"nonmonotonic={n_nonmonotonic} dup={n_duplicate} "
              f"(ETA {remaining/60:.1f} min)", flush=True)

    print("DONE", flush=True)
    df = pd.DataFrame(rows)
    print(f"\nTotals: {len(df)} files, {df['n_frames_decoded'].sum()} frames decoded, "
          f"{df['n_decode_errors'].sum()} decode errors, {df['n_nonmonotonic_pts'].sum()} non-monotonic PTS, "
          f"{df['n_duplicate_pts'].sum()} duplicate PTS")


if __name__ == "__main__":
    main()
