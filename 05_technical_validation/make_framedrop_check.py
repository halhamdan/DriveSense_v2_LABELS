"""
Video frame-drop rate check, matching UL-DD (Model_1)'s Technical Validation
convention: compares each session's actual stored frame count (container
metadata) against the expected count implied by declared duration x frame
rate. Uses the fast container-metadata field (ffprobe nb_frames) rather than
a full per-frame decode -- spot-checked against an actual decoded frame
count (ffprobe -count_frames) on two sessions first (D1_S1, D10_S2), both
matched exactly, before trusting the fast method for the full 79-session run.

Usage:
  python make_framedrop_check.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import PUBLISHED

OUT_DIR = Path(__file__).parent / "validation_output"
FFPROBE = os.environ.get("FFPROBE", "ffprobe")  # assumed on PATH unless overridden


def probe(path: Path) -> dict | None:
    cmd = [FFPROBE, "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=r_frame_rate,nb_frames",
           "-show_entries", "format=duration",
           "-of", "json", str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(out.stdout)
    except Exception:
        return None
    if not data.get("streams") or not data.get("format"):
        return None
    stream = data["streams"][0]
    if "nb_frames" not in stream or "r_frame_rate" not in stream:
        return None
    num, den = stream["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    duration = float(data["format"]["duration"])
    nb_frames = int(stream["nb_frames"])
    return {"fps": fps, "duration_s": duration, "actual_frames": nb_frames,
            "expected_frames": duration * fps}


def main():
    rows = []
    for d in range(1, 21):
        for s in range(1, 5):
            path = PUBLISHED / f"D{d}" / f"Session_{s}" / f"D{d}_S{s}_Front_blurred.mp4"
            if not path.exists():
                continue
            info = probe(path)
            if info is None:
                continue
            info["driver"] = d
            info["session"] = s
            info["completeness_ratio"] = info["actual_frames"] / info["expected_frames"]
            rows.append(info)
            print(f"D{d}_S{s}: {info['actual_frames']}/{info['expected_frames']:.0f} frames "
                  f"({info['completeness_ratio']*100:.2f}%)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "framedrop_results.csv", index=False)

    print(f"\n=== Frame-completeness summary, {len(df)} sessions ===")
    print(f"Mean completeness ratio: {df['completeness_ratio'].mean()*100:.3f}%")
    print(f"Min: {df['completeness_ratio'].min()*100:.3f}%  Max: {df['completeness_ratio'].max()*100:.3f}%")
    print(f"Sessions with any detectable frame loss (<99.9%): {(df['completeness_ratio'] < 0.999).sum()}")


if __name__ == "__main__":
    main()
