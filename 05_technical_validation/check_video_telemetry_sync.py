"""
Checks, for every session, whether the Front_blurred.mp4 / Side.mp4 video
duration matches the fused.csv (VBOX telemetry) duration. Discovered via
manual review of D4_S1/D9_S3/D13_S4 (video ran 20-126s longer than telemetry
in all three) that these two data sources may not stop recording at the same
real-world moment -- this checks whether that's a widespread pattern before
finalizing synchronized start/end trim points across modalities.

Analysis only, does not touch any files.

Usage:
  python check_video_telemetry_sync.py
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


def probe_duration(path: Path) -> float | None:
    cmd = [FFPROBE, "-v", "error", "-show_entries", "format=duration",
           "-of", "json", str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(out.stdout)
    except Exception:
        return None
    if not data.get("format") or "duration" not in data["format"]:
        return None
    return float(data["format"]["duration"])


def fused_duration(driver: int, session: int) -> float | None:
    path = PUBLISHED / f"D{driver}" / f"Session_{session}" / f"D{driver}_S{session}_fused.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, usecols=["elapsed_s"])
    if df.empty:
        return None
    return float(df["elapsed_s"].iloc[-1] - df["elapsed_s"].iloc[0])


def main():
    rows = []
    for d in range(1, 21):
        for s in range(1, 5):
            fused_dur = fused_duration(d, s)
            if fused_dur is None:
                continue

            front_path = PUBLISHED / f"D{d}" / f"Session_{s}" / f"D{d}_S{s}_Front_blurred.mp4"
            side_path = PUBLISHED / f"D{d}" / f"Session_{s}" / f"D{d}_S{s}_Side.mp4"

            front_dur = probe_duration(front_path) if front_path.exists() else None
            side_dur = probe_duration(side_path) if side_path.exists() else None

            row = {
                "driver": d, "session": s,
                "fused_dur_s": fused_dur,
                "front_dur_s": front_dur,
                "side_dur_s": side_dur,
                "front_minus_fused_s": (front_dur - fused_dur) if front_dur is not None else None,
                "side_minus_fused_s": (side_dur - fused_dur) if side_dur is not None else None,
            }
            rows.append(row)
            fdiff = f"{row['front_minus_fused_s']:+.1f}s" if row['front_minus_fused_s'] is not None else "N/A"
            sdiff = f"{row['side_minus_fused_s']:+.1f}s" if row['side_minus_fused_s'] is not None else "N/A"
            print(f"D{d}_S{s}: fused={fused_dur:.1f}s  front_diff={fdiff}  side_diff={sdiff}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "video_telemetry_sync_check.csv", index=False)

    print(f"\n=== Summary, {len(df)} sessions ===")
    for col, label in [("front_minus_fused_s", "Front video - fused"), ("side_minus_fused_s", "Side video - fused")]:
        vals = df[col].dropna()
        if vals.empty:
            continue
        print(f"\n{label}:")
        print(f"  mean={vals.mean():.2f}s  median={vals.median():.2f}s  "
              f"min={vals.min():.2f}s  max={vals.max():.2f}s  std={vals.std():.2f}s")
        print(f"  |diff|>5s: {(vals.abs() > 5).sum()}/{len(vals)}")
        print(f"  |diff|>30s: {(vals.abs() > 30).sum()}/{len(vals)}")
        worst = df.reindex(vals.abs().sort_values(ascending=False).index).head(10)
        print(f"  10 largest |diff|:")
        print(worst[["driver", "session", "fused_dur_s", col]].to_string(index=False))


if __name__ == "__main__":
    main()
