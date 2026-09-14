"""
Decisive raw-video-to-telemetry offset check using the VBOX HD2's burned-in
speedometer overlay (bottom-right of the native 1920x1080 frame).

The cyan arc of the overlay speedometer grows with speed, so the count of cyan
pixels in the gauge region of each raw frame is a monotonic proxy for the
speed the logger was displaying at that frame. Cross-correlating it with the
native .vbo/.csv speed locates the offset between raw video time (frame/30,
segments concatenated) and telemetry elapsed time. Racelogic's `Avi sync time`
column states this offset directly (video ms at each telemetry row); this
script verifies it from pixels.

Usage:
  python audit_video_overlay_offset.py --tag D1_S1 --videos a_0001.mp4 a_0002.mp4 --csv D1_S1.csv
Output: validation_output/video_overlay_offset.csv (appends one row per tag)
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "validation_output" / "video_overlay_offset.csv"
W, H = 480, 270  # decode size (quarter of 1920x1080)
FPS = 30


def cyan_count(video: Path) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"scale={W}:{H}", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10 ** 7); n = W * H * 3; out = []
    # gauge region in 960x540 coords ~ x 790-950, y 370-530 -> halve for 480x270
    y0, y1, x0, x1 = 185, 265, 395, 475
    while True:
        buf = p.stdout.read(n)
        if len(buf) < n: break
        f = np.frombuffer(buf, np.uint8).reshape(H, W, 3)[y0:y1, x0:x1].astype(np.int16)
        r, g, b = f[..., 0], f[..., 1], f[..., 2]
        out.append(int(((b > 140) & (g > 140) & (r < 120)).sum()))
    p.wait(); return np.array(out, float)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--videos", nargs="+", required=True); ap.add_argument("--csv", required=True)
    ap.add_argument("--max-lag", type=float, default=30.0); a = ap.parse_args()
    sig = np.concatenate([cyan_count(Path(v)) for v in a.videos]); tv = np.arange(len(sig)) / FPS
    d = pd.read_csv(a.csv, encoding="utf-8-sig"); te = d["Elapsed time (s)"].to_numpy(float); sp = d["Speed (km/h)"].to_numpy(float)
    avi0 = float(d["Avi sync time (s)"].iloc[0]) / 1000.0
    fs = 5.0; grid = np.arange(0, max(tv[-1], te[-1]), 1 / fs)
    x = pd.Series(sig, index=np.floor(tv * fs).astype(int)).groupby(level=0).mean().reindex(range(len(grid)))
    y = pd.Series(sp, index=np.floor(te * fs).astype(int)).groupby(level=0).mean().reindex(range(len(grid)))
    xv, yv = x.to_numpy(), y.to_numpy(); lags = np.arange(-int(a.max_lag * fs), int(a.max_lag * fs) + 1); rs = []
    for L in lags:  # positive L: video signal shifted later than telemetry
        aa, bb = (xv[L:], yv[:len(yv) - L]) if L >= 0 else (xv[:L], yv[-L:])
        ok = np.isfinite(aa) & np.isfinite(bb); rs.append(np.corrcoef(aa[ok], bb[ok])[0, 1] if ok.sum() > 100 else np.nan)
    rs = np.array(rs); k = int(np.nanargmax(rs))
    row = {"tag": a.tag, "video_frames": len(sig), "video_s": round(len(sig) / FPS, 2), "telemetry_rows": len(d), "telemetry_s": round(float(te[-1]) + 0.04, 2),
           "avi_sync_first_row_s": round(avi0, 3), "pixel_peak_lag_s": lags[k] / fs, "peak_r": round(float(rs[k]), 3), "r_at_0": round(float(rs[lags == 0][0]), 3),
           "meaning": "pixel_peak_lag_s = how much later the overlay speed appears in raw video time than in telemetry elapsed time; equals the video lead-in before telemetry row 0"}
    print(row)
    R = pd.DataFrame([row])
    if OUT.exists(): old = pd.read_csv(OUT); R = pd.concat([old[old.tag != a.tag], R], ignore_index=True)
    R.to_csv(OUT, index=False); print("->", OUT)


if __name__ == "__main__":
    main()
