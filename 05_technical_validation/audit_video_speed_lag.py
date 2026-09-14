"""
Decisive video-to-telemetry alignment check: background motion in the front
(driver-facing) camera vs GNSS speed.

When the vehicle moves, the scene visible through the side windows at the
left/right edges of the front-camera frame streams past; when it stops, it
freezes. The mean absolute frame-to-frame difference in those edge columns is
therefore a strong proxy for |speed|, independent of the driver. Cross-
correlating it with gps_speed_kmh from fused.csv over lags of +/-20 s locates
the true offset between the released video timeline (frame k <-> k/30 s) and
the telemetry timeline. A correct release gives a sharp peak at 0 s.

Frames are decoded with ffmpeg to 96x54 grey (no OpenCV needed).

Usage: python audit_video_speed_lag.py --sessions D1_S1 D11_S2 D5_S2 [--max-lag 20]
Output: validation_output/video_speed_lag.csv
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("DATASET_ROOT", "") or r"C:\Users\halha\OneDrive - Durham University\Documents\Published_Dataset_Final")
OUT = Path(__file__).resolve().parent / "validation_output" / "video_speed_lag.csv"
W, H = 96, 54
FPS = 30


def edge_motion(video: Path) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"scale={W}:{H},format=gray", "-f", "rawvideo", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10 ** 7)
    prev = None; out = []
    n = W * H
    while True:
        buf = p.stdout.read(n)
        if len(buf) < n:
            break
        f = np.frombuffer(buf, np.uint8).reshape(H, W).astype(np.int16)
        edges = np.concatenate([f[:, :W // 5], f[:, -W // 5:]], axis=1)
        if prev is not None:
            out.append(float(np.abs(edges - prev).mean()))
        prev = edges
    p.wait()
    return np.array(out)


def session_lag(tag: str, max_lag: float):
    d, s = tag[1:].split("_S")
    vid = ROOT / "Raw_Dataset" / f"D{d}" / f"Session_{s}" / f"{tag}_Front_blurred.mp4"
    fused = ROOT / "Preprocessed_Dataset" / f"D{d}" / f"Session_{s}" / f"{tag}_fused.csv"
    m = edge_motion(vid)  # m[i] = motion between frame i and i+1, time ~ (i+1)/30
    tm = (np.arange(len(m)) + 1) / FPS
    F = pd.read_csv(fused, usecols=["elapsed_s", "gps_speed_kmh"])
    fs = 5.0; grid = np.arange(0, min(tm[-1], F.elapsed_s.iloc[-1]), 1 / fs)
    mb = pd.Series(m, index=np.floor(tm * fs).astype(int)).groupby(level=0).mean().reindex(range(len(grid)))
    vb = pd.Series(F.gps_speed_kmh.to_numpy(), index=np.floor(F.elapsed_s.to_numpy() * fs).astype(int)).groupby(level=0).mean().reindex(range(len(grid)))
    x = mb.rolling(int(fs), center=True, min_periods=1).mean(); y = vb.rolling(int(fs), center=True, min_periods=1).mean()
    x = x - x.rolling(int(120 * fs), center=True, min_periods=1).mean(); y = y - y.rolling(int(120 * fs), center=True, min_periods=1).mean()
    xv, yv = x.to_numpy(), y.to_numpy(); lags = np.arange(-int(max_lag * fs), int(max_lag * fs) + 1); rs = []
    for L in lags:
        a, b = (xv[L:], yv[:len(yv) - L]) if L >= 0 else (xv[:L], yv[-L:])
        ok = np.isfinite(a) & np.isfinite(b); rs.append(np.corrcoef(a[ok], b[ok])[0, 1])
    rs = np.array(rs); k = int(np.nanargmax(rs))
    return {"tag": tag, "video_frames": len(m) + 1, "fused_rows": len(F), "video_s": round(len(m) / FPS + 1 / FPS, 2), "telemetry_s": round(float(F.elapsed_s.iloc[-1]) + 0.04, 2),
            "peak_lag_s": lags[k] / fs, "peak_r": round(float(rs[k]), 3), "r_at_0": round(float(rs[lags == 0][0]), 3),
            "lag_meaning": "positive = video background motion occurs LATER in the released timeline than the telemetry speed"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sessions", nargs="+", required=True); ap.add_argument("--max-lag", type=float, default=20.0)
    a = ap.parse_args(); rows = []
    for t in a.sessions:
        r = session_lag(t, a.max_lag); rows.append(r); print(r, flush=True)
    R = pd.DataFrame(rows)
    if OUT.exists():
        old = pd.read_csv(OUT); R = pd.concat([old[~old.tag.isin(R.tag)], R], ignore_index=True)
    R.to_csv(OUT, index=False); print("->", OUT)


if __name__ == "__main__":
    main()
