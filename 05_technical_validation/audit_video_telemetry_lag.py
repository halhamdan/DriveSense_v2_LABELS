"""
Empirical video-to-telemetry alignment check using released files only.

Steering moves the driver's wrists, and lateral acceleration follows steering
within a fraction of a second, so the frame-to-frame wrist displacement in
Side_pose.csv (30 fps, released frame index k <-> elapsed_s k/30) should be
maximally correlated with |lat_acc_g| from fused.csv (25 Hz) at lag ~0 if the
released video and telemetry timelines coincide. Any systematic lag (e.g. the
~10-11 s by which the VBOX HD2 starts its video before the first logged
telemetry sample, visible as `Avi sync time` in the native export) shows up as
a shifted cross-correlation peak, consistent in sign across sessions.

Writes validation_output/video_telemetry_lag.csv (per session: peak lag, peak r,
r at lag 0, number of usable seconds).

Usage: python audit_video_telemetry_lag.py [--max-lag 20] [--sessions D1_S1 D2_S2 ...]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("DATASET_ROOT", "") or r"C:\Users\halha\OneDrive - Durham University\Documents\DriveSense_Packages\3_OLD_versions\data\Published_Dataset_Final_v1_ORIGINAL_20260909")
OUT = Path(__file__).resolve().parent / "validation_output" / "video_telemetry_lag.csv"
FS = 5.0  # analysis rate, Hz


def session_lag(tag: str, max_lag: float):
    d, s = tag[1:].split("_S")
    pose_f = ROOT / "Preprocessed_Dataset" / f"D{d}" / f"Session_{s}" / f"{tag}_Side_pose.csv"
    fused_f = ROOT / "Preprocessed_Dataset" / f"D{d}" / f"Session_{s}" / f"{tag}_fused.csv"
    if not pose_f.exists():
        return None
    P = pd.read_csv(pose_f)
    wc = [c for c in P.columns if ("wrist" in c.lower()) and c.endswith(("_x", "_y"))]
    if "timestamp_s" not in P or not wc:
        return None
    det = P["pose_detected"].astype(bool) if "pose_detected" in P else P[wc[0]].notna()
    W = P.loc[det, ["timestamp_s"] + wc].copy()
    xy = W[wc].to_numpy(float); t = W["timestamp_s"].to_numpy(float)
    disp = np.sqrt(np.nansum(np.diff(xy, axis=0) ** 2, axis=1)); disp = np.where(np.diff(t) < 0.2, disp, np.nan)  # only consecutive frames
    tm = t[1:]
    F = pd.read_csv(fused_f, usecols=["elapsed_s", "lat_acc_g", "gps_speed_kmh"])
    T_end = min(tm[-1] if len(tm) else 0, F.elapsed_s.iloc[-1])
    grid = np.arange(0, T_end, 1 / FS)
    if len(grid) < 600 * FS:
        return None
    # bin both signals to the grid (mean per bin), then 1 s smoothing
    wb = pd.Series(disp, index=pd.cut(tm, np.r_[grid, grid[-1] + 1 / FS], labels=False, include_lowest=True)).groupby(level=0).mean()
    lb = pd.Series(F.lat_acc_g.abs().clip(upper=1.2).to_numpy(), index=pd.cut(F.elapsed_s.to_numpy(), np.r_[grid, grid[-1] + 1 / FS], labels=False, include_lowest=True)).groupby(level=0).mean()
    w = pd.Series(np.nan, index=range(len(grid))); w.loc[wb.index.astype(int)] = wb.values
    l = pd.Series(np.nan, index=range(len(grid))); l.loc[lb.index.astype(int)] = lb.values
    w = w.rolling(int(FS), center=True, min_periods=1).mean(); l = l.rolling(int(FS), center=True, min_periods=1).mean()
    w = (w - w.rolling(int(60 * FS), center=True, min_periods=1).mean()); l = (l - l.rolling(int(60 * FS), center=True, min_periods=1).mean())  # detrend
    wv = w.to_numpy(); lv = l.to_numpy()
    lags = np.arange(-int(max_lag * FS), int(max_lag * FS) + 1)
    rs = []
    for L in lags:
        if L >= 0: a, b = wv[L:], lv[:len(lv) - L]
        else: a, b = wv[:L], lv[-L:]
        ok = np.isfinite(a) & np.isfinite(b)
        rs.append(np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 100 else np.nan)
    rs = np.array(rs); k = int(np.nanargmax(rs))
    return {"tag": tag, "peak_lag_s": lags[k] / FS, "peak_r": round(float(rs[k]), 3), "r_at_0": round(float(rs[lags == 0][0]), 3),
            "usable_s": int(np.isfinite(wv).sum() / FS),
            "lag_meaning": "positive = wrist motion (video) occurs LATER in the released timeline than lateral acceleration (telemetry)"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--max-lag", type=float, default=20.0); ap.add_argument("--sessions", nargs="*")
    a = ap.parse_args()
    tags = a.sessions or [f"D{d}_S{s}" for d in range(1, 21) for s in range(1, 5)]
    rows = []
    for tag in tags:
        try:
            r = session_lag(tag, a.max_lag)
        except Exception as e:  # noqa: BLE001
            r = {"tag": tag, "error": str(e)}
        if r: rows.append(r); print(r, flush=True)
    R = pd.DataFrame(rows); R.to_csv(OUT, index=False)
    ok = R.dropna(subset=["peak_lag_s"]) if "peak_lag_s" in R else R
    if len(ok):
        print(f"\n{len(ok)} sessions: peak lag median {ok.peak_lag_s.median():+.1f} s (IQR {ok.peak_lag_s.quantile(.25):+.1f} to {ok.peak_lag_s.quantile(.75):+.1f}); "
              f"|lag| <= 1 s in {(ok.peak_lag_s.abs() <= 1).sum()}; peak r median {ok.peak_r.median():.2f}, r at 0 median {ok.r_at_0.median():.2f}")
    print("->", OUT)


if __name__ == "__main__":
    main()
