"""
Detects, for every session, the timestamp when the vehicle leaves its fixed
base/garage location (session start trim point) and when it returns to it
(session end trim point) -- ANALYSIS ONLY, does not touch any files.

Since this is a single study vehicle (not each driver's personal home), the
start/end GPS point should be nearly identical across all 79 sessions. We
exploit that: compute a "base location" as the median of every session's
first and last valid GPS fix, then for each session find the first/last
moment the vehicle is both (a) beyond a distance threshold from that base
location and (b) sustaining a real-driving speed -- not a garage-manoeuvre
speed blip.

Usage:
  python detect_garage_trim.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import STAGING_LABELED_DIR as RAW_DIR, PUBLISHED

OUT_DIR = Path(__file__).parent / "trim_review_output"
OUT_DIR.mkdir(exist_ok=True)

DIST_THRESHOLD_M = 120.0     # must be this far from base to count as "left"
SPEED_THRESHOLD_KMH = 15.0   # must sustain at least this speed
SUSTAIN_S = 8.0              # ...for at least this many seconds continuously

_DDM_RE = re.compile(r"(\d+)\D+([\d.]+)\s*([NSEW])")


def _parse_ddm(series: pd.Series) -> pd.Series:
    def parse_one(v):
        if not isinstance(v, str):
            return np.nan
        m = _DDM_RE.match(v.strip())
        if not m:
            return np.nan
        deg, minutes, hemi = m.groups()
        val = float(deg) + float(minutes) / 60.0
        return -val if hemi in ("S", "W") else val
    return series.map(parse_one)


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_session(driver: int, session: int) -> pd.DataFrame | None:
    raw_path = RAW_DIR / f"D{driver}_S{session}_LABELED.csv"
    fused_path = PUBLISHED / f"D{driver}" / f"Session_{session}" / f"D{driver}_S{session}_fused.csv"
    if not raw_path.exists() or not fused_path.exists():
        return None
    raw = pd.read_csv(raw_path, usecols=["Elapsed time (s)", "Latitude", "Longitude"])
    raw = raw.rename(columns={"Elapsed time (s)": "elapsed_s"})
    raw["Latitude"] = _parse_ddm(raw["Latitude"])
    raw["Longitude"] = _parse_ddm(raw["Longitude"])
    raw = raw.dropna(subset=["Latitude", "Longitude"])
    # Plausible Riyadh bounding box -- catches GPS cold-start / no-fix garbage
    # readings (e.g. lat=4.7,lon=61.2 or lat=-29.9,lon=47.4 seen in D3_S3,
    # D16_S1) that a simple "|value|>1" check does not.
    raw = raw[(raw["Latitude"].between(24.0, 25.5)) & (raw["Longitude"].between(46.0, 47.5))]
    if raw.empty:
        return None
    raw = raw.sort_values("elapsed_s")

    fused = pd.read_csv(fused_path, usecols=["elapsed_s", "gps_speed_kmh", "label"])
    if len(fused) < 25 * 40:
        return None
    fused = fused.sort_values("elapsed_s")

    merged = pd.merge_asof(fused, raw, on="elapsed_s", direction="nearest", tolerance=2.0)
    merged = merged.dropna(subset=["Latitude", "Longitude"])
    if merged.empty:
        return None
    merged["driver"], merged["session"] = driver, session
    return merged


def find_sustained_start(t, speed, dist, thresh_speed, thresh_dist, sustain_s):
    """First index where speed>=thresh AND dist>=thresh_dist, sustained for sustain_s."""
    ok = (speed >= thresh_speed) & (dist >= thresh_dist)
    n = len(t)
    for i in range(n):
        if not ok[i]:
            continue
        end_t = t[i] + sustain_s
        j = i
        while j < n and t[j] <= end_t:
            j += 1
        window = ok[i:j]
        if len(window) > 0 and window.mean() > 0.8:
            return i
    return None


def find_sustained_end(t, speed, dist, thresh_speed, thresh_dist, sustain_s):
    """Last index where speed>=thresh AND dist>=thresh_dist, sustained for sustain_s before it."""
    ok = (speed >= thresh_speed) & (dist >= thresh_dist)
    n = len(t)
    for i in range(n - 1, -1, -1):
        if not ok[i]:
            continue
        start_t = t[i] - sustain_s
        j = i
        while j >= 0 and t[j] >= start_t:
            j -= 1
        window = ok[j + 1:i + 1]
        if len(window) > 0 and window.mean() > 0.8:
            return i
    return None


def main():
    sessions = []
    for d in range(1, 21):
        for s in range(1, 5):
            df = load_session(d, s)
            if df is not None:
                sessions.append(df)
    print(f"Loaded {len(sessions)} sessions with usable GPS")

    # base location: median of each session's first and last valid GPS fix
    first_pts = np.array([[df["Latitude"].iloc[0], df["Longitude"].iloc[0]] for df in sessions])
    last_pts = np.array([[df["Latitude"].iloc[-1], df["Longitude"].iloc[-1]] for df in sessions])
    all_pts = np.vstack([first_pts, last_pts])
    base_lat, base_lon = np.median(all_pts[:, 0]), np.median(all_pts[:, 1])
    print(f"Estimated base location: {base_lat:.6f}, {base_lon:.6f}")

    dist_first = haversine_m(first_pts[:, 0], first_pts[:, 1], base_lat, base_lon)
    dist_last = haversine_m(last_pts[:, 0], last_pts[:, 1], base_lat, base_lon)
    print(f"Distance of session-start points from base: mean={dist_first.mean():.1f}m, "
          f"median={np.median(dist_first):.1f}m, max={dist_first.max():.1f}m")
    print(f"Distance of session-end points from base:   mean={dist_last.mean():.1f}m, "
          f"median={np.median(dist_last):.1f}m, max={dist_last.max():.1f}m")

    rows = []
    for df in sessions:
        t = df["elapsed_s"].to_numpy()
        speed = df["gps_speed_kmh"].to_numpy()
        dist = haversine_m(df["Latitude"].to_numpy(), df["Longitude"].to_numpy(), base_lat, base_lon)

        i_start = find_sustained_start(t, speed, dist, SPEED_THRESHOLD_KMH, DIST_THRESHOLD_M, SUSTAIN_S)
        i_end = find_sustained_end(t, speed, dist, SPEED_THRESHOLD_KMH, DIST_THRESHOLD_M, SUSTAIN_S)

        d, s = int(df["driver"].iloc[0]), int(df["session"].iloc[0])
        orig_dur = t[-1] - t[0]
        if i_start is None or i_end is None or i_start >= i_end:
            rows.append({"driver": d, "session": s, "status": "FAILED_DETECTION",
                         "orig_duration_s": orig_dur})
            continue

        trim_start_s = t[i_start] - t[0]
        trim_end_s = t[-1] - t[i_end]
        new_dur = t[i_end] - t[i_start]
        rows.append({
            "driver": d, "session": s, "status": "ok",
            "orig_duration_s": orig_dur,
            "trim_start_s": trim_start_s, "trim_end_s": trim_end_s,
            "new_duration_s": new_dur,
            "dist_at_start_m": dist[i_start], "dist_at_end_m": dist[i_end],
            "speed_at_start_kmh": speed[i_start], "speed_at_end_kmh": speed[i_end],
            "abs_start_s": t[i_start], "abs_end_s": t[i_end],
        })

    report = pd.DataFrame(rows)
    report.to_csv(OUT_DIR / "garage_trim_detection.csv", index=False)

    ok = report[report["status"] == "ok"]
    failed = report[report["status"] != "ok"]
    print(f"\n=== Detection results: {len(ok)}/{len(report)} sessions detected OK ===")
    if len(ok):
        print(f"trim_start_s: mean={ok['trim_start_s'].mean():.1f}, median={ok['trim_start_s'].median():.1f}, "
              f"min={ok['trim_start_s'].min():.1f}, max={ok['trim_start_s'].max():.1f}")
        print(f"trim_end_s:   mean={ok['trim_end_s'].mean():.1f}, median={ok['trim_end_s'].median():.1f}, "
              f"min={ok['trim_end_s'].min():.1f}, max={ok['trim_end_s'].max():.1f}")
    if len(failed):
        print(f"\nFAILED sessions (need manual review):")
        print(failed[["driver", "session", "orig_duration_s"]].to_string(index=False))

    print("\nSessions with unusually large trims (start or end > 90s) -- worth a spot-check:")
    outliers = ok[(ok["trim_start_s"] > 90) | (ok["trim_end_s"] > 90)]
    print(outliers[["driver", "session", "trim_start_s", "trim_end_s", "orig_duration_s"]].to_string(index=False) if len(outliers) else "(none)")

    print("\nSessions with near-zero trims (start or end < 5s) -- worth a spot-check:")
    tiny = ok[(ok["trim_start_s"] < 5) | (ok["trim_end_s"] < 5)]
    print(tiny[["driver", "session", "trim_start_s", "trim_end_s", "orig_duration_s"]].to_string(index=False) if len(tiny) else "(none)")


if __name__ == "__main__":
    main()
