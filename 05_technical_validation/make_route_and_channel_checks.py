"""
Two Technical Validation checks:

1. Route consistency (Technical Validation: "Route consistency") -- per-
   session GPS bounding box and total driven distance (great-circle
   distance summed over consecutive raw GPS fixes), verifying every one of
   the 79 sessions actually followed the same fixed ~17 km route, not only
   in aggregate (as the existing pooled bounding-box check already showed)
   but individually per session.

2. engine_rpm / thermopile-temperature plausible-range checks (Technical
   Validation: "Vehicle telemetry consistency" table; "Physiological
   signal characterisation"), added for parity with the other retained
   channels that already have a documented plausible-range check.

Usage:
  python make_route_and_channel_checks.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import DATASET_ROOT

RAW = DATASET_ROOT / "Raw_Dataset"
PREPROCESSED = DATASET_ROOT / "Preprocessed_Dataset"
OUT_DIR = Path(__file__).parent / "validation_output"
OUT_DIR.mkdir(exist_ok=True)


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def route_consistency():
    rows = []
    for fp in sorted(RAW.glob("D*/Session_*/D*_S*_VBOX_raw.csv")):
        tag = fp.stem.replace("_VBOX_raw", "")
        df = pd.read_csv(fp, usecols=["lat", "lon", "gps_speed_kmh"]).dropna(subset=["lat", "lon"])
        if len(df) < 2:
            continue
        d = haversine_m(df["lat"].values[:-1], df["lon"].values[:-1],
                         df["lat"].values[1:], df["lon"].values[1:])
        rows.append({
            "tag": tag, "total_km": d.sum() / 1000.0,
            "lat_min": df["lat"].min(), "lat_max": df["lat"].max(),
            "lon_min": df["lon"].min(), "lon_max": df["lon"].max(),
            "mean_speed_kmh": df["gps_speed_kmh"].mean(),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "route_consistency.csv", index=False)

    print("=== Route consistency ===")
    print(f"n sessions: {len(out)}")
    print(f"Shared bounding box: lat {out['lat_min'].min():.6f}-{out['lat_max'].max():.6f}, "
          f"lon {out['lon_min'].min():.6f}-{out['lon_max'].max():.6f}")
    ex_outlier = out.sort_values("total_km").iloc[1:]  # drop the shortest session for the "typical" stats
    print(f"Total distance excl. shortest session: mean={ex_outlier['total_km'].mean():.2f}km "
          f"sd={ex_outlier['total_km'].std():.2f}km range=[{ex_outlier['total_km'].min():.2f}, "
          f"{ex_outlier['total_km'].max():.2f}]km")
    shortest = out.sort_values("total_km").iloc[0]
    print(f"Shortest session: {shortest['tag']} = {shortest['total_km']:.2f}km, "
          f"mean speed {shortest['mean_speed_kmh']:.1f}km/h "
          f"(vs {ex_outlier['mean_speed_kmh'].mean():.1f}km/h mean for the rest)")


def channel_range_checks():
    print("\n=== engine_rpm / thermopile-temperature plausible-range checks ===")
    rpm_all, temp_all = [], []
    for fp in sorted(PREPROCESSED.glob("D*/Session_*/D*_S*_fused.csv")):
        rpm_all.append(pd.read_csv(fp, usecols=["engine_rpm"])["engine_rpm"])
        temp_all.append(pd.read_csv(fp, usecols=["TEMP_THERMOPILE"])["TEMP_THERMOPILE"])
    rpm = pd.concat(rpm_all, ignore_index=True)
    temp = pd.concat(temp_all, ignore_index=True)

    rpm_range = (500, 6500)
    temp_range = (20, 45)
    rpm_outside = ((rpm < rpm_range[0]) | (rpm > rpm_range[1])).mean()
    temp_outside = ((temp < temp_range[0]) | (temp > temp_range[1])).mean()

    print(f"engine_rpm: n={len(rpm)}, range={rpm_range}, observed=[{rpm.min():.0f}, {rpm.max():.0f}], "
          f"{rpm_outside*100:.2f}% outside")
    print(f"TEMP_THERMOPILE: n={len(temp)}, range={temp_range}, "
          f"observed=[{temp.min():.1f}, {temp.max():.1f}], {temp_outside*100:.2f}% outside")


if __name__ == "__main__":
    route_consistency()
    channel_range_checks()
