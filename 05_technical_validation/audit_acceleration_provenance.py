"""
Provenance of the released acceleration channels (Methods, "Data-acquisition
equipment"; Technical Validation, "Vehicle telemetry consistency").

The VBOX Video HD2 recordings (.vbo headers) contain no accelerometer or IMU
channel. The "Lateral acceleration (g)" / "Longitudinal acceleration (g)"
columns in the exported CSVs are Racelogic's standard GNSS-derived LatAcc /
LongAcc. This script verifies that derivation directly on the released
Raw_Dataset/*_VBOX_raw.csv files, for every valid session:

    lon_acc_g[t]  ~  (v[t] - v[t-1]) / dt / g            v = gps_speed_kmh / 3.6
    lat_acc_g[t]  ~  -v[t] * (psi[t] - psi[t-1]) / dt / g  psi = unwrapped heading (rad);
                                                          sign: positive to the left

reporting Pearson r and least-squares slope (moving samples, > 3 km/h),
together with the correlation against the CAN indicated speed's derivative --
an independent speed source -- which is a CONSISTENCY assessment, not a
validation against a reference accelerometer.

Usage:
    python audit_acceleration_provenance.py
Environment:
    DATASET_ROOT   release root containing Raw_Dataset/
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", ""))
OUT = Path(__file__).resolve().parent / "validation_output" / "acceleration_provenance.csv"
G = 9.81


def fit(a, b, m):
    m = m & np.isfinite(a) & np.isfinite(b)
    if m.sum() < 100:
        return np.nan, np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1]), float(np.polyfit(b[m], a[m], 1)[0])


def main():
    if not DATASET_ROOT.exists():
        raise SystemExit(f"DATASET_ROOT not found or unset: {DATASET_ROOT!r}")
    rows = []
    for f in sorted((DATASET_ROOT / "Raw_Dataset").rglob("*_VBOX_raw.csv")):
        tag = f.stem.replace("_VBOX_raw", "")
        x = pd.read_csv(f, usecols=["elapsed_s", "gps_speed_kmh", "speed_kph", "heading_deg", "lat_acc_g", "lon_acc_g"])
        dt = np.diff(x.elapsed_s.to_numpy(), prepend=np.nan)
        v = x.gps_speed_kmh.to_numpy(float) / 3.6
        vc = x.speed_kph.to_numpy(float) / 3.6
        psi = np.unwrap(np.deg2rad(x.heading_deg.to_numpy(float)))
        a_gnss = np.diff(v, prepend=np.nan) / dt / G
        a_can = np.diff(vc, prepend=np.nan) / dt / G
        a_lat = -v * np.diff(psi, prepend=np.nan) / dt / G
        lon = x.lon_acc_g.to_numpy(float); lat = x.lat_acc_g.to_numpy(float)
        m = v > 3.0
        r1, k1 = fit(lon, a_gnss, m); r2, k2 = fit(lat, a_lat, m); r3, k3 = fit(lon, a_can, m)
        rows.append(dict(tag=tag, n=len(x), lon_vs_dGNSSspeed_r=r1, lon_vs_dGNSSspeed_slope=k1,
                         lat_vs_v_headingrate_r=r2, lat_vs_v_headingrate_slope=k2,
                         lon_vs_dCANspeed_r=r3, lon_vs_dCANspeed_slope=k3))
    R = pd.DataFrame(rows); R.to_csv(OUT, index=False)
    print(f"sessions: {len(R)}")
    for c in ["lon_vs_dGNSSspeed_r", "lon_vs_dGNSSspeed_slope", "lat_vs_v_headingrate_r", "lat_vs_v_headingrate_slope",
              "lon_vs_dCANspeed_r", "lon_vs_dCANspeed_slope"]:
        print(f"  {c:28s} median {R[c].median():.3f}  min {R[c].min():.3f}  max {R[c].max():.3f}")
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
