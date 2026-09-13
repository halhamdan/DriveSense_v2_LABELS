"""
What would physically meaningful harsh-event labels look like? (decision aid,
2026-09-13; NOT part of the released pipeline)

Motivation: the released lat_acc_g / lon_acc_g are the unsmoothed one-sample
derivative of 25 Hz Doppler speed / heading (audit_acceleration_provenance).
event_exceedance_structure shows that ~80-87% of released events contain at
most ONE consecutive sample above the class threshold, and the CAN-speed-
derived 1 s deceleration inside "Harsh Braking" events is typically -0.08 g.
This script counts events under candidate redefinitions that require a
SMOOTHED acceleration to exceed the threshold for the full minimum duration,
keeping the existing speed / throttle gates, and reports how many braking
events are corroborated by the independent CAN-speed-derived deceleration.

Runs on the released fused.csv (25 Hz). Configurations:
  A  0.5 s smoothing, >= 0.5 s exceedance, thresholds 0.38 / 0.35 / 0.55 g (current)
  B  0.5 s smoothing, >= 0.5 s exceedance, thresholds 0.30 / 0.30 / 0.40 g
  C  1.0 s smoothing, >= 0.5 s exceedance, thresholds 0.25 / 0.25 / 0.35 g

Usage:
    python estimate_redefined_labels.py
Environment:
    DATASET_ROOT   release root containing Preprocessed_Dataset/
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", ""))
CONFIGS = {
    "A: 0.5s smooth, dur>=0.5s, thr 0.38/0.35/0.55": (13, 13, 0.38, 0.35, 0.55),
    "B: 0.5s smooth, dur>=0.5s, thr 0.30/0.30/0.40": (13, 13, 0.30, 0.30, 0.40),
    "C: 1.0s smooth, dur>=0.5s, thr 0.25/0.25/0.35": (25, 13, 0.25, 0.25, 0.35),
}


def runs(mask, minlen):
    out = []; i = 0; n = len(mask)
    while i < n:
        if mask[i]:
            k = i
            while k + 1 < n and mask[k + 1]: k += 1
            if k - i + 1 >= minlen: out.append((i, k))
            i = k + 1
        else: i += 1
    return out


def main():
    if not DATASET_ROOT.exists():
        raise SystemExit(f"DATASET_ROOT not found or unset: {DATASET_ROOT!r}")
    tot = {k: {"HA": 0, "HB": 0, "HT": 0, "HB_can_ok": 0} for k in CONFIGS}
    for f in sorted((DATASET_ROOT / "Preprocessed_Dataset").rglob("*_fused.csv")):
        d = pd.read_csv(f, usecols=["gps_speed_kmh", "speed_kph", "throttle_pct", "lon_acc_g", "lat_acc_g"])
        v = d.gps_speed_kmh.to_numpy(); vc = d.speed_kph.to_numpy() / 3.6; thr_pct = d.throttle_pct.to_numpy()
        a_can_1s = pd.Series(np.gradient(vc) / 0.04).rolling(25, center=True, min_periods=5).mean().to_numpy()
        for name, (w, minlen, tA, tB, tT) in CONFIGS.items():
            lon = pd.Series(d.lon_acc_g).rolling(w, center=True, min_periods=1).mean().to_numpy()
            lat = pd.Series(d.lat_acc_g.abs()).rolling(w, center=True, min_periods=1).mean().to_numpy()
            thr_s = pd.Series(thr_pct).rolling(w, center=True, min_periods=1).max().to_numpy()
            rb = runs((lon <= -tB) & (v > 15), minlen); ra = runs((lon >= tA) & (v > 15) & (thr_s > 25), minlen); rt = runs((lat >= tT) & (v > 30), minlen)
            tot[name]["HB"] += len(rb); tot[name]["HA"] += len(ra); tot[name]["HT"] += len(rt)
            tot[name]["HB_can_ok"] += sum(1 for i, k in rb if np.nanmin(a_can_1s[max(0, i - 13):k + 1]) <= -0.2 * 9.81)
    print("released baseline: HA 1429 | HB 6320 | HT 2645 | total 10394\n")
    for k, t in tot.items():
        print(f"{k}: HA {t['HA']:4d} | HB {t['HB']:4d} (CAN 1s-mean <= -0.2 g in {t['HB_can_ok']/max(1,t['HB'])*100:.0f}%) | "
              f"HT {t['HT']:4d} | total {t['HA']+t['HB']+t['HT']}")


if __name__ == "__main__":
    main()
