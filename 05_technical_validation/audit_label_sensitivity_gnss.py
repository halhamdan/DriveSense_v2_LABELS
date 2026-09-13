"""
Label sensitivity to the source quality of the GNSS-derived acceleration
channels (Technical Validation, "Vehicle telemetry consistency").

The released lat_acc_g / lon_acc_g are Racelogic's LatAcc / LongAcc: the
one-sample backward difference of the 25 Hz Doppler speed (longitudinal) and
speed x heading-rate (lateral), unsmoothed (audit_acceleration_provenance).
A single-sample GNSS speed or heading jump therefore produces a spurious
acceleration spike, and because the labelling rule uses rolling extrema over
a trailing 1 s window, one spike keeps a threshold condition true for ~25
samples -- long enough to pass the 0.4 s minimum duration. Braking and
acceleration labels have independent corroboration (CAN speed drop; throttle);
Harsh Turning does not.

This script re-runs the released labelling algorithm on every native session
(a) unchanged (baseline; reproduces the released labels) and (b) after each of
three documented source-quality treatments of the two acceleration columns:

  T1 mask_gnss    : set acceleration to NaN where the GNSS solution is suspect
                    (solution type 0, < 6 satellites, |heading jump| > 10 deg
                    or |speed jump| > 3 km/h in one sample). NaN never satisfies
                    a threshold, so this can only remove labels.
  T2 median3      : 3-sample median filter -- removes isolated single-sample
                    spikes, preserves any structure >= 2 samples.
  T3 mean5        : 5-sample (0.2 s) centred moving average -- the kind of
                    light smoothing Racelogic's own software offers.

and reports, per class, the number of discrete events before/after and how
many baseline events are lost / newly created (event = run of identical
non-Normal labels; an event is "retained" if any of its samples keeps the same
label after treatment).

Usage:
    python audit_label_sensitivity_gnss.py
Environment:
    STAGING_LABELED_DIR   directory of native D{d}_S{s}_LABELED.csv files
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "01_annotation"))
import label_harsh_events as lhe  # noqa: E402

LABELED_DIR = Path(os.environ.get("STAGING_LABELED_DIR", ""))
TRIM_CSV = REPO / "04_session_trimming" / "trim_review_output" / "trim_points_final.csv"
OUT = Path(__file__).resolve().parent / "validation_output" / "label_sensitivity_gnss.csv"

CLASSES = ["Harsh Acceleration", "Harsh Braking", "Harsh Turning"]
LAT, LON = "Lateral acceleration (g)", "Longitudinal acceleration (g)"


def events(lab: np.ndarray):
    """list of (cls, start, end) runs of identical non-Normal labels"""
    out = []; i = 0; n = len(lab)
    while i < n:
        if lab[i] != "Normal":
            k = i
            while k + 1 < n and lab[k + 1] == lab[i]: k += 1
            out.append((lab[i], i, k)); i = k + 1
        else: i += 1
    return out


def treat(df: pd.DataFrame, how: str) -> pd.DataFrame:
    d = df.copy()
    lat = pd.to_numeric(d[LAT], errors="coerce"); lon = pd.to_numeric(d[LON], errors="coerce")
    if how == "mask_gnss":
        sol = pd.to_numeric(d["Solution type"], errors="coerce"); sats = pd.to_numeric(d["Satellites"], errors="coerce")
        psi = np.unwrap(np.deg2rad(pd.to_numeric(d["Heading (Degrees)"], errors="coerce").to_numpy(float)))
        dpsi = np.rad2deg(np.abs(np.diff(psi, prepend=psi[0])))
        v = pd.to_numeric(d["Speed (km/h)"], errors="coerce").to_numpy(float); dv = np.abs(np.diff(v, prepend=v[0]))
        bad = (sol == 0) | (sats < 6) | (dpsi > 10) | (dv > 3)
        lat = lat.mask(bad); lon = lon.mask(bad)
    elif how == "median3":
        lat = lat.rolling(3, center=True, min_periods=1).median(); lon = lon.rolling(3, center=True, min_periods=1).median()
    elif how == "mean5":
        lat = lat.rolling(5, center=True, min_periods=1).mean(); lon = lon.rolling(5, center=True, min_periods=1).mean()
    d[LAT] = lat; d[LON] = lon
    return d


def main():
    if not LABELED_DIR.exists():
        raise SystemExit(f"STAGING_LABELED_DIR not found or unset: {LABELED_DIR!r}")
    tags = pd.read_csv(TRIM_CSV).query("status == 'ok'")["tag"].tolist()
    treatments = ["mask_gnss", "median3", "mean5"]
    rows = []
    for tag in tags:
        p = LABELED_DIR / f"{tag}_LABELED.csv"
        if not p.exists():
            print(f"  [missing] {tag}"); continue
        raw = pd.read_csv(p)
        if "Label" in raw.columns: raw = raw.drop(columns=["Label"])
        base = lhe.label_vehicle_dynamics(raw, lhe.CONFIG)["Label"].to_numpy()
        ev0 = events(base)
        rec = {"tag": tag}
        for c in CLASSES: rec[f"base_{c}"] = sum(1 for e in ev0 if e[0] == c)
        for how in treatments:
            lab = lhe.label_vehicle_dynamics(treat(raw, how), lhe.CONFIG)["Label"].to_numpy()
            ev1 = events(lab)
            for c in CLASSES:
                rec[f"{how}_{c}"] = sum(1 for e in ev1 if e[0] == c)
                lost = sum(1 for (cls, a, b) in ev0 if cls == c and not (lab[a:b + 1] == c).any())
                new = sum(1 for (cls, a, b) in ev1 if cls == c and not (base[a:b + 1] == c).any())
                rec[f"{how}_{c}_lost"] = lost; rec[f"{how}_{c}_new"] = new
            rec[f"{how}_changed_rows_pct"] = float((lab != base).mean() * 100)
        rows.append(rec)
        print(f"  {tag}: base T={rec['base_Harsh Turning']} B={rec['base_Harsh Braking']} A={rec['base_Harsh Acceleration']} | "
              f"median3 lost T/B/A = {rec['median3_Harsh Turning_lost']}/{rec['median3_Harsh Braking_lost']}/{rec['median3_Harsh Acceleration_lost']}", flush=True)
    R = pd.DataFrame(rows); R.to_csv(OUT, index=False)
    print("\n=== totals over", len(R), "sessions ===")
    for c in CLASSES:
        b = R[f"base_{c}"].sum()
        line = f"{c:19s} baseline {b:5d}"
        for how in treatments:
            line += f" | {how}: {R[f'{how}_{c}'].sum():5d} (lost {R[f'{how}_{c}_lost'].sum():4d} = {R[f'{how}_{c}_lost'].sum()/max(1,b)*100:4.1f}%, new {R[f'{how}_{c}_new'].sum():3d})"
        print(line)
    for how in treatments:
        print(f"{how}: rows with changed label {np.average(R[f'{how}_changed_rows_pct']):.3f}% (mean over sessions)")
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
