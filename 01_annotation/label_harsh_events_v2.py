"""
Harsh-event labelling, version 2 (2026-09-13): sustained exceedance of a
smoothed, GNSS-derived acceleration.

Why a version 2. The released acceleration channels are Racelogic's LatAcc /
LongAcc: the one-sample backward difference of the 25 Hz Doppler speed
(longitudinal) and speed x heading-rate (lateral), unsmoothed. Version 1
applied fixed thresholds to rolling extrema of that raw derivative over a
trailing 1 s window, so a single 40 ms spike of Doppler noise kept a condition
true for ~25 samples and passed the 0.4 s minimum duration. On the released
data 78-87% of v1 events contained at most one consecutive sample above their
threshold (05_technical_validation/event_exceedance_structure). Version 2
therefore (a) smooths the acceleration with a 0.5 s centred moving average and
(b) requires the smoothed signal itself to exceed the threshold for at least
0.5 s. Thresholds were calibrated on three deliberate-manoeuvre drives in the
study vehicle (05_technical_validation/estimate_redefined_labels.py and the
manuscript's Annotation validation): 0.35 g recovers 10/10 deliberate brakes,
0.45 g recovers 11/11 deliberate turns, and 0.20 g recovers the deliberate
accelerations, which peak at 0.23-0.29 g -- the vehicle's own ceiling.

Rule (all on the released 25 Hz fused.csv columns):
  a_lon_s, a_lat_s : 0.5 s (13-sample) centred moving average of lon_acc_g, |lat_acc_g|
  Harsh Braking      : a_lon_s <= -0.35 g  AND gps_speed > 15 km/h
                       AND CAN-speed-derived deceleration (1 s rolling min of
                       d(speed_kph)/dt) <= -0.3 m/s^2 (independent confirmation)
  Harsh Acceleration : a_lon_s >=  0.20 g  AND gps_speed > 15 km/h AND throttle (0.5 s max) > 25 %
  Harsh Turning      : a_lat_s >=  0.45 g  AND gps_speed > 30 km/h
  Each class mask: close gaps <= 0.2 s, keep runs >= 0.5 s. No dilation -- an
  event is the exceedance interval itself. Priority Turning > Braking >
  Acceleration where masks overlap. Everything else is Normal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CONFIG_V2 = {
    "FS_HZ": 25.0,
    "SMOOTH_S": 0.5,
    "MIN_DURATION_S": 0.5,
    "CLOSE_GAP_S": 0.2,
    "BRAKE_G": 0.35,
    "ACCEL_G": 0.20,
    "TURN_G": 0.45,
    "MIN_SPEED_KMH": 15.0,
    "MIN_SPEED_TURN_KMH": 30.0,
    "THROTTLE_MIN_PCT": 25.0,
    "BRAKE_A_SPEED_MIN_MS2": -0.3,
}
G0 = 9.81


def _runs(mask: np.ndarray):
    out = []; i = 0; n = len(mask)
    while i < n:
        if mask[i]:
            k = i
            while k + 1 < n and mask[k + 1]: k += 1
            out.append((i, k)); i = k + 1
        else:
            i += 1
    return out


def _clean(mask: np.ndarray, close_n: int, min_n: int) -> np.ndarray:
    m = mask.copy()
    # close short gaps
    rs = _runs(m)
    for (a0, a1), (b0, b1) in zip(rs, rs[1:]):
        if b0 - a1 - 1 <= close_n:
            m[a1 + 1:b0] = True
    # drop short runs
    out = np.zeros_like(m)
    for a, b in _runs(m):
        if b - a + 1 >= min_n:
            out[a:b + 1] = True
    return out


def label_fused_v2(df: pd.DataFrame, cfg: dict = CONFIG_V2) -> pd.Series:
    """Returns the v2 label series for a released fused.csv DataFrame."""
    fs = cfg["FS_HZ"]; w = max(1, int(round(cfg["SMOOTH_S"] * fs)))
    min_n = max(1, int(round(cfg["MIN_DURATION_S"] * fs))); close_n = int(round(cfg["CLOSE_GAP_S"] * fs))
    v = pd.to_numeric(df["gps_speed_kmh"], errors="coerce")
    lon = pd.to_numeric(df["lon_acc_g"], errors="coerce").rolling(w, center=True, min_periods=1).mean()
    lat = pd.to_numeric(df["lat_acc_g"], errors="coerce").abs().rolling(w, center=True, min_periods=1).mean()
    thr = pd.to_numeric(df["throttle_pct"], errors="coerce").rolling(w, center=True, min_periods=1).max()
    vc = pd.to_numeric(df["speed_kph"], errors="coerce") / 3.6
    a_can = vc.diff() * fs
    a_can_min = a_can.rolling(int(fs), min_periods=1).min()

    brake = ((lon <= -cfg["BRAKE_G"]) & (v > cfg["MIN_SPEED_KMH"]) & (a_can_min <= cfg["BRAKE_A_SPEED_MIN_MS2"])).fillna(False).to_numpy()
    accel = ((lon >= cfg["ACCEL_G"]) & (v > cfg["MIN_SPEED_KMH"]) & (thr > cfg["THROTTLE_MIN_PCT"])).fillna(False).to_numpy()
    turn = ((lat >= cfg["TURN_G"]) & (v > cfg["MIN_SPEED_TURN_KMH"])).fillna(False).to_numpy()

    brake = _clean(brake, close_n, min_n); accel = _clean(accel, close_n, min_n); turn = _clean(turn, close_n, min_n)
    label = np.array(["Normal"] * len(df), dtype=object)
    label[accel] = "Harsh Acceleration"
    label[brake] = "Harsh Braking"
    label[turn] = "Harsh Turning"
    return pd.Series(label, index=df.index, name="label")


def count_events(label: pd.Series) -> dict:
    lab = label.to_numpy(); out = {}
    for a, b in _runs(lab != "Normal"):
        # split runs where the class changes inside
        i = a
        while i <= b:
            k = i
            while k + 1 <= b and lab[k + 1] == lab[i]: k += 1
            out[lab[i]] = out.get(lab[i], 0) + 1; i = k + 1
    return out
