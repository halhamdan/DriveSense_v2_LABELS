"""
Parameterised harsh-event labeller used by the label-robustness audit
(05_technical_validation/audit_label_robustness_v2.py) to compare candidate
formulations of the version-2 rule on synthetic signals, the deliberate-manoeuvre
drives and the released data. All window lengths are INTEGER SAMPLE COUNTS at
25 Hz -- no float rounding at call time.

Options
-------
smooth_n        centred moving-average length (odd; 13 = k in -6..+6 = 0.52 s)
min_n           minimum run length of the exceedance mask (samples)
close_n         gaps of at most this many samples inside a run are closed
lateral         "abs_then_mean" : threshold  mean(|a_lat|)       (v2 draft of 2026-09-13)
                "mean_then_abs" : threshold |mean(a_lat)|        (rectified noise averages out)
median_n        odd length of a running-median prefilter applied to the SIGNED raw
                lateral and longitudinal channels before smoothing (0 = none)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

FS = 25
G0 = 9.81


@dataclass(frozen=True)
class RuleParams:
    smooth_n: int = 13
    min_n: int = 13
    close_n: int = 5
    lateral: str = "abs_then_mean"
    median_n: int = 0
    brake_g: float = 0.35
    accel_g: float = 0.20
    turn_g: float = 0.45
    min_speed_kmh: float = 15.0
    min_speed_turn_kmh: float = 30.0
    throttle_min_pct: float = 25.0
    brake_can_ms2: float = -0.3
    can_window_n: int = 25
    turn_min_heading_deg: float = 0.0   # >0: a turning run must change GNSS heading by at least this (net, wrapped)
    heading_margin_n: int = 3            # samples before/after the run over which the heading change is read

    def label(self) -> str:
        return (f"s{self.smooth_n}_m{self.min_n}_c{self.close_n}_{'A' if self.lateral == 'abs_then_mean' else 'S'}"
                f"{'' if not self.median_n else f'_med{self.median_n}'}"
                f"_A{self.accel_g:.2f}_B{self.brake_g:.2f}_T{self.turn_g:.2f}"
                f"{'' if not self.turn_min_heading_deg else f'_hd{self.turn_min_heading_deg:g}'}")

    def to_dict(self) -> dict:
        return asdict(self)


def runs(mask: np.ndarray):
    """[(start, end_inclusive), ...] of True runs."""
    m = np.asarray(mask, bool)
    if m.size == 0:
        return []
    d = np.diff(np.r_[0, m.astype(np.int8), 0])
    starts = np.flatnonzero(d == 1); ends = np.flatnonzero(d == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def clean_mask(mask: np.ndarray, close_n: int, min_n: int) -> np.ndarray:
    m = np.asarray(mask, bool).copy()
    rs = runs(m)
    for (a0, a1), (b0, b1) in zip(rs, rs[1:]):
        if b0 - a1 - 1 <= close_n:
            m[a1 + 1:b0] = True
    out = np.zeros_like(m)
    for a, b in runs(m):
        if b - a + 1 >= min_n:
            out[a:b + 1] = True
    return out


def heading_change_deg(heading: np.ndarray, a: int, b: int, margin: int) -> float:
    """Net (wrapped) change of GNSS heading from `margin` samples before run [a, b] to `margin` after."""
    i0 = max(a - margin, 0); i1 = min(b + margin, len(heading) - 1)
    h0, h1 = heading[i0], heading[i1]
    if not (np.isfinite(h0) and np.isfinite(h1)):
        return np.nan
    return float((h1 - h0 + 180.0) % 360.0 - 180.0)


def heading_gate(turn_mask: np.ndarray, heading: np.ndarray, min_deg: float, margin: int) -> np.ndarray:
    """Drops turning runs whose net heading change is below min_deg. Heading is the
    integral of the (noisy, derivative) heading rate, so sign-alternating noise
    leaves it unchanged while a genuine turn must move it: at the 0.45 g / 30 km/h
    thresholds a 0.52 s event changes heading by about 16 degrees."""
    out = turn_mask.copy()
    for a, b in runs(turn_mask):
        d = heading_change_deg(heading, a, b, margin)
        if np.isfinite(d) and abs(d) < min_deg:
            out[a:b + 1] = False
    return out


def _centred_mean(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, center=True, min_periods=1).mean()


def _median(x: pd.Series, n: int) -> pd.Series:
    if n <= 1:
        return x
    return x.rolling(n, center=True, min_periods=1).median()


def smoothed_signals(df: pd.DataFrame, p: RuleParams):
    lon_raw = pd.to_numeric(df["lon_acc_g"], errors="coerce")
    lat_raw = pd.to_numeric(df["lat_acc_g"], errors="coerce")
    lon_f = _median(lon_raw, p.median_n); lat_f = _median(lat_raw, p.median_n)
    lon_s = _centred_mean(lon_f, p.smooth_n)
    if p.lateral == "abs_then_mean":
        lat_s = _centred_mean(lat_f.abs(), p.smooth_n)
    elif p.lateral == "mean_then_abs":
        lat_s = _centred_mean(lat_f, p.smooth_n).abs()
    else:
        raise ValueError(p.lateral)
    return lon_s, lat_s


def label_df(df: pd.DataFrame, p: RuleParams) -> pd.Series:
    v = pd.to_numeric(df["gps_speed_kmh"], errors="coerce")
    lon_s, lat_s = smoothed_signals(df, p)
    thr = pd.to_numeric(df["throttle_pct"], errors="coerce").rolling(p.smooth_n, center=True, min_periods=1).max()
    vc = pd.to_numeric(df["speed_kph"], errors="coerce") / 3.6
    a_can_min = (vc.diff() * FS).rolling(p.can_window_n, min_periods=1).min()

    brake = ((lon_s <= -p.brake_g) & (v > p.min_speed_kmh) & (a_can_min <= p.brake_can_ms2)).fillna(False).to_numpy()
    accel = ((lon_s >= p.accel_g) & (v > p.min_speed_kmh) & (thr > p.throttle_min_pct)).fillna(False).to_numpy()
    turn = ((lat_s >= p.turn_g) & (v > p.min_speed_turn_kmh)).fillna(False).to_numpy()

    brake = clean_mask(brake, p.close_n, p.min_n); accel = clean_mask(accel, p.close_n, p.min_n); turn = clean_mask(turn, p.close_n, p.min_n)
    if p.turn_min_heading_deg > 0 and "heading_deg" in df.columns:
        turn = heading_gate(turn, pd.to_numeric(df["heading_deg"], errors="coerce").to_numpy(float), p.turn_min_heading_deg, p.heading_margin_n)
    label = np.full(len(df), "Normal", dtype=object)
    label[accel] = "Harsh Acceleration"; label[brake] = "Harsh Braking"; label[turn] = "Harsh Turning"
    return pd.Series(label, index=df.index, name="label")


def events(label: pd.Series | np.ndarray):
    """[(cls, start, end_inclusive)] splitting runs where the class changes."""
    lab = np.asarray(label, dtype=object); out = []
    for a, b in runs(lab != "Normal"):
        i = a
        while i <= b:
            k = i
            while k + 1 <= b and lab[k + 1] == lab[i]:
                k += 1
            out.append((lab[i], i, k)); i = k + 1
    return out


def count_events(label) -> dict:
    out = {"Harsh Acceleration": 0, "Harsh Braking": 0, "Harsh Turning": 0}
    for c, _, _ in events(label):
        out[c] += 1
    return out
