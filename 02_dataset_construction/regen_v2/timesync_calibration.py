"""
Robust EmotiBit device-clock calibration from timesyncs.csv.

Factored out of DriveSense/05_technical_validation/audit_emotibit_timesync_full.py
so the regeneration pipeline (emotibit_sync_v2.py) and the audit script share
the exact same, already-validated fitting logic rather than a second
reimplementation of the same delicate float-precision-sensitive math.

Returns a (device_ms -> unix_time) calibration built from ALL individual
NTP-style sync pings logged throughout a session (not just the 2-point
timeSyncMap.csv). Two models:

  model="anchor_only" (release default since 2026-09-09):
      unix_time = device_s + median(offset). Slope fixed at exactly 1. The
      ping logs show the device clock itself is stable to a few ppm within a
      session (median true drift ~4 ppm, i.e. ~15 ms/h), but a minority of
      sessions contain one or more discrete offset STEPS of 0.15-1.0 s (most
      plausibly host-clock adjustments on the logging laptop). A fitted slope
      cannot represent a step; it tilts through it and reports a spurious
      "drift" of hundreds of ppm (e.g. D4_S1 +552 ppm from a +326 ms step in
      an 8-min log), smearing the step across the whole session. The session-
      median offset is insensitive to such steps; the largest sustained step
      is reported as max_step_s and is the honest per-session bound on
      residual cross-device alignment error.

  model="linear":
      unix_time = a + b * device_ms, robust (MAD-based) iterative fit. This
      was the 2026-09-08 (regen_v2) model; retained for the audit script's
      diagnostics and for reproducing that intermediate release.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

UTC_OFFSET_HOURS = 3
TS_SENT_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})-(\d+)")
STEP_MIN_PINGS_EACH_SIDE = 5


def ts_sent_to_unix(s: str) -> float | None:
    m = TS_SENT_RE.match(str(s).strip())
    if not m:
        return None
    yr, mo, dy, hh, mi, ss, us = (int(x) for x in m.groups())
    us = int(str(us).ljust(6, "0")[:6])
    dt = datetime(yr, mo, dy, hh, mi, ss, us, tzinfo=timezone(timedelta(hours=UTC_OFFSET_HOURS)))
    return dt.timestamp()


class Calibration:
    """device_ms -> unix_time mapping, fit on offsets from the first ping for numerical stability."""

    def __init__(self, x0_ms: float, y0_unix: float, slope: float, intercept: float,
                 n_pings_used: int, n_outliers: int, span_s: float, max_resid_ms: float,
                 model: str, max_step_s: float, step_at_s: float, median_offset_s: float,
                 linear_slope_ppm: float):
        self.x0_ms = x0_ms
        self.y0_unix = y0_unix
        self.slope = slope
        self.intercept = intercept
        self.n_pings_used = n_pings_used
        self.n_outliers = n_outliers
        self.span_s = span_s
        self.max_resid_ms = max_resid_ms
        self.model = model
        self.max_step_s = max_step_s          # largest sustained offset step in the log (s)
        self.step_at_s = step_at_s            # device time (s since first ping) of that step
        self.median_offset_s = median_offset_s  # median(ref - device), relative to first ping
        self.linear_slope_ppm = linear_slope_ppm  # what a linear fit WOULD report (diagnostic)
        self.ppm_drift = (slope - 1.0) * 1e6  # 0.0 by construction for anchor_only

    def predict_unix(self, device_ms: float) -> float:
        return self.y0_unix + self.intercept + self.slope * (device_ms - self.x0_ms) / 1000.0

    def as_two_point_sync_map(self, span_ms: float = 1.0e7) -> dict:
        """
        Encodes this calibration as a synthetic 2-point (TE0,TE1,TL0,TL1) map
        compatible with emotibit_sync.py's existing, unmodified
        _emotibit_ts_to_unix() 2-point linear interpolation -- both points lie
        exactly on this calibration's line, so passing this map reproduces it
        exactly for any device_ms without modifying that function.
        """
        te0, te1 = 0.0, span_ms
        return {"TE0": te0, "TE1": te1,
                "TL0": self.predict_unix(te0), "TL1": self.predict_unix(te1)}


def _largest_sustained_step(x: np.ndarray, off: np.ndarray) -> tuple[float, float]:
    """Largest |median(after k) - median(before k)| over split points with >= 5 pings each side."""
    n = len(x)
    if n < 2 * STEP_MIN_PINGS_EACH_SIDE:
        return 0.0, float("nan")
    best, best_k = 0.0, None
    for k in range(STEP_MIN_PINGS_EACH_SIDE, n - STEP_MIN_PINGS_EACH_SIDE + 1):
        st = float(np.median(off[k:]) - np.median(off[:k]))
        if abs(st) > abs(best):
            best, best_k = st, k
    return best, (float(x[best_k]) if best_k is not None else float("nan"))


def fit_calibration(timesyncs_csv: Path, min_pings: int = 5, model: str = "anchor_only") -> Calibration | None:
    if model not in ("anchor_only", "linear"):
        raise ValueError(f"unknown model {model!r}")
    ts = pd.read_csv(timesyncs_csv, header=0,
                      names=["RD", "TS_received", "TS_sent", "AK", "RoundTrip", "_blank"],
                      dtype={"TS_sent": str})
    ts = ts.dropna(subset=["TS_received", "TS_sent", "RoundTrip"])
    ts["ref_unix"] = ts["TS_sent"].apply(ts_sent_to_unix) + ts["RoundTrip"].astype(float) / 2000.0
    ts = ts.dropna(subset=["ref_unix"])
    if len(ts) < min_pings:
        return None

    median_rt = ts["RoundTrip"].median()
    clean = ts[ts["RoundTrip"] <= max(3 * median_rt, median_rt + 20)]

    x_ms = clean["TS_received"].to_numpy(dtype=float)
    y_unix = clean["ref_unix"].to_numpy(dtype=float)
    x0, y0 = x_ms[0], y_unix[0]
    x = (x_ms - x0) / 1000.0          # seconds, device clock, from first ping
    y = y_unix - y0                    # seconds, reference clock, from first ping
    off = y - x                        # per-ping offset (s); constant if the clocks agree in rate

    # Robust linear fit (diagnostic for both models; the applied mapping for model="linear")
    keep = np.ones(len(x), dtype=bool)
    lin_slope, lin_intercept = np.polyfit(x, y, 1)
    for _ in range(3):
        lin_slope, lin_intercept = np.polyfit(x[keep], y[keep], 1)
        resid_all = y - (lin_slope * x + lin_intercept)
        mad = np.median(np.abs(resid_all[keep] - np.median(resid_all[keep]))) or 1e-6
        new_keep = np.abs(resid_all - np.median(resid_all[keep])) <= 5 * 1.4826 * mad
        if new_keep.sum() == keep.sum():
            keep = new_keep
            break
        keep = new_keep

    max_step_s, step_at_s = _largest_sustained_step(x, off)
    median_offset_s = float(np.median(off))

    if model == "anchor_only":
        slope, intercept = 1.0, median_offset_s
        resid = off - median_offset_s
        n_used, n_out = len(off), 0
        max_resid_ms = float(np.max(np.abs(resid)) * 1000)
    else:
        slope, intercept = float(lin_slope), float(lin_intercept)
        resid = y - (slope * x + intercept)
        n_used, n_out = int(keep.sum()), int((~keep).sum())
        max_resid_ms = float(np.max(np.abs(resid[keep])) * 1000) if keep.any() else float(np.max(np.abs(resid)) * 1000)

    return Calibration(
        x0_ms=x0, y0_unix=y0, slope=slope, intercept=intercept,
        n_pings_used=n_used, n_outliers=n_out,
        span_s=float(x.max() - x.min()), max_resid_ms=max_resid_ms,
        model=model, max_step_s=max_step_s, step_at_s=step_at_s,
        median_offset_s=median_offset_s, linear_slope_ppm=float((lin_slope - 1.0) * 1e6),
    )
