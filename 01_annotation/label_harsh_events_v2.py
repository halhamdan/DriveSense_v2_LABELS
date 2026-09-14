"""
Harsh-event labelling, version 2 (final form 2026-09-13): sustained exceedance
of a smoothed, GNSS-derived acceleration, robust to single-fix GNSS artefacts.

Why a version 2. The released acceleration channels are Racelogic's LatAcc /
LongAcc: the one-sample backward difference of the 25 Hz Doppler speed
(longitudinal) and speed x heading-rate (lateral), unsmoothed. Version 1
applied fixed thresholds to rolling extrema of that raw derivative over a
trailing 1 s window, so a single 40 ms spike of Doppler noise kept a condition
true for ~25 samples and passed the 0.4 s minimum duration. On the released
data 78-87% of v1 events contained at most one consecutive sample above their
threshold (05_technical_validation/event_exceedance_structure).

Rule (all on the released 25 Hz fused.csv columns; ALL windows are explicit
integer sample counts, see CONFIG_V2):
  1. running median, MEDIAN_N = 5 samples (0.20 s), centred, applied to the
     SIGNED lon_acc_g and lat_acc_g -- removes any excursion shorter than
     3 samples (the single-fix and two-fix GNSS jumps) without moving the
     edges of a sustained excursion;
  2. centred moving average, SMOOTH_N = 13 samples (k = -6..+6, 0.52 s), of
     the SIGNED channels; the lateral test uses |mean|, not mean|.|, so that
     alternating-sign heading noise averages to ~0 instead of rectifying to
     its amplitude;
  3. class conditions
       Harsh Braking      : lon_s <= -0.35 g AND gps_speed > 15 km/h AND the
                            CAN-speed-derived deceleration (1 s = 25-sample
                            rolling min of d(speed_kph)/dt) <= -0.3 m/s^2
       Harsh Acceleration : lon_s >=  0.20 g AND gps_speed > 15 km/h AND
                            throttle (13-sample centred max) > 25 %
       Harsh Turning      : |lat_s| >= 0.45 g AND gps_speed > 30 km/h AND, per
                            run, the net GNSS heading change from 3 samples
                            before to 3 samples after the run is >= 5 degrees
                            (heading is the integral of the noisy heading
                            rate: sign-alternating noise leaves it unchanged,
                            a genuine 0.45 g turn at 30 km/h moves it ~16
                            degrees in 0.52 s) -- the turning analogue of the
                            CAN-speed corroboration used for braking
  4. per class: close gaps <= CLOSE_N = 5 samples (0.20 s), keep runs of
     >= MIN_N = 13 samples (0.52 s). No dilation -- an event is the exceedance
     interval itself. Priority Turning > Braking > Acceleration where masks
     overlap; everything else Normal.

Guarantees (05_technical_validation/audit_label_robustness_v2.py, synthetic
cases): a single sample of any magnitude, a two-sample burst of any magnitude
and a three-sample burst up to 2.6 g produce no event; alternating +/-1.0 g
produces no turning event; a sustained 0.55 g excursion of 1 s does.
Thresholds were calibrated on the three deliberate-manoeuvre drives in the
study vehicle (10/10 deliberate brakes, 11/11 deliberate turns; deliberate
full-throttle accelerations peak at 0.21-0.29 g, the vehicle's own ceiling).

The first v2 draft of 2026-09-13 (mean of |lat|, and a 12-sample window from
round(0.5 * 25)) was superseded the same day after the robustness audit: it
labelled a single 6.06 g spike and alternating +/-0.6 g noise as turning.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_rule_variants import RuleParams, label_df, count_events as _count_events, events as _events  # noqa: E402

PARAMS_V2 = RuleParams(smooth_n=13, min_n=13, close_n=5, lateral="mean_then_abs", median_n=5,
                       brake_g=0.35, accel_g=0.20, turn_g=0.45, min_speed_kmh=15.0, min_speed_turn_kmh=30.0,
                       throttle_min_pct=25.0, brake_can_ms2=-0.3, can_window_n=25,
                       turn_min_heading_deg=5.0, heading_margin_n=3)

CONFIG_V2 = {
    "FS_HZ": 25.0,
    "MEDIAN_N": PARAMS_V2.median_n,        # samples, centred running median on signed channels
    "SMOOTH_N": PARAMS_V2.smooth_n,        # samples, centred moving average (k = -6..+6)
    "MIN_N": PARAMS_V2.min_n,              # samples, minimum run length
    "CLOSE_N": PARAMS_V2.close_n,          # samples, gaps closed inside a run
    "CAN_WINDOW_N": PARAMS_V2.can_window_n,
    "LATERAL": "abs(mean(signed lat_acc_g))",
    "TURN_MIN_HEADING_DEG": PARAMS_V2.turn_min_heading_deg, "HEADING_MARGIN_N": PARAMS_V2.heading_margin_n,
    "BRAKE_G": PARAMS_V2.brake_g, "ACCEL_G": PARAMS_V2.accel_g, "TURN_G": PARAMS_V2.turn_g,
    "MIN_SPEED_KMH": PARAMS_V2.min_speed_kmh, "MIN_SPEED_TURN_KMH": PARAMS_V2.min_speed_turn_kmh,
    "THROTTLE_MIN_PCT": PARAMS_V2.throttle_min_pct, "BRAKE_A_SPEED_MIN_MS2": PARAMS_V2.brake_can_ms2,
}


def label_fused_v2(df: pd.DataFrame, params: RuleParams = PARAMS_V2) -> pd.Series:
    """Returns the v2 label series for a released fused.csv DataFrame."""
    return label_df(df, params)


def count_events(label: pd.Series) -> dict:
    return _count_events(label)


def events(label):
    return _events(label)
