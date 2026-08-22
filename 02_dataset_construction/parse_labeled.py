"""
Parse the VBOX LABELED CSV (D2_S{n}_LABELED.csv).

Column layout (confirmed across all 4 sessions):
  0  UTC time          HHMMSS.ss  (GPS UTC, no date)
  1  Satellites
  2  Speed (km/h)
  3  Heading (Degrees)
  4  Latitude          "24°44.742936 N" DMS format
  5  Longitude         "46°40.228542 E" DMS format
  6  BrakeTrigger
  7  Dgps
  8  Wav file
  9  Dual Antenna Status
  10 Height (m)
  11 Vertical velocity (km/h)
  12 Solution type
  13 Avi sync time (s)
  14 Avi file index
  15 Elapsed time (s)      ← primary relative time axis
  16 Distance (m)
  17 Lateral acceleration (g)
  18 Longitudinal acceleration (g)
  19 Relative height (m)
  20 Gradient (%)
  21 Radius of turn (m)
  22 sampleperiod (s)      nominal = 0.04 s (25 Hz)
  23 Engine_Speed (rpm)
  24 Accelerator_Pedal_Position (%)
  25 Brake_Pedal_Position (%)
  26 Indicated_Vehicle_Speed_kph (km/h)
  27 Wheel_Speed_FR_kph (km/h)
  28 Wheel_Speed_FL_kph (km/h)
  29 Label              Normal | Harsh Braking | Harsh Acceleration | Harsh Turning

Synchronization anchor:
  unix_time = utc_epoch_of_session_start + elapsed_s
  where utc_epoch_of_session_start is derived from:
    (a) the EmotiBit filename (gives the session date), OR
    (b) an explicitly supplied date string.

GPS coordinates → (lat, lon) in decimal degrees.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE_RATE = 25.0   # Hz
UTC_OFFSET_HOURS = 3  # Saudi Arabia (Riyadh) is UTC+3

# Columns we keep in the output
_KEEP = {
    "elapsed_s": "elapsed_s",
    "UTC time": "utc_tod_s",          # time-of-day seconds
    "Speed (km/h)": "gps_speed_kmh",
    "Heading (Degrees)": "heading_deg",
    "Height (m)": "height_m",
    "Lateral acceleration (g)": "lat_acc_g",
    "Longitudinal acceleration (g)": "lon_acc_g",
    "Gradient (%)": "gradient_pct",
    "Engine_Speed (rpm)": "engine_rpm",
    "Accelerator_Pedal_Position (%)": "throttle_pct",
    "Brake_Pedal_Position (%)": "brake_pct",
    "Indicated_Vehicle_Speed_kph (km/h)": "speed_kph",
    "Wheel_Speed_FR_kph (km/h)": "wheel_fr_kph",
    "Wheel_Speed_FL_kph (km/h)": "wheel_fl_kph",
    "Label": "label",
}


def _parse_utc_time(t_str: str) -> float:
    """Convert 'HHMMSS.ss' string to seconds-of-day (float)."""
    t = str(t_str).strip()
    # Remove any decimal point ambiguity
    if "." in t:
        int_part, frac_part = t.split(".", 1)
    else:
        int_part, frac_part = t, "0"

    int_part = int_part.zfill(6)
    hh = int(int_part[0:2])
    mm = int(int_part[2:4])
    ss = int(int_part[4:6])
    frac = float("0." + frac_part)
    return hh * 3600 + mm * 60 + ss + frac


def _parse_dms(coord_str: str) -> float:
    """
    Convert DMS string like '24°44.742936 N' to decimal degrees.
    Also handles plain decimal floats.
    """
    s = str(coord_str).strip()
    m = re.match(r"(\d+)°([\d.]+)\s*([NSEW])", s)
    if m:
        deg = int(m.group(1))
        mins = float(m.group(2))
        hem = m.group(3)
        dec = deg + mins / 60.0
        if hem in ("S", "W"):
            dec = -dec
        return dec
    try:
        return float(s)
    except ValueError:
        return np.nan


def _date_from_emotibit_filename(session_dir: Path) -> str | None:
    """
    Find an EmotiBit file in the session directory and extract the date
    from its filename (format: YYYY-MM-DD_HH-MM-SS-...).
    Returns 'YYYY-MM-DD' string or None.
    """
    emb_dir = session_dir / "EmotiBit"
    if not emb_dir.exists():
        emb_dir = session_dir / "MotiBit"
    if emb_dir.exists():
        for f in emb_dir.iterdir():
            m = re.match(r"(\d{4}-\d{2}-\d{2})_", f.name)
            if m:
                return m.group(1)
    return None


def utc_hhmmss_to_unix(utc_tod_s: float, date_str: str) -> float:
    """
    Convert time-of-day seconds (UTC) + a date string 'YYYY-MM-DD' to Unix epoch (float).
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return dt.timestamp() + utc_tod_s


def load_labeled_csv(
    csv_path: str | Path,
    session_dir: str | Path | None = None,
    date_str: str | None = None,
) -> pd.DataFrame:
    """
    Load a D2_S{n}_LABELED.csv file.

    Parameters
    ----------
    csv_path:    path to the LABELED CSV
    session_dir: session folder (used to auto-detect date from EmotiBit filename)
    date_str:    explicit 'YYYY-MM-DD' override; takes precedence over session_dir

    Returns a DataFrame with:
        unix_time   – absolute Unix epoch seconds (float)
        elapsed_s   – elapsed seconds from session start (0-based)
        utc_tod_s   – UTC time-of-day in seconds
        gps_speed_kmh, heading_deg, height_m
        lat_acc_g, lon_acc_g, gradient_pct
        engine_rpm, throttle_pct, brake_pct
        speed_kph, wheel_fr_kph, wheel_fl_kph
        label       – one of {Normal, Harsh Braking, Harsh Acceleration, Harsh Turning}
        lat, lon    – decimal degrees (from GPS)
    """
    csv_path = Path(csv_path)

    # Auto-detect date
    if date_str is None and session_dir is not None:
        date_str = _date_from_emotibit_filename(Path(session_dir))
    if date_str is None:
        # Fall back: parse date from CSV filename pattern D2_S1_LABELED etc. — use parent dir name
        date_str = _date_from_emotibit_filename(csv_path.parent)

    raw = pd.read_csv(csv_path, dtype=str, low_memory=False)
    raw.columns = [c.strip() for c in raw.columns]

    # Parse UTC time-of-day
    raw["utc_tod_s"] = raw["UTC time"].apply(_parse_utc_time)

    # Elapsed time
    raw["elapsed_s"] = pd.to_numeric(raw["Elapsed time (s)"], errors="coerce")

    # Coordinates
    raw["lat"] = raw["Latitude"].apply(_parse_dms)
    raw["lon"] = raw["Longitude"].apply(_parse_dms)

    # Rename and coerce numeric columns
    rename_map = {k: v for k, v in _KEEP.items() if k in raw.columns and k not in ("UTC time", "Elapsed time (s)")}
    raw = raw.rename(columns=rename_map)

    numeric_cols = [
        "gps_speed_kmh", "heading_deg", "height_m",
        "lat_acc_g", "lon_acc_g", "gradient_pct",
        "engine_rpm", "throttle_pct", "brake_pct",
        "speed_kph", "wheel_fr_kph", "wheel_fl_kph",
    ]
    for col in numeric_cols:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    # Build unix_time
    if date_str:
        t0_unix = utc_hhmmss_to_unix(raw["utc_tod_s"].iloc[0], date_str)
        raw["unix_time"] = t0_unix + raw["elapsed_s"]
    else:
        # No date available: use elapsed_s only (relative time)
        raw["unix_time"] = raw["elapsed_s"].copy()
        print(f"  Warning: no date found for {csv_path.name}; unix_time is relative only.")

    # Select output columns
    keep_cols = ["unix_time", "utc_tod_s", "elapsed_s", "lat", "lon"]
    for col in numeric_cols + ["label"]:
        if col in raw.columns:
            keep_cols.append(col)

    out = raw[[c for c in keep_cols if c in raw.columns]].copy()
    out = out.sort_values("unix_time").reset_index(drop=True)

    # Drop rows where elapsed_s is NaN (bad GPS frames)
    out = out.dropna(subset=["elapsed_s"]).reset_index(drop=True)

    return out


def label_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary table of label counts and durations."""
    counts = df["label"].value_counts()
    # Duration assumes 25 Hz rows
    duration_s = counts / SAMPLE_RATE
    summary = pd.DataFrame({
        "count": counts,
        "duration_s": duration_s,
        "pct": (counts / len(df) * 100).round(1),
    })
    return summary


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    if csv_path is None:
        print("Usage: python parse_labeled.py <path_to_labeled.csv>")
        sys.exit(1)
    df = load_labeled_csv(csv_path, session_dir=Path(csv_path).parent)
    print(df.head(5).to_string())
    print(f"\nShape: {df.shape}")
    print(f"Duration: {df['elapsed_s'].max():.1f} s")
    print(f"unix_time range: {df['unix_time'].min():.3f} – {df['unix_time'].max():.3f}")
    print("\nLabel summary:")
    print(label_summary(df))
