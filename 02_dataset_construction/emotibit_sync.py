"""
EmotiBit → absolute Unix-epoch time synchronization.

Two paths depending on what files are present in the EmotiBit folder:

PATH A  (Session 1 — full export from EmotiBit Oscilloscope):
  Per-channel CSV files exist: *_HR.csv, *_AX.csv, etc.
  Each has a `LocalTimestamp` column (Unix epoch, float).
  We use those directly — no conversion needed.
  The `timeSyncMap.csv` is used for validation only.

PATH B  (Sessions 2-4 — raw CSV only):
  Only the main raw CSV (*_<timestamp>.csv) is present.
  Steps:
    1. Extract session datetime from the filename: YYYY-MM-DD_HH-MM-SS
       (this is local wall-clock time, UTC+3 for Riyadh)
    2. Convert to UTC Unix epoch: t_start_unix = t_local - 3*3600
    3. EmotiBit's first recorded EmotiBitTimestamp → maps to t_start_unix
    4. All other timestamps: unix_t = t_start_unix + (EmotiBitTs - EmotiBitTs_0) / 1000.0

The returned object is a dict  signal_name → DataFrame with columns:
    unix_time  (float, seconds)
    value      (float)   — for scalar signals
  or
    unix_time, x, y, z   — for 3-axis signals (ACC, GYRO, MAG)

Supported signals:
  PPG_IR, PPG_RED, PPG_GREEN
  EDA, EDA_LEVEL, SCR_FREQ
  TEMP_CONTACT, TEMP_THERMOPILE
  HR, IBI
  ACC, GYRO, MAG   (3-axis)
"""

from __future__ import annotations

import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

UTC_OFFSET_HOURS = 3   # Saudi Arabia (Riyadh) UTC+3

# Nominal sample rates per type tag (Hz)
SAMPLE_RATES = {
    "PI": 25.0, "PR": 25.0, "PG": 25.0,
    "AX": 25.0, "AY": 25.0, "AZ": 25.0,
    "GX": 25.0, "GY": 25.0, "GZ": 25.0,
    "MX": 25.0, "MY": 25.0, "MZ": 25.0,
    "EA": 15.0, "EL": 15.0, "SF": 3.0,
    "T1": 7.5, "TH": 7.5,
    "HR": 1.0, "BI": 1.0,
}
SKIP_TAGS = {"RB", "EM", "DC", "DO", "B%", "BV", "D+", "D-", "AK", "RD", "TL"}

TAG_TO_SIGNAL = {
    "PI": "PPG_IR", "PR": "PPG_RED", "PG": "PPG_GREEN",
    "EA": "EDA", "EL": "EDA_LEVEL", "SF": "SCR_FREQ",
    "T1": "TEMP_CONTACT", "TH": "TEMP_THERMOPILE",
    "HR": "HR", "BI": "IBI",
    "AX": "ACC_x", "AY": "ACC_y", "AZ": "ACC_z",
    "GX": "GYRO_x", "GY": "GYRO_y", "GZ": "GYRO_z",
    "MX": "MAG_x", "MY": "MAG_y", "MZ": "MAG_z",
}

PER_CHANNEL_TAG_COL = {
    "PI": "PI", "PR": "PR", "PG": "PG",
    "EA": "EA", "EL": "EL", "SF": "SF",
    "T1": "T1", "TH": "TH",
    "HR": "HR", "BI": "BI",
    "AX": "AX", "AY": "AY", "AZ": "AZ",
    "GX": "GX", "GY": "GY", "GZ": "GZ",
    "MX": "MX", "MY": "MY", "MZ": "MZ",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _filename_to_unix(filepath: Path) -> float | None:
    """
    Extract the session start Unix epoch from an EmotiBit filename.
    Format: YYYY-MM-DD_HH-MM-SS[-microseconds][_suffix]
    Assumes local time = UTC+3.
    """
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})", filepath.name)
    if not m:
        return None
    yr, mo, dy, hh, mi, ss = (int(x) for x in m.groups())
    local_dt = datetime(yr, mo, dy, hh, mi, ss, tzinfo=timezone(timedelta(hours=UTC_OFFSET_HOURS)))
    return local_dt.timestamp()


def _read_timesync_map(emotibit_dir: Path) -> dict | None:
    """
    Read the timeSyncMap.csv to get EmotiBit→local-time mapping coefficients.

    Returns dict with keys: TE0, TE1, TL0, TL1, or None if not found.
    """
    candidates = list(emotibit_dir.glob("*_timeSyncMap.csv"))
    if not candidates:
        return None
    try:
        df = pd.read_csv(candidates[0])
        row = df.iloc[0]
        return {
            "TE0": float(row["TE0"]),
            "TE1": float(row["TE1"]),
            "TL0": float(row["TL0"]),
            "TL1": float(row["TL1"]),
        }
    except Exception:
        return None


def _emotibit_ts_to_unix(
    ts_ms: np.ndarray,
    ts_start: float,
    t_start_unix: float,
    sync_map: dict | None = None,
) -> np.ndarray:
    """
    Convert EmotiBit millisecond timestamps to Unix epoch (seconds).

    If sync_map is provided (timeSyncMap.csv data), use linear interpolation
    between (TE0, TL0) and (TE1, TL1).  Otherwise use the start-point offset.
    """
    if sync_map is not None:
        te0, te1 = sync_map["TE0"], sync_map["TE1"]
        tl0, tl1 = sync_map["TL0"], sync_map["TL1"]
        slope = (tl1 - tl0) / (te1 - te0)
        return tl0 + (ts_ms - te0) * slope
    else:
        return t_start_unix + (ts_ms - ts_start) / 1000.0


# ─── Path A: per-channel CSV files (Session 1) ─────────────────────────────

def _load_per_channel_csvs(emotibit_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Load per-channel exported CSVs (format: *_<TAG>.csv).
    These already contain LocalTimestamp (Unix epoch).
    Returns dict tag → DataFrame(unix_time, value).
    """
    result: dict[str, pd.DataFrame] = {}
    for tag, sig_name in TAG_TO_SIGNAL.items():
        pattern = f"*_{tag}.csv"
        matches = list(emotibit_dir.glob(pattern))
        if not matches:
            continue
        try:
            df = pd.read_csv(matches[0], low_memory=False)
            if "LocalTimestamp" not in df.columns:
                continue
            val_col = PER_CHANNEL_TAG_COL.get(tag)
            if val_col not in df.columns:
                # Try last column
                val_col = df.columns[-1]
            out = pd.DataFrame({
                "unix_time": pd.to_numeric(df["LocalTimestamp"], errors="coerce"),
                "value": pd.to_numeric(df[val_col], errors="coerce"),
            }).dropna().sort_values("unix_time").reset_index(drop=True)
            result[sig_name] = out
        except Exception as e:
            print(f"    Warning: could not load {matches[0].name}: {e}")

    return result


# ─── Path B: raw main CSV (all sessions, required for 2-4) ─────────────────

def _load_raw_csv(
    csv_path: Path,
    t_start_unix: float,
    sync_map: dict | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Parse an EmotiBit raw CSV and return time-indexed signals.

    csv_path:     path to the main *_<timestamp>.csv
    t_start_unix: Unix epoch of the recording start (from filename or timeSyncMap)
    sync_map:     optional timeSyncMap params for accurate ts conversion
    """
    raw_channels: dict[str, list[tuple[float, float]]] = defaultdict(list)
    ts_start: float | None = None

    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 7:
                continue
            try:
                ts_ms = float(parts[0])
                count = int(parts[2])
                tag = parts[3].strip()
            except ValueError:
                continue

            if tag in SKIP_TAGS or tag not in SAMPLE_RATES:
                continue

            if ts_start is None:
                ts_start = ts_ms

            values_raw = parts[6: 6 + count]
            if len(values_raw) < count:
                continue
            try:
                values = [float(v) for v in values_raw]
            except ValueError:
                continue

            fs = SAMPLE_RATES[tag]
            dt_ms = 1000.0 / fs
            for i, v in enumerate(values):
                raw_channels[tag].append((ts_ms + i * dt_ms, v))

    if ts_start is None:
        return {}

    # Convert EmotiBit ms → Unix epoch
    result: dict[str, pd.DataFrame] = {}
    for tag, rows in raw_channels.items():
        sig_name = TAG_TO_SIGNAL.get(tag)
        if sig_name is None:
            continue
        ts_arr = np.array([r[0] for r in rows])
        v_arr = np.array([r[1] for r in rows])
        unix_arr = _emotibit_ts_to_unix(ts_arr, ts_start, t_start_unix, sync_map)
        df = pd.DataFrame({"unix_time": unix_arr, "value": v_arr})
        df = df.sort_values("unix_time").reset_index(drop=True)
        result[sig_name] = df

    return result


# ─── 3-axis merger ──────────────────────────────────────────────────────────

def _merge_xyz_signals(
    scalars: dict[str, pd.DataFrame],
    x_name: str, y_name: str, z_name: str,
    out_name: str,
) -> dict[str, pd.DataFrame]:
    """Merge three scalar signals into one 3-axis DataFrame."""
    if not all(n in scalars for n in (x_name, y_name, z_name)):
        return scalars
    fs = {"ACC_x": 25.0, "GYRO_x": 25.0, "MAG_x": 25.0}.get(x_name, 25.0)
    tol = 0.5 / fs  # seconds

    dx = scalars[x_name].rename(columns={"value": "x"})
    dy = scalars[y_name].rename(columns={"value": "y"})
    dz = scalars[z_name].rename(columns={"value": "z"})

    merged = pd.merge_asof(
        dx.sort_values("unix_time"),
        dy.sort_values("unix_time"),
        on="unix_time", tolerance=tol, direction="nearest"
    )
    merged = pd.merge_asof(
        merged,
        dz.sort_values("unix_time"),
        on="unix_time", tolerance=tol, direction="nearest"
    )
    merged = merged.dropna(subset=["x", "y", "z"]).reset_index(drop=True)

    del scalars[x_name], scalars[y_name], scalars[z_name]
    scalars[out_name] = merged
    return scalars


# ─── Public API ─────────────────────────────────────────────────────────────

def load_emotibit(session_dir: str | Path) -> tuple[dict[str, pd.DataFrame], dict]:
    """
    Load and time-synchronize all EmotiBit signals for a session.

    Returns:
        signals : dict  signal_name → DataFrame(unix_time, value)
                        or (unix_time, x, y, z) for ACC/GYRO/MAG
        meta    : dict  with sync info (method, t_start_unix, offset_to_vbox, ...)
    """
    session_dir = Path(session_dir)
    emotibit_dir = session_dir / "EmotiBit"
    if not emotibit_dir.exists():
        emotibit_dir = session_dir / "MotiBit"

    meta: dict = {"session_dir": str(session_dir)}

    # Check which path to use
    per_channel_files = list(emotibit_dir.glob("*_HR.csv"))
    sync_map = _read_timesync_map(emotibit_dir)

    # Find the main CSV
    main_csvs = sorted(
        [f for f in emotibit_dir.glob("*.csv")
         if not any(f.name.endswith(f"_{tag}.csv") for tag in list(TAG_TO_SIGNAL) + ["timeSyncMap", "timesyncs", "AK", "RD", "TL"])
         and "_info" not in f.name],
        key=lambda f: f.stat().st_size, reverse=True
    )
    main_csv = main_csvs[0] if main_csvs else None

    if per_channel_files and emotibit_dir.glob("*_timeSyncMap.csv"):
        # Path A: per-channel CSVs with LocalTimestamp
        meta["sync_method"] = "per_channel_LocalTimestamp"
        signals = _load_per_channel_csvs(emotibit_dir)
        if sync_map and signals:
            # Validation: compare first HR timestamp against timeSyncMap
            hr = signals.get("HR")
            if hr is not None and not hr.empty:
                meta["t_start_unix"] = float(hr["unix_time"].iloc[0])
        print(f"  EmotiBit sync: PATH A (per-channel CSVs + LocalTimestamp)")
    else:
        # Path B: raw CSV + filename-based or timeSyncMap sync
        if main_csv is None:
            raise FileNotFoundError(f"No EmotiBit CSV found in {emotibit_dir}")

        t_start_unix = _filename_to_unix(main_csv)
        if t_start_unix is None:
            raise ValueError(f"Cannot parse datetime from filename: {main_csv.name}")

        meta["sync_method"] = "filename_UTC+3" if sync_map is None else "timeSyncMap_linear"
        meta["t_start_unix"] = t_start_unix
        meta["t_start_local"] = main_csv.name[:19]

        signals = _load_raw_csv(main_csv, t_start_unix, sync_map)
        print(f"  EmotiBit sync: PATH B ({meta['sync_method']}) | "
              f"start={meta['t_start_local']} -> unix={t_start_unix:.3f}")

    # Merge x/y/z axes
    for prefix, (x, y, z) in {
        "ACC": ("ACC_x", "ACC_y", "ACC_z"),
        "GYRO": ("GYRO_x", "GYRO_y", "GYRO_z"),
        "MAG": ("MAG_x", "MAG_y", "MAG_z"),
    }.items():
        signals = _merge_xyz_signals(signals, x, y, z, prefix)

    # Summary
    meta["signals"] = {name: len(df) for name, df in signals.items()}
    if signals:
        all_starts = [df["unix_time"].iloc[0] for df in signals.values()]
        all_ends = [df["unix_time"].iloc[-1] for df in signals.values()]
        meta["emotibit_start_unix"] = float(np.min(all_starts))
        meta["emotibit_end_unix"] = float(np.max(all_ends))
        meta["emotibit_duration_s"] = float(np.max(all_ends) - np.min(all_starts))

    return signals, meta


if __name__ == "__main__":
    import sys, json
    sess_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if sess_dir is None:
        print("Usage: python emotibit_sync.py <session_dir>")
        sys.exit(1)
    sigs, meta = load_emotibit(sess_dir)
    print(json.dumps({k: v for k, v in meta.items() if k != "signals"}, indent=2))
    print("\nSignals:")
    for name, df in sigs.items():
        print(f"  {name:20s}: {len(df):7d} samples | "
              f"t=[{df['unix_time'].iloc[0]:.3f}, {df['unix_time'].iloc[-1]:.3f}]")
