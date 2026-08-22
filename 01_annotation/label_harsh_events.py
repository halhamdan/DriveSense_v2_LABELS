"""
Harsh Driving Labelling Pipeline (Physics Thresholds + Temporal Cleaning)
=======================================================================

Goal
----
Label each sample into exactly one of:
    - Normal
    - Harsh Acceleration
    - Harsh Braking
    - Harsh Turning

Key Improvements (no logic gaps / no silent failures)
-----------------------------------------------------
1) Threshold policy options:
   - FIXED physics thresholds for ground-truth consistency (recommended).
   - Adaptive thresholds available, but guarded to avoid "normalising" aggressive drivers.

2) Turning quality fixes:
   - Separate turning speed gate (MIN_SPEED_TURN_KMH) to ignore parking manoeuvres.
   - Event expansion (hysteresis via dilation) to capture ramp-up/ramp-down around peaks.

3) Noise control:
   - Run-length filtering removes short blips (drop_short_runs) BEFORE and AFTER expansion.

4) Safety + determinism:
   - Always produces a Label column (even if some channels missing).
   - Braking still requires speed-drop confirmation (hard rule).
   - Priority: Turning > Braking > Acceleration.

Requirements
------------
pip install pandas numpy scipy
"""

import argparse
import os
from typing import Optional, List, Tuple, Dict
import numpy as np
import pandas as pd
from scipy.ndimage import binary_closing, binary_dilation


# ============================================================
# 1) PATHS
# ============================================================
# Both directories are supplied at runtime (CLI arg, falling back to an
# environment variable) rather than hardcoded, since this script is released
# alongside the dataset for reproducibility on any machine.
INPUT_DIR = os.environ.get("DATASET_RAW_DIR", "")
OUTPUT_DIR = os.environ.get("DATASET_LABELED_DIR", "")


# ============================================================
# 2) CONFIG
# ============================================================
CONFIG: Dict[str, object] = {
    # -------------------------------
    # Threshold policy
    # -------------------------------
    # Recommended for ground-truth creation
    "USE_ADAPTIVE_THRESHOLDS": False,

    # Physics thresholds (g)
    "BRAKE_G": -0.35,
    "ACCEL_G": 0.38,
    "TURN_G": 0.55,

    # Intent channels
    "BRAKE_PEDAL_MIN": 5.0,
    "ACCEL_PEDAL_MIN": 25.0,
    "USE_BRAKE_TRIGGER": True,

    # -------------------------------
    # Motion / smoothing
    # -------------------------------
    "MIN_SPEED_KMH": 15.0,        # used for accel + brake
    "MIN_SPEED_TURN_KMH": 30.0,   # used for turning (ignores low-speed steering)
    "WINDOW_SECONDS": 1.0,
    "MIN_DURATION_S": 0.40,

    # Event expansion (hysteresis) seconds around detected events
    "EVENT_EXPANSION_SECONDS": 1.0,

    # -------------------------------
    # Adaptive thresholds (if enabled)
    # -------------------------------
    "Q_ACCEL": 0.975,
    "Q_BRAKE": 0.025,
    "Q_TURN": 0.98,

    # Physical caps (adaptive thresholds clipped here)
    "ACCEL_G_CAP_MIN": 0.25,
    "ACCEL_G_CAP_MAX": 0.55,
    "BRAKE_G_CAP_MIN": -0.60,
    "BRAKE_G_CAP_MAX": -0.35,
    "TURN_G_CAP_MIN": 0.30,
    "TURN_G_CAP_MAX": 0.55,

    # Guard: prevent adaptive thresholds from drifting too high for aggressive drivers
    # (keeps "harsh" as a physical notion, not a percentile)
    "ADAPTIVE_ANCHOR_TURN_G_MAX": 0.45,
    "ADAPTIVE_ANCHOR_ACCEL_G_MAX": 0.40,
    "ADAPTIVE_ANCHOR_BRAKE_G_MIN": -0.38,  # brake thresholds are negative: "min" means less negative bound

    # Braking confirmation from speed signal (m/s², negative)
    "BRAKE_A_SPEED_MIN_MS2": -0.3,

    # Label priority
    "PRIORITY": ["Harsh Turning", "Harsh Braking", "Harsh Acceleration"],
}

LABEL_ORDER = ["Normal", "Harsh Acceleration", "Harsh Braking", "Harsh Turning"]


# ============================================================
# 3) UTILITIES
# ============================================================
def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to stable snake_case-like tokens."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r"[^\w\s%()/.-]", "", regex=True)
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[()/%.-]", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    return df


def to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def safe_dt(df: pd.DataFrame, default: float = 0.04) -> float:
    """Median sampling period; fallback ~25Hz if missing/invalid."""
    if "sampleperiod_s" not in df.columns:
        return default
    dt = pd.to_numeric(df["sampleperiod_s"], errors="coerce").median()
    if pd.isna(dt) or dt <= 0:
        return default
    return float(dt)


def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Pick the first existing column from candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def clean_mask(mask: pd.Series, min_samples: int) -> pd.Series:
    """
    Morphological closing:
      - fills short gaps
      - enforces local continuity
    """
    mask = mask.fillna(False).astype(bool)
    if min_samples <= 1:
        return mask
    cleaned = binary_closing(mask.values, structure=np.ones(min_samples, dtype=bool))
    return pd.Series(cleaned, index=mask.index)


def drop_short_runs(mask: pd.Series, min_len: int) -> pd.Series:
    """
    Remove True segments shorter than min_len samples.
    This is a strict minimum-duration enforcement.
    """
    mask_val = mask.fillna(False).astype(bool).values
    if min_len <= 1:
        return mask.fillna(False).astype(bool)

    padded = np.concatenate(([False], mask_val, [False]))
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    out = mask_val.copy()
    for s, e in zip(starts, ends):
        if (e - s) < min_len:
            out[s:e] = False

    return pd.Series(out, index=mask.index)


def expand_mask(mask: pd.Series, samples: int) -> pd.Series:
    """
    Expand True regions by 'samples' in both directions (dilation).
    Captures ramp-up / ramp-down of harsh events.
    """
    mask = mask.fillna(False).astype(bool)
    if samples <= 0:
        return mask
    expanded = binary_dilation(mask.values, structure=np.ones(samples, dtype=bool))
    return pd.Series(expanded, index=mask.index)


def count_distinct_events(df: pd.DataFrame) -> dict:
    """Count continuous segments per non-Normal label."""
    if df.empty or "Label" not in df.columns:
        return {}
    tmp = df[["Label"]].copy()
    tmp["grp"] = (tmp["Label"] != tmp["Label"].shift()).cumsum()
    return tmp[tmp["Label"] != "Normal"].groupby("Label")["grp"].nunique().to_dict()


# ============================================================
# 4) THRESHOLDS (FIXED OR ADAPTIVE + GUARDS)
# ============================================================
def estimate_thresholds(df: pd.DataFrame, cfg: dict) -> dict:
    """
    If USE_ADAPTIVE_THRESHOLDS is False: return fixed physics thresholds.

    If True: quantile-based thresholds clipped to physical caps PLUS anchors
    to prevent "normalising" aggressive drivers (the core failure mode).
    """
    brake_g = float(cfg["BRAKE_G"])
    accel_g = float(cfg["ACCEL_G"])
    turn_g = float(cfg["TURN_G"])

    if not bool(cfg["USE_ADAPTIVE_THRESHOLDS"]):
        return {"BRAKE_G": brake_g, "ACCEL_G": accel_g, "TURN_G": turn_g}

    # Adaptive estimation
    if "Longitudinal_acceleration_g" in df.columns:
        long_g = pd.to_numeric(df["Longitudinal_acceleration_g"], errors="coerce").dropna()
        if not long_g.empty:
            brake_q = long_g.quantile(float(cfg["Q_BRAKE"]))
            brake_g = float(np.clip(brake_q, float(cfg["BRAKE_G_CAP_MIN"]), float(cfg["BRAKE_G_CAP_MAX"])))

            pos = long_g[long_g > 0]
            accel_q = pos.quantile(float(cfg["Q_ACCEL"])) if not pos.empty else long_g.quantile(float(cfg["Q_ACCEL"]))
            accel_g = float(np.clip(accel_q, float(cfg["ACCEL_G_CAP_MIN"]), float(cfg["ACCEL_G_CAP_MAX"])))

    if "Lateral_acceleration_g" in df.columns:
        lat_abs = pd.to_numeric(df["Lateral_acceleration_g"], errors="coerce").dropna().abs()
        if not lat_abs.empty:
            turn_q = lat_abs.quantile(float(cfg["Q_TURN"]))
            turn_g = float(np.clip(turn_q, float(cfg["TURN_G_CAP_MIN"]), float(cfg["TURN_G_CAP_MAX"])))

    # Guards/anchors against aggressive-driver normalisation
    turn_g = float(min(turn_g, float(cfg["ADAPTIVE_ANCHOR_TURN_G_MAX"])))
    accel_g = float(min(accel_g, float(cfg["ADAPTIVE_ANCHOR_ACCEL_G_MAX"])))
    brake_g = float(max(brake_g, float(cfg["ADAPTIVE_ANCHOR_BRAKE_G_MIN"])))

    return {"BRAKE_G": brake_g, "ACCEL_G": accel_g, "TURN_G": turn_g}


# ============================================================
# 5) PHYSICS BRAKING MASK
# ============================================================
def compute_physics_braking_mask(
    df: pd.DataFrame,
    *,
    win_n: int,
    min_speed_kmh: float,
    brake_g_threshold: float,      # negative g
    a_speed_min_ms2: float,        # negative m/s² (speed-derived confirmation)
    use_gradient_correction: bool = True,
) -> Tuple[pd.Series, pd.Series]:
    """
    Returns:
      - mask_brake_strong: (a_corr <= brake threshold) AND (speed decel confirms)
      - roll_a_speed_min : speed-derived rolling min decel (useful for debug/rescue)
    """
    g0 = 9.81
    dt_med = safe_dt(df)

    col_speed = pick_col(
        df,
        [
            "Indicated_Vehicle_Speed_kph_km_h",
            "Indicated_Vehicle_Speed_kph_kmh",
            "Indicated_Vehicle_Speed_kph",
            "Indicated_Vehicle_Speed",
        ],
    )
    if col_speed is None:
        raise ValueError("Missing indicated speed column (Indicated_Vehicle_Speed_...).")
    if "Longitudinal_acceleration_g" not in df.columns:
        raise ValueError("Missing Longitudinal_acceleration_g.")

    speed_kmh = pd.to_numeric(df[col_speed], errors="coerce")
    speed_mps = speed_kmh * (1000.0 / 3600.0)

    if "sampleperiod_s" in df.columns:
        dt_series = pd.to_numeric(df["sampleperiod_s"], errors="coerce").replace(0, np.nan).fillna(dt_med)
    else:
        dt_series = pd.Series(dt_med, index=df.index)

    a_speed = speed_mps.diff() / dt_series
    a_speed.iloc[0] = np.nan

    a_long_ms2 = pd.to_numeric(df["Longitudinal_acceleration_g"], errors="coerce") * g0

    a_corr = a_long_ms2
    if use_gradient_correction:
        col_grad = pick_col(df, ["Gradient_pct", "Gradient"])
        if col_grad is not None:
            grad_pct = pd.to_numeric(df[col_grad], errors="coerce").fillna(0.0)
            theta = np.arctan(grad_pct / 100.0)
            a_corr = a_long_ms2 - (g0 * np.sin(theta))

    roll_a_corr_min = a_corr.rolling(win_n, min_periods=1).min()
    roll_a_speed_min = a_speed.rolling(win_n, min_periods=1).min()

    moving = speed_kmh > float(min_speed_kmh)
    speed_drop_confirm = roll_a_speed_min <= float(a_speed_min_ms2)

    a_corr_thresh_ms2 = float(brake_g_threshold) * g0
    mask_brake_strong = moving & speed_drop_confirm & (roll_a_corr_min <= a_corr_thresh_ms2)

    return mask_brake_strong.fillna(False), roll_a_speed_min


# ============================================================
# 6) CORE LABELING
# ============================================================
def label_vehicle_dynamics(df_raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = standardise_columns(df_raw)

    # Coerce numeric for known channels (only if present)
    df = to_numeric(
        df,
        [
            "sampleperiod_s",
            "Speed_kmh",
            "Longitudinal_acceleration_g",
            "Lateral_acceleration_g",
            "Accelerator_Pedal_Position_",
            "Brake_Pedal_Position_",
            "BrakeTrigger",
            "Gradient_pct",
            "Gradient",
            "Indicated_Vehicle_Speed_kph_km_h",
            "Indicated_Vehicle_Speed_kph_kmh",
            "Indicated_Vehicle_Speed_kph",
            "Indicated_Vehicle_Speed",
        ],
    )

    # Always provide Label column
    df["Label"] = "Normal"

    # Minimum required for harsh-event logic; if missing, return Normal labels
    if "Longitudinal_acceleration_g" not in df.columns or "Lateral_acceleration_g" not in df.columns:
        return df

    dt = safe_dt(df)
    win_n = max(1, int(round(float(cfg["WINDOW_SECONDS"]) / dt)))
    min_event_samples = max(1, int(round(float(cfg["MIN_DURATION_S"]) / dt)))
    expansion_samples = max(0, int(round(float(cfg["EVENT_EXPANSION_SECONDS"]) / dt)))

    thr = estimate_thresholds(df, cfg)

    # Rolling extrema for accel/turn in g
    df["roll_long_max_g"] = df["Longitudinal_acceleration_g"].rolling(win_n, min_periods=1).max()
    df["roll_lat_max_g"] = df["Lateral_acceleration_g"].abs().rolling(win_n, min_periods=1).max()

    # Optional throttle smoothing
    # NOTE: standardise_columns() maps "Accelerator_Pedal_Position (%)" to
    # "Accelerator_Pedal_Position" (no trailing underscore) -- .str.strip("_")
    # removes it. A previous version of this check looked for
    # "Accelerator_Pedal_Position_" (with trailing underscore), which never
    # matched, silently disabling throttle-pedal corroboration for every
    # released Harsh Acceleration label (see Technical Validation).
    if "Accelerator_Pedal_Position" in df.columns:
        df["roll_throttle_max"] = df["Accelerator_Pedal_Position"].rolling(win_n, min_periods=1).max()
    else:
        df["roll_throttle_max"] = np.nan

    # --------------------------------------------------------
    # A) Harsh Braking (physics + speed confirmation)
    # --------------------------------------------------------
    mask_brake_strong, roll_a_speed_min = compute_physics_braking_mask(
        df,
        win_n=win_n,
        min_speed_kmh=float(cfg["MIN_SPEED_KMH"]),
        brake_g_threshold=float(thr["BRAKE_G"]),
        a_speed_min_ms2=float(cfg["BRAKE_A_SPEED_MIN_MS2"]),
        use_gradient_correction=False,
    )

    # NOTE: brake-pedal/trigger corroboration was removed from the weak-path
    # condition below after Technical Validation found both raw CAN channels
    # (Brake_Pedal_Position_, BrakeTrigger) non-functional on this vehicle --
    # Brake_Pedal_Position_ is stuck near 100% regardless of actual braking
    # (median 100.00 across every released session) and BrakeTrigger never
    # fires. The weak path now relies solely on the physically-grounded
    # speed-drop + relaxed-magnitude confirmation below, which is exactly
    # what it already reduced to in practice (brake_intent was, in effect,
    # always True given the broken pedal channel) -- this change does not
    # alter any previously-released label.

    # Rescue rule (weak braking) still requires speed-drop confirmation
    g0 = 9.81
    brake_g_weak = min(float(thr["BRAKE_G"]) + 0.05, -0.15)  # relax a bit but keep meaningful
    a_corr_thresh_weak_ms2 = brake_g_weak * g0

    # NOTE: gradient correction was removed from this weak-path magnitude
    # check after Technical Validation found gradient_pct contains a small
    # tail of physically implausible spikes (up to several hundred percent
    # grade); quantified impact was 3 of 4,212 released Harsh Braking events
    # (0.07%) that existed only via a gradient-corrected threshold crossing.
    # The weak path now uses raw longitudinal acceleration directly, matching
    # the primary/strong path (compute_physics_braking_mask,
    # use_gradient_correction=False), for a gradient-independent pipeline.
    a_long_ms2 = pd.to_numeric(df["Longitudinal_acceleration_g"], errors="coerce") * g0
    roll_a_corr_min = a_long_ms2.rolling(win_n, min_periods=1).min()

    col_ind_speed = pick_col(
        df,
        [
            "Indicated_Vehicle_Speed_kph_km_h",
            "Indicated_Vehicle_Speed_kph_kmh",
            "Indicated_Vehicle_Speed_kph",
            "Indicated_Vehicle_Speed",
        ],
    )
    speed_kmh = pd.to_numeric(df[col_ind_speed], errors="coerce")
    moving_brake = speed_kmh > float(cfg["MIN_SPEED_KMH"])
    speed_drop_confirm = roll_a_speed_min <= float(cfg["BRAKE_A_SPEED_MIN_MS2"])

    mask_brake_weak = (
        moving_brake
        & speed_drop_confirm
        & (roll_a_corr_min <= a_corr_thresh_weak_ms2)
    )

    mask_braking = (mask_brake_strong | mask_brake_weak).fillna(False)

    # --------------------------------------------------------
    # B) Harsh Acceleration
    # --------------------------------------------------------
    col_move_speed = pick_col(
        df,
        [
            "Speed_kmh",
            "Indicated_Vehicle_Speed_kph_km_h",
            "Indicated_Vehicle_Speed_kph_kmh",
            "Indicated_Vehicle_Speed_kph",
            "Indicated_Vehicle_Speed",
        ],
    )
    if col_move_speed is None:
        # Without a speed gate, return with braking-only labels (still valid)
        df.loc[mask_braking, "Label"] = "Harsh Braking"
        return df

    moving = pd.to_numeric(df[col_move_speed], errors="coerce") > float(cfg["MIN_SPEED_KMH"])

    # Throttle intent (strict if available; permissive otherwise)
    if "Accelerator_Pedal_Position" in df.columns:
        accel_intent = df["roll_throttle_max"] > float(cfg["ACCEL_PEDAL_MIN"])
    else:
        accel_intent = pd.Series(True, index=df.index)

    mask_accel = (
        moving
        & (df["roll_long_max_g"] >= float(thr["ACCEL_G"]))
        & accel_intent
        & (~mask_braking)
    )

    # --------------------------------------------------------
    # C) Harsh Turning (separate speed gate)
    # --------------------------------------------------------
    moving_turn = pd.to_numeric(df[col_move_speed], errors="coerce") > float(cfg["MIN_SPEED_TURN_KMH"])
    mask_turning = moving_turn & (df["roll_lat_max_g"] >= float(thr["TURN_G"]))

    # --------------------------------------------------------
    # D) Temporal cleaning + noise control + hysteresis
    # --------------------------------------------------------
    # 1) Close gaps
    mask_braking = clean_mask(mask_braking, min_event_samples)
    mask_accel = clean_mask(mask_accel, min_event_samples)
    mask_turning = clean_mask(mask_turning, min_event_samples)

    # 2) Enforce min-duration (remove blips)
    mask_braking = drop_short_runs(mask_braking, min_event_samples)
    mask_accel = drop_short_runs(mask_accel, min_event_samples)
    mask_turning = drop_short_runs(mask_turning, min_event_samples)

    # 3) Expand events (ramp-up/ramp-down capture)
    # Turning tends to be slower; expand fully. Brake/accel expand moderately.
    mask_turning = expand_mask(mask_turning, expansion_samples)
    mask_braking = expand_mask(mask_braking, int(round(expansion_samples * 0.5)))
    mask_accel = expand_mask(mask_accel, int(round(expansion_samples * 0.5)))

    # 4) Re-enforce min-duration after expansion (prevents expansion from creating fake events)
    mask_braking = drop_short_runs(mask_braking, min_event_samples)
    mask_accel = drop_short_runs(mask_accel, min_event_samples)
    mask_turning = drop_short_runs(mask_turning, min_event_samples)

    # --------------------------------------------------------
    # E) Priority assignment (Turning > Braking > Accel)
    # --------------------------------------------------------
    # Start all Normal, then apply in increasing priority
    df["Label"] = "Normal"
    df.loc[mask_accel, "Label"] = "Harsh Acceleration"
    df.loc[mask_braking, "Label"] = "Harsh Braking"
    df.loc[mask_turning, "Label"] = "Harsh Turning"

    # Audit fields
    df["thr_brake_g"] = float(thr["BRAKE_G"])
    df["thr_accel_g"] = float(thr["ACCEL_G"])
    df["thr_turn_g"] = float(thr["TURN_G"])

    return df


# ============================================================
# 7) RUN PIPELINE
# ============================================================
def main() -> None:
    global INPUT_DIR, OUTPUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=INPUT_DIR,
                         help="Directory containing raw D{driver}_S{session}.csv files "
                              "(default: $DATASET_RAW_DIR)")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                         help="Directory to write D{driver}_S{session}_LABELED.csv files "
                              "(default: $DATASET_LABELED_DIR)")
    args = parser.parse_args()
    INPUT_DIR, OUTPUT_DIR = args.input_dir, args.output_dir
    if not INPUT_DIR or not OUTPUT_DIR:
        parser.error("--input-dir/--output-dir must be set (or DATASET_RAW_DIR/DATASET_LABELED_DIR env vars)")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("STARTING LABELING...")
    print(f"INPUT_DIR: {INPUT_DIR}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"Adaptive thresholds: {CONFIG['USE_ADAPTIVE_THRESHOLDS']}")
    if not bool(CONFIG["USE_ADAPTIVE_THRESHOLDS"]):
        print("Mode: FIXED physics thresholds (recommended for ground truth).")
    else:
        print("Mode: ADAPTIVE thresholds with anchor guards (prevents normalising aggression).")
    print(f"Turning speed gate: {CONFIG['MIN_SPEED_TURN_KMH']} km/h")
    print(f"Event expansion: {CONFIG['EVENT_EXPANSION_SECONDS']} s")
    print("=" * 70)

    overall_counts = pd.Series(0, index=LABEL_ORDER, dtype=int)
    processed = 0

    for d in range(1, 21):
        for s in range(1, 5):
            in_name = f"D{d}_S{s}.csv"
            in_path = os.path.join(INPUT_DIR, in_name)
            if not os.path.exists(in_path):
                continue

            try:
                df_raw = pd.read_csv(in_path)
                df_lab = label_vehicle_dynamics(df_raw, CONFIG)

                out_name = f"D{d}_S{s}_LABELED.csv"
                out_path = os.path.join(OUTPUT_DIR, out_name)

                # Preserve raw schema exactly; only append Label
                df_out = df_raw.copy()
                df_out["Label"] = df_lab["Label"].values
                df_out.to_csv(out_path, index=False)

                dist = df_lab["Label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
                overall_counts += dist.astype(int)

                thr_br = float(df_lab["thr_brake_g"].iloc[0]) if "thr_brake_g" in df_lab.columns else float("nan")
                thr_ac = float(df_lab["thr_accel_g"].iloc[0]) if "thr_accel_g" in df_lab.columns else float("nan")
                thr_tu = float(df_lab["thr_turn_g"].iloc[0]) if "thr_turn_g" in df_lab.columns else float("nan")
                ev = count_distinct_events(df_lab)

                print(f"\n{in_name} -> {out_name}")
                print(f"  thresholds(g): brake={thr_br:.3f}, accel={thr_ac:.3f}, turn={thr_tu:.3f}")
                print(f"  events: {ev if ev else 'none'}")

                processed += 1

            except Exception as e:
                print(f"\n{in_name}: ERROR -> {e}")

    print("\n" + "=" * 70)
    print(f"FINISHED. Processed files: {processed}")
    print("FINAL LABEL DISTRIBUTION:")
    print(overall_counts.to_string())
    print("=" * 70)


if __name__ == "__main__":
    main()
