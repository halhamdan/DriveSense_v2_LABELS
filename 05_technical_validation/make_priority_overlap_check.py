"""
Technical Validation: exact turning/braking/acceleration overlap count
BEFORE priority resolution (Turning > Braking > Acceleration, Methods
Eq. eq:priority in the manuscript).

Reproduces the mask logic in 01_annotation/label_harsh_events.py against the
FULL, UNTRIMMED native per-session VBOX export -- not the released, already
trimmed/resampled Preprocessed_Dataset/*_fused.csv files. An earlier version
of this script ran on the trimmed fused.csv directly, which the manuscript's
own Table-sensitivity caption already warns against ("clipping first...
starves rolling-window calculations of context at the clip boundary"): doing
so gave pre-priority mask sizes that contradicted the released class totals
in Table 4 (tab:classdist) outright (e.g. pre-priority Turning smaller than
the released, top-priority Harsh Turning count, which is impossible under the
priority rule). This version:

  1. Labels each full, untrimmed session (same input the original annotation
     pipeline used: DATASET_RAW_DIR/D{d}_S{s}.csv), reusing
     01_annotation/label_harsh_events.py's own functions directly rather than
     a separate reimplementation.
  2. Locates the released window empirically within that full session, via
     cross-correlation of lon_acc_g against the released fused.csv (the
     fused release's own elapsed-time origin is anchored to the later of the
     two devices' start times, not the raw VBOX file's own t=0, so the two
     time axes cannot simply be lined up from stored metadata alone).
  3. Reports how closely the reconstructed class totals match Table 4 before
     reporting any overlap number, exactly as Table~tab:sensitivity already
     does for its own reconstruction.

Requires DATASET_RAW_DIR (native per-session VBOX CSVs, D{d}_S{s}.csv, NOT
the trimmed Raw_Dataset/ release tier) in addition to DATASET_ROOT.

Usage:
  python make_priority_overlap_check.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "01_annotation"))
import label_harsh_events as leh  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))
from _paths import DATASET_ROOT  # noqa: E402

RAW_NATIVE_DIR = Path(os.environ.get("DATASET_RAW_DIR", ""))  # full, untrimmed per-session CSVs
PUB = DATASET_ROOT
OUT_DIR = Path(__file__).parent / "validation_output"
OUT_DIR.mkdir(exist_ok=True)

CFG = leh.CONFIG


def find_offset(raw_lon: np.ndarray, fused_lon: np.ndarray, max_search: int = 6000):
    n_probe = min(1000, len(fused_lon))
    probe = fused_lon[:n_probe]
    max_off = min(len(raw_lon) - n_probe, max_search)
    best_offset, best_score = 0, np.inf
    for off in range(0, max_off):
        window = raw_lon[off:off + n_probe]
        score = np.abs(window - probe).sum()
        if score < best_score:
            best_score = score
            best_offset = off
    return best_offset, best_score


def compute_masks_full_session(df_raw: pd.DataFrame):
    """Reproduces label_vehicle_dynamics() up to (not including) the final
    priority-overwrite step, returning the three cleaned pre-priority masks,
    using label_harsh_events.py's own helper functions throughout."""
    df = leh.standardise_columns(df_raw)
    df = leh.to_numeric(df, [
        "sampleperiod_s", "Speed_kmh", "Longitudinal_acceleration_g", "Lateral_acceleration_g",
        "Accelerator_Pedal_Position_", "Brake_Pedal_Position_", "BrakeTrigger",
        "Gradient_pct", "Gradient", "Indicated_Vehicle_Speed_kph_km_h",
        "Indicated_Vehicle_Speed_kph_kmh", "Indicated_Vehicle_Speed_kph", "Indicated_Vehicle_Speed",
    ])

    dt = leh.safe_dt(df)
    win_n = max(1, int(round(float(CFG["WINDOW_SECONDS"]) / dt)))
    min_event_samples = max(1, int(round(float(CFG["MIN_DURATION_S"]) / dt)))
    expansion_samples = max(0, int(round(float(CFG["EVENT_EXPANSION_SECONDS"]) / dt)))

    thr = leh.estimate_thresholds(df, CFG)

    df["roll_long_max_g"] = df["Longitudinal_acceleration_g"].rolling(win_n, min_periods=1).max()
    df["roll_lat_max_g"] = df["Lateral_acceleration_g"].abs().rolling(win_n, min_periods=1).max()
    if "Accelerator_Pedal_Position" in df.columns:
        df["roll_throttle_max"] = df["Accelerator_Pedal_Position"].rolling(win_n, min_periods=1).max()
    else:
        df["roll_throttle_max"] = np.nan

    mask_brake_strong, roll_a_speed_min = leh.compute_physics_braking_mask(
        df, win_n=win_n, min_speed_kmh=float(CFG["MIN_SPEED_KMH"]),
        brake_g_threshold=float(thr["BRAKE_G"]), a_speed_min_ms2=float(CFG["BRAKE_A_SPEED_MIN_MS2"]),
        use_gradient_correction=False,
    )

    g0 = 9.81
    brake_g_weak = min(float(thr["BRAKE_G"]) + 0.05, -0.15)
    a_corr_thresh_weak_ms2 = brake_g_weak * g0
    a_long_ms2 = pd.to_numeric(df["Longitudinal_acceleration_g"], errors="coerce") * g0
    roll_a_corr_min = a_long_ms2.rolling(win_n, min_periods=1).min()

    col_ind_speed = leh.pick_col(df, [
        "Indicated_Vehicle_Speed_kph_km_h", "Indicated_Vehicle_Speed_kph_kmh",
        "Indicated_Vehicle_Speed_kph", "Indicated_Vehicle_Speed",
    ])
    speed_kmh = pd.to_numeric(df[col_ind_speed], errors="coerce")
    moving_brake = speed_kmh > float(CFG["MIN_SPEED_KMH"])
    speed_drop_confirm = roll_a_speed_min <= float(CFG["BRAKE_A_SPEED_MIN_MS2"])

    mask_brake_weak = moving_brake & speed_drop_confirm & (roll_a_corr_min <= a_corr_thresh_weak_ms2)
    mask_braking = (mask_brake_strong | mask_brake_weak).fillna(False)

    col_move_speed = leh.pick_col(df, [
        "Speed_kmh", "Indicated_Vehicle_Speed_kph_km_h", "Indicated_Vehicle_Speed_kph_kmh",
        "Indicated_Vehicle_Speed_kph", "Indicated_Vehicle_Speed",
    ])
    moving = pd.to_numeric(df[col_move_speed], errors="coerce") > float(CFG["MIN_SPEED_KMH"])
    if "Accelerator_Pedal_Position" in df.columns:
        accel_intent = df["roll_throttle_max"] > float(CFG["ACCEL_PEDAL_MIN"])
    else:
        accel_intent = pd.Series(True, index=df.index)
    mask_accel = moving & (df["roll_long_max_g"] >= float(thr["ACCEL_G"])) & accel_intent & (~mask_braking)

    moving_turn = pd.to_numeric(df[col_move_speed], errors="coerce") > float(CFG["MIN_SPEED_TURN_KMH"])
    mask_turning = moving_turn & (df["roll_lat_max_g"] >= float(thr["TURN_G"]))

    mask_braking = leh.clean_mask(mask_braking, min_event_samples)
    mask_accel = leh.clean_mask(mask_accel, min_event_samples)
    mask_turning = leh.clean_mask(mask_turning, min_event_samples)
    mask_braking = leh.drop_short_runs(mask_braking, min_event_samples)
    mask_accel = leh.drop_short_runs(mask_accel, min_event_samples)
    mask_turning = leh.drop_short_runs(mask_turning, min_event_samples)
    mask_turning = leh.expand_mask(mask_turning, expansion_samples)
    mask_braking = leh.expand_mask(mask_braking, int(round(expansion_samples * 0.5)))
    mask_accel = leh.expand_mask(mask_accel, int(round(expansion_samples * 0.5)))
    mask_braking = leh.drop_short_runs(mask_braking, min_event_samples)
    mask_accel = leh.drop_short_runs(mask_accel, min_event_samples)
    mask_turning = leh.drop_short_runs(mask_turning, min_event_samples)

    return mask_accel, mask_braking, mask_turning, df["Longitudinal_acceleration_g"].values


def main():
    meta = json.load(open(PUB / "dataset_metadata.json"))
    rows_map = {s["tag"]: s["rows"] for s in meta["sessions"]}

    rows = []
    tot = dict(normal=0, accel=0, braking=0, turning=0, turn_brake_ov=0, turn_accel_ov=0,
               pre_turn=0, pre_brake=0, pre_accel=0, n=0)

    files = sorted(RAW_NATIVE_DIR.glob("D*_S*.csv"))
    print(f"{len(files)} native per-session CSVs found in DATASET_RAW_DIR")

    for fp in files:
        tag = fp.stem
        if tag not in rows_map:
            continue
        d, s = tag[1:].split("_S")

        df_raw = pd.read_csv(fp)
        mask_accel, mask_braking, mask_turning, raw_lon = compute_masks_full_session(df_raw)

        fused_fp = PUB / "Preprocessed_Dataset" / f"D{d}" / f"Session_{s}" / f"{tag}_fused.csv"
        fused_lon = pd.to_numeric(pd.read_csv(fused_fp, usecols=["lon_acc_g"])["lon_acc_g"],
                                   errors="coerce").fillna(0).values
        offset, score = find_offset(np.nan_to_num(raw_lon), fused_lon)
        released_rows = rows_map[tag]

        ma = mask_accel.values[offset:offset + released_rows]
        mb = mask_braking.values[offset:offset + released_rows]
        mt = mask_turning.values[offset:offset + released_rows]

        final_braking = mb & (~mt)
        final_accel = ma & (~mb) & (~mt)
        final_normal = ~(mt | mb | ma)

        tot["n"] += len(mt)
        tot["normal"] += int(final_normal.sum())
        tot["accel"] += int(final_accel.sum())
        tot["braking"] += int(final_braking.sum())
        tot["turning"] += int(mt.sum())
        tot["turn_brake_ov"] += int((mt & mb).sum())
        tot["turn_accel_ov"] += int((mt & ma).sum())
        tot["pre_turn"] += int(mt.sum())
        tot["pre_brake"] += int(mb.sum())
        tot["pre_accel"] += int(ma.sum())

        rows.append({"tag": tag, "offset": offset, "match_score": score,
                      "final_normal": int(final_normal.sum()), "final_accel": int(final_accel.sum()),
                      "final_braking": int(final_braking.sum()), "final_turning": int(mt.sum())})

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "priority_overlap_check.csv", index=False)

    print("=== Reconstruction accuracy vs. Table 4 (tab:classdist) ===")
    print(f"Normal:   {tot['normal']:,} vs. 2,961,620 released ({100*tot['normal']/2961620:.2f}%)")
    print(f"Accel:    {tot['accel']:,} vs. 52,088 released ({100*tot['accel']/52088:.2f}%)")
    print(f"Braking:  {tot['braking']:,} vs. 303,499 released ({100*tot['braking']/303499:.2f}%)")
    print(f"Turning:  {tot['turning']:,} vs. 186,864 released ({100*tot['turning']/186864:.2f}%)")
    print("\n=== Pre-priority overlap (within this reconstruction) ===")
    print(f"Turning & Braking overlap: {tot['turn_brake_ov']:,} "
          f"({100*tot['turn_brake_ov']/tot['pre_brake']:.1f}% of pre-priority braking)")
    print(f"Turning & Accel overlap: {tot['turn_accel_ov']:,} "
          f"({100*tot['turn_accel_ov']/tot['pre_accel']:.1f}% of pre-priority accel)")
    total_reassigned = tot['turn_brake_ov'] + tot['turn_accel_ov']
    print(f"Total reassigned: {total_reassigned:,} ({100*total_reassigned/tot['turning']:.1f}% of Turning)")


if __name__ == "__main__":
    main()
