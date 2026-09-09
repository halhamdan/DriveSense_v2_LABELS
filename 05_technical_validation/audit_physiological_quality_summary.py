"""
Item 5 (Important) -- Generate physiological quality summaries.

Single consolidated script summarising missingness, long gaps, out-of-range
values and flatlining for every biometric channel, per session and dataset-
wide -- replacing the current situation where this evidence is scattered
across several one-off audits (acceleration artefacts, sync evidence, the
new-channel characterisation) with different, ad hoc methodology each time.

Plausible/hardware ranges are read directly from data_schema.json's own
"range" field per column (the canonical, already-published source used
throughout the Technical Validation section) rather than re-declared here,
so this script cannot silently drift from what the manuscript documents.

Four checks per channel, per session:
  1. Missingness   -- % NaN (dropout-flagged where a companion flag exists,
                       otherwise raw NaN for channels with no dropout flag).
  2. Long gaps      -- for present (non-missing) samples, the longest and
                       total duration of gaps between consecutive samples
                       exceeding 3x the channel's nominal sample period
                       (matching the convention already used for IBI/HR).
  3. Out-of-range   -- % of PRESENT samples outside data_schema.json's
                       documented plausible/hardware range, where one exists.
  4. Flatlining     -- longest run of bit-identical consecutive present
                       values, and the % of present samples inside any run
                       of at least FLATLINE_MIN_S seconds. FLATLINE_MIN_S is
                       0.4 s = 10 consecutive identical samples at 25 Hz, the
                       rule already disclosed in the manuscript's EDA flatline
                       statement ("10 of 79 sessions >30% flatlined"); an
                       earlier draft of this script used 30 s, which is a
                       different (looser) rule and gave 4 sessions instead.

Usage:
    python audit_physiological_quality_summary.py

Environment variables:
    DATASET_ROOT   release root containing Preprocessed_Dataset/
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TRIM_CSV = REPO_ROOT / "04_session_trimming" / "trim_review_output" / "trim_points_final.csv"
DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", ""))

OUT_DIR = Path(__file__).resolve().parent / "validation_output"
OUT_DIR.mkdir(exist_ok=True)

FUSE_HZ = 25.0
GAP_THRESHOLD_X = 3.0       # matches the existing IBI/HR gap-aware convention
FLATLINE_MIN_S = 0.4        # 10 identical consecutive samples at 25 Hz -- the manuscript's disclosed rule

# Nominal per-channel sample period, for gap detection (seconds). Channels
# resampled onto the shared 25 Hz grid still have a NATIVE rate that governs
# what counts as a "gap" vs ordinary sample-to-sample spacing.
NATIVE_HZ = {
    "EDA": 15, "EDA_LEVEL": 15, "SCR_FREQ": 3,
    "SCR_AMPLITUDE": 1, "SCR_RISE_TIME": 1,   # event-triggered; treated like HR/IBI
    "TEMP_CONTACT": 7.5, "TEMP_THERMOPILE": 7.5,
    "PPG_IR": 25, "PPG_RED": 25, "PPG_GREEN": 25,
    "IBI": 1, "HR": 1,
    "ACC_x": 25, "ACC_y": 25, "ACC_z": 25, "ACC_mag": 25,
    "GYRO_x": 25, "GYRO_y": 25, "GYRO_z": 25, "GYRO_mag": 25,
    "MAG_x": 25, "MAG_y": 25, "MAG_z": 25, "MAG_mag": 25,
}
DROPOUT_FLAG_COL = {"HR": "HR_dropout_flag", "IBI": "IBI_dropout_flag",
                     "SCR_AMPLITUDE": "SCR_AMPLITUDE_dropout_flag",
                     "SCR_RISE_TIME": "SCR_RISE_TIME_dropout_flag"}


def load_ranges() -> dict:
    with open(DATASET_ROOT / "data_schema.json") as f:
        schema = json.load(f)
    cols = schema["files"]["{tag}_fused.csv"]["columns"]
    return {name: tuple(info["range"]) for name, info in cols.items()
            if "range" in info and info["range"][0] is not None and info["range"][1] is not None}


def longest_run_and_total_gap(t: np.ndarray, threshold_s: float) -> tuple[float, float]:
    if len(t) < 2:
        return 0.0, 0.0
    diffs = np.diff(t)
    gap_mask = diffs > threshold_s
    if not gap_mask.any():
        return 0.0, 0.0
    return float(diffs[gap_mask].max()), float(diffs[gap_mask].sum())


def flatline_stats(present_mask: np.ndarray, values: np.ndarray, dt: float, min_run_s: float) -> tuple[float, float]:
    """Returns (longest_flatline_s, pct_samples_in_run_ge_min_run_s)."""
    v = values.copy()
    v[~present_mask] = np.nan
    same = np.isfinite(v) & (np.concatenate(([False], np.diff(v) == 0)))
    # run-length encode `same` to find contiguous flat runs (each run includes its start sample)
    runs = []
    run_len = 1
    for i in range(1, len(same)):
        if same[i]:
            run_len += 1
        else:
            if run_len > 1:
                runs.append(run_len)
            run_len = 1
    if run_len > 1:
        runs.append(run_len)
    if not runs:
        return 0.0, 0.0
    runs = np.array(runs)
    longest_s = float(runs.max() * dt)
    min_run_samples = min_run_s / dt
    pct_in_long_runs = float(runs[runs >= min_run_samples].sum() / max(1, present_mask.sum()) * 100)
    return longest_s, pct_in_long_runs


def analyse_channel(df: pd.DataFrame, elapsed_s: np.ndarray, channel: str, ranges: dict) -> dict:
    dropout_col = DROPOUT_FLAG_COL.get(channel)
    values = df[channel].to_numpy(dtype=float)
    if dropout_col and dropout_col in df.columns:
        present_mask = ~df[dropout_col].to_numpy(dtype=bool)
    else:
        present_mask = np.isfinite(values)

    n = len(values)
    missing_pct = float((~present_mask).mean() * 100)

    t_present = elapsed_s[present_mask]
    native_hz = NATIVE_HZ.get(channel, FUSE_HZ)
    native_dt = 1.0 / native_hz
    longest_gap_s, total_gap_s = longest_run_and_total_gap(t_present, GAP_THRESHOLD_X * native_dt)

    rng = ranges.get(channel)
    if rng is not None:
        v_present = values[present_mask]
        oor_pct = float((~((v_present >= rng[0]) & (v_present <= rng[1]))).mean() * 100) if len(v_present) else np.nan
    else:
        oor_pct = np.nan

    longest_flat_s, pct_in_long_flat = flatline_stats(present_mask, values, 1.0 / FUSE_HZ, FLATLINE_MIN_S)

    return {
        "channel": channel, "n": n, "missing_pct": round(missing_pct, 3),
        "longest_gap_s": round(longest_gap_s, 2), "total_gap_s": round(total_gap_s, 2),
        "oor_pct": round(oor_pct, 3) if not np.isnan(oor_pct) else None,
        "longest_flatline_s": round(longest_flat_s, 2),
        "pct_in_flatline": round(pct_in_long_flat, 3),
    }


def main():
    if not DATASET_ROOT.exists():
        raise SystemExit(f"DATASET_ROOT not found or unset: {DATASET_ROOT!r}")

    ranges = load_ranges()
    trim_df = pd.read_csv(TRIM_CSV)
    trim_df = trim_df[trim_df["status"] == "ok"][["tag", "driver", "session"]]

    channels = list(NATIVE_HZ.keys())
    per_session_rows = []
    for _, r in trim_df.iterrows():
        tag, d, s = r["tag"], int(r["driver"]), int(r["session"])
        path = DATASET_ROOT / "Preprocessed_Dataset" / f"D{d}" / f"Session_{s}" / f"D{d}_S{s}_fused.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=lambda c: c == "elapsed_s" or c in channels or c in DROPOUT_FLAG_COL.values())
        elapsed_s = df["elapsed_s"].to_numpy(dtype=float)
        for ch in channels:
            if ch not in df.columns:
                continue
            row = analyse_channel(df, elapsed_s, ch, ranges)
            row["tag"] = tag
            per_session_rows.append(row)
        print(f"  {tag} done")

    per_session = pd.DataFrame(per_session_rows)
    per_session.to_csv(OUT_DIR / "physiological_quality_per_session.csv", index=False)

    # Dataset-wide aggregate per channel
    agg_rows = []
    for ch, g in per_session.groupby("channel"):
        agg_rows.append({
            "channel": ch,
            "mean_missing_pct": round(g["missing_pct"].mean(), 2),
            "max_missing_pct": round(g["missing_pct"].max(), 2),
            "max_longest_gap_s": round(g["longest_gap_s"].max(), 1),
            "mean_oor_pct": round(g["oor_pct"].mean(), 3) if g["oor_pct"].notna().any() else None,
            "max_oor_pct": round(g["oor_pct"].max(), 3) if g["oor_pct"].notna().any() else None,
            "n_sessions_gt30pct_flatline": int((g["pct_in_flatline"] > 30).sum()),
            "max_longest_flatline_s": round(g["longest_flatline_s"].max(), 1),
        })
    summary = pd.DataFrame(agg_rows).sort_values("channel")
    summary.to_csv(OUT_DIR / "physiological_quality_summary.csv", index=False)

    print("\n=== Dataset-wide physiological quality summary ===")
    print(summary.to_string(index=False))
    print(f"\nPer-session detail: {OUT_DIR / 'physiological_quality_per_session.csv'}")
    print(f"Summary table:      {OUT_DIR / 'physiological_quality_summary.csv'}")


if __name__ == "__main__":
    main()
