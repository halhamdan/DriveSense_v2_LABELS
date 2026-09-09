"""
Regenerates fused.csv for all 79 valid sessions with:
  1. Corrected EmotiBit-to-VBOX absolute-time alignment (robust, full-session
     calibration from timesyncs.csv instead of the whole-second filename
     anchor), and
  2. Two additional biometric channels (SCR_AMPLITUDE, SCR_RISE_TIME) that
     exist in the raw device data but were never previously captured.

Deliberately narrow in scope and low-risk: the released grid (unix_time,
elapsed_s, all VBOX/GPS columns, the label column, and the trim window) is
NOT recomputed -- none of it depends on the EmotiBit stream at all, so it is
copied byte-for-byte from the currently-released fused.csv. Only the
biometric columns are recomputed, using the corrected EmotiBit loader
(emotibit_sync_v2.py) and the ORIGINAL, unmodified resampling/despiking code
from fuse_stage1.py (_resample_emotibit_signal) -- reused via import, not
reimplemented, to avoid introducing a second, un-validated resampling path.

Writes to a completely separate output tree; the released dataset
(Published_Dataset_Final) is never opened for writing.

Usage:
    python regenerate_fused_v2.py

Environment variables:
    STAGING_DATA_ROOT   raw per-driver session folders (containing EmotiBit/)
    DATASET_ROOT        current release root (read-only; source of the
                         non-biometric columns and target output grid)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "02_dataset_construction"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fuse_stage1 as fs1          # noqa: E402 -- original, unmodified resampling code
import data_checks as dc           # noqa: E402
from emotibit_sync_v2 import load_emotibit_v2, NEW_CHANNELS  # noqa: E402

STAGING_DATA_ROOT = Path(os.environ.get("STAGING_DATA_ROOT", ""))
DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", ""))

# SYNC_MODEL: "anchor_only" (release default since 2026-09-09) or "linear"
# (the 2026-09-08 regen_v2 model) -- see timesync_calibration.py.
# REGEN_OUT_ROOT: output tree; defaults to regenerated_v3 for anchor_only and
# regenerated_v2 for linear so the two regenerations never overwrite each other.
SYNC_MODEL = os.environ.get("SYNC_MODEL", "anchor_only")
_default_out = "regenerated_v3" if SYNC_MODEL == "anchor_only" else "regenerated_v2"
_env_out = os.environ.get("REGEN_OUT_ROOT", "").strip()
OUT_ROOT = Path(_env_out) if _env_out else (Path(__file__).resolve().parents[3] / _default_out)
OUT_FUSED = OUT_ROOT / "Preprocessed_Dataset"
REPORT_PATH = OUT_ROOT / "regeneration_report.csv"

TRIM_CSV = REPO_ROOT / "04_session_trimming" / "trim_review_output" / "trim_points_final.csv"

# Extend the two lookup tables fuse_stage1._resample_emotibit_signal reads at
# call time, so the new channels get the SAME gap-aware, no-despike, dropout-
# flagged treatment already used for HR/IBI (both event-triggered, dropout-
# prone signals) -- without touching either module's own source.
dc.EMOTIBIT_FS.setdefault("SCR_AMPLITUDE", 1.0)
dc.EMOTIBIT_FS.setdefault("SCR_RISE_TIME", 1.0)
fs1._NO_DESPIKE_SIGNALS = fs1._NO_DESPIKE_SIGNALS | set(NEW_CHANNELS)

# Original per-signal names as returned by load_emotibit_v2 (raw, pre-suffix).
# "ACC"/"GYRO"/"MAG" are the merged 3-axis groups; _resample_emotibit_signal
# expands each into _x/_y/_z/_mag columns itself.
BIOMETRIC_SIGNAL_NAMES = [
    "EDA", "EDA_LEVEL", "TEMP_CONTACT", "TEMP_THERMOPILE",
    "PPG_IR", "PPG_RED", "PPG_GREEN", "SCR_FREQ", "IBI", "HR",
    "SCR_AMPLITUDE", "SCR_RISE_TIME",
    "ACC", "GYRO", "MAG",
]


def load_sessions() -> pd.DataFrame:
    df = pd.read_csv(TRIM_CSV)
    return df[df["status"] == "ok"][["tag", "driver", "session"]]


def find_session_dir(driver: int, session: int) -> Path | None:
    for sname in (f"Session {session}", f"Session_{session}", f"Session{session}", f"Seesion {session}"):
        d = STAGING_DATA_ROOT / f"D{driver}" / sname
        if d.exists():
            return d
    return None


def regenerate_session(tag: str, driver: int, session: int) -> dict:
    released_path = DATASET_ROOT / "Preprocessed_Dataset" / f"D{driver}" / f"Session_{session}" / f"D{driver}_S{session}_fused.csv"
    if not released_path.exists():
        return {"tag": tag, "status": "NO_RELEASED_FILE"}

    released = pd.read_csv(released_path)
    t_out = released["unix_time"].to_numpy(dtype=float)

    sess_dir = find_session_dir(driver, session)
    if sess_dir is None:
        return {"tag": tag, "status": "NO_RAW_SESSION_DIR"}

    try:
        signals, meta = load_emotibit_v2(sess_dir, sync_model=SYNC_MODEL)
    except Exception as e:
        return {"tag": tag, "status": f"LOAD_ERROR: {e}"}

    new_bio: dict[str, np.ndarray] = {}
    for sig_name in BIOMETRIC_SIGNAL_NAMES:
        df = signals.get(sig_name)
        if df is None or df.empty:
            continue
        new_bio.update(fs1._resample_emotibit_signal(df, t_out, sig_name))

    # Schema consistency: a session with zero detected SCR events genuinely
    # has no SA/SR raw files at all (confirmed on D1_S4 -- not a load error),
    # but every session's fused.csv should carry the same column set so
    # downstream code doesn't have to special-case one session's schema.
    # Missing here means "no signal for the whole session", i.e. fully
    # NaN/dropout, matching what the gap-aware resampler would have produced
    # from an entirely-empty input series.
    for ch in NEW_CHANNELS:
        if ch not in new_bio:
            new_bio[ch] = np.full(len(t_out), np.nan)
            new_bio[f"{ch}_dropout_flag"] = np.ones(len(t_out), dtype=bool)

    # Non-biometric columns copied verbatim from the release (untouched --
    # nothing here depends on the EmotiBit stream).
    non_bio_cols = [c for c in released.columns
                    if c not in new_bio and not any(
                        c == old or c.startswith(old + "_")
                        for old in ["EDA", "TEMP_CONTACT", "TEMP_THERMOPILE",
                                    "PPG_IR", "PPG_RED", "PPG_GREEN", "SCR_FREQ",
                                    "IBI", "HR", "ACC", "GYRO", "MAG"])]
    out_df = released[non_bio_cols].copy()
    for col, arr in new_bio.items():
        out_df[col] = arr

    out_dir = OUT_FUSED / f"D{driver}" / f"Session_{session}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"D{driver}_S{session}_fused.csv"
    out_df.to_csv(out_path, index=False, float_format="%.6f")

    return {
        "tag": tag, "status": "ok",
        "sync_method": meta.get("sync_method"),
        "ppm_drift": meta.get("ppm_drift"),
        "linear_slope_ppm": meta.get("linear_slope_ppm"),
        "median_offset_s": meta.get("median_offset_s"),
        "max_step_s": meta.get("max_step_s"),
        "step_at_s": meta.get("step_at_s"),
        "calibration_span_s": meta.get("calibration_span_s"),
        "n_pings_used": meta.get("n_pings_used"),
        "n_rows_released": len(released), "n_rows_v2": len(out_df),
        "n_cols_released": released.shape[1], "n_cols_v2": out_df.shape[1],
        "new_channels_present": [c for c in NEW_CHANNELS if c in out_df.columns],
        "out_path": str(out_path),
    }


def main():
    if not STAGING_DATA_ROOT.exists():
        raise SystemExit(f"STAGING_DATA_ROOT not found or unset: {STAGING_DATA_ROOT!r}")
    if not DATASET_ROOT.exists():
        raise SystemExit(f"DATASET_ROOT not found or unset: {DATASET_ROOT!r}")
    OUT_FUSED.mkdir(parents=True, exist_ok=True)

    sessions = load_sessions()
    print(f"Regenerating {len(sessions)} sessions into {OUT_FUSED}")
    print(f"(released dataset at {DATASET_ROOT} is opened READ-ONLY and never modified)\n")

    rows = []
    for i, r in sessions.iterrows():
        tag, d, s = r["tag"], int(r["driver"]), int(r["session"])
        res = regenerate_session(tag, d, s)
        status = res["status"]
        flag = "OK" if status == "ok" else f"** {status}"
        print(f"  [{len(rows)+1:2d}/{len(sessions)}] {tag}: {flag}")
        rows.append(res)

    report = pd.DataFrame(rows)
    report.to_csv(REPORT_PATH, index=False)

    ok = report[report["status"] == "ok"]
    print(f"\n{len(ok)} / {len(report)} sessions regenerated successfully.")
    if len(ok) < len(report):
        print("Failures:")
        print(report[report["status"] != "ok"][["tag", "status"]].to_string(index=False))

    print(f"\nSync method breakdown:")
    print(ok["sync_method"].value_counts().to_string())
    print(f"\nFull report: {REPORT_PATH}")
    print(f"Regenerated files under: {OUT_FUSED}")


if __name__ == "__main__":
    main()
