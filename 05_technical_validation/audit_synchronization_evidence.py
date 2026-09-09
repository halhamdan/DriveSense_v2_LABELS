"""
Item 4 (Important) -- Clarify synchronization using existing evidence.

Checks clock-setting records, timestamp conversions and existing common
timing anchors actually present in this project's own code and raw staging
data, rather than re-deriving synchronization from scratch. Three checks:

  (a) EmotiBit clock-drift calibration coverage. Scans every valid session's
      raw EmotiBit folder for a timeSyncMap.csv (the only mechanism in
      02_dataset_construction/emotibit_sync.py that corrects for EmotiBit
      onboard-clock drift over a session -- see that script's Path A vs
      Path B). Where found, computes the implied clock-rate error in ppm
      from (TE1-TE0) [EmotiBit ms] vs (TL1-TL0) [reference seconds].

  (b) Filename timestamp precision. Checks whether emotibit_sync.py's
      _filename_to_unix regex (which parses only YYYY-MM-DD_HH-MM-SS) is
      discarding a sub-second suffix actually present in the raw EmotiBit
      filenames.

  (c) GPS anchor reliability. For every valid session's native LABELED CSV,
      checks whether the first inter-sample UTC time-of-day gap is close to
      the nominal 0.04 s period -- an anomalously large first gap indicates
      the session's absolute-time anchor may have been taken during a GPS
      time-of-day stall (the mechanism confirmed on D17_S2 in the label-
      reproducibility audit).

Usage:
    python audit_synchronization_evidence.py

Environment variables:
    STAGING_DATA_ROOT     raw per-driver session folders (containing EmotiBit/)
    STAGING_LABELED_DIR   native D{driver}_S{session}_LABELED.csv directory
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TRIM_CSV = REPO_ROOT / "04_session_trimming" / "trim_review_output" / "trim_points_final.csv"

STAGING_DATA_ROOT = Path(os.environ.get("STAGING_DATA_ROOT", ""))
LABELED_DIR = Path(os.environ.get("STAGING_LABELED_DIR", ""))

sys.path.insert(0, str(REPO_ROOT / "02_dataset_construction"))
import parse_labeled as pl  # noqa: E402


def load_sessions() -> pd.DataFrame:
    df = pd.read_csv(TRIM_CSV)
    return df[df["status"] == "ok"][["tag", "driver", "session"]]


# ------------------------------------------------------------------
# (a) + (b): EmotiBit calibration coverage and filename precision
# ------------------------------------------------------------------
def find_emotibit_dir(driver: int, session: int) -> Path | None:
    for sname in (f"Session {session}", f"Session_{session}"):
        d = STAGING_DATA_ROOT / f"D{driver}" / sname / "EmotiBit"
        if d.exists():
            return d
    return None


def check_emotibit_calibration(sessions: pd.DataFrame) -> None:
    n_with_syncmap = 0
    drift_estimates = []
    n_checked = 0
    for _, r in sessions.iterrows():
        d, s = int(r["driver"]), int(r["session"])
        emb_dir = find_emotibit_dir(d, s)
        if emb_dir is None:
            continue
        n_checked += 1
        sync_files = list(emb_dir.glob("*_timeSyncMap.csv"))
        if not sync_files:
            continue
        n_with_syncmap += 1
        row = pd.read_csv(sync_files[0]).iloc[0]
        te0, te1, tl0, tl1 = row["TE0"], row["TE1"], row["TL0"], row["TL1"]
        emotibit_elapsed_s = (te1 - te0) / 1000.0
        reference_elapsed_s = tl1 - tl0
        ppm_error = 1e6 * (reference_elapsed_s - emotibit_elapsed_s) / emotibit_elapsed_s
        drift_estimates.append((r["tag"], emotibit_elapsed_s, reference_elapsed_s, ppm_error))

    print(f"(a) EmotiBit clock-drift calibration (timeSyncMap.csv) coverage:")
    print(f"    {n_with_syncmap} / {n_checked} valid sessions with a raw EmotiBit folder have a timeSyncMap.csv")
    for tag, emb_s, ref_s, ppm in drift_estimates:
        print(f"    {tag}: calibration span {emb_s:.1f}s (EmotiBit clock) vs {ref_s:.3f}s (reference) "
              f"-> {ppm:.1f} ppm ({'slow' if ppm > 0 else 'fast'})")
        for extrap_s, label in [(1200, "20 min"), (4015, "longest session, ~67 min")]:
            print(f"      extrapolated uncorrected drift over {label}: {abs(ppm) * 1e-6 * extrap_s:.3f} s")
    if n_with_syncmap < n_checked:
        print(f"    -> the other {n_checked - n_with_syncmap} sessions have NO drift correction: "
              f"EmotiBit's onboard millisecond counter is trusted uncorrected for the full session duration.")


def check_filename_precision(sessions: pd.DataFrame) -> None:
    print(f"\n(b) EmotiBit filename timestamp precision check:")
    pattern_used = re.compile(r"(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})")
    pattern_full = re.compile(r"(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})-(\d+)")
    n_with_subsecond = 0
    n_checked = 0
    for _, r in sessions.iterrows():
        d, s = int(r["driver"]), int(r["session"])
        emb_dir = find_emotibit_dir(d, s)
        if emb_dir is None:
            continue
        main_csvs = [f for f in emb_dir.glob("*.csv")
                     if pattern_used.match(f.name) and "_" not in f.name[19:20]]
        candidates = sorted(emb_dir.glob("2*.csv"), key=lambda f: f.stat().st_size, reverse=True)
        if not candidates:
            continue
        n_checked += 1
        name = candidates[0].name
        m_used = pattern_used.match(name)
        m_full = pattern_full.match(name)
        if m_full:
            n_with_subsecond += 1
    print(f"    {n_with_subsecond} / {n_checked} sessions' main EmotiBit filename has a parseable sub-second "
          f"suffix (e.g. '-985608') that emotibit_sync.py's current regex does NOT capture "
          f"(it stops at whole seconds).")
    print(f"    -> up to ~1 s of avoidable start-time quantisation per session, on top of any genuine "
          f"device-clock offset, for every session using this filename-based anchor.")


# ------------------------------------------------------------------
# (c) GPS anchor reliability
# ------------------------------------------------------------------
def check_gps_anchor(sessions: pd.DataFrame) -> None:
    print(f"\n(c) GPS first-sample anchor reliability check:")
    anomalies = []
    for _, r in sessions.iterrows():
        d, s, tag = int(r["driver"]), int(r["session"]), r["tag"]
        path = LABELED_DIR / f"D{d}_S{s}_LABELED.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=["UTC time"], nrows=6)
        tods = df["UTC time"].apply(pl._parse_utc_time).to_numpy()
        diffs = tods[1:] - tods[:-1]
        if len(diffs) == 0:
            continue
        if abs(diffs[0] - 0.04) > 0.02 or diffs[0] <= 0:
            anomalies.append((tag, diffs.tolist()))
    print(f"    {len(anomalies)} / {len(sessions)} sessions have a first-to-second GPS-timestamp gap "
          f"anomalously far from the nominal 0.04 s period:")
    for tag, diffs in anomalies:
        print(f"      {tag}: first 5 inter-sample gaps (s) = {[round(x, 4) for x in diffs]}")
    if anomalies:
        print(f"    -> consistent with the intermittent GPS time-of-day stall mechanism confirmed on D17_S2 "
              f"(label-reproducibility audit); these sessions' absolute-time anchor may itself be off by "
              f"up to the size of the anomalous gap.")


def main():
    sessions = load_sessions()
    print(f"Loaded {len(sessions)} valid sessions.\n")

    if STAGING_DATA_ROOT.exists():
        check_emotibit_calibration(sessions)
        check_filename_precision(sessions)
    else:
        print(f"(a)/(b) skipped: STAGING_DATA_ROOT not found or unset: {STAGING_DATA_ROOT!r}")

    if LABELED_DIR.exists():
        check_gps_anchor(sessions)
    else:
        print(f"(c) skipped: STAGING_LABELED_DIR not found or unset: {LABELED_DIR!r}")


if __name__ == "__main__":
    main()
