"""
Adds per-session provenance to the released dataset_metadata.json:

  sessions[*].sync        -- EmotiBit-to-VBOX alignment actually applied to the
                             released fused.csv (from the regeneration report):
                             method, median offset, largest sustained offset step
                             (the per-session bound on residual alignment error),
                             number of sync pings, and -- diagnostic only -- the
                             slope a linear fit would have reported.
  sessions[*].exceptions  -- short machine-readable tags for every manual
                             exception affecting that session (full prose in
                             MANUAL_EXCEPTIONS.md).
  manual_exceptions_doc   -- pointer to MANUAL_EXCEPTIONS.md
  date                    -- bumped

Backs up the existing metadata file first. Idempotent.

Usage:
    python add_session_provenance.py            # dry run: prints what would change
    python add_session_provenance.py --apply

Environment variables:
    DATASET_ROOT    release root containing dataset_metadata.json
    REGEN_REPORT    regeneration report CSV (default: ../../regenerated_v3/regeneration_report.csv)
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", "") or
                    r"C:\Users\halha\OneDrive - Durham University\Documents\Published_Dataset_Final")
REGEN_REPORT = Path(os.environ.get("REGEN_REPORT", "") or
                    (Path(__file__).resolve().parents[2] / "regenerated_v3" / "regeneration_report.csv"))
META = DATASET_ROOT / "dataset_metadata.json"

# Tags are deliberately terse; MANUAL_EXCEPTIONS.md carries the reasons.
EXCEPTIONS = {
    "D6_S2":  ["session_excluded_insufficient_overlap"],
    "D6_S3":  ["side_camera_disconnected"],
    "D10_S3": ["side_camera_disconnected"],
    "D16_S1": ["side_camera_disconnected"],
    "D2_S3":  ["side_video_and_pose_removed_camera_unmounted", "route_partial_6km_congestion"],
    "D11_S3": ["side_video_withheld_deidentification_failure_pose_retained"],
    "D18_S4": ["side_video_face_exposure_corrected_2_intervals"],
    "D4_S2":  ["front_video_face_exposure_corrected_3_intervals"],
    "D10_S1": ["front_video_face_exposure_corrected", "gnss_altitude_excursion_disclosed"],
    "D15_S1": ["front_video_face_exposure_corrected"],
    "D9_S3":  ["side_video_face_exposure_corrected", "regenerated_2026-08-22_time_reference_fix"],
    "D1_S1":  ["two_raw_emotibit_recordings_largest_used"],
    "D3_S2":  ["two_raw_emotibit_recordings_largest_used"],
    "D4_S1":  ["two_raw_emotibit_recordings_largest_used"],
    "D16_S3": ["two_raw_emotibit_recordings_largest_used"],
    "D1_S4":  ["vbox_first_sample_gps_time_gap_disclosed"],
    "D12_S2": ["vbox_first_sample_gps_time_gap_disclosed"],
    "D17_S2": ["noisier_telemetry_28g_glitch_despiked"],
    "D12_S1": ["eda_out_of_range_whole_session"],
    "D7_S1":  ["midsession_stop_not_trimmed"],
    "D8_S3":  ["midsession_stop_not_trimmed"],
    "D20_S3": ["midsession_stop_not_trimmed"],
}


def main():
    apply = "--apply" in sys.argv
    meta = json.load(open(META, encoding="utf-8"))
    rep = pd.read_csv(REGEN_REPORT).set_index("tag")
    if len(rep) != 79 or (rep["status"] != "ok").any():
        raise SystemExit("regeneration report is not a clean 79-session report")

    n_sync, n_exc = 0, 0
    for s in meta["sessions"]:
        tag = s["tag"]
        if tag in rep.index:
            r = rep.loc[tag]
            s["sync"] = {
                "method": str(r["sync_method"]),
                "description": ("EmotiBit device clock mapped to UTC with slope fixed at 1 and the "
                                "session-median offset over all logged sync pings; max_step_s is the "
                                "largest sustained offset step in the ping log and bounds the residual "
                                "cross-device alignment error for this session"),
                "median_offset_s": round(float(r["median_offset_s"]), 4),
                "max_step_s": round(float(r["max_step_s"]), 3),
                "step_at_s": (None if pd.isna(r["step_at_s"]) else round(float(r["step_at_s"]), 1)),
                "n_pings_used": int(r["n_pings_used"]),
                "calibration_span_s": round(float(r["calibration_span_s"]), 1),
                "linear_fit_slope_ppm_diagnostic": round(float(r["linear_slope_ppm"]), 1),
            }
            n_sync += 1
        exc = EXCEPTIONS.get(tag, [])
        if s.get("status") != "ok" and "session_excluded_insufficient_overlap" not in exc:
            exc = exc + ["session_excluded_insufficient_overlap"]
        s["exceptions"] = exc
        n_exc += bool(exc)

    meta["manual_exceptions_doc"] = "DriveSense/MANUAL_EXCEPTIONS.md (code repository)"
    meta["sync_model"] = ("anchor_only_median (2026-09-09): see sessions[*].sync; previous fused files "
                          "preserved locally in Preprocessed_Dataset_PRE_EMOTIBIT_SYNC_FIX_BACKUP/ and "
                          "Preprocessed_Dataset_PRE_ANCHOR_ONLY_FIX_BACKUP/")
    meta["date"] = datetime.now(timezone.utc).isoformat()

    print(f"sessions with sync block: {n_sync} | sessions with >=1 exception tag: {n_exc}")
    if not apply:
        print("dry run -- re-run with --apply to write")
        return
    backup = META.with_name(f"dataset_metadata_PRE_PROVENANCE_BACKUP_{datetime.now():%Y%m%d}.json")
    if not backup.exists():
        shutil.copy2(META, backup)
    json.dump(meta, open(META, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"written {META}\nbackup {backup}")


if __name__ == "__main__":
    main()
