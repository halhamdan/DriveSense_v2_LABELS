"""
Registers the release-level extras of the v2 label release in its descriptive
files (run after regenerate_labels_v2.py --apply, make_calibration_table.py,
audit_label_robustness_v2.py events and make_scr_native_event_table.py):

  harsh_events_v2.csv                      per-event table (176 rows)
  scr_events_native.csv                    native EmotiBit SCR events on the released timeline
  Calibration_Drives/*.csv                 three native VBOX exports of the calibration drives
                                           + calibration_manoeuvres.csv + calibration_drives.csv

Adds a `files` entry with column descriptions to data_schema.json, a manifest.csv
row (tier "release", size, rows, sha256) per file, and a `release_extras` note to
dataset_metadata.json. Idempotent. Environment: V2_ROOT.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path

import pandas as pd

V2 = Path(os.environ.get("V2_ROOT", "") or r"C:\Users\halha\OneDrive - Durham University\Documents\Published_Dataset_Final_v2_LABELS")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


EXTRA_SCHEMA = {
    "harsh_events_v2.csv": {
        "description": "One row per version-2 harsh event in the release (05_technical_validation/audit_label_robustness_v2.py events). Reference for the class sizes in Table 4 and the per-event diagnostics discussed in Technical Validation.",
        "columns": {
            "tag": "session D{driver}_S{session}", "class": "Harsh Acceleration | Harsh Braking | Harsh Turning",
            "start_s": "event start, elapsed_s of the fused file", "end_s": "event end (inclusive sample), elapsed_s", "duration_s": "n_samples / 25",
            "n_samples": "samples labelled with the class in this run", "peak_smoothed_g": "largest smoothed (median-5 then 13-sample mean) acceleration inside the event, g",
            "mean_speed_kmh": "mean GNSS speed inside the event", "n_raw_above_thr": "raw samples inside the event whose |acceleration| exceeds the class threshold",
            "frac_raw_above_half_thr": "fraction of raw samples above half the threshold", "max_raw_g": "largest raw |acceleration| inside the event",
            "n_raw_gt_1p2g": "raw samples inside the event beyond +/-1.2 g (implausible-magnitude class, Technical Validation)",
            "rectification_ratio": "|mean(signed raw)| / mean(|raw|) over the event; ~1 for a consistent-sign excursion, ~0 for alternating noise",
            "heading_change_deg": "net GNSS heading change from 3 samples before to 3 samples after the event (wrapped)",
            "heading_change_expected_deg": "heading change implied by the smoothed lateral acceleration and speed over the event",
            "clip_1p2g_outcome": "what happens to the event when raw |acceleration| is clipped to 1.2 g and the rule re-run: unchanged | boundary N samples | disappears | class change | split | merge",
            "present_at_plus10pct": "an event of the same class overlaps this one when all thresholds are raised 10 %",
            "present_at_minus10pct": "... when all thresholds are lowered 10 %", "v1_overlap": "overlaps a non-Normal label_v1_peak run",
        }},
    "scr_events_native.csv": {
        "description": "The EmotiBit's own skin-conductance-response events (one row per device event; SA amplitude and SR rise time) for every valid session, converted to the released absolute timeline with the same anchor-only model as fused.csv and restricted to the released session window (05_technical_validation/make_scr_native_event_table.py). This is the native event table from which the 25 Hz SCR_AMPLITUDE / SCR_RISE_TIME columns are derived; use it for event-level analyses.",
        "columns": {"tag": "session", "unix_time": "event time, s (released timeline)", "elapsed_s": "seconds from the trimmed session start",
                    "SCR_AMPLITUDE": "device-reported onset-to-peak amplitude, uS", "SCR_RISE_TIME": "device-reported rise time, s"}},
    "Calibration_Drives/calibration_drive{n}_VBOX_native.csv": {
        "description": "Native VBOX HD2 exports (25 Hz) of the three threshold-calibration drives in the study vehicle (corresponding author driving). No CAN channels were logged on these drives; no video, no physiology. Used to choose the version-2 thresholds -- calibration, not independent validation (Technical Validation, 'Threshold calibration and manoeuvre recovery').",
        "columns": {"UTC time": "HHMMSS.ss", "Speed (km/h)": "GNSS Doppler speed", "Heading (Degrees)": "GNSS heading", "Latitude/Longitude": "degrees-minutes, N/E",
                    "Lateral acceleration (g)": "GNSS-derived (speed x heading rate)", "Longitudinal acceleration (g)": "GNSS-derived (one-sample speed difference)", "Elapsed time (s)": "from first row"}},
    "Calibration_Drives/calibration_manoeuvres.csv": {
        "description": "Events labelled on the calibration drives by the released rule (GNSS speed substituted for CAN speed; throttle gate disabled): drive, class, start_s, end_s, duration_s, peak_smoothed_g, mean_speed_kmh -- the manoeuvre reference intervals.", "columns": {}},
    "Calibration_Drives/calibration_drives.csv": {
        "description": "One row per calibration drive: date, UTC start, duration, manoeuvres performed, labelled counts per class, largest smoothed lateral acceleration outside turning events, rule modifications.", "columns": {}},
}


def main():
    schema = json.load(open(V2 / "data_schema.json", encoding="utf-8"))
    schema.setdefault("files", {}).update(EXTRA_SCHEMA)
    note = f"Release-level extras registered {date.today().isoformat()}: harsh_events_v2.csv, scr_events_native.csv, Calibration_Drives/ (see files)."
    if not any(note[:40] in c.get("change", "") for c in schema.get("changelog", [])):
        schema["changelog"].append({"date": date.today().isoformat(), "change": note})
    json.dump(schema, open(V2 / "data_schema.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    man = pd.read_csv(V2 / "manifest.csv")
    files = [V2 / "harsh_events_v2.csv", V2 / "scr_events_native.csv"] + sorted((V2 / "Calibration_Drives").glob("*.csv"))
    rows = []
    for f in files:
        if not f.exists():
            print("missing:", f); continue
        rel = str(f.relative_to(V2)).replace("\\", "/")
        man = man[man.relative_path.str.replace("\\", "/") != rel]
        rows.append({"tier": "release", "relative_path": rel, "size_bytes": f.stat().st_size, "rows": sum(1 for _ in open(f, "rb")) - 1, "sha256": sha256(f)})
    extra = pd.DataFrame(rows)
    for c in man.columns:
        if c not in extra.columns: extra[c] = ""
    man = pd.concat([man, extra[man.columns]], ignore_index=True); man.to_csv(V2 / "manifest.csv", index=False)

    meta = json.load(open(V2 / "dataset_metadata.json", encoding="utf-8"))
    meta["release_extras"] = {"harsh_events_v2.csv": "per-event table, version-2 labels", "scr_events_native.csv": "native EmotiBit SCR events on the released timeline",
                              "Calibration_Drives/": "threshold-calibration drives (native VBOX exports + manoeuvre intervals); calibration, not independent validation",
                              "registered": date.today().isoformat()}
    json.dump(meta, open(V2 / "dataset_metadata.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"registered {len(rows)} extra files; manifest now {len(man)} rows")
    print(extra[["relative_path", "rows", "size_bytes"]].to_string(index=False))


if __name__ == "__main__":
    main()
