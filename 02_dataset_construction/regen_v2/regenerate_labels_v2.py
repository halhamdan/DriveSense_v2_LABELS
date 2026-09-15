"""
Builds the v2-label release as a NEW, separately named tree (nothing in the
v1 release is modified):

    <V2_ROOT>/Preprocessed_Dataset/D{d}/Session_{s}/{tag}_fused.csv
    <V2_ROOT>/data_schema.json, dataset_metadata.json, manifest.csv, README_v2_LABELS.md

Each v2 fused.csv is the v1 file with:
  * `label`          replaced by the version-2 rule (01_annotation/label_harsh_events_v2.py)
  * `label_v1_peak`  = the previous label column, kept for traceability
All other columns are byte-for-byte the v1 values. Front_emotions / Side_pose /
Raw_Dataset are unchanged and are NOT duplicated: the v2 tree is a label-only
delta over v1 and README_v2_LABELS.md says so.

Usage:
    python regenerate_labels_v2.py            # dry run: counts only
    python regenerate_labels_v2.py --apply    # write the v2 tree
Environment:
    DATASET_ROOT   v1 release root
    V2_ROOT        output root (default: sibling "Published_Dataset_Final_v2_LABELS")
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from datetime import date
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "01_annotation"))
from label_harsh_events_v2 import label_fused_v2, count_events, CONFIG_V2  # noqa: E402

V1 = Path(os.environ.get("DATASET_ROOT", "") or r"C:\Users\halha\OneDrive - Durham University\Documents\DriveSense_Packages\3_OLD_versions\data\Published_Dataset_Final_v1_ORIGINAL_20260909")
V2 = Path(os.environ.get("V2_ROOT", "") or (V1.parent / "Published_Dataset_Final_v2_LABELS"))
REPORT = REPO.parent / "regenerated_v3" / "labels_v2_report.csv"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    apply = "--apply" in sys.argv
    files = sorted((V1 / "Preprocessed_Dataset").rglob("*_fused.csv"))
    rows = []; tot = {}
    for f in files:
        tag = f.stem.replace("_fused", "")
        df = pd.read_csv(f)
        v1 = df["label"].copy()
        v2 = label_fused_v2(df)
        c = count_events(v2)
        for k, n in c.items(): tot[k] = tot.get(k, 0) + n
        rows.append({"tag": tag, "rows": len(df), **{f"v2_{k.replace(' ', '_')}": c.get(k, 0) for k in ["Harsh Acceleration", "Harsh Braking", "Harsh Turning"]},
                     "v1_nonnormal_rows": int((v1 != "Normal").sum()), "v2_nonnormal_rows": int((v2 != "Normal").sum())})
        if apply:
            out = df.copy()
            out["label"] = v2.values
            out["label_v1_peak"] = v1.values
            rel = f.relative_to(V1)
            dst = V2 / rel; dst.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(dst, index=False, float_format="%.6f")
        print(f"  {tag}: v2 events {c}", flush=True)
    R = pd.DataFrame(rows); REPORT.parent.mkdir(exist_ok=True); R.to_csv(REPORT, index=False)
    print(f"\nTOTAL v2 events: {tot} | non-Normal rows v1 {R.v1_nonnormal_rows.sum():,} -> v2 {R.v2_nonnormal_rows.sum():,}")
    if not apply:
        print("dry run -- re-run with --apply to write the v2 tree"); return

    # schema
    schema = json.load(open(V1 / "data_schema.json", encoding="utf-8"))
    cols = schema["files"]["{tag}_fused.csv"]["columns"]
    cols["label"] = {"dtype": "string", "unit": None,
                     "description": ("Deterministic harsh-event label, version 2 (final 2026-09-13): Normal | Harsh Acceleration | Harsh Braking | "
                                     "Harsh Turning. Computed from the released columns by 01_annotation/label_harsh_events_v2.py with explicit "
                                     "integer windows at 25 Hz: 5-sample centred running median of the SIGNED lon_acc_g / lat_acc_g, then 13-sample "
                                     "centred mean (k=-6..+6); lateral test on |mean| (not mean|.|); exceedance sustained >= 13 samples (0.52 s), gaps "
                                     "<= 5 samples closed, no dilation; thresholds 0.20 g (acceleration, throttle > 25 %, speed > 15 km/h), 0.35 g "
                                     "(braking, speed > 15 km/h, CAN-speed-derived deceleration <= -0.3 m/s^2), 0.45 g (|lateral|, speed > 30 km/h, "
                                     "and net GNSS heading change >= 5 deg across the run); priority Turning > Braking > Acceleration. Verified on "
                                     "synthetic signals: no single sample, two-sample burst, three-sample burst <= 2.6 g or alternating +/-1 g sequence "
                                     "creates an event. Thresholds calibrated on deliberate-manoeuvre drives in the study vehicle (10/10 brakes, "
                                     "11/11 turns; accelerations peak at 0.21-0.29 g, the vehicle's ceiling). Per-event diagnostics: harsh_events_v2.csv.")}
    cols["label_v1_peak"] = {"dtype": "string", "unit": None,
                             "description": ("Previous (version 1) label, retained for traceability and for continuity with analyses that used "
                                             "it (e.g. CBANet). v1 applied fixed thresholds (0.38 / -0.35 / 0.55 g) to rolling extrema of the "
                                             "UNSMOOTHED GNSS-derived acceleration over a trailing 1 s window, so a single 40 ms spike could "
                                             "create an event: on the released data 78-87 % of v1 events contain at most one consecutive sample "
                                             "above threshold and the CAN-derived deceleration inside v1 'Harsh Braking' is typically -0.08 g. "
                                             "Do not use as a measure of sustained harsh driving; use `label`.")}
    schema["changelog"].append({"date": date.today().isoformat(), "change": (
        "LABELS VERSION 2 (separate release tree Published_Dataset_Final_v2_LABELS): `label` recomputed with a sustained-exceedance rule on "
        "median-filtered, 13-sample-averaged signed acceleration with a heading-change corroboration for turning (see column description); "
        "previous labels kept as `label_v1_peak`. Reason: the acceleration channels are unsmoothed one-sample derivatives of 25 Hz Doppler "
        "speed/heading, and v1's rolling-extremum rule let single-sample spikes create events. A first v2 draft (same day; mean of |lat| over a "
        "12-sample window) was superseded after a robustness audit showed it labelled single spikes and alternating-sign noise as turning. "
        f"Event totals v1 -> v2: {json.dumps(tot)}. No other column changed.")})
    schema["generated"] = date.today().isoformat(); schema["schema_version"] = "2.0"
    json.dump(schema, open(V2 / "data_schema.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    # metadata
    meta = json.load(open(V1 / "dataset_metadata.json", encoding="utf-8"))
    meta["label_version"] = {"version": 2, "rule": CONFIG_V2, "code": "01_annotation/label_harsh_events_v2.py",
                             "event_totals": tot, "previous_label_column": "label_v1_peak"}
    per = {r["tag"]: r for r in rows}
    for s in meta["sessions"]:
        if s["tag"] in per:
            r = per[s["tag"]]
            s["events_v2"] = {"Harsh Acceleration": r["v2_Harsh_Acceleration"], "Harsh Braking": r["v2_Harsh_Braking"], "Harsh Turning": r["v2_Harsh_Turning"]}
    meta["date"] = date.today().isoformat()
    json.dump(meta, open(V2 / "dataset_metadata.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    # manifest: copy v1 rows, re-hash the 79 fused files
    man = pd.read_csv(V1 / "manifest.csv")
    for f in files:
        rel = str(f.relative_to(V1)); dst = V2 / f.relative_to(V1)
        mask = man["relative_path"].str.replace("/", "\\") == rel.replace("/", "\\")
        man.loc[mask, "size_bytes"] = dst.stat().st_size
        man.loc[mask, "rows"] = float(sum(1 for _ in open(dst, "rb")) - 1)
        man.loc[mask, "sha256"] = sha256(dst)
    man.to_csv(V2 / "manifest.csv", index=False)

    (V2 / "README_v2_LABELS.md").write_text(f"""# Published_Dataset_Final_v2_LABELS

Label-only delta over the v1 release (`Published_Dataset_Final`), created {date.today().isoformat()}.

* `Preprocessed_Dataset/**/{{tag}}_fused.csv` -- the 79 v1 files with `label` recomputed by
  `01_annotation/label_harsh_events_v2.py` and the previous labels kept as `label_v1_peak`.
  Every other column is identical to v1.
* `data_schema.json`, `dataset_metadata.json`, `manifest.csv` -- v2 versions (manifest rows for
  files not in this tree refer to the unchanged v1 files).
* NOT duplicated here (unchanged, take from v1): `Raw_Dataset/`, `Front_emotions.csv`, `Side_pose.csv`.

Event totals v1 -> v2: {json.dumps(tot)}.
""", encoding="utf-8")
    print(f"\nv2 tree written to {V2}")


if __name__ == "__main__":
    main()
