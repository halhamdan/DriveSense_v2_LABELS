"""
Builds the VERSION-1-LABEL variant of release v3:

    <OUT> = Published_Dataset_Final_v3_LABELS_v1      (package 2, "latest before the relabelling")

= a complete copy of the v3 tree (videos aligned to telemetry and full-length, D17_S2 truncated at its
logging pause, recomputed hands_on_wheel_proxy, Calibration_Drives/, scr_events_native.csv, fresh manifest)
in which the primary `label` column carries the VERSION-1 (peak-exceedance) labels and the version-2
labels are retained as `label_v2_sustained`. Everything else is byte-identical to v3.

Also writes harsh_events_v1.csv (one row per v1 event with the structural diagnostics that the
manuscript reports: duration, consecutive raw samples above threshold, peak raw acceleration, CAN
deceleration, >1.2 g samples, overlap with a v2 event) and updates data_schema.json,
dataset_metadata.json, README and manifest.csv.

Usage: python build_v1label_variant.py [--skip-copy]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

PK = Path(r"C:\Users\halha\OneDrive - Durham University\Documents\DriveSense_Packages")
V3 = PK / "1_LATEST__labels_v2__release_v3__paper_v6" / "data" / "Published_Dataset_Final_v3_VIDEO_ALIGNED"
OUT = PK / "2_PREVIOUS__labels_v1__release_v1__paper_v5" / "data" / "Published_Dataset_Final_v3_LABELS_v1"
HERE = Path(__file__).resolve().parent
THR = {"Harsh Acceleration": 0.38, "Harsh Braking": 0.35, "Harsh Turning": 0.55}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def runs(mask):
    m = np.asarray(mask, bool); d = np.diff(np.r_[0, m.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1).tolist(), (np.flatnonzero(d == -1) - 1).tolist()))


def events(lab):
    out = []
    for a, b in runs(lab != "Normal"):
        i = a
        while i <= b:
            k = i
            while k + 1 <= b and lab[k + 1] == lab[i]:
                k += 1
            out.append((lab[i], i, k)); i = k + 1
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--skip-copy", action="store_true"); a = ap.parse_args()
    today = date.today().isoformat()
    if not a.skip_copy:
        OUT.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["robocopy", str(V3), str(OUT), "/E", "/MT:16", "/NFL", "/NDL", "/NJH", "/NP", "/R:2", "/W:2"], capture_output=True, text=True)
        print("robocopy exit", r.returncode, "(0-7 = ok)"); print(r.stdout[-800:])
    # ---- fused files: swap label roles, build the v1 event table
    rows = []; tot = {"Harsh Acceleration": 0, "Harsh Braking": 0, "Harsh Turning": 0}; per = {}
    for f in sorted((OUT / "Preprocessed_Dataset").rglob("*_fused.csv")):
        tag = f.stem.replace("_fused", ""); df = pd.read_csv(f)
        v2 = df["label"].to_numpy(dtype=object); v1 = df["label_v1_peak"].to_numpy(dtype=object)
        df["label"] = v1; df["label_v2_sustained"] = v2; df = df.drop(columns=["label_v1_peak"])
        df.to_csv(f, index=False, float_format="%.6f")
        t = df.elapsed_s.to_numpy(); lat = df.lat_acc_g.to_numpy(); lon = df.lon_acc_g.to_numpy()
        a_can = np.r_[np.nan, np.diff(df.speed_kph.to_numpy() / 3.6) * 25]
        big = (np.abs(lat) > 1.2) | (np.abs(lon) > 1.2); c = {"Harsh Acceleration": 0, "Harsh Braking": 0, "Harsh Turning": 0}
        for cls, i, k in events(v1):
            c[cls] += 1; g = slice(i, k + 1); raw = np.abs(lat[g]) if cls == "Harsh Turning" else (lon[g] if cls == "Harsh Acceleration" else -lon[g])
            above = raw >= THR[cls]; maxrun = max([b - a_ + 1 for a_, b in runs(above)], default=0)
            rows.append({"tag": tag, "class": cls, "start_s": round(float(t[i]), 2), "end_s": round(float(t[k]), 2), "duration_s": round((k - i + 1) / 25, 2),
                         "n_samples": k - i + 1, "n_raw_above_thr": int(above.sum()), "max_consecutive_above_thr": maxrun,
                         "peak_raw_g": round(float(raw.max()), 3), "n_raw_gt_1p2g": int(big[g].sum()),
                         "can_accel_min_ms2": round(float(np.nanmin(a_can[g])), 2) if np.isfinite(a_can[g]).any() else np.nan,
                         "can_accel_mean_ms2": round(float(np.nanmean(a_can[g])), 2) if np.isfinite(a_can[g]).any() else np.nan,
                         "overlaps_v2_event": bool((v2[g] != "Normal").any())})
        for k2 in tot: tot[k2] += c[k2]
        per[tag] = c; print(f"  {tag}: {c}", flush=True)
    E = pd.DataFrame(rows); E.to_csv(OUT / "harsh_events_v1.csv", index=False)
    print("v1 events on the v3 timeline:", tot, "| single-sample fraction:", {k: round(float((E[E["class"] == k].max_consecutive_above_thr <= 1).mean()), 3) for k in tot})
    # ---- schema
    sch = json.load(open(OUT / "data_schema.json", encoding="utf-8")); key = next(k for k in sch["files"] if "fused" in k); cols = sch["files"][key]["columns"]
    v2desc = cols["label"]["description"]; v1desc = cols.pop("label_v1_peak")["description"]
    cols["label"] = {"dtype": "string", "unit": None, "description": "VERSION-1 harsh-event label (primary column of this variant). " + v1desc.replace("Previous (version 1) label, retained for traceability and for continuity with analyses that used it (e.g. CBANet). ", "")}
    cols["label_v2_sustained"] = {"dtype": "string", "unit": None, "description": "Version-2 (sustained-exceedance) label, retained for comparison. " + v2desc}
    sch["schema_version"] = "3.0-L1"; sch["generated"] = today
    sch["changelog"].append({"date": today, "change": f"VERSION-1-LABEL VARIANT of release v3 (Published_Dataset_Final_v3_LABELS_v1): identical to v3 except that `label` carries the version-1 peak-exceedance labels and the version-2 labels are kept as `label_v2_sustained`; harsh_events_v1.csv added. v1 event totals on the v3 timeline: {json.dumps(tot)}."})
    sch["files"]["harsh_events_v1.csv"] = {"description": "One row per version-1 event: tag, class, start_s, end_s, duration_s, n_samples, n_raw_above_thr, max_consecutive_above_thr (longest run of consecutive raw samples above the class threshold; 1 = single-sample event), peak_raw_g, n_raw_gt_1p2g, can_accel_min/mean_ms2, overlaps_v2_event.", "columns": {}}
    json.dump(sch, open(OUT / "data_schema.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    # ---- metadata
    meta = json.load(open(OUT / "dataset_metadata.json", encoding="utf-8"))
    meta["release_version"] = "3-L1"; meta["date"] = today
    meta["label_version_v2"] = meta.pop("label_version", None)
    meta["label_version"] = {"version": 1, "rule": "01_annotation/label_harsh_events.py (rolling 1 s extrema of the raw GNSS-derived acceleration; thresholds 0.38 / -0.35 / 0.55 g; 0.4 s min duration; dilation 0.5/1.0 s)",
                             "event_totals_on_v3_timeline": tot, "retained_v2_column": "label_v2_sustained", "event_table": "harsh_events_v1.csv",
                             "caveat": "78-87 % of these events contain at most one consecutive raw sample above threshold; see harsh_events_v1.csv max_consecutive_above_thr"}
    for s in meta["sessions"]:
        if s["tag"] in per: s["events_v1"] = per[s["tag"]]
    json.dump(meta, open(OUT / "dataset_metadata.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    # ---- README
    rd = OUT / "README_v3_LABELS_v1.md"; old = OUT / "README_v3_VIDEO_ALIGNED.md"
    txt = f"""# Published_Dataset_Final_v3_LABELS_v1 — release v3 with VERSION-1 labels   ({today})

Identical to `Published_Dataset_Final_v3_VIDEO_ALIGNED` (release v3: videos aligned to the telemetry and spanning the full session,
D17_S2 truncated at its 123 s logging pause, recomputed `hands_on_wheel_proxy`, `Calibration_Drives/`, `scr_events_native.csv`),
except that in every `fused.csv` the primary `label` column carries the **version-1 peak-exceedance labels** and the version-2
sustained-exceedance labels are kept as `label_v2_sustained`. `harsh_events_v1.csv` lists every v1 event with its structural
diagnostics; `harsh_events_v2.csv` is the v2 table. Caveat carried from the audits: 78-87 % of v1 events rest on a single raw
sample above threshold (GNSS-derivative spikes). v1 event totals on this timeline: {json.dumps(tot)}.

"""
    rd.write_text(txt + (old.read_text(encoding="utf-8") if old.exists() else ""), encoding="utf-8")
    if old.exists(): old.unlink()
    # ---- manifest
    mrows = []
    for p in sorted(OUT.rglob("*")):
        if not p.is_file() or p.name == "manifest.csv": continue
        rel = str(p.relative_to(OUT)).replace("\\", "/"); tier = "raw" if rel.startswith("Raw_Dataset/") else ("processed" if rel.startswith("Preprocessed_Dataset/") else "release")
        n = (sum(1 for _ in open(p, "rb")) - 1) if p.suffix.lower() == ".csv" else ""
        mrows.append({"tier": tier, "relative_path": rel, "size_bytes": p.stat().st_size, "rows": n, "sha256": sha256(p)})
    pd.DataFrame(mrows).to_csv(OUT / "manifest.csv", index=False); print("manifest", len(mrows), "files; done ->", OUT)


if __name__ == "__main__":
    main()
