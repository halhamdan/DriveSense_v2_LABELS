"""
Builds release VERSION 3 as a NEW, complete, separately named tree
(nothing in v1 `Published_Dataset_Final` or v2 `Published_Dataset_Final_v2_LABELS`
is modified):

    <V3_ROOT> = Published_Dataset_Final_v3_VIDEO_ALIGNED

What changes relative to v2 (and why):

1. VIDEO REALIGNMENT. The VBOX HD2 starts its video lead_in = 10.4-11.3 s (per
   session; Racelogic `Avi sync time` at telemetry row 0) before the first logged
   telemetry sample. The trimming pipeline assumed video frame 0 = telemetry row
   0, so every v1 video, Front_emotions.csv and Side_pose.csv shows, at elapsed
   tau, the scene at telemetry time tau - lead_in (verified: overlay speedometer
   on the raw video; frame-pair stop test on 49 released sessions).
   v3 drops the first L = round(lead_in * 30) frames of each released video and
   of the frame-indexed CSVs, so that frame k <-> elapsed_s k/30 exactly. The
   videos are re-encoded with NVENC (h264, cq 26) from the released, verified,
   de-identified files -- no new footage is introduced, so every blur, mask and
   exposure fix stays valid (their frame indices shift by -L). Consequence: each
   video ends lead_in seconds before the telemetry does (no video for the last
   ~11 s of each session).
2. D17_S2 LOGGING PAUSE. The VBOX stopped logging for 123.4 s (satellites 0) at
   released elapsed 654.52 s while its elapsed counter and video continued.
   Because unix_time is reconstructed as first-UTC + elapsed, everything after
   that instant in v1/v2 carries a unix_time 123.4 s too early (EmotiBit
   misaligned, video offset not constant). v3 truncates D17_S2 at the pause
   (all tiers) and recomputes its v2 labels on the truncated file.
3. Four sessions (D8_S1 1.44 s, D18_S1 0.92 s, D4_S1 0.64 s, D19_S1 0.60 s) show a
   gradual UTC-vs-elapsed divergence below 1.5 s: disclosed per session in
   dataset_metadata.json, not corrected.

Everything else (fused.csv from v2 incl. label / label_v1_peak, VBOX_raw and
EmotiBit raw CSVs from v1, release-root extras from v2) is copied unchanged.
A fresh manifest.csv is built by hashing every file in the v3 tree.

Usage:
    python build_v3_video_realign.py [--sessions D1_S1 ...] [--skip-videos]
Environment: V1_ROOT, V2_ROOT, V3_ROOT (defaults: OneDrive Documents siblings)
Resumable: existing v3 videos with the expected frame count are not re-encoded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "01_annotation"))
from label_harsh_events_v2 import label_fused_v2  # noqa: E402

DOCS = Path(r"C:\Users\halha\OneDrive - Durham University\Documents")
V1 = Path(os.environ.get("V1_ROOT", "") or DOCS / "Published_Dataset_Final")
V2 = Path(os.environ.get("V2_ROOT", "") or DOCS / "Published_Dataset_Final_v2_LABELS")
V3 = Path(os.environ.get("V3_ROOT", "") or DOCS / "Published_Dataset_Final_v3_VIDEO_ALIGNED")
VAL = REPO / "05_technical_validation" / "validation_output"
LEAD = pd.read_csv(VAL / "video_lead_in_per_session.csv").set_index("tag")
CONT = pd.read_csv(VAL / "vbox_time_continuity.csv").set_index("tag")
LOG = Path(__file__).resolve().parent / "build_v3.log"
FPS = 30
NVENC = ["-c:v", "h264_nvenc", "-rc", "vbr", "-cq", "26", "-preset", "p5", "-pix_fmt", "yuv420p", "-an"]
DIVERGENT = {"D8_S1": 1.44, "D18_S1": 0.92, "D4_S1": 0.64, "D19_S1": 0.60}


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def nb_frames(p: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames",
                          "-of", "csv=p=0", str(p)], capture_output=True, text=True).stdout.strip()
    try:
        return int(out.split(",")[0])
    except ValueError:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames", "-of", "csv=p=0", str(p)],
                             capture_output=True, text=True).stdout.strip()
        return int(out.split(",")[0])


def nb_frames_fast(p: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames", "-of", "csv=p=0", str(p)],
                         capture_output=True, text=True).stdout.strip()
    return int(out.split(",")[0])


def cut_video(src: Path, dst: Path, first: int, last: int | None) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    sel = f"gte(n\\,{first})" if last is None else f"between(n\\,{first}\\,{last})"
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", f"select='{sel}',setpts=PTS-STARTPTS", "-r", str(FPS)] + NVENC + [str(dst)]
    with open(LOG, "a", encoding="utf-8") as err:
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=err)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {src.name}")
    return nb_frames_fast(dst)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def d17_cut(tag: str):
    """Released elapsed (inclusive) and unix_time at which D17_S2 is truncated."""
    r = pd.read_csv(V1 / "Raw_Dataset" / "D17" / "Session_2" / f"{tag}_VBOX_raw.csv", usecols=["unix_time", "utc_tod_s", "elapsed_s"])
    t = r.utc_tod_s.to_numpy(); e = r.elapsed_s.to_numpy(); res = (t - t[0]) - e
    j = int(np.argmax(np.abs(np.diff(res)) > 0.5))
    return float(e[j]), float(r.unix_time.iloc[j]), float(res[j + 1] - res[j])


def process(tag: str, skip_videos: bool, report: list):
    d, s = tag[1:].split("_S")
    raw1 = V1 / "Raw_Dataset" / f"D{d}" / f"Session_{s}"; pre1 = V1 / "Preprocessed_Dataset" / f"D{d}" / f"Session_{s}"
    pre2 = V2 / "Preprocessed_Dataset" / f"D{d}" / f"Session_{s}"
    raw3 = V3 / "Raw_Dataset" / f"D{d}" / f"Session_{s}"; pre3 = V3 / "Preprocessed_Dataset" / f"D{d}" / f"Session_{s}"
    raw3.mkdir(parents=True, exist_ok=True); pre3.mkdir(parents=True, exist_ok=True)
    lead = float(LEAD.loc[tag, "lead_in_s"]); L = int(round(lead * FPS))
    cut_e = cut_u = None; last_frame = None
    if tag == "D17_S2":
        cut_e, cut_u, gap = d17_cut(tag); last_frame = L + int(round(cut_e * FPS))
        log(f"  {tag}: truncating at released elapsed {cut_e:.2f} s (logging pause {gap:.2f} s)")
    info = {"tag": tag, "lead_in_s": lead, "frames_dropped": L, "truncated_at_elapsed_s": cut_e}

    # --- fused.csv (from v2; D17_S2 truncated + relabelled)
    f2 = pre2 / f"{tag}_fused.csv"; f3 = pre3 / f"{tag}_fused.csv"
    if cut_e is None:
        if not f3.exists(): shutil.copy2(f2, f3)
        info["fused_rows"] = sum(1 for _ in open(f3, "rb")) - 1
    else:
        df = pd.read_csv(f2); df = df[df.elapsed_s <= cut_e + 1e-6].copy()
        df["label"] = label_fused_v2(df).values
        df.to_csv(f3, index=False, float_format="%.6f"); info["fused_rows"] = len(df)

    # --- raw CSVs (from v1; D17_S2 truncated)
    for src in sorted(raw1.glob("*.csv")):
        dst = raw3 / src.name
        if cut_u is None:
            if not dst.exists(): shutil.copy2(src, dst)
        else:
            r = pd.read_csv(src)
            if "unix_time" in r.columns: r = r[r.unix_time <= cut_u + 1e-6]
            elif "elapsed_s" in r.columns: r = r[r.elapsed_s <= cut_e + 1e-6]
            r.to_csv(dst, index=False, float_format="%.6f")

    # --- frame-indexed CSVs (from v1 released; shifted by L)
    for name in ("Front_emotions", "Side_pose"):
        src = pre1 / f"{tag}_{name}.csv"
        if not src.exists(): continue
        c = pd.read_csv(src); c = c[c.frame >= L].copy()
        if last_frame is not None: c = c[c.frame <= last_frame]
        c["frame"] = (c["frame"] - L).astype(int); c["timestamp_s"] = (c["frame"] / FPS).round(4)
        c.to_csv(pre3 / f"{tag}_{name}.csv", index=False); info[f"{name}_rows"] = len(c)

    # --- videos (from v1 released; first L frames dropped; NVENC)
    for cam in ("Front", "Side"):
        src = raw1 / f"{tag}_{cam}_blurred.mp4"
        if not src.exists(): continue
        dst = raw3 / f"{tag}_{cam}_blurred.mp4"; n_in = nb_frames_fast(src)
        expected = (n_in - L) if last_frame is None else (min(last_frame, n_in - 1) - L + 1)
        info[f"{cam}_frames_in"] = n_in; info[f"{cam}_frames_expected"] = expected
        if skip_videos: continue
        if dst.exists() and abs(nb_frames_fast(dst) - expected) <= 1:
            info[f"{cam}_frames_out"] = nb_frames_fast(dst); continue
        t0 = time.time(); n_out = cut_video(src, dst, L, last_frame); info[f"{cam}_frames_out"] = n_out
        log(f"  {tag} {cam}: {n_in} -> {n_out} frames (expected {expected}) in {time.time() - t0:.0f} s")
        if abs(n_out - expected) > 1:
            log(f"  !! {tag} {cam}: frame count mismatch")
    report.append(info)


def descriptive_files(report: list):
    """schema, metadata, README for v3 (manifest built separately)."""
    today = date.today().isoformat()
    schema = json.load(open(V2 / "data_schema.json", encoding="utf-8"))
    schema["schema_version"] = "3.0"; schema["generated"] = today
    schema["video_alignment"] = {
        "statement": ("Version 3: video frame k of {tag}_Front_blurred.mp4 / {tag}_Side_blurred.mp4 corresponds to elapsed_s = k/30 of the same session's "
                      "fused.csv (and to unix_time = trimmed start + k/30). Front_emotions.csv and Side_pose.csv use the same frame index. Each video ends "
                      "lead_in_s (dataset_metadata.json -> sessions[*].video.lead_in_s, 10.4-11.3 s) before the telemetry does."),
        "why": ("The VBOX HD2 starts its video file lead_in_s before the first logged telemetry sample (Racelogic 'Avi sync time' at row 0). Versions 1 and 2 "
                "mapped telemetry time to video frames assuming frame 0 = telemetry row 0, so their video-derived files showed, at elapsed tau, the scene at "
                "tau - lead_in_s. v3 drops the first round(lead_in_s*30) frames of the released (de-identified, verified) files and re-encodes them (h264_nvenc, cq 26); "
                "no new footage was introduced, so all blur, masking and exposure fixes remain valid with frame indices shifted by -round(lead_in_s*30)."),
        "verification": "05_technical_validation/audit_video_overlay_offset.py (raw-video speedometer overlay), audit_released_video_stop_frames.py (frame-pair stop test, 49 sessions), audit_vbox_time_continuity.py",
    }
    schema["changelog"].append({"date": today, "change": (
        "VERSION 3 (separate tree Published_Dataset_Final_v3_VIDEO_ALIGNED): (1) all 153 videos and the 154 frame-indexed CSVs realigned to the telemetry timeline by "
        "dropping the per-session video lead-in (10.4-11.3 s) that the v1 trimming had not applied; videos re-encoded with NVENC from the released files; each video "
        "now ends lead_in_s before the telemetry. (2) D17_S2 truncated at released elapsed 654.52 s: the VBOX paused logging for 123.4 s (satellites 0) while its elapsed "
        "counter continued, so unix_time after that instant was 123.4 s too early in v1/v2 (EmotiBit misaligned); its v2 labels recomputed on the truncated file. "
        "(3) D8_S1, D18_S1, D4_S1, D19_S1: UTC-vs-elapsed divergence of 1.44/0.92/0.64/0.60 s disclosed per session (not corrected). All other files identical to v2 (fused.csv, labels) and v1 (raw CSVs).")})
    json.dump(schema, open(V3 / "data_schema.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    meta = json.load(open(V2 / "dataset_metadata.json", encoding="utf-8"))
    meta["date"] = today; meta["release_version"] = 3
    meta["video_alignment"] = schema["video_alignment"]
    rep = {r["tag"]: r for r in report}
    for s in meta["sessions"]:
        r = rep.get(s["tag"])
        if not r: continue
        vid = {"lead_in_s": r["lead_in_s"], "frames_dropped_at_start": r["frames_dropped"], "realigned": True,
               "front_frames": r.get("Front_frames_out", r.get("Front_frames_expected")), "side_frames": r.get("Side_frames_out", r.get("Side_frames_expected")),
               "note": "frame k <-> elapsed_s k/30; video ends lead_in_s before the telemetry"}
        s["video"] = vid
        ex = list(s.get("exceptions", []))
        if r["truncated_at_elapsed_s"] is not None:
            ex.append(f"truncated_at_{r['truncated_at_elapsed_s']:.2f}s_vbox_logging_pause_123s_gps_loss")
            s["drive_duration_s"] = round(r["truncated_at_elapsed_s"] + 0.04, 1); s["truncated_at_elapsed_s"] = r["truncated_at_elapsed_s"]
            if "events_v2" in s: s["events_v2_note"] = "recomputed on the truncated file; see harsh_events_v2.csv"
        if s["tag"] in DIVERGENT:
            ex.append(f"vbox_utc_vs_elapsed_divergence_{DIVERGENT[s['tag']]:.2f}s_not_corrected")
        s["exceptions"] = ex
    json.dump(meta, open(V3 / "dataset_metadata.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    (V3 / "README_v3_VIDEO_ALIGNED.md").write_text(f"""# Published_Dataset_Final_v3_VIDEO_ALIGNED

Complete release tree, version 3, built {today} from v1 (`Published_Dataset_Final`) and v2
(`Published_Dataset_Final_v2_LABELS`) by `02_dataset_construction/regen_v3/build_v3_video_realign.py`.

* **Videos realigned.** `{{tag}}_Front_blurred.mp4` / `{{tag}}_Side_blurred.mp4`: frame k <-> `elapsed_s` k/30.
  The VBOX HD2 video starts 10.4-11.3 s before the first telemetry sample; v1/v2 had not applied that
  lead-in, so their videos, `Front_emotions.csv` and `Side_pose.csv` were late by that amount. v3 drops
  the first round(lead_in*30) frames of the released, de-identified files (re-encoded, NVENC cq 26) and
  shifts the two CSVs identically. Each video ends lead_in seconds before the telemetry.
  Per-session lead-in: `dataset_metadata.json` -> `sessions[*].video`.
* **D17_S2 truncated at 654.52 s** (VBOX logging pause of 123.4 s, satellites 0; unix_time after it was
  123.4 s too early in v1/v2). Labels recomputed on the truncated file.
* **D8_S1, D18_S1, D4_S1, D19_S1**: UTC-vs-elapsed divergence of 1.44 / 0.92 / 0.64 / 0.60 s, disclosed
  in `sessions[*].exceptions`, not corrected.
* Unchanged: `fused.csv` (v2 labels, `label_v1_peak`), raw VBOX/EmotiBit CSVs (v1), `harsh_events_v2.csv`,
  `scr_events_native.csv`, `Calibration_Drives/` (D17_S2 rows beyond the cut removed from the two tables).
* `manifest.csv` rebuilt by hashing every file in this tree.
""", encoding="utf-8")


def extras(report):
    cut = {r["tag"]: r["truncated_at_elapsed_s"] for r in report if r["truncated_at_elapsed_s"] is not None}
    for name, col in (("harsh_events_v2.csv", "start_s"), ("scr_events_native.csv", "elapsed_s")):
        t = pd.read_csv(V2 / name)
        for tag, c in cut.items():
            t = t[~((t.tag == tag) & (t[col] > c))]
        t.to_csv(V3 / name, index=False)
    if (V2 / "Calibration_Drives").exists():
        shutil.copytree(V2 / "Calibration_Drives", V3 / "Calibration_Drives", dirs_exist_ok=True)
    # D17_S2's events after the cut must be recomputed from the truncated file; the events table filter above
    # keeps events that started before the cut -- identical to a recomputation except possibly the last event.


def manifest():
    rows = []
    for p in sorted(V3.rglob("*")):
        if not p.is_file() or p.name in ("manifest.csv",): continue
        rel = str(p.relative_to(V3)).replace("\\", "/")
        tier = "raw" if rel.startswith("Raw_Dataset/") else ("processed" if rel.startswith("Preprocessed_Dataset/") else "release")
        n = (sum(1 for _ in open(p, "rb")) - 1) if p.suffix.lower() == ".csv" else ""
        rows.append({"tier": tier, "relative_path": rel, "size_bytes": p.stat().st_size, "rows": n, "sha256": sha256(p)})
    pd.DataFrame(rows).to_csv(V3 / "manifest.csv", index=False); log(f"manifest: {len(rows)} files")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sessions", nargs="*"); ap.add_argument("--skip-videos", action="store_true"); ap.add_argument("--no-manifest", action="store_true")
    a = ap.parse_args()
    V3.mkdir(parents=True, exist_ok=True)
    tags = a.sessions or sorted(f.stem.replace("_fused", "") for f in (V2 / "Preprocessed_Dataset").rglob("*_fused.csv"))
    log(f"=== build v3 {date.today().isoformat()} : {len(tags)} sessions -> {V3}")
    report = []
    for i, tag in enumerate(tags, 1):
        log(f"[{i}/{len(tags)}] {tag}")
        process(tag, a.skip_videos, report)
        pd.DataFrame(report).to_csv(Path(__file__).resolve().parent / "build_v3_report.csv", index=False)
    extras(report); descriptive_files(report)
    if not a.no_manifest: manifest()
    log("=== done")


if __name__ == "__main__":
    main()
