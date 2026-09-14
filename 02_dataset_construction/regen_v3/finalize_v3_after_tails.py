"""
After restore_video_tails_v3.py: records the restored video coverage in the v3
descriptive files and rebuilds manifest.csv.

  dataset_metadata.json  sessions[*].video: covers_full_session, front/side frame counts,
                         restored_tail_frames, side_tail_detections; D1_S1 side history;
                         video_alignment statement updated
  data_schema.json       changelog entry
  README_v3_VIDEO_ALIGNED.md  coverage paragraph rewritten
  manifest.csv           rebuilt by hashing every file
"""
from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd

DOCS = Path(r"C:\Users\halha\OneDrive - Durham University\Documents")
V3 = DOCS / "Published_Dataset_Final_v3_VIDEO_ALIGNED"
HERE = Path(__file__).resolve().parent
T = pd.read_csv(HERE / "build_v3_tails_report.csv").set_index("tag")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    today = date.today().isoformat()
    meta = json.load(open(V3 / "dataset_metadata.json", encoding="utf-8"))
    stmt = ("Version 3: video frame k of {tag}_Front_blurred.mp4 / {tag}_Side_blurred.mp4 corresponds to elapsed_s = k/30 of the same session's "
            "fused.csv (and to unix_time = trimmed start + k/30), and the videos span the full telemetry window. The first round(lead_in_s*30) frames "
            "of the earlier (v1/v2) videos were dropped and the same number of frames restored at the end: front from the full-length output of the "
            "same blurring run; side by blurring the raw frames with the same detector, carry-forward rule and researcher mask, then visual review. "
            "Front_emotions.csv and Side_pose.csv use the same frame index but END lead_in_s BEFORE the telemetry (their rows for the restored frames "
            "were not recomputed). D1_S1's side video had additionally been cut twice in v1 (frame k = raw frame 2*fs_old + k); its missing first 41 s "
            "were restored from the raw recording by the same procedure.")
    meta["video_alignment"]["statement"] = stmt
    meta["video_alignment"]["restoration"] = "02_dataset_construction/regen_v3/restore_video_tails_v3.py; per-file report build_v3_tails_report.csv; review sheets 05_technical_validation/validation_output/video_tail_review/"
    for s in meta["sessions"]:
        t = s["tag"]
        if t not in T.index or "video" not in s: continue
        r = T.loc[t]; v = s["video"]
        if isinstance(r.get("note"), str) and "truncated" in r["note"]:
            v["covers_full_session"] = True; v["note"] = "frame k <-> elapsed_s k/30; session truncated at the logging pause, video and telemetry end together"; continue
        v.update({"covers_full_session": True, "front_frames": int(r["front_frames_out"]), "restored_tail_frames": int(r["tail_frames"]),
                  "tail_raw_frame_range": r["tail_raw_range"], "note": "frame k <-> elapsed_s k/30; video spans the full telemetry window; feature CSVs end lead_in_s earlier"})
        if not pd.isna(r.get("side_frames_out")):
            v["side_frames"] = int(r["side_frames_out"]); v["side_tail_detections"] = r["side_tail_detect"]; v["side_researcher_mask_px"] = int(r["side_mask_px"])
        if isinstance(r.get("side_head_range"), str):
            v["side_history"] = (f"v1 side video was cut twice (frame k = raw frame 2*fs_old + k, 51.5 s late, 1,546 frames short); raw frames {r['side_head_range']} "
                                 f"restored and blurred in v3 ({r['side_head_detect']} detections)")
            s["exceptions"] = list(s.get("exceptions", [])) + ["v1_side_video_double_trimmed_restored_in_v3"]
    meta["date"] = today
    json.dump(meta, open(V3 / "dataset_metadata.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    schema = json.load(open(V3 / "data_schema.json", encoding="utf-8"))
    schema["video_alignment"]["statement"] = stmt
    schema["changelog"].append({"date": today, "change": (
        "VERSION 3, second pass: video coverage restored to the full telemetry window (front tails from the full-length blurred recording; side tails and "
        "D1_S1's missing first 41 s blurred from the raw frames with the release's own detector and rules, then visually reviewed). Front_emotions.csv / "
        "Side_pose.csv unchanged (end lead_in_s before the telemetry). manifest.csv rebuilt.")})
    json.dump(schema, open(V3 / "data_schema.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    readme = V3 / "README_v3_VIDEO_ALIGNED.md"
    txt = readme.read_text(encoding="utf-8")
    txt = txt.replace("shifts the two CSVs identically. Each video ends lead_in seconds before the telemetry.",
                      "shifts the two CSVs identically. The frames the earlier cut had left off the END of each video were restored "
                      "(front: full-length output of the same blurring run; side: raw frames blurred with the same detector, carry-forward "
                      "rule and researcher mask, visually reviewed), so the videos span the full telemetry window. `Front_emotions.csv` and "
                      "`Side_pose.csv` still end lead_in seconds before the telemetry. D1_S1's side video, which v1 had cut twice (51.5 s late, "
                      "41 s short), was restored the same way.")
    readme.write_text(txt, encoding="utf-8")

    rows = []
    for p in sorted(V3.rglob("*")):
        if not p.is_file() or p.name == "manifest.csv": continue
        rel = str(p.relative_to(V3)).replace("\\", "/")
        tier = "raw" if rel.startswith("Raw_Dataset/") else ("processed" if rel.startswith("Preprocessed_Dataset/") else "release")
        n = (sum(1 for _ in open(p, "rb")) - 1) if p.suffix.lower() == ".csv" else ""
        rows.append({"tier": tier, "relative_path": rel, "size_bytes": p.stat().st_size, "rows": n, "sha256": sha256(p)})
    pd.DataFrame(rows).to_csv(V3 / "manifest.csv", index=False); print(f"manifest: {len(rows)} files; metadata/schema/README updated")


if __name__ == "__main__":
    main()
