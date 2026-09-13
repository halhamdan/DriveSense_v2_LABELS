"""
Generates Supplementary Table S1 (per-driver, per-session data availability)
directly from the RELEASED dataset: dataset_metadata.json for status,
duration, modality presence and detection rates, plus the actual presence of
each session's side video and pose file in the release tree -- so the table
distinguishes, for the side camera, what was recorded, what video is released
and what pose file is released, and gives the duration column the main text
refers to.

Usage:
  python make_availability_table.py            # writes validation_output/si_availability_table.tex
Environment:
  DATASET_ROOT   release root containing dataset_metadata.json, Raw_Dataset/, Preprocessed_Dataset/
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", "") or
                    r"C:\Users\halha\OneDrive - Durham University\Documents\Published_Dataset_Final")
OUT = Path(__file__).resolve().parent / "validation_output" / "si_availability_table.tex"

# Side camera recorded but nothing usable released (camera unmounted mid-session; 14.5 % pose detection)
SIDE_REMOVED = {"D2_S3": "recorded; removed (camera unmounted)"}
# Side camera recorded; video withheld after the de-identification audit; pose file retained
SIDE_WITHHELD = {"D11_S3": "recorded; video withheld"}


def esc(x):
    return str(x).replace("_", "\\_")


def ck(b):
    return r"\checkmark" if b else r"$\times$"


def main():
    meta = json.load(open(DATASET_ROOT / "dataset_metadata.json", encoding="utf-8"))
    rows = []
    for s in sorted(meta["sessions"], key=lambda s: (s["driver"], s["session"])):
        d, n, tag = s["driver"], s["session"], s["tag"]
        raw = DATASET_ROOT / "Raw_Dataset" / f"D{d}" / f"Session_{n}"
        pre = DATASET_ROOT / "Preprocessed_Dataset" / f"D{d}" / f"Session_{n}"
        rows.append(dict(
            tag=tag, status=s["status"], dur_min=(s.get("drive_duration_s") or 0) / 60.0,
            phys=bool(s.get("has_physiology")), facial=bool(s.get("has_facial")),
            face_cov=s.get("face_detection_rate"),
            side_video=(raw / f"{tag}_Side_blurred.mp4").exists(),
            pose_file=(pre / f"{tag}_Side_pose.csv").exists(),
            pose_rate=s.get("pose_detection_rate"),
        ))
    df = pd.DataFrame(rows)
    ok = df[df.status == "ok"]
    n_side_rec = int(ok.side_video.sum()) + len(SIDE_REMOVED) + len(SIDE_WITHHELD)

    hdr = (r"Session & Dur.\ (min) & Veh. & Bio. & Facial & Face cov. & "
           r"Side cam. & Side video & Pose file & Pose rate \\")
    cap = (r"\caption{Per-driver, per-session data availability in the release. Duration is the trimmed driving "
           r"interval. Vehicle, Biometric and Facial are boolean (present/absent). Facial output cov.\ is the "
           r"fraction of front-camera frames for which the facial-processing pipeline produced an output row. "
           r"The three side-camera columns distinguish whether the side camera recorded at all, whether its "
           r"face-blurred video is released, and whether a pose file is released; Pose det.\ rate is the fraction "
           r"of frames with a usable pose estimate in the released pose file. D6\_S2 is excluded entirely "
           r"(insufficient VBOX/EmotiBit temporal overlap). D6\_S3, D10\_S3 and D16\_S1: side camera disconnected "
           r"(nothing recorded). D2\_S3: side camera recorded but became unmounted mid-session (pose detection "
           r"14.5\%), so its video and pose file were removed from the release. D11\_S3: side camera recorded; "
           r"its video was withheld after the de-identification audit, but its pose file is released. "
           f"Totals: {len(ok)} valid sessions; side camera recorded in {n_side_rec}; side video released for "
           f"{int(ok.side_video.sum())}; pose files released for {int(ok.pose_file.sum())} "
           f"(mean pose-detection rate {ok[ok.pose_file].pose_rate.mean()*100:.1f}\\%).}}")

    lines = [r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\begin{longtable}{@{}lrcccrcccl@{}}", cap,
             r"\label{tab:si-availability}\\", r"\toprule", hdr, r"\midrule", r"\endfirsthead",
             r"\multicolumn{10}{c}{\tablename\ \thetable{} -- continued}\\", r"\toprule", hdr, r"\midrule",
             r"\endhead", r"\bottomrule", r"\endfoot"]
    for _, r in df.iterrows():
        tag = esc(r.tag)
        if r.status != "ok":
            lines.append(f"{tag} & -- & -- & -- & -- & -- & -- & -- & -- & excluded (insufficient overlap) \\\\")
            continue
        face = f"{r.face_cov:.3f}" if pd.notna(r.face_cov) else "--"
        if r.tag in SIDE_REMOVED:
            side, vid, pose, rate = "recorded", ck(False), ck(False), "removed (unmounted)"
        elif r.tag in SIDE_WITHHELD:
            side, vid, pose, rate = "recorded", ck(False), ck(True), f"{r.pose_rate:.3f}"
        elif not r.side_video and not r.pose_file:
            side, vid, pose, rate = ck(False), ck(False), ck(False), "camera disconnected"
        else:
            side, vid, pose, rate = ck(True), ck(r.side_video), ck(r.pose_file), (f"{r.pose_rate:.3f}" if pd.notna(r.pose_rate) else "--")
        lines.append(" & ".join([tag, f"{r.dur_min:.1f}", ck(True), ck(r.phys), ck(r.facial), face, side, vid, pose, rate]) + r" \\")
    lines.append(r"\end{longtable}")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(df)} rows to {OUT}")
    print(f"valid {len(ok)} | side recorded {n_side_rec} | side video released {int(ok.side_video.sum())} | "
          f"pose files {int(ok.pose_file.sum())} | mean pose rate over released pose files "
          f"{ok[ok.pose_file].pose_rate.mean()*100:.1f}% | total {ok.dur_min.sum()/60:.2f} h")


if __name__ == "__main__":
    main()
