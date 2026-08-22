"""
Generates the per-driver x session data-availability table (Supplementary
Table S1 in the published manuscript), matching UL-DD (Model_1)'s Table 5
convention, plus refined
summary counts for the main Data Records section -- this also corrects an
imprecise claim already in the manuscript ("76/79 have valid in-cabin data")
by distinguishing camera-disconnected sessions (3) from camera-present-but-
pose-detection-degraded sessions (1 additional: D2_S3, 11% detection rate,
found by actually computing per-session rates rather than asserting a
blanket completeness number).

Usage:
  python make_availability_table.py
"""
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
# data_availability_full.csv is derived from dataset_metadata.json (released
# at the top level of the dataset, see README's "Regenerating this table" note).
df = pd.read_csv(BASE / "validation_output" / "data_availability_full.csv")
df = df.sort_values(["driver", "session"])


def esc(x):
    return str(x).replace("_", "\\_")


def ck(b):
    return r"\checkmark" if b else r"$\times$"


lines = []
lines.append(r"\begin{longtable}{@{}lccccrc@{}}")
lines.append(r"\caption{Per-driver, per-session data availability. Behaviour, Biometric, and Facial columns are boolean (present/absent); Facial and In-Cabin columns additionally give the per-session detection rate (fraction of frames with a usable face or pose estimate) where available. D6\_S2 is excluded entirely (status: skipped, insufficient VBOX/EmotiBit temporal overlap). D6\_S3, D10\_S3, and D16\_S1 have no side-camera signal at all (camera disconnected). D2\_S3 is a case Table~S3 itself surfaces: the side camera was present, but pose detection succeeded for only 11\% of frames (vs.\ a 98\% mean elsewhere) -- flagged here as unreliable for in-cabin analysis despite nominally having a file, rather than silently included in a blanket completeness count.}")
lines.append(r"\label{tab:si-availability}\\")
lines.append(r"\toprule")
lines.append(r"Session & Behaviour & Biometric & Facial & Face det.\ rate & In-cabin & Pose det.\ rate \\")
lines.append(r"\midrule")
lines.append(r"\endfirsthead")
lines.append(r"\multicolumn{7}{c}{\tablename\ \thetable{} -- continued}\\")
lines.append(r"\toprule")
lines.append(r"Session & Behaviour & Biometric & Facial & Face det.\ rate & In-cabin & Pose det.\ rate \\")
lines.append(r"\midrule")
lines.append(r"\endhead")
lines.append(r"\bottomrule")
lines.append(r"\endfoot")

for _, row in df.iterrows():
    tag = esc(row["tag"])
    if row["status"] != "ok":
        cells = [tag, "--", "--", "--", "--", "--", "excluded (insufficient overlap)"]
        lines.append(" & ".join(cells) + r" \\")
        continue
    behaviour = ck(True)  # all status==ok sessions have telemetry by construction
    biometric = ck(bool(row["has_physiology"]))
    facial = ck(bool(row["has_facial"]))
    face_rate = f"{row['face_detection_rate']:.3f}" if pd.notna(row["face_detection_rate"]) else "--"
    has_cabin_signal = bool(row["has_road_camera"])  # metadata field name for the side/cabin camera
    pose_rate = row["pose_detection_rate"]
    if not has_cabin_signal:
        cabin_cell = ck(False)
        pose_cell = "camera disconnected"
    elif pd.notna(pose_rate) and pose_rate < 0.5:
        cabin_cell = r"\textbf{$\times$}"
        pose_cell = f"\\textbf{{{pose_rate:.3f} -- unreliable}}"
    else:
        cabin_cell = ck(True)
        pose_cell = f"{pose_rate:.3f}" if pd.notna(pose_rate) else "--"
    cells = [tag, behaviour, biometric, facial, face_rate, cabin_cell, pose_cell]
    lines.append(" & ".join(cells) + r" \\")

lines.append(r"\end{longtable}")

out_path = BASE / "validation_output" / "si_availability_table.tex"
out_path.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {len(df)} rows to {out_path}")

# Refined summary counts for main Data Records
ok = df[df["status"] == "ok"]
n_cabin_signal = ok["has_road_camera"].sum()
n_cabin_reliable = ((ok["has_road_camera"] == True) & (ok["pose_detection_rate"] >= 0.5)).sum()
n_cabin_unreliable = ((ok["has_road_camera"] == True) & (ok["pose_detection_rate"] < 0.5)).sum()
print(f"\nValid sessions: {len(ok)}/80")
print(f"Sessions with side-camera signal at all: {n_cabin_signal}")
print(f"Sessions with reliable (>=50% pose detection) in-cabin data: {n_cabin_reliable}")
print(f"Sessions with camera present but unreliable pose detection: {n_cabin_unreliable}")
print(f"Sessions with no camera signal: {(ok['has_road_camera'] == False).sum()}")
