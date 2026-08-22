"""
Final reorganization: copies everything from the flat Published_Dataset_Trimmed
tree into the two-tier Raw_Dataset/Preprocessed_Dataset structure described in
the manuscript's Data Records section.

  Raw_Dataset/D{n}/Session_{s}/
      {tag}_VBOX_raw.csv
      {tag}_EmotiBit_{SIGNAL}.csv  (13 files)
      {tag}_Front_blurred.mp4
      {tag}_Side_blurred.mp4        (76/79 sessions -- 3 invalid side camera)

  Preprocessed_Dataset/D{n}/Session_{s}/
      {tag}_fused.csv
      {tag}_Front_emotions.csv
      {tag}_Side_pose.csv           (76/79 sessions)

This COPIES (does not move/delete) from Published_Dataset_Trimmed, so the
original trimmed tree -- which represents hours of video re-encoding -- is
left untouched until the copy is verified.

Usage:
  python reorganize_final.py
"""
import shutil
from pathlib import Path

import pandas as pd

from apply_trim import OUT_ROOT, TRIM_CSV

FINAL_ROOT = OUT_ROOT.parent / "Published_Dataset_Final"
RAW_DIR = FINAL_ROOT / "Raw_Dataset"
PRE_DIR = FINAL_ROOT / "Preprocessed_Dataset"

EMOTIBIT_SIGNALS = [
    "EDA", "EDA_LEVEL", "SCR_FREQ", "HR", "IBI",
    "TEMP_CONTACT", "TEMP_THERMOPILE",
    "PPG_IR", "PPG_RED", "PPG_GREEN",
    "ACC", "GYRO", "MAG",
]


def copy_if_exists(src: Path, dst: Path, required: bool, report: dict) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        if required:
            report["missing_required"].append(str(src))
        return
    shutil.copy2(src, dst)
    if src.stat().st_size != dst.stat().st_size:
        report["size_mismatch"].append(str(src))
    else:
        report["copied"] += 1


def process_session(driver: int, session: int, report: dict) -> None:
    tag = f"D{driver}_S{session}"
    src_dir = OUT_ROOT / f"D{driver}" / f"Session_{session}"
    raw_out = RAW_DIR / f"D{driver}" / f"Session_{session}"
    pre_out = PRE_DIR / f"D{driver}" / f"Session_{session}"

    if not src_dir.exists():
        report["missing_session"].append(tag)
        return

    # ---- Raw_Dataset ----
    copy_if_exists(src_dir / f"{tag}_VBOX_raw.csv", raw_out / f"{tag}_VBOX_raw.csv", True, report)
    for sig in EMOTIBIT_SIGNALS:
        copy_if_exists(src_dir / f"{tag}_EmotiBit_{sig}.csv", raw_out / f"{tag}_EmotiBit_{sig}.csv", False, report)
    copy_if_exists(src_dir / f"{tag}_Front_blurred.mp4", raw_out / f"{tag}_Front_blurred.mp4", True, report)
    copy_if_exists(src_dir / f"{tag}_Side_blurred.mp4", raw_out / f"{tag}_Side_blurred.mp4", False, report)

    # ---- Preprocessed_Dataset ----
    copy_if_exists(src_dir / f"{tag}_fused.csv", pre_out / f"{tag}_fused.csv", True, report)
    copy_if_exists(src_dir / f"{tag}_Front_emotions.csv", pre_out / f"{tag}_Front_emotions.csv", True, report)
    copy_if_exists(src_dir / f"{tag}_Side_pose.csv", pre_out / f"{tag}_Side_pose.csv", False, report)


def main():
    trims = pd.read_csv(TRIM_CSV)
    report = {"copied": 0, "missing_required": [], "missing_session": [], "size_mismatch": []}

    for _, row in trims.iterrows():
        process_session(int(row["driver"]), int(row["session"]), report)

    print(f"\n=== Reorganization summary ===")
    print(f"Files copied: {report['copied']}")
    print(f"Sessions missing entirely: {len(report['missing_session'])}")
    for x in report["missing_session"]:
        print(f"  {x}")
    print(f"Required files missing: {len(report['missing_required'])}")
    for x in report["missing_required"]:
        print(f"  {x}")
    print(f"Size mismatches (copy may be corrupt): {len(report['size_mismatch'])}")
    for x in report["size_mismatch"]:
        print(f"  {x}")

    if not report["missing_required"] and not report["missing_session"] and not report["size_mismatch"]:
        print("\nAll required files copied cleanly. Original Published_Dataset_Trimmed left untouched.")


if __name__ == "__main__":
    main()
