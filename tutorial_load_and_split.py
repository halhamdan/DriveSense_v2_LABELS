"""
Worked usage example for the RELEASED dataset (distinct from the
construction pipeline in 01-06): loads one session, applies documented QC
masks, aligns the facial/pose modalities onto the fused 25 Hz timeline, and
builds a leave-one-driver-out split with no cross-driver leakage.

This is a tutorial over the *released* Preprocessed_Dataset tier -- it does
not reproduce the dataset from raw sensor exports (see README's "Scope and
honesty note" for why that is a separate, larger undertaking).

Usage:
  python tutorial_load_and_split.py --dataset-root /path/to/Published_Dataset_Final
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_session(dataset_root: Path, driver: str, session: int) -> pd.DataFrame:
    """Load one session's fused.csv, apply the documented QC masks, and
    left-join the facial/pose derived features onto the fused 25 Hz grid by
    nearest timestamp (their native rates are lower than 25 Hz)."""
    tag = f"{driver}_S{session}"
    sess_dir = dataset_root / "Preprocessed_Dataset" / driver / f"Session_{session}"
    fused = pd.read_csv(sess_dir / f"{tag}_fused.csv")

    # QC masks documented in the Data Descriptor's Technical Validation:
    # exclude resampled rows falling inside a disclosed IBI/HR dropout, and
    # apply the documented plausible-range checks for EDA and lat/lon accel.
    fused["IBI_valid"] = ~fused["IBI_dropout_flag"].astype(bool)
    fused["HR_valid"] = ~fused["HR_dropout_flag"].astype(bool)
    fused["EDA_valid"] = fused["EDA"].between(0, 50)
    fused["accel_valid"] = fused["lat_acc_g"].between(-1.2, 1.2) & fused["lon_acc_g"].between(-1.2, 1.2)

    for name, fp in [("Front_emotions", sess_dir / f"{tag}_Front_emotions.csv"),
                      ("Side_pose", sess_dir / f"{tag}_Side_pose.csv")]:
        if not fp.exists():
            continue
        modality = pd.read_csv(fp).sort_values("timestamp_s")
        fused = pd.merge_asof(
            fused.sort_values("elapsed_s"), modality,
            left_on="elapsed_s", right_on="timestamp_s",
            direction="nearest", suffixes=("", f"_{name.lower()}"),
        )

    fused["driver"] = driver
    fused["session"] = session
    return fused


def build_grouped_split(dataset_root: Path, drivers: list[str]) -> None:
    """Demonstrate a leave-one-driver-out split: the held-out driver's rows
    never appear in the training set, avoiding the cross-driver-session
    leakage a random or time-based split would risk (Data Descriptor,
    Usage Notes)."""
    from sklearn.model_selection import LeaveOneGroupOut

    all_sessions = []
    for driver in drivers:
        for session in range(1, 5):
            sess_dir = dataset_root / "Preprocessed_Dataset" / driver / f"Session_{session}"
            if not (sess_dir / f"{driver}_S{session}_fused.csv").exists():
                continue
            df = load_session(dataset_root, driver, session)
            all_sessions.append(df[["driver", "session", "elapsed_s", "label"]])

    combined = pd.concat(all_sessions, ignore_index=True)
    groups = combined["driver"].to_numpy()

    logo = LeaveOneGroupOut()
    print(f"Loaded {len(combined):,} rows across {len(drivers)} drivers.")
    print(f"LeaveOneGroupOut would produce {logo.get_n_splits(groups=groups)} folds "
          f"(one per driver); each fold's held-out driver contributes zero rows to training.")
    # Sanity check: no driver appears in both a fold's train and test indices.
    for fold_i, (train_idx, test_idx) in enumerate(logo.split(combined, groups=groups)):
        train_drivers = set(combined.iloc[train_idx]["driver"])
        test_drivers = set(combined.iloc[test_idx]["driver"])
        assert train_drivers.isdisjoint(test_drivers), "leakage detected"
    print("Verified: no driver appears in both train and test in any fold.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path,
                         help="Root containing Raw_Dataset/ and Preprocessed_Dataset/")
    parser.add_argument("--drivers", nargs="+", default=[f"D{i}" for i in range(1, 6)],
                         help="Drivers to include (default: D1-D5, for a quick example)")
    args = parser.parse_args()
    build_grouped_split(args.dataset_root, args.drivers)
