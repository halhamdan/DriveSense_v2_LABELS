"""
Generates the Data Descriptor's own Technical Validation / Data Overview
figures directly from the released dataset (Preprocessed_Dataset/*/fused.csv)
-- no companion-paper intermediate files or modelling code required.

Figures produced (PDF + PNG, 300 DPI) in paper_figures/:
  fig_biometric_quality.{pdf,png}     -- EDA/HR outlier characterization
  fig_biometric_by_driver.{pdf,png}   -- EDA/HR/IBI distributions per driver

(Class distribution is reported as a table in the manuscript, not a figure --
see Table 4, "Data Overview".)

(Two further figures appear in the manuscript -- the route map and the
route harsh-density map -- these live in make_route_map_figure.py and
make_route_harsh_density_figure.py respectively, since they need the raw
GPS files rather than fused.csv.)

Usage:
  python make_technical_validation_figures.py --dataset-root /path/to/Published_Dataset_Final
  (or set DATASET_ROOT)
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
# Arial/Helvetica per Nature's figure typography requirement (matplotlib's
# default DejaVu Sans does not satisfy this).
plt.rcParams["font.family"] = "Arial"

BASE = Path(__file__).parent
OUT_DIR = BASE / "paper_figures"
OUT_DIR.mkdir(exist_ok=True)

def _save(fig, name: str):
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  saved {name}.pdf / .png")


def _fused_files(dataset_root: Path):
    return sorted(glob.glob(str(dataset_root / "Preprocessed_Dataset" / "D*" / "Session_*" / "D*_S*_fused.csv")))


def _robust_outlier_pct(x: np.ndarray, lo: float, hi: float) -> float:
    # Percentage of PRESENT readings outside [lo, hi]. Rows inside a disclosed
    # HR/IBI dropout are NaN in the release and are excluded from the
    # denominator, matching the manuscript's "x% of readings" convention
    # (counting them would understate the rate by a factor of 1 - dropout).
    x = x[np.isfinite(x)]
    return float(((x < lo) | (x > hi)).mean() * 100)


def fig_biometric_quality(dataset_root: Path):
    files = _fused_files(dataset_root)
    if not files:
        print("  [skip] no Preprocessed_Dataset/D*/Session_*/D*_fused.csv found")
        return
    eda_vals, hr_vals = [], []
    for f in files:
        df = pd.read_csv(f, usecols=["EDA", "HR"])
        eda_vals.append(df["EDA"].to_numpy())
        hr_vals.append(df["HR"].to_numpy())
    eda = np.concatenate(eda_vals)
    hr = np.concatenate(hr_vals)

    eda_out_pct = _robust_outlier_pct(eda, 0, 50)
    hr_out_pct = _robust_outlier_pct(hr, 40, 200)
    print(f"  EDA outside 0-50 uS: {eda_out_pct:.2f}%  |  HR outside 40-200 bpm: {hr_out_pct:.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    eda_core = eda[(eda >= 0) & (eda <= 20)]
    sns.histplot(eda_core, bins=60, ax=ax, color="#4C72B0")
    ax.set_xlabel("EDA (uS) -- view clipped to the 0-20 uS core distribution", fontsize=13)
    ax.set_ylabel(ax.get_ylabel(), fontsize=13)
    ax.tick_params(axis="both", labelsize=12)

    ax = axes[1]
    sns.histplot(hr[(hr >= 0) & (hr <= 250)], bins=60, ax=ax, color="#55A868")
    ax.axvline(40, color="#C44E52", linestyle="--", linewidth=1.5)
    ax.axvline(200, color="#C44E52", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Heart rate (bpm)", fontsize=13)
    ax.set_ylabel(ax.get_ylabel(), fontsize=13)
    ax.tick_params(axis="both", labelsize=12)

    fig.tight_layout()
    _save(fig, "fig_biometric_quality")


def fig_biometric_by_driver(dataset_root: Path):
    files = _fused_files(dataset_root)
    if not files:
        print("  [skip] no Preprocessed_Dataset/D*/Session_*/D*_fused.csv found")
        return
    rows = []
    for f in files:
        m = re.search(r"D(\d+)_S\d+_fused\.csv$", f)
        driver = int(m.group(1))
        df = pd.read_csv(f, usecols=["EDA", "HR", "IBI"])
        df["driver"] = driver
        rows.append(df)
    all_data = pd.concat(rows, ignore_index=True)
    # EDA/HR and IBI are filtered to their own plausible ranges independently (300-1500 ms for
    # IBI, equivalent to 40-200 bpm): a dropout-driven IBI outlier shouldn't be dropped just
    # because that same timestep's EDA or HR happens to look fine, or vice versa.
    data = all_data[(all_data["EDA"] >= 0) & (all_data["EDA"] <= 50) & (all_data["HR"] >= 40) & (all_data["HR"] <= 200)]
    ibi_data = all_data[(all_data["IBI"] >= 300) & (all_data["IBI"] <= 1500)]

    fig, axes = plt.subplots(3, 1, figsize=(13, 13))

    ax = axes[0]
    sns.boxplot(data=data, x="driver", y="EDA", ax=ax, color="#4C72B0", fliersize=1, linewidth=0.8)
    ax.set_xlabel("")
    ax.set_ylabel("EDA (uS)", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)

    ax = axes[1]
    sns.boxplot(data=data, x="driver", y="HR", ax=ax, color="#55A868", fliersize=1, linewidth=0.8)
    ax.set_xlabel("")
    ax.set_ylabel("Heart rate (bpm)", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)

    ax = axes[2]
    sns.boxplot(data=ibi_data, x="driver", y="IBI", ax=ax, color="#C44E52", fliersize=1, linewidth=0.8)
    ax.set_xlabel("Driver", fontsize=13)
    ax.set_ylabel("IBI (ms)", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)

    fig.tight_layout()
    _save(fig, "fig_biometric_by_driver")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=os.environ.get("DATASET_ROOT", ""),
                         help="Path to the released dataset root (containing Preprocessed_Dataset/); "
                              "default: $DATASET_ROOT")
    args = parser.parse_args()
    if not args.dataset_root:
        parser.error("--dataset-root must be set (or DATASET_ROOT env var)")
    dataset_root = Path(args.dataset_root)

    print("Biometric signal quality:")
    fig_biometric_quality(dataset_root)
    print("Biometric signal quality by driver:")
    fig_biometric_by_driver(dataset_root)
    print(f"\nAll available figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
