"""
Spatial density of harsh-event labels along the fixed route: fraction of
25 Hz samples labelled harsh per ~15m grid cell, restricted to cells visited
by at least 10 distinct drivers (a generalization requirement so a single
driver's idiosyncratic route deviation doesn't dominate a cell).

This is purely a spatial redistribution of the released label column against
the released GPS coordinates -- it does not use, and deliberately excludes,
any physiological signal. (A related analysis in the companion research
paper additionally maps median session-normalized SCR-frequency arousal per
cell; that finding is out of scope for the Data Descriptor and is not
reproduced here.)

Reads directly from the released dataset (Raw_Dataset for GPS, Preprocessed_
Dataset/fused.csv for the label), merged by nearest unix_time.

Usage:
  python make_route_harsh_density_figure.py --dataset-root /path/to/Published_Dataset_Final
  (or set DATASET_ROOT)
"""
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
OUT_DIR = BASE / "paper_figures"
OUT_DIR.mkdir(exist_ok=True)

GRID_DEG = 0.00015  # ~15-17m at this latitude
MIN_DRIVERS_FOR_CELL = 10  # generalization bar: at least half the drivers


def load_session_gps_label(dataset_root: Path, driver: int, session: int) -> pd.DataFrame | None:
    raw_path = dataset_root / "Raw_Dataset" / f"D{driver}" / f"Session_{session}" / f"D{driver}_S{session}_VBOX_raw.csv"
    # PREPROCESSED_ROOT lets the labels come from a separate label-only tree
    # (e.g. Published_Dataset_Final_v2_LABELS) while GPS stays in the v1 raw tier.
    pre_root = Path(os.environ.get("PREPROCESSED_ROOT", "") or dataset_root)
    fused_path = pre_root / "Preprocessed_Dataset" / f"D{driver}" / f"Session_{session}" / f"D{driver}_S{session}_fused.csv"
    if not raw_path.exists() or not fused_path.exists():
        return None

    raw = pd.read_csv(raw_path, usecols=["unix_time", "lat", "lon"]).dropna(subset=["lat", "lon"])
    raw = raw[(raw["lat"].abs() > 1) & (raw["lon"].abs() > 1)]
    if raw.empty:
        return None
    raw = raw.sort_values("unix_time")

    fused = pd.read_csv(fused_path, usecols=["unix_time", "label"]).sort_values("unix_time")
    if len(fused) < 25 * 40:
        return None

    merged = pd.merge_asof(fused, raw, on="unix_time", direction="nearest", tolerance=1.0)
    merged = merged.dropna(subset=["lat", "lon"])
    if merged.empty:
        return None
    merged["driver"] = driver
    return merged[["driver", "lat", "lon", "label"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=os.environ.get("DATASET_ROOT", ""),
                         help="Path to the released dataset root (containing Raw_Dataset/ and "
                              "Preprocessed_Dataset/); default: $DATASET_ROOT")
    args = parser.parse_args()
    if not args.dataset_root:
        parser.error("--dataset-root must be set (or DATASET_ROOT env var)")
    dataset_root = Path(args.dataset_root)

    frames = []
    for d in range(1, 21):
        for s in range(1, 5):
            df = load_session_gps_label(dataset_root, d, s)
            if df is not None:
                frames.append(df)
    if not frames:
        print("No sessions with usable GPS + label data found.")
        return
    all_df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(all_df)} GPS-tagged samples from {all_df['driver'].nunique()} drivers")

    all_df["grid_lat"] = (all_df["lat"] / GRID_DEG).round().astype(int)
    all_df["grid_lon"] = (all_df["lon"] / GRID_DEG).round().astype(int)

    cell = all_df.groupby(["grid_lat", "grid_lon"]).agg(
        n_drivers=("driver", "nunique"),
        lat=("lat", "mean"),
        lon=("lon", "mean"),
        harsh_frac=("label", lambda x: (x != "Normal").mean()),
    ).reset_index()

    gen = cell[cell["n_drivers"] >= MIN_DRIVERS_FOR_CELL].copy()
    print(f"{len(cell)} grid cells total; {len(gen)} visited by >= {MIN_DRIVERS_FOR_CELL} distinct drivers")
    gen.to_csv(OUT_DIR / "route_harsh_density_cells.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(gen["lon"], gen["lat"], c=gen["harsh_frac"], cmap="magma_r",
                     vmin=0, vmax=np.nanpercentile(gen["harsh_frac"], 98), s=10, alpha=0.85)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_aspect("equal")
    ax.tick_params(axis="both", labelsize=10)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Fraction of samples labelled harsh", fontsize=11)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_route_harsh_density.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig_route_harsh_density.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved to {OUT_DIR}/fig_route_harsh_density.{{pdf,png}}")


if __name__ == "__main__":
    main()
