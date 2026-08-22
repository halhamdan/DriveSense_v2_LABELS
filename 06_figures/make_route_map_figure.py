"""
Study-route map: plots the actual GPS trace of the fixed 17km route on a
real street basemap (OpenStreetMap-family tiles via contextily), overlaying
several sessions lightly to visually confirm the route is genuinely
fixed/repeated across drivers, with start/end markers.

Reads GPS coordinates directly from the released Raw_Dataset VBOX export
(lat/lon columns) and merges the harsh-event label from the released
Preprocessed_Dataset fused.csv by nearest unix_time -- both files are on the
same 25 Hz VBOX timeline, so this does not require any of the dataset's own
processing/fusion code, only pandas.

Usage:
  python make_route_map_figure.py --dataset-root /path/to/Published_Dataset_Final
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

try:
    import contextily as cx
    HAVE_CX = True
except ImportError:
    HAVE_CX = False

BASE = Path(__file__).parent
OUT_DIR = BASE / "paper_figures"
OUT_DIR.mkdir(exist_ok=True)


def load_session_gps(dataset_root: Path, driver: int, session: int) -> pd.DataFrame | None:
    raw_path = dataset_root / "Raw_Dataset" / f"D{driver}" / f"Session_{session}" / f"D{driver}_S{session}_VBOX_raw.csv"
    fused_path = dataset_root / "Preprocessed_Dataset" / f"D{driver}" / f"Session_{session}" / f"D{driver}_S{session}_fused.csv"
    if not raw_path.exists() or not fused_path.exists():
        return None

    raw = pd.read_csv(raw_path, usecols=["unix_time", "lat", "lon"]).dropna(subset=["lat", "lon"])
    raw = raw[(raw["lat"].abs() > 1) & (raw["lon"].abs() > 1)]  # drop 0/0 no-fix rows
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
    return merged[["lat", "lon", "label"]]


def lonlat_to_webmercator(lon, lat):
    k = 6378137
    x = lon * (k * np.pi / 180.0)
    y = np.log(np.tan((90 + lat) * np.pi / 360.0)) * k
    return x, y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=os.environ.get("DATASET_ROOT", ""),
                         help="Path to the released dataset root (containing Raw_Dataset/ and "
                              "Preprocessed_Dataset/); default: $DATASET_ROOT")
    args = parser.parse_args()
    if not args.dataset_root:
        parser.error("--dataset-root must be set (or DATASET_ROOT env var)")
    dataset_root = Path(args.dataset_root)

    # A handful of sessions from different drivers -- enough to visually
    # confirm the route is fixed, not so many the plot becomes an
    # indistinguishable blob.
    sample_sessions = [(1, 1), (5, 2), (9, 3), (13, 1), (17, 4), (3, 2), (11, 3)]
    frames = []
    for d, s in sample_sessions:
        df = load_session_gps(dataset_root, d, s)
        if df is not None:
            frames.append(df)

    if not frames:
        print("No sessions loaded.")
        return

    fig, ax = plt.subplots(figsize=(9, 9))
    ROUTE_COLOR = "#1e6e5c"  # matches the Behaviour/vehicle-dynamics colour used elsewhere in the paper

    all_x, all_y = [], []
    for i, df in enumerate(frames):
        x, y = lonlat_to_webmercator(df["lon"].to_numpy(), df["lat"].to_numpy())
        # White casing under the route for contrast against a busy basemap,
        # then the coloured line on top.
        ax.plot(x, y, color="white", linewidth=6.0, alpha=0.85, solid_capstyle="round", zorder=3)
        ax.plot(x, y, color=ROUTE_COLOR, linewidth=4.0, alpha=0.75, solid_capstyle="round", zorder=4,
                label="Session route" if i == 0 else None)
        all_x.append(x)
        all_y.append(y)

    # start/end markers from the first sampled session
    x0, y0 = lonlat_to_webmercator(frames[0]["lon"].to_numpy()[:1], frames[0]["lat"].to_numpy()[:1])
    x1, y1 = lonlat_to_webmercator(frames[0]["lon"].to_numpy()[-1:], frames[0]["lat"].to_numpy()[-1:])
    ax.scatter(x0, y0, marker="^", s=180, color="#1e6e5c", edgecolor="white", linewidth=1.6, zorder=5, label="Start")
    ax.scatter(x1, y1, marker="s", s=160, color="#9c3d3a", edgecolor="white", linewidth=1.6, zorder=5, label="End")

    all_x, all_y = np.concatenate(all_x), np.concatenate(all_y)
    pad_x = (all_x.max() - all_x.min()) * 0.08
    pad_y = (all_y.max() - all_y.min()) * 0.08
    ax.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
    ax.set_ylim(all_y.min() - pad_y, all_y.max() + pad_y)

    if HAVE_CX:
        # Standard (non-satellite) basemap. OSM/CartoDB tiles render Riyadh
        # street/district names in Arabic script, which doesn't reproduce
        # legibly in the manuscript; Esri's World Topo Map renders
        # district-level labels predominantly in English at this zoom, with
        # a clean, minimalist style that keeps the route line legible.
        try:
            cx.add_basemap(ax, source=cx.providers.Esri.WorldTopoMap, zoom=14, attribution_size=6)
        except Exception as e:
            print(f"Basemap fetch failed ({e}); plotting without basemap.")

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
    # No in-figure title -- caption in the manuscript covers this.
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    # Simple scale bar (approx, based on Web Mercator meters at this latitude).
    # Anchored top-left rather than bottom-left so it doesn't overlap the
    # basemap tile-provider attribution text, which contextily always draws
    # along the bottom edge (Esri's tile terms of use require that credit to
    # remain legible).
    bar_len_m = 2000
    bar_x0 = all_x.min() + pad_x * 0.3
    bar_y0 = all_y.max() - pad_y * 0.3
    ax.plot([bar_x0, bar_x0 + bar_len_m], [bar_y0, bar_y0], color="black", linewidth=3, zorder=6)
    ax.text(bar_x0 + bar_len_m / 2, bar_y0 + pad_y * 0.05, "2 km", ha="center", fontsize=9, zorder=6)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_route_map.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig_route_map.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved to {OUT_DIR}/fig_route_map.{{pdf,png}} (basemap={'yes' if HAVE_CX else 'no'})")


if __name__ == "__main__":
    main()
