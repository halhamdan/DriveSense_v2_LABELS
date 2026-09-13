"""
Generates the Data Overview's multimodal-alignment figure
(fig_multimodal_alignment.{png,pdf}) directly from the released fused.csv:
a ~2-minute segment of one session showing vehicle speed, longitudinal and
lateral acceleration, EDA and heart rate on the shared 25 Hz timeline, with
the deterministic harsh-event label shaded.

Default segment: D5_S2, 122 s starting 249.8 s after the trimmed session
start -- a standstill followed by a compound accelerate-then-turn manoeuvre
and a cluster of closely spaced braking/turning events (the segment used in
the manuscript; located by its label pattern in the current release so the
figure always reflects the released timeline).

Usage:
  python make_alignment_figure.py --dataset-root /path/to/Published_Dataset_Final
      [--driver 5 --session 2 --start 249.8 --duration 122]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent / "paper_figures"
COLORS = {"Harsh Acceleration": "#F2C94C", "Harsh Braking": "#E8956D", "Harsh Turning": "#7FD1B9"}


def label_spans(t: np.ndarray, lab: np.ndarray):
    spans = []
    s = 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[s]:
            if lab[s] != "Normal":
                spans.append((lab[s], t[s], t[i - 1]))
            s = i
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=os.environ.get("DATASET_ROOT", ""))
    ap.add_argument("--driver", type=int, default=5)
    ap.add_argument("--session", type=int, default=2)
    ap.add_argument("--start", type=float, default=249.8, help="segment start, s after trimmed session start")
    ap.add_argument("--duration", type=float, default=122.0)
    ap.add_argument("--out-name", default="fig_multimodal_alignment", help="output file stem (e.g. fig_multimodal_alignment_v2)")
    a = ap.parse_args()
    if not a.dataset_root:
        ap.error("--dataset-root must be set (or DATASET_ROOT)")

    tag = f"D{a.driver}_S{a.session}"
    f = Path(a.dataset_root) / "Preprocessed_Dataset" / f"D{a.driver}" / f"Session_{a.session}" / f"{tag}_fused.csv"
    df = pd.read_csv(f, usecols=["elapsed_s", "gps_speed_kmh", "lon_acc_g", "lat_acc_g", "EDA", "HR", "HR_dropout_flag", "label"])
    seg = df[(df.elapsed_s >= a.start) & (df.elapsed_s <= a.start + a.duration)].copy()
    seg["t"] = seg.elapsed_s - a.start
    hr = seg["HR"].where(~seg["HR_dropout_flag"].astype(bool))

    plt.rcParams.update({"font.size": 13, "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12})
    fig, axes = plt.subplots(4, 1, figsize=(8, 10.5), sharex=True)
    spans = label_spans(seg["t"].to_numpy(), seg["label"].to_numpy())
    for ax in axes:
        for cls, s0, s1 in spans:
            ax.axvspan(s0, s1, color=COLORS[cls], alpha=0.55, lw=0)

    axes[0].plot(seg.t, seg.gps_speed_kmh, color="#333333", lw=1.4)
    axes[0].set_ylabel("Speed\n(km/h)")
    axes[1].plot(seg.t, seg.lon_acc_g, color="#4C72B0", lw=1.0, label="Longitudinal")
    axes[1].plot(seg.t, seg.lat_acc_g, color="#8172B2", lw=1.0, label="Lateral")
    axes[1].axhline(0, color="#999999", lw=0.8)
    axes[1].set_ylabel("Acceleration\n(g)")
    axes[1].legend(loc="upper right", ncol=2, frameon=False)
    axes[2].plot(seg.t, seg.EDA, color="#4C72B0", lw=1.4)
    axes[2].set_ylabel("EDA\n(µS)")
    axes[3].plot(seg.t, hr, color="#55A868", lw=1.4)
    axes[3].set_ylabel("Heart rate\n(bpm)")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_xlim(0, a.duration)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)

    handles = [Patch(facecolor=c, alpha=0.55, label=k) for k, c in COLORS.items()]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.965))

    OUT_DIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{a.out_name}.{ext}", dpi=300, bbox_inches="tight")
    print(f"{tag} segment {a.start:.1f}-{a.start + a.duration:.1f} s: {len(seg)} rows, "
          f"{len(spans)} labelled spans -> {OUT_DIR / (a.out_name + '.png')}")


if __name__ == "__main__":
    main()
