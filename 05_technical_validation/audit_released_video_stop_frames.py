"""
Released-file check of the video/telemetry offset that needs no overlay and no
raw files: pick instants where the RELEASED telemetry says the vehicle is
stationary (speed < 1 km/h for >= 3 s) but was moving fast (> 30 km/h) lead_in
seconds earlier, then measure how much the released SIDE video (road-facing
half) changes between frame k and frame k+15 (0.5 s) at

    tau            -- if video and telemetry were aligned, the scene is frozen
    tau + lead_in  -- where the offset hypothesis predicts the actual stop

and, as a control, at an instant where telemetry says > 40 km/h.  Frame pairs
are saved as PNG montages for visual inspection.

Usage: python audit_released_video_stop_frames.py --sessions D1_S1 D5_S2 D11_S2 [--n 3]
Output: validation_output/video_offset_evidence/<tag>_stop_*.png and
        validation_output/released_video_stop_check.csv
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("DATASET_ROOT", "") or r"C:\Users\halha\OneDrive - Durham University\Documents\DriveSense_Packages\3_OLD_versions\data\Published_Dataset_Final_v1_ORIGINAL_20260909")
HERE = Path(__file__).resolve().parent
LEAD = pd.read_csv(HERE / "validation_output" / "video_lead_in_per_session.csv").set_index("tag")["lead_in_s"]
EVID = HERE / "validation_output" / "video_offset_evidence"; EVID.mkdir(exist_ok=True, parents=True)
FPS = 30


def frame(video: Path, t: float, out: Path):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1", "-vf", "scale=480:270", str(out)], check=True)


def grey(png: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(png), "-vf", "format=gray", "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(270, 480).astype(np.int16)


def montage(files, out):
    args = ["ffmpeg", "-v", "error", "-y"]
    for f in files: args += ["-i", str(f)]
    args += ["-filter_complex", "".join(f"[{i}]" for i in range(len(files))) + f"hstack=inputs={len(files)}", str(out)]
    subprocess.run(args, check=True)


def _verdict(d):
    """Scene change at the telemetry stop instant (tau) and one lead-in later, relative to a
    moving control. Frozen = < 35 % of the moving control; moving = > 50 % of it."""
    c = max(d["control_moving"], 1e-6); at, later = d["tau"] / c, d["tau_plus_lead"] / c
    if at < 0.35 and later > 0.5: return "ALIGNED (frozen at tau, moving at tau+lead)"
    if at < 0.35: return "ALIGNED (frozen at tau)"
    if at > 0.5 and later < 0.35: return "OFFSET (scene moves at tau, frozen at tau+lead)"
    return "inconclusive"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sessions", nargs="+", required=True); ap.add_argument("--n", type=int, default=2); a = ap.parse_args()
    rows = []
    for tag in a.sessions:
        d, s = tag[1:].split("_S"); lead = float(LEAD.get(tag, np.nan))
        vid = ROOT / "Raw_Dataset" / f"D{d}" / f"Session_{s}" / f"{tag}_Side_blurred.mp4"
        if not vid.exists(): vid = ROOT / "Raw_Dataset" / f"D{d}" / f"Session_{s}" / f"{tag}_Front_blurred.mp4"
        fused_f = ROOT / "Preprocessed_Dataset" / f"D{d}" / f"Session_{s}" / f"{tag}_fused.csv"
        if not fused_f.exists() or not vid.exists() or not np.isfinite(lead):
            print(f"{tag}: skipped (missing files)"); continue
        F = pd.read_csv(fused_f, usecols=["elapsed_s", "gps_speed_kmh"])
        e = F.elapsed_s.to_numpy(); v = F.gps_speed_kmh.to_numpy(); n_lead = int(round(lead * 25))
        cands = []
        i = n_lead + 75
        while i < len(e) - 75 and len(cands) < a.n:
            if v[i - 75:i + 75].max() < 1.0 and v[i - n_lead] > 30 and e[i] + lead + 1 < e[-1]:
                cands.append(e[i]); i += 25 * 60
            else:
                i += 1
        ctrl = e[np.argmax(v > 40)]
        for k, tau in enumerate(cands):
            files = []; diffs = {}
            for name, t in [("tau", tau), ("tau_plus_lead", tau + lead), ("control_moving", ctrl)]:
                p0 = EVID / f"{tag}_stop{k}_{name}_a.png"; p1 = EVID / f"{tag}_stop{k}_{name}_b.png"
                frame(vid, t, p0); frame(vid, t + 0.5, p1); diffs[name] = float(np.abs(grey(p0) - grey(p1)).mean()); files += [p0, p1]
            montage(files, EVID / f"{tag}_stop{k}_montage.png")
            for f in files: f.unlink()
            rows.append({"tag": tag, "tau_s": round(tau, 2), "lead_in_s": lead, "telemetry_speed_at_tau": round(float(v[np.searchsorted(e, tau)]), 1),
                         "telemetry_speed_at_tau_minus_lead": round(float(v[np.searchsorted(e, tau) - n_lead]), 1),
                         "frame_change_at_tau": round(diffs["tau"], 2), "frame_change_at_tau_plus_lead": round(diffs["tau_plus_lead"], 2),
                         "frame_change_control_moving": round(diffs["control_moving"], 2),
                         "verdict": _verdict(diffs)})
            print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(HERE / "validation_output" / "released_video_stop_check.csv", index=False)
        if not cands:
            print(f"{tag}: no qualifying stop (stationary >= 3 s with > 30 km/h lead_in earlier)")
    R = pd.DataFrame(rows); R.to_csv(HERE / "validation_output" / "released_video_stop_check.csv", index=False); print(R.verdict.value_counts()); print("sessions with a verdict:", R.tag.nunique())


if __name__ == "__main__":
    main()
