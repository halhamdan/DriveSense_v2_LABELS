"""
Blind rating pack for validating hands_on_wheel_proxy with a second rater.

Draws 100 side-camera frames from the v3 release (seed 20260914): 50 while the
vehicle is moving (> 30 km/h) and 50 while stopped or slow (< 5 km/h), spread over
sessions (at most 2 per session), from rows with a pose detection. Frames are saved
WITHOUT any overlay so the rater is not influenced by the released keypoints or
proxy value, tiled 10 per contact sheet in a fixed order, and listed in
rating_template.csv with a blank `hands_on_wheel_rater` column.

Rater instruction (write 0, 1 or 2 per frame): number of the driver's hands that
are on the steering wheel (touching or gripping the rim, hub or spokes). If a hand
is hidden behind the wheel and cannot be judged, write "u". Do not open the
released CSVs while rating.

The released values are written to a SEPARATE file (rating_key.csv) for the
scoring step (score_hands_on_wheel_ratings.py), not shown on the sheets.

Usage: DATASET_ROOT=/path/to/v3 python make_hands_on_wheel_rating_pack.py
Output: validation_output/hands_on_wheel_rating/{sheet_*.png, rating_template.csv, rating_key.csv}
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("DATASET_ROOT", "") or r"C:\Users\halha\OneDrive - Durham University\Documents\Published_Dataset_Final_v3_VIDEO_ALIGNED")
OUT = Path(__file__).resolve().parent / "validation_output" / "hands_on_wheel_rating"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260914
N_MOVING = 50
N_STOPPED = 50
FPS = 30


def main():
    rng = np.random.default_rng(SEED)
    sessions = sorted(f.stem.replace("_fused", "") for f in (ROOT / "Preprocessed_Dataset").rglob("*_fused.csv"))
    cands = {"moving": [], "stopped": []}
    for t in sessions:
        d, s = t[1:].split("_S")
        pre = ROOT / "Preprocessed_Dataset" / f"D{d}" / f"Session_{s}"
        raw = ROOT / "Raw_Dataset" / f"D{d}" / f"Session_{s}"
        pf, vf = pre / f"{t}_Side_pose.csv", raw / f"{t}_Side_blurred.mp4"
        if not (pf.exists() and vf.exists()):
            continue
        p = pd.read_csv(pf)
        f = pd.read_csv(pre / f"{t}_fused.csv", usecols=["elapsed_s", "gps_speed_kmh"])
        p["speed"] = np.interp(p.frame / FPS, f.elapsed_s, f.gps_speed_kmh)
        p = p[p.pose_detected.astype(bool)]
        for cond, mask in (("moving", p.speed > 30), ("stopped", p.speed < 5)):
            q = p[mask]
            if len(q) == 0:
                continue
            for _ in range(2):  # at most 2 candidates per session per condition
                r = q.iloc[int(rng.integers(len(q)))]
                cands[cond].append({"tag": t, "frame": int(r.frame), "speed_kmh": round(float(r.speed), 1), "condition": cond,
                                    "video": str(vf), "released_proxy": float(r.hands_on_wheel_proxy),
                                    "released_proxy_v1": float(r.get("hands_on_wheel_proxy_v1", np.nan))})
    sel = []
    for cond, n in (("moving", N_MOVING), ("stopped", N_STOPPED)):
        c = pd.DataFrame(cands[cond]).drop_duplicates(["tag", "frame"])
        # one per session first, then a second round if needed
        first = c.groupby("tag").head(1)
        pick = first.sample(min(n, len(first)), random_state=SEED)
        if len(pick) < n:
            rest = c.drop(pick.index)
            pick = pd.concat([pick, rest.sample(n - len(pick), random_state=SEED)])
        sel.append(pick)
    S = pd.concat(sel).sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffle so conditions are mixed
    S.insert(0, "frame_id", [f"F{i + 1:03d}" for i in range(len(S))])
    files = []
    for _, r in S.iterrows():
        png = OUT / f"{r.frame_id}.png"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{r.frame / FPS:.4f}", "-i", r.video, "-frames:v", "1", "-vf", "scale=640:360", str(png)], check=True)
        files.append(png)
    sheets = []
    for k in range(0, len(files), 10):
        chunk = files[k:k + 10]
        args = ["ffmpeg", "-v", "error", "-y"]
        for f in chunk:
            args += ["-i", str(f)]
        n = len(chunk)
        layout = "|".join(f"{(j % 5) * 640}_{(j // 5) * 360}" for j in range(n))
        out = OUT / f"sheet_{k // 10 + 1:02d}_frames_{k + 1:03d}-{k + n:03d}.png"
        args += ["-filter_complex", "".join(f"[{j}]" for j in range(n)) + f"xstack=inputs={n}:layout={layout}:fill=black", str(out)]
        subprocess.run(args, check=True)
        sheets.append(out.name)
    S["sheet"] = [sheets[i // 10] for i in range(len(S))]
    S["position_on_sheet"] = [f"row {i % 10 // 5 + 1}, column {i % 5 + 1}" for i in range(len(S))]
    S[["frame_id", "sheet", "position_on_sheet"]].assign(hands_on_wheel_rater="").to_csv(OUT / "rating_template.csv", index=False)
    S.drop(columns=["video"]).to_csv(OUT / "rating_key.csv", index=False)
    (OUT / "INSTRUCTIONS.md").write_text(
        "# Hands-on-wheel rating (blind)\n\n"
        "Open each `sheet_*.png` (10 frames: 2 rows x 5 columns; frame IDs run left to right, top row first, as listed in "
        "`rating_template.csv`). For every frame write in the `hands_on_wheel_rater` column the number of the driver's hands "
        "that are ON the steering wheel (touching or gripping rim, hub or spokes): 0, 1 or 2. If a hand cannot be judged "
        "(hidden), write u. Do not look at `rating_key.csv` or the released CSVs until you have finished. "
        "Save the template as `ratings_<yourinitials>.csv` in this folder.\n", encoding="utf-8")
    for f in files:
        f.unlink()
    print(f"{len(S)} frames ({(S.condition == 'moving').sum()} moving, {(S.condition == 'stopped').sum()} stopped) from {S.tag.nunique()} sessions -> {OUT}")


if __name__ == "__main__":
    main()
