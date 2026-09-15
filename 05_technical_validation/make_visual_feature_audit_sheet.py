"""
Reproducibly selected 60-frame qualitative audit of the released visual features:
draws, on the released (de-identified) frames, the released face box
(Front_emotions.csv) and the released wrist/shoulder keypoints with the
hands_on_wheel_proxy value (Side_pose.csv), and writes contact sheets for
visual review. Selection: seed 20260913, 30 front + 30 side frames, stratified
over sessions (each session at most once per camera), restricted to rows with a
detection so that the *quality* of the features is judged, not their coverage
(coverage is reported separately in Table 7).

Usage: DATASET_ROOT=/path/to/release python make_visual_feature_audit_sheet.py
Output: validation_output/visual_feature_audit/{selection.csv, sheet_front_*.png, sheet_side_*.png}
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("DATASET_ROOT", "") or r"C:\Users\halha\OneDrive - Durham University\Documents\DriveSense_Packages\1_LATEST__labels_v2__release_v3__paper_v6\data\Published_Dataset_Final_v3_VIDEO_ALIGNED")
OUT = Path(__file__).resolve().parent / "validation_output" / "visual_feature_audit"; OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260913; N_PER_CAM = 30; FPS = 30; W, H = 960, 540


def grab(video: Path, frame: int, out: Path):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{frame / FPS:.4f}", "-i", str(video), "-frames:v", "1", str(out)], check=True)


def draw(png: Path, boxes, points, label):
    """Annotate with ffmpeg drawbox/drawtext (no OpenCV/PIL needed)."""
    # drawbox only (ffmpeg's drawtext needs a font configuration that is absent on this machine);
    # the frame label is carried in the file name and selection.csv. Colours: face box yellow,
    # wrists red, shoulders cyan.
    filt = []
    for (x, y, w, h) in boxes: filt.append(f"drawbox=x={x}:y={y}:w={w}:h={h}:color=yellow@0.9:t=3")
    for (x, y, name) in points:
        col = "red" if "wrist" in name else "cyan"
        filt.append(f"drawbox=x={x - 7}:y={y - 7}:w=14:h=14:color={col}@0.9:t=fill")
    tmp = png.with_suffix(".ann.png")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(png), "-vf", ",".join(filt), str(tmp)], check=True); tmp.replace(png)


def main():
    rng = np.random.default_rng(SEED)
    sessions = sorted(f.stem.replace("_fused", "") for f in (ROOT / "Preprocessed_Dataset").rglob("*_fused.csv"))
    sel = []
    for cam, n in (("front", N_PER_CAM), ("side", N_PER_CAM)):
        pool = [t for t in sessions if (ROOT / "Raw_Dataset" / f"D{t[1:].split('_S')[0]}" / f"Session_{t.split('_S')[1]}" / f"{t}_{'Front' if cam == 'front' else 'Side'}_blurred.mp4").exists()
                and (cam == "front" or (ROOT / "Preprocessed_Dataset" / f"D{t[1:].split('_S')[0]}" / f"Session_{t.split('_S')[1]}" / f"{t}_Side_pose.csv").exists())]
        for t in rng.choice(pool, size=min(n, len(pool)), replace=False):
            d, s = t[1:].split("_S"); pre = ROOT / "Preprocessed_Dataset" / f"D{d}" / f"Session_{s}"; raw = ROOT / "Raw_Dataset" / f"D{d}" / f"Session_{s}"
            if cam == "front":
                e = pd.read_csv(pre / f"{t}_Front_emotions.csv"); e = e[e.face_x.notna()]
                r = e.iloc[int(rng.integers(len(e)))]; sel.append({"camera": cam, "tag": t, "frame": int(r.frame), "video": str(raw / f"{t}_Front_blurred.mp4"),
                                                                    "face_box": [float(r.face_x), float(r.face_y), float(r.face_w), float(r.face_h)], "dominant": str(r.get("dominant_emotion", ""))})
            else:
                p = pd.read_csv(pre / f"{t}_Side_pose.csv"); p = p[p.pose_detected.astype(bool)]
                r = p.iloc[int(rng.integers(len(p)))]
                pts = {k: (float(r[f"{k}_x"]), float(r[f"{k}_y"])) for k in ("wrist_l", "wrist_r", "shoulder_l", "shoulder_r") if f"{k}_x" in r}
                sel.append({"camera": cam, "tag": t, "frame": int(r.frame), "video": str(raw / f"{t}_Side_blurred.mp4"), "points": pts,
                            "hands_on_wheel_proxy": float(r.get("hands_on_wheel_proxy", np.nan))})
    pd.DataFrame(sel).to_csv(OUT / "selection.csv", index=False)
    files = {"front": [], "side": []}
    for i, s in enumerate(sel):
        png = OUT / f"{s['camera']}_{i:02d}_{s['tag']}_f{s['frame']}.png"; grab(Path(s["video"]), s["frame"], png)
        if s["camera"] == "front":
            x, y, w, h = s["face_box"]
            if w <= 1.5 and h <= 1.5: x, y, w, h = x * W, y * H, w * W, h * H  # normalised -> pixels
            draw(png, [(int(x), int(y), int(w), int(h))], [], f"{s['tag']} f{s['frame']} {s['dominant']}")
        else:
            pts = [(int(v[0] * W) if v[0] <= 1.5 else int(v[0]), int(v[1] * H) if v[1] <= 1.5 else int(v[1]), k.replace("_", "")) for k, v in s["points"].items()]
            draw(png, [], pts, f"{s['tag']} f{s['frame']} how={s['hands_on_wheel_proxy']:.1f}")
        files[s["camera"]].append(png)
    for cam, fl in files.items():
        for k in range(0, len(fl), 6):
            chunk = fl[k:k + 6]; args = ["ffmpeg", "-v", "error", "-y"]
            for f in chunk: args += ["-i", str(f)]
            n = len(chunk); layout = "|".join(f"{(j % 3) * W}_{(j // 3) * H}" for j in range(n))
            args += ["-filter_complex", "".join(f"[{j}]" for j in range(n)) + f"xstack=inputs={n}:layout={layout}:fill=black,scale=1440:-1", str(OUT / f"sheet_{cam}_{k // 6:02d}.png")]
            subprocess.run(args, check=True)
    print(f"{len(sel)} frames -> {OUT}")


if __name__ == "__main__":
    main()
