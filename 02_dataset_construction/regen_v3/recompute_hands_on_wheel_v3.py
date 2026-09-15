"""
Re-derives hands_on_wheel_proxy for release v3 from the RELEASED wrist coordinates
(Side_pose.csv) with a wheel region fitted to this camera framing, and scores the
old and new definitions against the 30 audited side frames
(05_technical_validation/validation_output/visual_feature_audit/, seed 20260913),
whose hand placement was judged by eye (TRUTH below: hands on the wheel / 2).

The released files carry no landmark-visibility column, so the recomputed proxy
is (in_region(left wrist) + in_region(right wrist)) / 2 over the released
coordinates; a missing wrist counts as not on the wheel; rows without a pose
detection stay missing. The old column is kept as hands_on_wheel_proxy_v1.

Usage:
  python recompute_hands_on_wheel_v3.py            # analysis only
  python recompute_hands_on_wheel_v3.py --apply    # rewrite the 75 v3 Side_pose.csv files
"""
from __future__ import annotations

import argparse, itertools, json, os
from pathlib import Path
import numpy as np, pandas as pd

V3 = Path(os.environ.get("V3_ROOT", "") or r"C:\Users\halha\OneDrive - Durham University\Documents\DriveSense_Packages\1_LATEST__labels_v2__release_v3__paper_v6\data\Published_Dataset_Final_v3_VIDEO_ALIGNED")
REPO = Path(__file__).resolve().parents[2]
AUD = REPO / "05_technical_validation" / "validation_output" / "visual_feature_audit"
OUT = REPO / "05_technical_validation" / "validation_output" / "hands_on_wheel_recalibration.csv"
OLD = (0.25, 0.75, 0.55, 1.0)   # x0, x1, y0, y1 (normalised image, origin top-left)
# reviewer's judgement of the 30 audited side frames (2026-09-14): number of hands on the wheel / 2
TRUTH = {"D15_S2": .5, "D18_S3": .5, "D17_S4": 1, "D4_S1": 1, "D16_S2": .5, "D3_S4": 1, "D17_S3": 1, "D15_S1": 0, "D4_S2": .5, "D19_S4": .5,
         "D15_S3": .5, "D4_S4": 1, "D9_S1": .5, "D5_S2": 1, "D14_S4": .5, "D5_S1": 1, "D7_S1": 1, "D16_S3": 1, "D12_S4": 1, "D17_S1": 0,
         "D13_S1": .5, "D18_S2": 1, "D5_S4": .5, "D8_S4": .5, "D3_S1": .5, "D12_S1": 1, "D8_S2": .5, "D9_S2": .5, "D7_S3": 1, "D3_S3": 1}


def pose_path(tag):
    d, s = tag[1:].split("_S"); return V3 / "Preprocessed_Dataset" / f"D{d}" / f"Session_{s}" / f"{tag}_Side_pose.csv"


def proxy(df, box):
    x0, x1, y0, y1 = box
    def inr(x, y): return ((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)).astype(float)
    v = (inr(df.wrist_l_x, df.wrist_l_y).fillna(0) + inr(df.wrist_r_x, df.wrist_r_y).fillna(0)) / 2.0
    return v.where(df.pose_detected.astype(bool), np.nan)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--apply", action="store_true"); a = ap.parse_args()
    sel = pd.read_csv(AUD / "selection.csv"); sel = sel[sel.camera == "side"]
    rows = []
    for _, r in sel.iterrows():
        p = pd.read_csv(pose_path(r.tag)); q = p[p.frame == r.frame].iloc[0]
        rows.append({"tag": r.tag, "frame": int(r.frame), "truth": TRUTH[r.tag], "old": float(q.hands_on_wheel_proxy),
                     "wl": (q.wrist_l_x, q.wrist_l_y), "wr": (q.wrist_r_x, q.wrist_r_y)})
    F = pd.DataFrame(rows)
    def score(box):
        def inr(pt): x, y = pt; return float(box[0] <= x <= box[1] and box[2] <= y <= box[3]) if np.isfinite(x) else 0.0
        pred = F.apply(lambda r: (inr(r.wl) + inr(r.wr)) / 2, axis=1); return int((pred == F.truth).sum()), pred
    s_old, _ = score(OLD); print(f"old rectangle {OLD}: agrees {s_old}/30 (from CSV values: {(F.old == F.truth).sum()}/30; under {(F.old < F.truth).sum()}, over {(F.old > F.truth).sum()})")
    # wrists that are on the wheel according to the judgement, to see where the wheel is
    on = []
    for _, r in F.iterrows():
        pts = [r.wl, r.wr]
        if r.truth == 1: on += pts
        elif r.truth == .5: on += [min(pts, key=lambda p: (p[0] - 0.55) ** 2 + (p[1] - 0.45) ** 2)]  # the wrist nearer the wheel centre
    on = np.array([p for p in on if np.isfinite(p[0])]); print(f"wrists judged on the wheel: n={len(on)}, x {on[:,0].min():.2f}-{on[:,0].max():.2f} (p10-p90 {np.percentile(on[:,0],10):.2f}-{np.percentile(on[:,0],90):.2f}), y {on[:,1].min():.2f}-{on[:,1].max():.2f} (p10-p90 {np.percentile(on[:,1],10):.2f}-{np.percentile(on[:,1],90):.2f})")
    apriori = (0.40, 0.70, 0.28, 0.62); s_ap, _ = score(apriori); print(f"a-priori wheel box {apriori}: agrees {s_ap}/30")
    best = (0, None)
    for x0, x1, y0, y1 in itertools.product(np.arange(0.30, 0.56, 0.05), np.arange(0.55, 0.81, 0.05), np.arange(0.20, 0.46, 0.05), np.arange(0.50, 0.76, 0.05)):
        s, _ = score((x0, x1, y0, y1))
        if s > best[0] or (s == best[0] and best[1] and (x1 - x0) * (y1 - y0) < (best[1][1] - best[1][0]) * (best[1][3] - best[1][2])): best = (s, (round(x0, 2), round(x1, 2), round(y0, 2), round(y1, 2)))
    print(f"grid search (0.05 steps): best {best[1]} agrees {best[0]}/30")
    # adopt the a-priori box measured from the frames (wheel rim spans ~x 0.42-0.68, y 0.30-0.58 in this framing),
    # not the grid optimum: same agreement, and it was not tuned on the judgements
    box = apriori; s_new, pred = score(box); F["new"] = pred
    print(f"chosen box {box}: agrees {s_new}/30; under {(F.new < F.truth).sum()}, over {(F.new > F.truth).sum()}"); print(F[["tag", "truth", "old", "new"]].to_string(index=False))
    F.to_csv(OUT, index=False)
    if not a.apply: return
    # ---- apply to all v3 pose files
    files = sorted((V3 / "Preprocessed_Dataset").rglob("*_Side_pose.csv")); stats = []
    for f in files:
        p = pd.read_csv(f)
        if "hands_on_wheel_proxy_v1" not in p.columns: p["hands_on_wheel_proxy_v1"] = p["hands_on_wheel_proxy"]
        p["hands_on_wheel_proxy"] = proxy(p, box)
        cols = [c for c in p.columns if c != "hands_on_wheel_proxy_v1"] + ["hands_on_wheel_proxy_v1"]; p = p[cols]; p.to_csv(f, index=False)
        det = p.pose_detected.astype(bool); stats.append({"tag": f.stem.replace("_Side_pose", ""), "mean_old": p.loc[det, "hands_on_wheel_proxy_v1"].mean(), "mean_new": p.loc[det, "hands_on_wheel_proxy"].mean(),
                                                          "frac_new_1": (p.loc[det, "hands_on_wheel_proxy"] == 1).mean(), "frac_new_0": (p.loc[det, "hands_on_wheel_proxy"] == 0).mean()})
    S = pd.DataFrame(stats); S.to_csv(OUT.with_name("hands_on_wheel_recomputed_per_session.csv"), index=False)
    print(f"\napplied to {len(files)} files: mean proxy old {S.mean_old.mean():.2f} -> new {S.mean_new.mean():.2f}; new ==1 in {S.frac_new_1.mean()*100:.0f}% of detected rows, ==0 in {S.frac_new_0.mean()*100:.0f}%")
    json.dump({"box_x0_x1_y0_y1": box, "agreement_30_frames": s_new, "old_box": OLD, "old_agreement": s_old}, open(OUT.with_name("hands_on_wheel_box.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
