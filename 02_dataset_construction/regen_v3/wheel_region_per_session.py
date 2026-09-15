"""
Per-session hands-on-wheel region from the RELEASED data (v3).

A first attempt fitted a circle to the wrist positions (assuming the wrists trace the
rim); it does not hold -- drivers hold fixed grips, so the wrist cloud is a compact
blob and the circle lands on the grip cluster, not the wheel. What the released data
does provide per session is the driver's habitual ON-WHEEL hand position: while the
vehicle moves faster than 30 km/h the hands are on the wheel most of the time. This
script therefore fits a 3-component Gaussian mixture to the left+right wrist positions
at speed > 30 km/h, keeps the components whose centre lies inside the (slightly
enlarged) fixed wheel area and that carry >= 15 % of the points, and defines
    on-wheel(wrist) = inside any kept component at Mahalanobis distance <= 2.5
                      OR inside the fixed region x 0.40-0.70, y 0.28-0.62.
The union keeps the fixed region's coverage of unusual grips and adds the driver's own
grip position where it falls outside that box. Every session's ellipses are drawn on a
mid-session frame (validation_output/wheel_fit_review/{tag}.png) for visual checking.

Outputs
  validation_output/wheel_region_per_session.csv  per-session components (weight, centre, axes) and status
  --score : agreement of fixed-only vs union rule on the 30 audited frames
  --apply : Side_pose.csv -> hands_on_wheel_proxy recomputed with the union rule, then a 1 s (5-sample) median
            filter; fixed-region value kept as hands_on_wheel_proxy_v2, original as hands_on_wheel_proxy_v1
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

V3 = Path(os.environ.get("V3_ROOT", "") or r"C:\Users\halha\OneDrive - Durham University\Documents\DriveSense_Packages\1_LATEST__labels_v2__release_v3__paper_v6\data\Published_Dataset_Final_v3_VIDEO_ALIGNED")
REPO = Path(__file__).resolve().parents[2]
OUTD = REPO / "05_technical_validation" / "validation_output"
REV = OUTD / "wheel_fit_review"
REV.mkdir(parents=True, exist_ok=True)
FIXED = (0.40, 0.70, 0.28, 0.62)
ENL = (0.30, 0.78, 0.20, 0.70)   # enlarged area in which a driving-hand cluster may sit
SPEED = 30.0
MAHAL = 2.5
K = 3
RNG = np.random.default_rng(20260914)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from recompute_hands_on_wheel_v3 import TRUTH  # noqa: E402  (30 audited frames: hands on wheel / 2)


def gmm(P, k=K, iters=60):
    """Small EM for a k-component full-covariance Gaussian mixture (no sklearn dependency)."""
    n = len(P)
    mu = P[RNG.choice(n, k, replace=False)]
    cov = np.array([np.cov(P.T) + 1e-4 * np.eye(2)] * k)
    w = np.full(k, 1 / k)
    for _ in range(iters):
        ll = np.zeros((n, k))
        for j in range(k):
            d = P - mu[j]
            inv = np.linalg.inv(cov[j])
            ll[:, j] = np.log(w[j] + 1e-12) - 0.5 * np.einsum("ij,jk,ik->i", d, inv, d) - 0.5 * np.log(np.linalg.det(cov[j]) + 1e-18)
        r = np.exp(ll - ll.max(1, keepdims=True))
        r /= r.sum(1, keepdims=True)
        nk = r.sum(0) + 1e-9
        w = nk / n
        mu = (r.T @ P) / nk[:, None]
        for j in range(k):
            d = P - mu[j]
            cov[j] = (r[:, j, None] * d).T @ d / nk[j] + 1e-5 * np.eye(2)
    return w, mu, cov


def components(tag):
    d, s = tag[1:].split("_S")
    pre = V3 / "Preprocessed_Dataset" / f"D{d}" / f"Session_{s}"
    p = pd.read_csv(pre / f"{tag}_Side_pose.csv")
    f = pd.read_csv(pre / f"{tag}_fused.csv", usecols=["elapsed_s", "gps_speed_kmh"])
    sp = np.interp(p.frame / 30.0, f.elapsed_s, f.gps_speed_kmh)
    det = p.pose_detected.astype(bool) & (sp > SPEED)
    P = np.vstack([p.loc[det & p[f"{w}_x"].notna(), [f"{w}_x", f"{w}_y"]].to_numpy(float) for w in ("wrist_l", "wrist_r")])
    P = P[(P[:, 0] > 0) & (P[:, 0] < 1) & (P[:, 1] > 0) & (P[:, 1] < 1)]
    if len(P) > 8000:
        P = P[RNG.choice(len(P), 8000, replace=False)]
    comps = []
    if len(P) >= 100:
        w, mu, cov = gmm(P)
        for j in range(K):
            inside = ENL[0] <= mu[j, 0] <= ENL[1] and ENL[2] <= mu[j, 1] <= ENL[3]
            ev = np.sqrt(np.linalg.eigvalsh(cov[j]))
            comps.append({"weight": float(w[j]), "cx": float(mu[j, 0]), "cy": float(mu[j, 1]), "sd_major": float(ev[1]), "sd_minor": float(ev[0]),
                          "cov": cov[j].tolist(), "kept": bool(inside and w[j] >= 0.15 and ev[1] < 0.12)})
    return p, P, comps


def on_wheel(x, y, comps):
    """Boolean Series: inside the fixed box OR within MAHAL of a kept component."""
    on = (x >= FIXED[0]) & (x <= FIXED[1]) & (y >= FIXED[2]) & (y <= FIXED[3])
    for c in comps:
        if not c["kept"]:
            continue
        inv = np.linalg.inv(np.array(c["cov"]))
        dx, dy = x - c["cx"], y - c["cy"]
        m2 = inv[0, 0] * dx * dx + 2 * inv[0, 1] * dx * dy + inv[1, 1] * dy * dy
        on = on | (m2 <= MAHAL ** 2)
    return on.fillna(False)


def draw(tag, p, P, comps):
    import cv2
    d, s = tag[1:].split("_S")
    vid = V3 / "Raw_Dataset" / f"D{d}" / f"Session_{s}" / f"{tag}_Side_blurred.mp4"
    if not vid.exists():
        return
    mid = int(p.frame.iloc[len(p) // 2])
    png = REV / f"{tag}.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{mid / 30:.3f}", "-i", str(vid), "-frames:v", "1", str(png)], check=True)
    im = cv2.imread(str(png))
    for (x, y) in P[:: max(1, len(P) // 800)]:
        cv2.circle(im, (int(x * 960), int(y * 540)), 2, (0, 0, 255), -1)
    cv2.rectangle(im, (int(FIXED[0] * 960), int(FIXED[2] * 540)), (int(FIXED[1] * 960), int(FIXED[3] * 540)), (0, 255, 255), 1)
    for c in comps:
        cov = np.array(c["cov"]) * np.array([[960 * 960, 960 * 540], [960 * 540, 540 * 540]])
        vals, vecs = np.linalg.eigh(cov)
        ang = float(np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1])))
        axes = (int(MAHAL * np.sqrt(vals[1])), int(MAHAL * np.sqrt(vals[0])))
        cv2.ellipse(im, (int(c["cx"] * 960), int(c["cy"] * 540)), axes, ang, 0, 360, (0, 255, 0) if c["kept"] else (128, 128, 128), 2 if c["kept"] else 1)
    cv2.imwrite(str(png), im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    tags = a.sessions or sorted(f.stem.replace("_fused", "") for f in (V3 / "Preprocessed_Dataset").rglob("*_fused.csv")
                                if (f.parent / f"{f.stem.replace('_fused', '')}_Side_pose.csv").exists())
    rows, per = [], {}
    for t in tags:
        p, P, comps = components(t)
        per[t] = comps
        draw(t, p, P, comps)
        kept = [c for c in comps if c["kept"]]
        rows.append({"tag": t, "n_points": len(P), "n_kept": len(kept), "kept_weight": round(sum(c["weight"] for c in kept), 2),
                     "kept_centres": ";".join(f"({c['cx']:.2f},{c['cy']:.2f})" for c in kept)})
        if a.apply:
            det = p.pose_detected.astype(bool)
            v = ((on_wheel(p.wrist_l_x, p.wrist_l_y, comps).astype(float) + on_wheel(p.wrist_r_x, p.wrist_r_y, comps).astype(float)) / 2).where(det, np.nan)
            sm = v.rolling(5, center=True, min_periods=1).median().round(1).where(det, np.nan)
            if "hands_on_wheel_proxy_v2" not in p.columns:
                p["hands_on_wheel_proxy_v2"] = p["hands_on_wheel_proxy"]
            p["hands_on_wheel_proxy"] = sm
            order = [c for c in p.columns if c not in ("hands_on_wheel_proxy_v1", "hands_on_wheel_proxy_v2")] + ["hands_on_wheel_proxy_v2", "hands_on_wheel_proxy_v1"]
            d, s = t[1:].split("_S")
            p[order].to_csv(V3 / "Preprocessed_Dataset" / f"D{d}" / f"Session_{s}" / f"{t}_Side_pose.csv", index=False)
            rows[-1]["mean_proxy"] = round(float(sm.mean()), 3)
    R = pd.DataFrame(rows)
    R.to_csv(OUTD / "wheel_region_per_session.csv", index=False)
    json.dump(per, open(OUTD / "wheel_region_components.json", "w"), indent=1)
    print(R.to_string(index=False))
    if a.score:
        sel = pd.read_csv(OUTD / "visual_feature_audit" / "selection.csv")
        sel = sel[sel.camera == "side"]
        out = []
        for _, r in sel.iterrows():
            if r.tag not in per:
                continue
            d, s = r.tag[1:].split("_S")
            q = pd.read_csv(V3 / "Preprocessed_Dataset" / f"D{d}" / f"Session_{s}" / f"{r.tag}_Side_pose.csv")
            q = q[q.frame == r.frame].iloc[[0]]
            fixed = ((on_wheel(q.wrist_l_x, q.wrist_l_y, []).astype(float) + on_wheel(q.wrist_r_x, q.wrist_r_y, []).astype(float)) / 2).iloc[0]
            union = ((on_wheel(q.wrist_l_x, q.wrist_l_y, per[r.tag]).astype(float) + on_wheel(q.wrist_r_x, q.wrist_r_y, per[r.tag]).astype(float)) / 2).iloc[0]
            out.append({"tag": r.tag, "truth": TRUTH[r.tag], "fixed": fixed, "union": union})
        S = pd.DataFrame(out)
        print(f"\n30-frame audit: fixed region agrees {(S.fixed == S.truth).sum()}/{len(S)}; union (fixed + driver's grip clusters) agrees {(S.union == S.truth).sum()}/{len(S)} "
              f"(under {(S.union < S.truth).sum()}, over {(S.union > S.truth).sum()})")
        print(S[S.fixed != S.union].to_string(index=False))
        S.to_csv(OUTD / "hands_on_wheel_union_score.csv", index=False)


if __name__ == "__main__":
    main()
