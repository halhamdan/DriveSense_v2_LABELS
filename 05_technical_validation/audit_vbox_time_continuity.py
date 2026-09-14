"""
Continuity of the VBOX timeline in every native export: does the receiver's
UTC time-of-day advance in step with the `Elapsed time` counter, and does the
video time (`Avi sync time`, per segment) advance in step with elapsed time?

A logging pause would show as a jump in (UTC - UTC0) - elapsed; a video pause or
segment boundary shows in avi_sync - elapsed. The released unix_time is
reconstructed as first UTC + elapsed, so any UTC-vs-elapsed jump is a timing
error in the release for that session (EmotiBit alignment AND video).

Output: validation_output/vbox_time_continuity.csv
"""
from pathlib import Path
import numpy as np, pandas as pd

NATIVE = Path(r"C:\Users\halha\OneDrive - Durham University\Documents\VBOX_Data\Raw")
OUT = Path(__file__).resolve().parent / "validation_output" / "vbox_time_continuity.csv"


def tod(u):
    hh = np.floor(u / 10000); mm = np.floor((u % 10000) / 100); ss = u % 100
    return hh * 3600 + mm * 60 + ss


rows = []
for f in sorted(NATIVE.glob("D*_S*.csv")):
    d = pd.read_csv(f, encoding="utf-8-sig", usecols=["UTC time", "Elapsed time (s)", "Avi sync time (s)", "Avi file index", "Satellites"])
    t = tod(d["UTC time"].to_numpy(float)); t = np.where(t < t[0], t + 86400, t)  # midnight wrap
    e = d["Elapsed time (s)"].to_numpy(float); a = d["Avi sync time (s)"].to_numpy(float) / 1000; idx = d["Avi file index"].to_numpy()
    resid = (t - t[0]) - e
    dr = np.diff(resid); big = np.flatnonzero(np.abs(dr) > 0.5)
    # per-segment video residual
    seg_notes = []
    for k in np.unique(idx):
        m = idx == k; r = a[m] - e[m]; seg_notes.append(f"seg{int(k)}: n={m.sum()}, avi-elapsed range {np.ptp(r):.2f}s")
        jumps = np.flatnonzero(np.abs(np.diff(r)) > 0.5)
        if len(jumps): seg_notes.append(f"  video/elapsed jumps at elapsed {np.round(e[m][jumps][:5], 1).tolist()} sizes {np.round(np.diff(r)[jumps][:5], 1).tolist()}")
    rows.append({"tag": f.stem, "rows": len(d), "elapsed_end_s": round(e[-1], 2), "utc_span_s": round(t[-1] - t[0], 2),
                 "utc_minus_elapsed_end_s": round(resid[-1], 2), "max_abs_utc_minus_elapsed_s": round(np.abs(resid).max(), 2),
                 "n_utc_jumps_gt_0p5s": len(big), "first_utc_jump_at_elapsed_s": round(e[big[0]], 1) if len(big) else np.nan,
                 "utc_jump_sizes_s": np.round(dr[big][:5], 2).tolist(), "n_segments": len(np.unique(idx)), "min_satellites": int(d["Satellites"].min()),
                 "video_notes": " | ".join(seg_notes)})
R = pd.DataFrame(rows); R.to_csv(OUT, index=False)
bad = R[R.max_abs_utc_minus_elapsed_s > 0.5]
print(f"{len(R)} sessions; UTC-vs-elapsed discrepancy > 0.5 s in {len(bad)}:")
print(bad[["tag", "elapsed_end_s", "utc_span_s", "utc_minus_elapsed_end_s", "max_abs_utc_minus_elapsed_s", "n_utc_jumps_gt_0p5s", "first_utc_jump_at_elapsed_s", "utc_jump_sizes_s", "min_satellites"]].to_string(index=False))
print("\nvideo notes for flagged sessions:")
for _, r in bad.iterrows(): print(r.tag, r.video_notes)
print("->", OUT)
