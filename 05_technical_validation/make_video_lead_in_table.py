"""
Per-session video lead-in: the VBOX HD2 starts its video file(s) before the
first logged telemetry sample; Racelogic's `Avi sync time` column gives, for
each telemetry row, the corresponding time in the video (ms). Its value at row 0
is therefore the number of seconds of video that precede telemetry row 0.

The trimming pipeline (04_session_trimming/apply_trim.py, build_raw_tier.py)
mapped telemetry time to video frames as round((t - t0_raw) * fps), i.e. it
assumed video frame 0 coincides with telemetry row 0. Verified on D1_S1 against
the burned-in speedometer overlay of the raw video (validation_output/
video_offset_evidence/): the released video, Front_emotions.csv and
Side_pose.csv therefore show, at released elapsed time tau, the scene at
telemetry time tau - lead_in. This script tabulates lead_in for every session
from the native VBOX CSV exports so that the release can be corrected
(re-trim with frame = round((t - t0_raw + lead_in) * fps)) or, failing that,
documented per session.

Output: validation_output/video_lead_in_per_session.csv
Environment: VBOX_NATIVE_DIR (default OneDrive .../Documents/VBOX_Data/Raw)
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

NATIVE = Path(os.environ.get("VBOX_NATIVE_DIR", "") or r"C:\Users\halha\OneDrive - Durham University\Documents\VBOX_Data\Raw")
OUT = Path(__file__).resolve().parent / "validation_output" / "video_lead_in_per_session.csv"

rows = []
for f in sorted(NATIVE.glob("D*_S*.csv")):
    d = pd.read_csv(f, encoding="utf-8-sig", usecols=["Avi sync time (s)", "Avi file index", "Elapsed time (s)"])
    a = d["Avi sync time (s)"].to_numpy(float) / 1000.0; e = d["Elapsed time (s)"].to_numpy(float); idx = d["Avi file index"].to_numpy()
    first_seg = idx == idx[0]
    # within the first video segment, avi_sync - elapsed should be constant (= lead-in)
    resid = a[first_seg] - e[first_seg]
    rows.append({"tag": f.stem, "lead_in_s": round(float(a[0]), 3), "lead_in_median_first_segment_s": round(float(np.median(resid)), 3),
                 "lead_in_spread_ms": round(float(np.ptp(resid)) * 1000, 1), "n_video_segments": int(len(np.unique(idx))),
                 "telemetry_rows": len(d), "telemetry_s": round(float(e[-1]), 2)})
R = pd.DataFrame(rows); R.to_csv(OUT, index=False)
print(R.to_string(index=False))
print(f"\n{len(R)} sessions: lead-in median {R.lead_in_s.median():.2f} s, range {R.lead_in_s.min():.2f}-{R.lead_in_s.max():.2f} s; "
      f"within-segment spread max {R.lead_in_spread_ms.max():.0f} ms; sessions with >1 video segment: {(R.n_video_segments > 1).sum()}")
print("->", OUT)
