"""
Per-session cadence of the EmotiBit NTP-style sync pings: what "coverage"
in Supplementary Table S3 does and does not mean. Coverage there is the span
between the first and last ping used, relative to the drive (bracketing);
this script adds how densely that span is sampled -- median inter-ping
interval and the longest gap between consecutive pings -- so that a reader
can see that a log may bracket a session with relatively few pings.

Reuses the file pairing and timestamp parsing of audit_emotibit_timesync_full.py.
Output: validation_output/ping_cadence_per_session.csv and a summary line.
Environment: STAGING_DATA_ROOT (per-session EmotiBit folders).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from audit_emotibit_timesync_full import find_emotibit_dir, load_sessions, ts_sent_to_unix, OUT_DIR

TQ = OUT_DIR / "timing_quality_per_session.csv"


def main():
    tq = pd.read_csv(TQ).set_index("tag")
    rows = []
    for _, s in load_sessions().iterrows():
        tag, d, n = s.tag, int(s.driver), int(s.session)
        emb = find_emotibit_dir(d, n)
        if emb is None:
            rows.append({"tag": tag, "status": "NO_DIR"}); continue
        mains = sorted([f for f in emb.glob("*.csv") if re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", f.name) and "_" not in f.stem[19:20]],
                       key=lambda f: f.stat().st_size, reverse=True)
        sync = emb / f"{mains[0].stem}_timesyncs.csv"
        ts = pd.read_csv(sync, header=0, names=["RD", "TS_received", "TS_sent", "AK", "RoundTrip", "_blank"], dtype={"TS_sent": str})
        ts = ts.dropna(subset=["TS_received", "TS_sent", "RoundTrip"])
        ts["ref_unix"] = ts["TS_sent"].apply(ts_sent_to_unix)
        ts = ts.dropna(subset=["ref_unix"]).sort_values("ref_unix")
        t = ts["ref_unix"].to_numpy(float); gaps = np.diff(t)
        dur_s = float(tq.loc[tag, "dur_min"]) * 60 if tag in tq.index else np.nan
        rows.append({"tag": tag, "n_pings_logged": len(t), "n_pings_used": int(tq.loc[tag, "pings"]) if tag in tq.index else np.nan,
                     "span_s": round(float(t[-1] - t[0]), 1) if len(t) > 1 else 0.0, "drive_s": round(dur_s, 1),
                     "coverage_pct": float(tq.loc[tag, "coverage_pct"]) if tag in tq.index else np.nan,
                     "median_interval_s": round(float(np.median(gaps)), 2) if len(gaps) else np.nan,
                     "p90_interval_s": round(float(np.percentile(gaps, 90)), 2) if len(gaps) else np.nan,
                     "longest_gap_s": round(float(gaps.max()), 1) if len(gaps) else np.nan,
                     "pings_per_min_in_span": round(len(t) / max(t[-1] - t[0], 1) * 60, 1) if len(t) > 1 else np.nan})
    R = pd.DataFrame(rows); R.to_csv(OUT_DIR / "ping_cadence_per_session.csv", index=False)
    ok = R.dropna(subset=["median_interval_s"])
    print(R.to_string(index=False))
    print(f"\n{len(ok)} sessions: pings logged median {ok.n_pings_logged.median():.0f} (range {ok.n_pings_logged.min()}-{ok.n_pings_logged.max()}); "
          f"median inter-ping interval: median {ok.median_interval_s.median():.1f} s (range {ok.median_interval_s.min():.1f}-{ok.median_interval_s.max():.1f}); "
          f"longest gap: median {ok.longest_gap_s.median():.0f} s, 90th pct {ok.longest_gap_s.quantile(.9):.0f} s, max {ok.longest_gap_s.max():.0f} s ({ok.loc[ok.longest_gap_s.idxmax(),'tag']}); "
          f"sessions with longest gap > 60 s: {(ok.longest_gap_s > 60).sum()}, > 300 s: {(ok.longest_gap_s > 300).sum()}")
    print("->", OUT_DIR / "ping_cadence_per_session.csv")


if __name__ == "__main__":
    main()
