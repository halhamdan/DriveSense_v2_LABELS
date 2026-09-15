"""
Generates Supplementary Table S3: per-session timing quality, stating for each
session what is MEASURED, what is ESTIMATED and what is UNKNOWN about the
alignment between the two independent recording systems (VBOX HD2 on GNSS
time; EmotiBit anchored to the logging laptop's clock via NTP-style sync
pings).

Inputs (all produced by other scripts in this repository):
  regenerated_v3/regeneration_report.csv          -- anchor actually applied per session
  validation_output/emotibit_timesync_full_report.csv -- ping statistics per session
  <DATASET_ROOT>/dataset_metadata.json            -- drive duration

Columns:
  Duration (min)      trimmed drive duration                                   [measured]
  Pings               number of sync pings used for the median anchor          [measured]
  Coverage (%)        span of the ping log as % of the drive                   [measured]
  Ping MAD (ms)       robust scatter of ping offsets about the anchor          [measured]
  Max step (s)        largest sustained offset step in the ping log            [measured]
  Longest gap (s)     longest interval between consecutive pings (from
                      make_ping_cadence_table.py; pings come in 5.1 s bursts
                      with long silent intervals)                                [measured]
  Variability (s)     observed timing-variability indicator
                      max(step, 2 x MAD, 0.10 s)  (CSV column kept as bound_s)   [derived]
  Notes               span < 50 % (anchor from a short window), VBOX first-
                      sample GPS-time gap (D1_S4, D12_S2), two raw recordings

The laptop-clock-to-UTC error at recording time is UNKNOWN for every session
and is not in the table. The variability indicator is NOT a bound on the
absolute EmotiBit-to-VBOX alignment error (a consistently wrong laptop clock
shows no scatter and no step); it was called an "indicative bound" in
manuscript v5 and the first v6 draft and renamed on 2026-09-13.

Usage:
    python make_timing_quality_table.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", "") or
                    r"C:\Users\halha\OneDrive - Durham University\Documents\DriveSense_Packages\3_OLD_versions\data\Published_Dataset_Final_v1_ORIGINAL_20260909")
REGEN = BASE.parents[1] / "regenerated_v3" / "regeneration_report.csv"
AUDIT = BASE / "validation_output" / "emotibit_timesync_full_report.csv"
OUT = BASE / "validation_output" / "si_timing_quality_table.tex"
OUT_CSV = BASE / "validation_output" / "timing_quality_per_session.csv"

GPS_FIRST_SAMPLE_GAP = {"D1_S4", "D12_S2"}
TWO_RECORDINGS = {"D1_S1", "D3_S2", "D4_S1", "D16_S3"}


def esc(x):
    return str(x).replace("_", "\\_")


def main():
    rep = pd.read_csv(REGEN).set_index("tag")
    aud = pd.read_csv(AUDIT).set_index("tag")
    meta = {s["tag"]: s for s in json.load(open(DATASET_ROOT / "dataset_metadata.json", encoding="utf-8"))["sessions"]}
    T = pd.DataFrame(index=rep.index)
    T["driver"] = [int(t[1:].split("_")[0]) for t in T.index]
    T["session"] = [int(t.split("_S")[1]) for t in T.index]
    T["dur_min"] = [meta[t]["drive_duration_s"] / 60.0 for t in T.index]
    T["pings"] = rep.n_pings_used.astype(int)
    T["coverage_pct"] = (rep.calibration_span_s / 60.0 / T.dur_min * 100).clip(upper=100)
    T["mad_ms"] = aud.anchor_only_mad_ms.reindex(T.index)
    T["step_s"] = rep.max_step_s.abs()
    T["bound_s"] = np.maximum.reduce([T.step_s.to_numpy(), 2 * T.mad_ms.to_numpy() / 1000.0, np.full(len(T), 0.10)])
    # Column name kept as bound_s in the CSV for backwards compatibility; it is reported as the
    # "observed timing-variability indicator" -- NOT a bound on absolute alignment error.
    cad = BASE / "validation_output" / "ping_cadence_per_session.csv"
    if cad.exists():
        C = pd.read_csv(cad).set_index("tag")
        T["longest_gap_s"] = C.longest_gap_s.reindex(T.index)
    else:
        T["longest_gap_s"] = np.nan
    notes = []
    for t, r in T.iterrows():
        n = []
        if r.coverage_pct < 50: n.append(f"ping span covers {r.coverage_pct:.0f}\\% of drive")
        if t in GPS_FIRST_SAMPLE_GAP: n.append("VBOX first-sample GPS-time gap")
        if t in TWO_RECORDINGS: n.append("two raw EmotiBit recordings; larger used")
        notes.append("; ".join(n))
    T["notes"] = notes
    T = T.sort_values(["driver", "session"])
    T.round(3).to_csv(OUT_CSV)

    hdr = r"Session & Dur.\ (min) & Pings & Span (\%) & Longest gap (s) & Ping MAD (ms) & Max step (s) & Variability (s) & Notes \\"
    cap = (r"\caption{Per-session timing evidence for the alignment between the two independent recording systems. "
           r"Within the VBOX HD2, telemetry, GNSS and both video streams share one GNSS-disciplined clock and are exact "
           r"to the sample/frame. The EmotiBit device clock is mapped to UTC by the session-median offset over NTP-style "
           r"sync pings exchanged with the logging laptop (Methods). Measured columns: number of pings used; the span "
           r"bracketed by the first and last ping as a percentage of the drive (bracketing, not continuous sampling: pings "
           r"arrive in bursts at a 5.1\,s cadence separated by silent intervals when the Wi-Fi link dropped); the longest "
           r"gap between consecutive pings; robust scatter (median absolute deviation) of the per-ping offsets about the "
           r"anchor; largest sustained offset step in the log. Derived column: the observed timing-variability indicator "
           r"$\max(\text{step},\,2\times\text{MAD},\,0.10\,\text{s})$, which summarises how much the laptop-to-device offset "
           r"was seen to vary during the recording. It is not a bound on the EmotiBit-to-VBOX alignment error: the laptop "
           r"clock's own offset from UTC at recording time was not measured for any session, so the absolute cross-device "
           r"alignment uncertainty is unknown and is not tabulated. Notes flag sessions whose ping span covers less than half "
           r"of the drive (anchor estimated from a short window), the two sessions with a 0.12--0.16\,s gap between the first "
           r"two VBOX GPS timestamps (a VBOX-side anchor uncertainty of that size), and sessions with two raw EmotiBit "
           r"recordings. Bold rows: variability indicator above 0.30\,s.}")
    lines = [r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}", r"\renewcommand{\arraystretch}{0.9}", r"\begin{longtable}{@{}lrrrrrrrp{3.4cm}@{}}", cap,
             r"\label{tab:si-timing}\\", r"\toprule", hdr, r"\midrule", r"\endfirsthead",
             r"\multicolumn{9}{c}{\tablename\ \thetable{} -- continued}\\", r"\toprule", hdr, r"\midrule", r"\endhead",
             r"\bottomrule", r"\endfoot"]
    for t, r in T.iterrows():
        gap = f"{r.longest_gap_s:.0f}" if np.isfinite(r.longest_gap_s) else "--"
        cells = [esc(t), f"{r.dur_min:.1f}", f"{r.pings}", f"{r.coverage_pct:.0f}", gap, f"{r.mad_ms:.0f}", f"{r.step_s:.2f}", f"{r.bound_s:.2f}", r.notes]
        if r.bound_s > 0.3:
            cells = [rf"\textbf{{{c}}}" if c else c for c in cells]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\end{longtable}")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(T)} rows to {OUT}")
    print(f"bound (s): median {T.bound_s.median():.2f}, 90th pct {T.bound_s.quantile(.9):.2f}, max {T.bound_s.max():.2f} ({T.bound_s.idxmax()}); "
          f"sessions with bound <= 0.15 s: {(T.bound_s <= 0.15).sum()}, 0.15-0.30: {((T.bound_s > 0.15) & (T.bound_s <= 0.30)).sum()}, > 0.30: {(T.bound_s > 0.30).sum()}")
    print(f"ping MAD (ms): median {T.mad_ms.median():.0f}, 90th pct {T.mad_ms.quantile(.9):.0f}; coverage < 50%: {(T.coverage_pct < 50).sum()} sessions")


if __name__ == "__main__":
    main()
