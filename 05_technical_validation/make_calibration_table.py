"""
Threshold-calibration drives: manoeuvre table for the manuscript and the
release (NOT an independent validation of the released rule -- the same drives
were used to choose the thresholds, the native exports carry no CAN speed or
throttle channel, so GNSS speed stands in for the CAN corroboration and the
throttle gate is disabled; see the manuscript's "Threshold calibration and
manoeuvre recovery").

Inputs: validation_session{,2,3}.csv in the paper folder (native VBOX exports of
the three drives, same Renault Koleos as the study).
Outputs:
  validation_output/calibration_manoeuvres.csv   one row per event the final rule
                                                 labels on each drive (reference
                                                 intervals users can inspect)
  validation_output/calibration_drives.csv       one row per drive
  validation_output/si_calibration_table.tex     compact LaTeX table
  <RELEASE_ROOT>/Calibration_Drives/             copies of the three native CSVs
                                                 + the two tables (if RELEASE_ROOT set)
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01_annotation"))
from label_harsh_events_v2 import PARAMS_V2  # noqa: E402
from label_rule_variants import label_df, events, smoothed_signals  # noqa: E402

PAPER = HERE.parents[1]
OUT = HERE / "validation_output"
RELEASE = Path(os.environ.get("RELEASE_ROOT", "") or "")

DRIVES = [
    ("validation_session.csv", "Drive 1", "2026-08-05", "10 hard brakes, 10 full-throttle accelerations, 10 attempted hard turns (not achieved)"),
    ("validation_session2.csv", "Drive 2", "2026-08-17", "full-throttle accelerations and attempted hard turns (not achieved); counts not recorded"),
    ("validation_session3.csv", "Drive 3", "2026-08-17", "11 hard turns, counted aloud"),
]


def load(name):
    r = pd.read_csv(PAPER / name, encoding="utf-8-sig")
    d = pd.DataFrame({"gps_speed_kmh": r["Speed (km/h)"], "speed_kph": r["Speed (km/h)"], "throttle_pct": 100.0,
                      "lon_acc_g": r["Longitudinal acceleration (g)"], "lat_acc_g": r["Lateral acceleration (g)"],
                      "heading_deg": r["Heading (Degrees)"], "elapsed_s": r["Elapsed time (s)"], "utc": r["UTC time"]})
    return d


def main():
    man, drv = [], []
    for fn, name, date, performed in DRIVES:
        d = load(fn); lab = label_df(d, PARAMS_V2); lon_s, lat_s = smoothed_signals(d, PARAMS_V2)
        t = d.elapsed_s.to_numpy(); u0 = d.utc.iloc[0]; hh, mm, ss = int(u0 // 10000), int((u0 % 10000) // 100), u0 % 100
        ev = events(lab); counts = {"Harsh Acceleration": 0, "Harsh Braking": 0, "Harsh Turning": 0}
        for cls, a, b in ev:
            counts[cls] += 1
            sm = (lat_s if cls == "Harsh Turning" else lon_s.abs())[a:b + 1]
            man.append({"drive": name, "file": fn, "class": cls, "start_s": round(float(t[a]), 2), "end_s": round(float(t[b]), 2),
                        "duration_s": round((b - a + 1) / 25, 2), "peak_smoothed_g": round(float(sm.max()), 3),
                        "mean_speed_kmh": round(float(d.gps_speed_kmh.iloc[a:b + 1].mean()), 1)})
        # attempted-turn evidence: largest smoothed |lateral| at v > 30 outside labelled turns
        m = (lat_s >= 0.20) & (d.gps_speed_kmh > 30) & (lab != "Harsh Turning")
        drv.append({"drive": name, "file": fn, "date_utc": date, "start_utc": f"{hh:02d}:{mm:02d}:{ss:05.2f}", "duration_min": round(float(t[-1]) / 60, 1),
                    "manoeuvres_performed": performed, "labelled_A": counts["Harsh Acceleration"], "labelled_B": counts["Harsh Braking"],
                    "labelled_T": counts["Harsh Turning"], "max_smoothed_lateral_outside_turn_events_g": round(float(lat_s[m].max()) if m.any() else 0.0, 2),
                    "rule_modifications": "GNSS speed substituted for CAN speed in the braking corroboration; throttle gate disabled (no CAN channels in these exports)"})
        print(name, counts)
    M = pd.DataFrame(man); D = pd.DataFrame(drv)
    M.to_csv(OUT / "calibration_manoeuvres.csv", index=False); D.to_csv(OUT / "calibration_drives.csv", index=False)
    rows = [r"\begin{table}[h]", r"\centering", r"\small",
            r"\caption{Threshold-calibration drives (same vehicle and instrumentation as the study; corresponding author driving, a passenger coordinating). "
            r"The final rule was applied with two modifications forced by the native exports, which carry no CAN channels: GNSS speed replaces CAN speed in the braking corroboration and the throttle gate is disabled. "
            r"These drives were used to choose the thresholds, so the recovery figures are calibration results, not an independent performance estimate of the complete released rule. "
            r"Event intervals: \texttt{Calibration\_Drives/calibration\_manoeuvres.csv} in the release.}",
            r"\label{tab:calibration}", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}llrp{4.6cm}rrrr@{}}", r"\toprule",
            r"Drive & Date (UTC start) & Min & Manoeuvres performed & A & B & T & Max lat.\ outside T (g) \\", r"\midrule"]
    for _, r in D.iterrows():
        rows.append(f"{r.drive} & {r.date_utc} ({r.start_utc[:5]}) & {r.duration_min:.1f} & {r.manoeuvres_performed} & {r.labelled_A} & {r.labelled_B} & {r.labelled_T} & {r.max_smoothed_lateral_outside_turn_events_g:.2f} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "si_calibration_table.tex").write_text("\n".join(rows), encoding="utf-8")
    if RELEASE and RELEASE.exists():
        dst = RELEASE / "Calibration_Drives"; dst.mkdir(exist_ok=True)
        for fn, name, _, _ in DRIVES:
            shutil.copy(PAPER / fn, dst / f"calibration_{name.lower().replace(' ', '')}_VBOX_native.csv")
        shutil.copy(OUT / "calibration_manoeuvres.csv", dst / "calibration_manoeuvres.csv"); shutil.copy(OUT / "calibration_drives.csv", dst / "calibration_drives.csv")
        print("copied to", dst)
    print(M.groupby(["drive", "class"]).size())


if __name__ == "__main__":
    main()
