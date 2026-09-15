"""
Final-gate consistency check: re-derives every data-derivable number stated in
the Data Descriptor from the RELEASED files and compares it with the value
the manuscript states (the EXPECTED dict below, kept in step with the .tex by
hand -- if the manuscript changes, change EXPECTED).

Covers: totals (sessions, rows, hours, files), Table 4 (class distribution,
event runs, median durations), Data Overview statistics (IQRs, events per
driver/session), Table 5 (speed-channel agreement), Table 6 (implausible-value
rates and extremes), Table 7 / video and pose completeness, physiological
figures (EDA/HR/IBI/SCR), and the label total. Sensitivity (Table 8) and the
synchronisation figures come from their own audit scripts and are not
repeated here.

Usage:
    python audit_manuscript_numbers.py
Environment:
    DATASET_ROOT   release root
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", "") or
                    r"C:\Users\halha\OneDrive - Durham University\Documents\DriveSense_Packages\3_OLD_versions\data\Published_Dataset_Final_v1_ORIGINAL_20260909")
# LABELS_V2=1 checks manuscript v6 (version-2 labels): the fused files come from the
# label-only v2 tree (PREPROCESSED_ROOT, default sibling Published_Dataset_Final_v2_LABELS),
# everything else still from the v1 release.
LABELS_V2 = os.environ.get("LABELS_V2", "") == "1"
PRE_ROOT = Path(os.environ.get("PREPROCESSED_ROOT", "") or
                (DATASET_ROOT.parent / "Published_Dataset_Final_v2_LABELS" if LABELS_V2 else DATASET_ROOT))
PRE = PRE_ROOT / "Preprocessed_Dataset"            # fused.csv (labels)
PRE_V1 = DATASET_ROOT / "Preprocessed_Dataset"     # Front_emotions / Side_pose (not duplicated in the v2 tree)
RAW = DATASET_ROOT / "Raw_Dataset"
OUT = Path(__file__).resolve().parent / "validation_output" / ("manuscript_numbers_check_v2labels.csv" if LABELS_V2 else "manuscript_numbers_check.csv")

# value stated in the manuscript, and the tolerance used to compare
EXPECTED = {
    "valid_sessions": (79, 0), "fused_rows": (3504071, 0), "hours": (38.93, 0.005),
    "front_videos": (79, 0), "side_videos": (74, 0), "pose_files": (75, 0), "video_files": (153, 0),
    "normal_ts": (2961620, 0), "accel_ts": (52088, 0), "brake_ts": (303499, 0), "turn_ts": (186864, 0),
    "normal_pct": (84.52, 0.005), "accel_pct": (1.49, 0.005), "brake_pct": (8.66, 0.005), "turn_pct": (5.33, 0.005),
    "accel_events": (1429, 0), "brake_events": (6320, 0), "turn_events": (2645, 0), "total_events": (10394, 0),
    "accel_med_dur": (1.44, 0.005), "brake_med_dur": (1.44, 0.005), "turn_med_dur": (2.00, 0.005),
    "accel_iqr": ("1.08-1.64", None), "brake_iqr": ("1.36-2.28", None), "turn_iqr": ("1.96-3.00", None),
    "events_per_driver_median": (510, 0), "events_per_driver_min": (436, 0), "events_per_driver_max": (650, 0),
    "events_per_session_median": (127, 0), "events_per_session_min": (71, 0), "events_per_session_max": (219, 0),
    "speed_r": (0.9997, 0.00005), "speed_med_abs": (0.40, 0.005), "speed_p95_abs": (1.08, 0.005), "speed_gt5_pct": (0.33, 0.005),
    "wheel_r": (0.9999, 0.00005), "wheel_med_abs": (0.10, 0.005), "wheel_p95_abs": (1.13, 0.005), "wheel_gt5_pct": (0.14, 0.005),
    "gradient_oor_pct": (0.30, 0.005), "gradient_min": (-2606, 1), "gradient_max": (902, 1),
    "heading_oor_pct": (0.00, 0.005), "throttle_oor_pct": (0.00, 0.005),
    "acc_oor_pct": (0.04, 0.005), "lat_max_abs": (6.06, 0.005), "lon_max_abs": (3.38, 0.005),
    "height_oor_pct": (0.0067, 0.00005), "height_min": (625.7, 0.05), "height_max": (1060.5, 0.05),
    "rpm_oor_pct": (0.00, 0.005), "rpm_min": (699, 1), "rpm_max": (5795, 1),
    "acc_outside_samples": (1312, 0),
    "facial_rows": (4204442, 0), "facial_sum_ok_pct": (98.95, 0.005), "pose_mean_cov_pct": (99.2, 0.05),
    "pose_out_of_frame_min_pct": (0.0, 0.00005), "pose_out_of_frame_max_pct": (0.006, 0.0005),
    "eda_median": (1.55, 0.005), "hr_median": (78.7, 0.05), "eda_oor_pct": (5.61, 0.005), "hr_oor_pct": (1.92, 0.005),
    "ibi_dropout_pct": (17.80, 0.005), "ibi_oor_pct": (24.3, 0.05), "ibi_usable_pct": (62.2, 0.05),
    "scr_present_pct": (12.02, 0.005), "eda_flat_sessions": (10, 0), "scr_flat_sessions": (11, 0), "hr_drop_sessions": (12, 0),
    "ibi_gap_max_s": (84.0, 0.05), "ibi_gap_gt40_sessions": (8, 0), "ibi_gap_median_s": (12.1, 0.05),
    "scr_rise_median": (0.26, 0.005), "scr_rise_max": (3.06, 0.05), "scr_amp_gt100_pct": (0.048, 0.0005), "scr_amp_max": (9925, 1),
}
# manuscript v6 / labels version 2 (Table 4, Data Overview, Usage Notes)
EXPECTED_V2_LABELS = {
    "normal_ts": (3499991, 0), "accel_ts": (3245, 0), "brake_ts": (571, 0), "turn_ts": (264, 0),
    "normal_pct": (99.88, 0.005), "accel_pct": (0.09, 0.005), "brake_pct": (0.02, 0.005), "turn_pct": (0.01, 0.005),
    "accel_events": (134, 0), "brake_events": (30, 0), "turn_events": (12, 0), "total_events": (176, 0),
    "accel_med_dur": (0.80, 0.005), "brake_med_dur": (0.60, 0.005), "turn_med_dur": (0.76, 0.005),
    "accel_iqr": ("0.60-1.23", None), "brake_iqr": ("0.56-0.91", None), "turn_iqr": ("0.65-1.14", None),
    "events_per_driver_median": (6, 0), "events_per_driver_min": (0, 0), "events_per_driver_max": (37, 0),
    "events_per_session_median": (1, 0), "events_per_session_min": (0, 0), "events_per_session_max": (16, 0),
    "zero_event_sessions": (26, 0),
}
if LABELS_V2:
    EXPECTED.update(EXPECTED_V2_LABELS)
# release version 3 (2026-09-14): D17_S2 truncated at its logging pause (7,920 rows fewer); videos realigned.
# Run with DATASET_ROOT=PREPROCESSED_ROOT=<v3 tree>, LABELS_V2=1, RELEASE_V3=1.
EXPECTED_V3 = {
    "fused_rows": (3496151, 0), "hours": (38.85, 0.005), "normal_ts": (3492071, 0),
    "wheel_p95_abs": (1.12, 0.005), "wheel_gt5_pct": (0.13, 0.005), "height_oor_pct": (0.0068, 0.00005),
    "facial_rows": (4170203, 0), "facial_sum_ok_pct": (98.96, 0.005),
    "eda_median": (1.56, 0.005), "eda_oor_pct": (5.63, 0.005), "ibi_dropout_pct": (17.81, 0.005), "scr_present_pct": (12.03, 0.005),
}
# release v3 with VERSION-1 labels (Published_Dataset_Final_v3_LABELS_v1; manuscript v5.2): v1 EXPECTED + v3-timeline values
EXPECTED_V3L1 = {
    "fused_rows": (3496151, 0), "hours": (38.85, 0.005), "normal_ts": (2955270, 0), "accel_ts": (52006, 0), "brake_ts": (302215, 0), "turn_ts": (186660, 0),
    "normal_pct": (84.53, 0.005), "brake_pct": (8.64, 0.005), "turn_pct": (5.34, 0.005),
    "accel_events": (1427, 0), "brake_events": (6301, 0), "turn_events": (2641, 0), "total_events": (10369, 0),
    "accel_iqr": ("1.08-1.62", None), "turn_iqr": ("1.96-3.04", None), "events_per_driver_median": (508, 0),
    "wheel_p95_abs": (1.12, 0.005), "wheel_gt5_pct": (0.13, 0.005), "height_oor_pct": (0.0068, 0.00005),
    "facial_rows": (4170203, 0), "facial_sum_ok_pct": (98.96, 0.005),
    "eda_median": (1.56, 0.005), "eda_oor_pct": (5.63, 0.005), "ibi_dropout_pct": (17.81, 0.005), "scr_present_pct": (12.03, 0.005),
}
if os.environ.get("RELEASE_V3L1", "") == "1":
    EXPECTED.update(EXPECTED_V3L1)
    OUT = OUT.with_name("manuscript_numbers_check_v3L1.csv")
if os.environ.get("RELEASE_V3", "") == "1":
    EXPECTED.update(EXPECTED_V3)
    OUT = OUT.with_name("manuscript_numbers_check_v3.csv")


def flat_pct(v, min_run=10):
    v = np.asarray(v, float); ok = np.isfinite(v)
    same = ok & np.concatenate(([False], np.diff(v) == 0))
    runs = []; L = 1
    for s in same[1:]:
        if s: L += 1
        else:
            if L > 1: runs.append(L)
            L = 1
    if L > 1: runs.append(L)
    runs = np.array(runs) if runs else np.array([0])
    return runs[runs >= min_run].sum() / max(1, ok.sum()) * 100


def runs_of(lab, t, cls):
    m = (lab == cls); out = []
    i = 0; n = len(m)
    while i < n:
        if m[i]:
            j = i
            while j + 1 < n and m[j + 1]: j += 1
            out.append(t[j] - t[i] + 0.04); i = j + 1
        else: i += 1
    return out


def main():
    got = {}
    files = sorted(PRE.rglob("*_fused.csv"))
    got["valid_sessions"] = len(files)
    rows = 0; hours = 0.0
    ts = {"Normal": 0, "Harsh Acceleration": 0, "Harsh Braking": 0, "Harsh Turning": 0}
    durs = {"Harsh Acceleration": [], "Harsh Braking": [], "Harsh Turning": []}
    ev_per_driver = {}; ev_per_session = {}
    sp = {"d_speed": [], "d_wheel": []}; speed_pairs = []; wheel_pairs = []
    anom = {k: [] for k in ["gradient_pct", "heading_deg", "throttle_pct", "lat_acc_g", "lon_acc_g", "height_m", "engine_rpm"]}
    acc_outside = 0
    eda_all = []; hr_all = []; ibi_drop = 0; ibi_pres = 0; ibi_oor = 0; hr_pres = 0; hr_oor = 0
    scr_pres = 0; sr_vals = []; sa_vals = []
    eda_flat_n = 0; scr_flat_n = 0; hr_drop_n = 0; ibi_gaps = []
    for f in files:
        d = pd.read_csv(f)
        tag = f.stem.replace("_fused", ""); drv = int(tag[1:].split("_")[0])
        rows += len(d); t = d.elapsed_s.to_numpy(); hours += (t[-1] - t[0] + 0.04) / 3600
        lab = d.label.to_numpy()
        for k in ts: ts[k] += int((lab == k).sum())
        nev = 0
        for k in durs:
            r = runs_of(lab, t, k); durs[k].extend(r); nev += len(r)
        ev_per_driver[drv] = ev_per_driver.get(drv, 0) + nev; ev_per_session[tag] = nev
        speed_pairs.append(d[["gps_speed_kmh", "speed_kph"]].to_numpy()); wheel_pairs.append(d[["wheel_fl_kph", "wheel_fr_kph"]].to_numpy())
        for k in anom: anom[k].append(d[k].to_numpy())
        acc_outside += int(((d.lat_acc_g.abs() > 1.2) | (d.lon_acc_g.abs() > 1.2)).sum())
        eda = d.EDA.to_numpy(); eda_all.append(eda)
        m = ~d.IBI_dropout_flag.astype(bool).to_numpy(); ibi_drop += (~m).sum(); ibi_pres += m.sum(); ibi_oor += (~d.IBI[m].between(300, 1500)).sum()
        mh = ~d.HR_dropout_flag.astype(bool).to_numpy(); hr_pres += mh.sum(); hr_oor += (~d.HR[mh].between(40, 200)).sum(); hr_all.append(d.HR[mh].to_numpy())
        ms = ~d.SCR_AMPLITUDE_dropout_flag.astype(bool).to_numpy(); scr_pres += ms.sum(); sa_vals.append(d.SCR_AMPLITUDE[ms].to_numpy()); sr_vals.append(d.SCR_RISE_TIME[ms].to_numpy())
        eda_flat_n += flat_pct(eda) > 30; scr_flat_n += flat_pct(d.SCR_FREQ) > 30; hr_drop_n += ((~m).mean() * 100) > 30
        tp = t[m]; gaps = np.diff(tp); ibi_gaps.append(float(gaps.max()) if len(gaps) and (gaps > 3).any() else 0.0)
    got.update(fused_rows=rows, hours=round(hours, 2))
    tot = sum(ts.values())
    got.update(normal_ts=ts["Normal"], accel_ts=ts["Harsh Acceleration"], brake_ts=ts["Harsh Braking"], turn_ts=ts["Harsh Turning"],
               normal_pct=round(ts["Normal"] / tot * 100, 2), accel_pct=round(ts["Harsh Acceleration"] / tot * 100, 2),
               brake_pct=round(ts["Harsh Braking"] / tot * 100, 2), turn_pct=round(ts["Harsh Turning"] / tot * 100, 2))
    for k, key in [("Harsh Acceleration", "accel"), ("Harsh Braking", "brake"), ("Harsh Turning", "turn")]:
        a = np.array(durs[k]); got[f"{key}_events"] = len(a); got[f"{key}_med_dur"] = round(float(np.median(a)), 2)
        got[f"{key}_iqr"] = f"{np.percentile(a,25):.2f}-{np.percentile(a,75):.2f}"
    got["total_events"] = sum(len(durs[k]) for k in durs)
    pdv = np.array(list(ev_per_driver.values())); psv = np.array(list(ev_per_session.values()))
    got.update(events_per_driver_median=int(np.median(pdv)), events_per_driver_min=int(pdv.min()), events_per_driver_max=int(pdv.max()),
               events_per_session_median=int(np.median(psv)), events_per_session_min=int(psv.min()), events_per_session_max=int(psv.max()),
               zero_event_sessions=int((psv == 0).sum()))
    S = np.vstack(speed_pairs); W = np.vstack(wheel_pairs)
    for name, M in [("speed", S), ("wheel", W)]:
        ok = np.isfinite(M).all(1); a, b = M[ok, 0], M[ok, 1]; dd = np.abs(a - b)
        got[f"{name}_r"] = round(float(np.corrcoef(a, b)[0, 1]), 4); got[f"{name}_med_abs"] = round(float(np.median(dd)), 2)
        got[f"{name}_p95_abs"] = round(float(np.percentile(dd, 95)), 2); got[f"{name}_gt5_pct"] = round(float((dd > 5).mean() * 100), 2)
    A = {k: np.concatenate(v) for k, v in anom.items()}
    got.update(gradient_oor_pct=round(float((np.abs(A["gradient_pct"]) > 30).mean() * 100), 2), gradient_min=round(float(np.nanmin(A["gradient_pct"]))), gradient_max=round(float(np.nanmax(A["gradient_pct"]))),
               heading_oor_pct=round(float(((A["heading_deg"] < 0) | (A["heading_deg"] >= 360)).mean() * 100), 2),
               throttle_oor_pct=round(float(((A["throttle_pct"] < 0) | (A["throttle_pct"] > 100)).mean() * 100), 2),
               acc_oor_pct=round(float(((np.abs(A["lat_acc_g"]) > 1.2) | (np.abs(A["lon_acc_g"]) > 1.2)).mean() * 100), 2),
               lat_max_abs=round(float(np.nanmax(np.abs(A["lat_acc_g"]))), 2), lon_max_abs=round(float(np.nanmax(np.abs(A["lon_acc_g"]))), 2),
               height_oor_pct=round(float(((A["height_m"] < 550) | (A["height_m"] > 800)).mean() * 100), 4), height_min=round(float(np.nanmin(A["height_m"])), 1), height_max=round(float(np.nanmax(A["height_m"])), 1),
               rpm_oor_pct=round(float(((A["engine_rpm"] < 500) | (A["engine_rpm"] > 6500)).mean() * 100), 2), rpm_min=round(float(np.nanmin(A["engine_rpm"]))), rpm_max=round(float(np.nanmax(A["engine_rpm"]))),
               acc_outside_samples=acc_outside)
    eda = np.concatenate(eda_all); hr = np.concatenate(hr_all); sa = np.concatenate(sa_vals); sr = np.concatenate(sr_vals)
    drop = ibi_drop / rows * 100; oor = ibi_oor / ibi_pres * 100
    got.update(eda_median=round(float(np.nanmedian(eda)), 2), hr_median=round(float(np.nanmedian(hr)), 1),
               eda_oor_pct=round(float(((eda < 0) | (eda > 50))[np.isfinite(eda)].mean() * 100), 2), hr_oor_pct=round(hr_oor / hr_pres * 100, 2),
               ibi_dropout_pct=round(drop, 2), ibi_oor_pct=round(oor, 1), ibi_usable_pct=round((1 - drop / 100) * (1 - oor / 100) * 100, 1),
               scr_present_pct=round(scr_pres / rows * 100, 2), eda_flat_sessions=int(eda_flat_n), scr_flat_sessions=int(scr_flat_n), hr_drop_sessions=int(hr_drop_n),
               ibi_gap_max_s=round(max(ibi_gaps), 1), ibi_gap_gt40_sessions=int(sum(g > 40 for g in ibi_gaps)), ibi_gap_median_s=round(float(np.median(ibi_gaps)), 1),
               scr_rise_median=round(float(np.nanmedian(sr)), 2), scr_rise_max=round(float(np.nanmax(sr)), 2),
               scr_amp_gt100_pct=round(float((sa > 100).mean() * 100), 3), scr_amp_max=round(float(np.nanmax(sa))))
    # video / facial / pose
    fronts = sorted(RAW.rglob("*_Front_blurred.mp4")); sides = sorted(RAW.rglob("*_Side_blurred.mp4"))
    got.update(front_videos=len(fronts), side_videos=len(sides), video_files=len(fronts) + len(sides))
    frows = 0; fok = 0
    for f in sorted(PRE_V1.rglob("*_Front_emotions.csv")):
        e = pd.read_csv(f)
        cols = [c for c in e.columns if c.lower().split("_")[-1] in ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")]
        s = e[cols].sum(axis=1); det = e[cols].notna().all(axis=1); frows += len(e)
        fok += int((((s - 100).abs() <= 1) & det).sum())
    got.update(facial_rows=frows, facial_sum_ok_pct=round(fok / frows * 100, 2))
    poses = sorted(PRE_V1.rglob("*_Side_pose.csv")); cov = []; oob = {}
    for f in poses:
        p = pd.read_csv(f); det = p["pose_detected"].astype(bool) if "pose_detected" in p else p.iloc[:, 2].notna()
        cov.append(det.mean() * 100)
        for c in [c for c in p.columns if c.endswith(("_x", "_y")) and ("wrist" in c or "shoulder" in c)]:
            v = p.loc[det, c].to_numpy(dtype=float); oob.setdefault(c, [0, 0]); oob[c][0] += int(((v < 0) | (v > 1)).sum()); oob[c][1] += len(v)
    pct = [a / b * 100 for a, b in oob.values() if b]
    got.update(pose_files=len(poses), pose_mean_cov_pct=round(float(np.mean(cov)), 1), pose_out_of_frame_min_pct=round(min(pct), 4), pose_out_of_frame_max_pct=round(max(pct), 3))

    recs = []
    for k, (exp, tol) in EXPECTED.items():
        g = got.get(k, "MISSING")
        if tol is None: ok = (str(g) == str(exp))
        else:
            try: ok = abs(float(g) - float(exp)) <= tol
            except Exception: ok = False
        recs.append((k, exp, g, "PASS" if ok else "FAIL"))
    R = pd.DataFrame(recs, columns=["quantity", "manuscript", "recomputed", "status"]); R.to_csv(OUT, index=False)
    print(R.to_string(index=False)); print(f"\n{(R.status=='FAIL').sum()} FAIL / {len(R)} checks -> {OUT}")


if __name__ == "__main__":
    main()
