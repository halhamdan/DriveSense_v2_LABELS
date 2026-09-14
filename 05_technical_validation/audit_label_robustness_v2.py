"""
Label-robustness audit for the version-2 harsh-event rule (2026-09-13 critique).

Part A  synthetic  -- one spike, short bursts, alternating-sign noise, sustained
                     excursion, sub-duration excursion, longitudinal spike: which
                     formulations of the rule label them?
Part B  drives     -- recovery of the deliberate manoeuvres on the three calibration
                     drives under each formulation.
Part C  release    -- event totals per formulation on the released data (needs the
                     column cache from cache_release_columns.py).
Part D  events     -- for the ADOPTED formulation: per-event table with artefact-
                     overlap, rectification, heading-change and sensitivity flags, and
                     a classification of what clipping raw |acc| to 1.2 g does to each
                     event (unchanged / boundary / disappears / class change / split-merge).

Usage:
    python audit_label_robustness_v2.py synthetic
    python audit_label_robustness_v2.py drives
    python audit_label_robustness_v2.py release          # totals for all variants
    python audit_label_robustness_v2.py events           # per-event table, adopted rule
Environment:
    RELEASE_CACHE  npz written by cache_release_columns.py (default validation_output/release_columns_cache.npz)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01_annotation"))
from label_rule_variants import RuleParams, label_df, events, count_events, smoothed_signals, runs  # noqa: E402

OUT = HERE / "validation_output"; OUT.mkdir(exist_ok=True)
CACHE = Path(os.environ.get("RELEASE_CACHE", "") or (OUT / "release_columns_cache.npz"))
DRIVES = HERE.parent.parent  # validation_session*.csv live in the paper folder

VARIANTS = {
    "V0 draft (12/12/5, |.| then mean)":           RuleParams(smooth_n=12, min_n=12, close_n=5, lateral="abs_then_mean"),
    "V1 13/13/5, |.| then mean":                   RuleParams(smooth_n=13, min_n=13, close_n=5, lateral="abs_then_mean"),
    "V2 13/13/5, mean then |.|":                   RuleParams(smooth_n=13, min_n=13, close_n=5, lateral="mean_then_abs"),
    "V3 13/15/5, mean then |.|":                   RuleParams(smooth_n=13, min_n=15, close_n=5, lateral="mean_then_abs"),
    "V4 13/13/5, mean then |.|, median-5 prefilter": RuleParams(smooth_n=13, min_n=13, close_n=5, lateral="mean_then_abs", median_n=5),
    "V5 13/15/5, mean then |.|, median-5 prefilter": RuleParams(smooth_n=13, min_n=15, close_n=5, lateral="mean_then_abs", median_n=5),
    "V6 = V4 + heading change >= 5 deg (ADOPTED)":  RuleParams(smooth_n=13, min_n=13, close_n=5, lateral="mean_then_abs", median_n=5, turn_min_heading_deg=5.0),
}
ADOPTED = "V6 = V4 + heading change >= 5 deg (ADOPTED)"


# ----------------------------------------------------------------------------- synthetic
def _base(n=400, v=50.0):
    # heading is left constant (no heading_deg column -> heading gate inactive), so the
    # synthetic cases test the smoothing/duration logic alone; the heading gate can
    # only remove events, never add one.
    return pd.DataFrame({"gps_speed_kmh": np.full(n, v), "speed_kph": np.full(n, v), "throttle_pct": np.zeros(n),
                         "lat_acc_g": np.zeros(n), "lon_acc_g": np.zeros(n)})


def synthetic_cases():
    cases = {}
    d = _base(); d.loc[200, "lat_acc_g"] = 6.06; cases["single 6.06 g lateral spike (dataset max)"] = d
    d = _base(); d.loc[200, "lat_acc_g"] = 5.5; cases["single 5.5 g lateral spike"] = d
    d = _base(); d.loc[200:201, "lat_acc_g"] = 3.0; cases["2-sample 3.0 g lateral burst"] = d
    d = _base(); d.loc[200:202, "lat_acc_g"] = 2.0; cases["3-sample 2.0 g lateral burst"] = d
    d = _base(); d.loc[200:202, "lat_acc_g"] = 2.6; cases["3-sample 2.6 g lateral burst (p90 of class)"] = d
    d = _base(); d.loc[200:249, "lat_acc_g"] = 0.6 * np.where(np.arange(50) % 2 == 0, 1, -1); cases["alternating +/-0.6 g for 2 s"] = d
    d = _base(); d.loc[200:249, "lat_acc_g"] = 1.0 * np.where(np.arange(50) % 2 == 0, 1, -1); cases["alternating +/-1.0 g for 2 s"] = d
    rng = np.random.default_rng(0)
    d = _base(); d.loc[200:224, "lat_acc_g"] = 0.55 + rng.normal(0, 0.15, 25); cases["sustained 0.55 g for 1.0 s + noise sd 0.15"] = d
    d = _base(); d.loc[200:224, "lat_acc_g"] = 0.55 + 0.3 * np.where(np.arange(25) % 2 == 0, 1, -1); cases["sustained 0.55 g for 1.0 s + alternating +/-0.3 g"] = d
    d = _base(); d.loc[200:209, "lat_acc_g"] = 0.60; cases["sustained 0.60 g for 0.40 s (below min duration)"] = d
    d = _base(); d.loc[200:214, "lat_acc_g"] = 0.60; cases["sustained 0.60 g for 0.60 s"] = d
    d = _base(); d.loc[200:212, "lat_acc_g"] = 0.60; cases["sustained 0.60 g for 0.52 s"] = d
    # longitudinal: one -3.38 g spike (dataset max) while CAN speed drops 1 m/s^2 (gate passes)
    d = _base(); d.loc[200, "lon_acc_g"] = -3.38; d["speed_kph"] = 50 - np.clip(np.arange(400) - 150, 0, None) * (1.0 / 25) * 3.6
    cases["single -3.38 g longitudinal spike, CAN decel 1 m/s^2"] = d
    d = _base(); d.loc[200:224, "lon_acc_g"] = -0.45; d["speed_kph"] = 50 - np.clip(np.arange(400) - 150, 0, None) * (1.0 / 25) * 3.6
    cases["sustained -0.45 g braking 1.0 s, CAN decel 1 m/s^2"] = d
    return cases


def part_synthetic():
    rows = []
    for name, d in synthetic_cases().items():
        r = {"case": name}
        for vn, p in VARIANTS.items():
            c = count_events(label_df(d, p)); r[vn] = f"T{c['Harsh Turning']} B{c['Harsh Braking']} A{c['Harsh Acceleration']}"
        rows.append(r)
    R = pd.DataFrame(rows); R.to_csv(OUT / "label_robustness_synthetic.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 60):
        print(R.to_string(index=False))


# ----------------------------------------------------------------------------- drives
def _load_drive(name):
    r = pd.read_csv(DRIVES / name, encoding="utf-8-sig")
    return pd.DataFrame({"gps_speed_kmh": r["Speed (km/h)"], "speed_kph": r["Speed (km/h)"], "throttle_pct": 100.0,
                         "lon_acc_g": r["Longitudinal acceleration (g)"], "lat_acc_g": r["Lateral acceleration (g)"],
                         "heading_deg": r["Heading (Degrees)"]})


def part_drives():
    truth = {"validation_session.csv": "S1: 10 brakes, 10 accels, 10 attempted turns (failed)",
             "validation_session2.csv": "S2: accels + attempted turns (failed)",
             "validation_session3.csv": "S3: 11 turns"}
    rows = []
    for f, t in truth.items():
        d = _load_drive(f); r = {"drive": t}
        for vn, p in VARIANTS.items():
            c = count_events(label_df(d, p)); r[vn] = f"T{c['Harsh Turning']} B{c['Harsh Braking']} A{c['Harsh Acceleration']}"
        rows.append(r)
    R = pd.DataFrame(rows); R.to_csv(OUT / "label_robustness_drives.csv", index=False)
    with pd.option_context("display.width", 250):
        print(R.to_string(index=False))


# ----------------------------------------------------------------------------- release
def _load_cache():
    z = np.load(CACHE, allow_pickle=True)
    df = pd.DataFrame({k: z[k] for k in ["session", "elapsed_s", "gps_speed_kmh", "speed_kph", "throttle_pct",
                                          "lat_acc_g", "lon_acc_g", "heading_deg", "label_draft", "label_v1_peak"]})
    tags = list(z["tags"])
    return df, tags


def _label_all(df, tags, p, clip=None):
    out = np.full(len(df), "Normal", dtype=object)
    for si, tag in enumerate(tags):
        idx = np.flatnonzero(df["session"].to_numpy() == si)
        d = df.iloc[idx]
        if clip is not None:
            d = d.copy(); d["lat_acc_g"] = d["lat_acc_g"].clip(-clip, clip); d["lon_acc_g"] = d["lon_acc_g"].clip(-clip, clip)
        out[idx] = label_df(d.reset_index(drop=True), p).to_numpy()
    return out


def part_release():
    df, tags = _load_cache(); rows = []
    for vn, p in VARIANTS.items():
        lab = _label_all(df, tags, p); c = count_events(lab)
        rows.append({"variant": vn, **c, "total": sum(c.values()), "nonnormal_rows": int((lab != "Normal").sum())})
        print(rows[-1], flush=True)
    R = pd.DataFrame(rows); R.to_csv(OUT / "label_robustness_release_totals.csv", index=False)
    print(R.to_string(index=False))


# ----------------------------------------------------------------------------- events
def _match(ev_a, ev_b):
    """For each event in A, overlapping events in B (index list)."""
    out = []
    for ca, a0, a1 in ev_a:
        out.append([j for j, (cb, b0, b1) in enumerate(ev_b) if b0 <= a1 and a0 <= b1])
    return out


def part_sensitivity(p=None):
    """Table 8: all three thresholds scaled by -10 % / 0 / +10 % (rule otherwise unchanged)."""
    p = p or VARIANTS[ADOPTED]; df, tags = _load_cache(); rows = []
    for name, f in [("-10% (more lenient)", 0.9), ("Primary (as released)", 1.0), ("+10% (stricter)", 1.1)]:
        q = RuleParams(**{**p.to_dict(), "brake_g": p.brake_g * f, "accel_g": p.accel_g * f, "turn_g": p.turn_g * f})
        c = count_events(_label_all(df, tags, q)); rows.append({"setting": name, **c, "total": sum(c.values())}); print(rows[-1], flush=True)
    R = pd.DataFrame(rows); R.to_csv(OUT / "label_sensitivity_v2.csv", index=False); print(R.to_string(index=False))


def part_stats(p=None):
    """Table 4 / Data Overview statistics for the adopted rule, from the cache."""
    p = p or VARIANTS[ADOPTED]; df, tags = _load_cache(); lab = _label_all(df, tags, p)
    sess = df["session"].to_numpy(); tot = len(lab)
    print("timesteps:", {k: (int((lab == k).sum()), round((lab == k).mean() * 100, 2)) for k in ["Normal", "Harsh Acceleration", "Harsh Braking", "Harsh Turning"]})
    dur = {"Harsh Acceleration": [], "Harsh Braking": [], "Harsh Turning": []}; per_s = {}; per_d = {}
    for si, tag in enumerate(tags):
        idx = np.flatnonzero(sess == si); ev = events(lab[idx]); per_s[tag] = len(ev)
        d = int(tag[1:].split("_")[0]); per_d[d] = per_d.get(d, 0) + len(ev)
        for c, a, b in ev: dur[c].append((b - a + 1) / 25)
    for c, v in dur.items():
        v = np.array(v); print(f"{c}: n={len(v)} median {np.median(v):.2f} IQR {np.percentile(v,25):.2f}-{np.percentile(v,75):.2f}")
    ps = np.array(list(per_s.values())); pdv = np.array(list(per_d.values()))
    print(f"per driver median {np.median(pdv):.0f} range {pdv.min()}-{pdv.max()}; per session median {np.median(ps):.0f} range {ps.min()}-{ps.max()}; zero-event sessions {(ps==0).sum()}")
    print("overlap T over A / T over B / B over A:", end=" ")
    # recompute masks to report overlaps
    n = {"TA": 0, "TB": 0, "BA": 0}
    for si, tag in enumerate(tags):
        idx = np.flatnonzero(sess == si); d = df.iloc[idx].reset_index(drop=True)
        from label_rule_variants import smoothed_signals as _ss, clean_mask as _cm
        lon_s, lat_s = _ss(d, p); v = d.gps_speed_kmh; thr = d.throttle_pct.rolling(p.smooth_n, center=True, min_periods=1).max()
        a_can = ((d.speed_kph / 3.6).diff() * 25).rolling(p.can_window_n, min_periods=1).min()
        B = _cm(((lon_s <= -p.brake_g) & (v > p.min_speed_kmh) & (a_can <= p.brake_can_ms2)).fillna(False).to_numpy(), p.close_n, p.min_n)
        A = _cm(((lon_s >= p.accel_g) & (v > p.min_speed_kmh) & (thr > p.throttle_min_pct)).fillna(False).to_numpy(), p.close_n, p.min_n)
        T = (lab[idx] == "Harsh Turning")
        n["TA"] += int((T & A).sum()); n["TB"] += int((T & B).sum()); n["BA"] += int((B & A).sum())
    print(n)


def part_segment(p=None, duration=122.0):
    """Finds windows of `duration` s containing one event of each class (for Fig. 3)."""
    p = p or VARIANTS[ADOPTED]; df, tags = _load_cache(); lab = _label_all(df, tags, p); sess = df["session"].to_numpy(); t = df["elapsed_s"].to_numpy()
    found = []
    for si, tag in enumerate(tags):
        idx = np.flatnonzero(sess == si); ev = events(lab[idx]); tt = t[idx]
        for c, a, b in ev:
            if c != "Harsh Turning": continue
            t0 = max(tt[a] - duration / 2, 0); w = [(cc, tt[aa], tt[bb]) for cc, aa, bb in ev if tt[aa] >= t0 and tt[bb] <= t0 + duration]
            cls = set(cc for cc, _, _ in w)
            if len(cls) == 3: found.append((tag, round(t0, 1), len(w), sorted(cls)))
    print("windows with all three classes:", found if found else "none")
    if not found:
        for si, tag in enumerate(tags):
            idx = np.flatnonzero(sess == si); ev = events(lab[idx]); tt = t[idx]
            for c, a, b in ev:
                if c != "Harsh Turning": continue
                t0 = max(tt[a] - duration / 2, 0); w = [(cc, tt[aa], tt[bb]) for cc, aa, bb in ev if tt[aa] >= t0 and tt[bb] <= t0 + duration]
                print(tag, round(t0, 1), sorted(set(cc for cc, _, _ in w)), len(w))


def part_events(p=None, name=None):
    p = p or VARIANTS[ADOPTED]; name = name or ADOPTED
    df, tags = _load_cache()
    lab = _label_all(df, tags, p)
    lab_clip = _label_all(df, tags, p, clip=1.2)
    p_lo = RuleParams(**{**p.to_dict(), "brake_g": p.brake_g * 0.9, "accel_g": p.accel_g * 0.9, "turn_g": p.turn_g * 0.9})
    p_hi = RuleParams(**{**p.to_dict(), "brake_g": p.brake_g * 1.1, "accel_g": p.accel_g * 1.1, "turn_g": p.turn_g * 1.1})
    lab_lo = _label_all(df, tags, p_lo); lab_hi = _label_all(df, tags, p_hi)
    sess = df["session"].to_numpy(); t = df["elapsed_s"].to_numpy(); v = df["gps_speed_kmh"].to_numpy()
    lat = df["lat_acc_g"].to_numpy(); lon = df["lon_acc_g"].to_numpy(); hd = df["heading_deg"].to_numpy()
    vcan = df["speed_kph"].to_numpy() / 3.6; a_can = np.r_[np.nan, np.diff(vcan) * 25]  # CAN-speed-derived acceleration, m/s^2 (per session boundaries handled below)
    a_can[np.r_[True, np.diff(sess) != 0]] = np.nan
    thr_pct = df["throttle_pct"].to_numpy()
    v1 = df["label_v1_peak"].to_numpy(); big = (np.abs(lat) > 1.2) | (np.abs(lon) > 1.2)
    rows = []
    for si, tag in enumerate(tags):
        idx = np.flatnonzero(sess == si); o = idx[0]
        d = df.iloc[idx].reset_index(drop=True)
        lon_s, lat_s = smoothed_signals(d, p); lon_s = lon_s.to_numpy(); lat_s = lat_s.to_numpy()
        ev = events(lab[idx]); ev_c = events(lab_clip[idx]); ev_lo = events(lab_lo[idx]); ev_hi = events(lab_hi[idx])
        m_c = _match(ev, ev_c); m_lo = _match(ev, ev_lo); m_hi = _match(ev, ev_hi)
        # events in clipped set not matched by any primary event (new events)
        for k, (cls, a, b) in enumerate(ev):
            g = slice(o + a, o + b + 1); n = b - a + 1
            thr = {"Harsh Turning": p.turn_g, "Harsh Braking": p.brake_g, "Harsh Acceleration": p.accel_g}[cls]
            raw = lat[g] if cls == "Harsh Turning" else lon[g]
            sm = lat_s[a:b + 1] if cls == "Harsh Turning" else np.abs(lon_s[a:b + 1])
            # rectification: |signed mean| / mean|.| of raw lateral inside the event
            rect = float(abs(raw.mean()) / max(np.abs(raw).mean(), 1e-9))
            # heading change across the event (wrapped), and expected from smoothed lateral: dpsi = sum(a/v dt)
            h0 = hd[o + max(a - 3, 0)]; h1 = hd[o + min(b + 3, len(idx) - 1)]
            dpsi = (h1 - h0 + 180) % 360 - 180
            vv = np.maximum(v[g], 1) / 3.6
            dpsi_exp = float(np.degrees(np.sum(np.abs(lat_s[a:b + 1]) * 9.81 / vv * (1 / 25))))
            # clipping outcome
            mc = m_c[k]
            if not mc: clip_out = "disappears"
            else:
                same = [j for j in mc if ev_c[j][0] == cls]
                if not same: clip_out = "class change"
                elif len(mc) > 1: clip_out = "split"
                else:
                    j = same[0]; cb0, cb1 = ev_c[j][1], ev_c[j][2]
                    others = [kk for kk, mm in enumerate(m_c) if j in mm and kk != k]
                    clip_out = "merge" if others else ("unchanged" if (cb0, cb1) == (a, b) else f"boundary {abs(cb0 - a) + abs(cb1 - b)} samples")
            gw = slice(max(o + a - 25, o), o + b + 1)  # event plus the preceding second (the CAN gate's window)
            ac = a_can[gw]; ac = ac[np.isfinite(ac)]
            can_1s_mean = float(np.nanmean(a_can[max(o + a - 12, o):o + b + 1])) if cls != "Harsh Turning" else np.nan
            rows.append({
                "tag": tag, "class": cls, "start_s": round(float(t[o + a]), 2), "end_s": round(float(t[o + b]), 2), "duration_s": round(n / 25, 2),
                "n_samples": n, "peak_smoothed_g": round(float(sm.max()), 3), "mean_speed_kmh": round(float(v[g].mean()), 1),
                "can_accel_min_ms2": round(float(ac.min()), 2) if len(ac) else np.nan, "can_accel_max_ms2": round(float(ac.max()), 2) if len(ac) else np.nan,
                "can_accel_mean_ms2_event": round(float(np.nanmean(a_can[g])), 2) if np.isfinite(a_can[g]).any() else np.nan,
                "can_speed_change_kmh": round(float(df["speed_kph"].to_numpy()[o + b] - df["speed_kph"].to_numpy()[max(o + a - 1, o)]), 1),
                "gnss_speed_change_kmh": round(float(v[o + b] - v[max(o + a - 1, o)]), 1),
                "throttle_max_pct": round(float(np.nanmax(thr_pct[g])), 0) if np.isfinite(thr_pct[g]).any() else np.nan,
                "n_raw_above_thr": int((np.abs(raw) >= thr).sum()), "frac_raw_above_half_thr": round(float((np.abs(raw) >= thr / 2).mean()), 2),
                "max_raw_g": round(float(np.abs(raw).max()), 2), "n_raw_gt_1p2g": int(big[g].sum()),
                "rectification_ratio": round(rect, 2), "heading_change_deg": round(float(dpsi), 1), "heading_change_expected_deg": round(dpsi_exp, 1),
                "clip_1p2g_outcome": clip_out,
                "present_at_plus10pct": bool(any(ev_hi[j][0] == cls for j in m_hi[k])),
                "present_at_minus10pct": bool(any(ev_lo[j][0] == cls for j in m_lo[k])),
                "v1_overlap": bool((v1[g] != "Normal").any()),
            })
        # clipped-only events
        matched_c = set(j for mm in m_c for j in mm)
        for j, (cls, a, b) in enumerate(ev_c):
            if j not in matched_c:
                rows.append({"tag": tag, "class": cls, "start_s": round(float(t[o + a]), 2), "end_s": round(float(t[o + b]), 2), "duration_s": round((b - a + 1) / 25, 2),
                             "n_samples": b - a + 1, "clip_1p2g_outcome": "appears only after clipping"})
    R = pd.DataFrame(rows)
    fn = OUT / "harsh_events_v2_table.csv"; R.to_csv(fn, index=False)
    prim = R[R.clip_1p2g_outcome != "appears only after clipping"]
    print(f"rule: {name}\n{len(prim)} events; totals {prim['class'].value_counts().to_dict()}")
    print("clipping outcome by class:\n", pd.crosstab(prim["class"], prim["clip_1p2g_outcome"].str.replace(r"boundary \d+ samples", "boundary", regex=True)))
    print("clipped-only events:", int((R.clip_1p2g_outcome == "appears only after clipping").sum()))
    T = prim[prim["class"] == "Harsh Turning"]
    print(f"turning: rectification ratio median {T.rectification_ratio.median():.2f}, min {T.rectification_ratio.min():.2f}, n<0.5: {(T.rectification_ratio < 0.5).sum()}")
    print(f"turning: |heading change| median {T.heading_change_deg.abs().median():.1f} deg; n with |dpsi| < 5 deg: {(T.heading_change_deg.abs() < 5).sum()}; "
          f"ratio observed/expected median {(T.heading_change_deg.abs() / T.heading_change_expected_deg.clip(lower=0.1)).median():.2f}")
    print(f"events with any raw sample > 1.2 g: {(prim.n_raw_gt_1p2g > 0).sum()} ({(prim.n_raw_gt_1p2g > 0).mean()*100:.1f}%); turning: {(T.n_raw_gt_1p2g > 0).sum()}/{len(T)}")
    print(f"n_raw_above_thr == 0 (event carried entirely by smoothing): {(prim.n_raw_above_thr == 0).sum()}")
    print(f"present at +10%: {prim.present_at_plus10pct.sum()}; present at -10%: {prim.present_at_minus10pct.sum()}; v1 overlap: {prim.v1_overlap.sum()}")
    print("->", fn)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
    {"synthetic": part_synthetic, "drives": part_drives, "release": part_release, "events": part_events,
     "sensitivity": part_sensitivity, "stats": part_stats, "segment": part_segment}[cmd]()
