"""
Does the version-2 label rule change if the median(5)+mean(13) smoothing step is
replaced by an FFT low-pass filter (as the supervisor asked, 2026-09-16), or by
wavelet denoising? Everything else in the rule (thresholds, speed/throttle/CAN/
heading corroboration, min-run-length, gap-closing, priority) is left exactly as
label_rule_variants.label_df implements it -- only smoothed_signals() is swapped,
via monkeypatching label_rule_variants.smoothed_signals so label_df's internal
call picks up the replacement.

Three tests, mirroring audit_label_robustness_v2.py's own structure:
  A. synthetic  -- the same stress cases used to choose the adopted rule (single
                   spike, short bursts, alternating noise, sustained excursion):
                   does each smoothing choice still reject spikes / accept the
                   genuine event?
  B. drives     -- the three deliberate-manoeuvre calibration drives: does each
                   smoothing choice still recover 10/10 brakes and 11/11 turns?
  C. release    -- full release (79 sessions, v3 timeline): event totals per
                   smoothing choice, and event-by-event overlap against the
                   176 events of the adopted (released) rule.

FFT filter: zero-phase (fft -> zero bins with |freq| > cutoff, both signs ->
ifft), applied to the SIGNED channel over the WHOLE session (not window-by-
window, to avoid edge artefacts), exactly as in compare_filters_v2.m. Lateral
test uses mean_then_abs (abs of the filtered signed channel), matching the
adopted rule's own convention.

Wavelet filter: db4-equivalent is unavailable in this environment (no
PyWavelets), so the SAME manual 4-level Haar DWT + soft-threshold + inverse
DWT used as compare_filters_v2.m's fallback (MATLAB's licensed Wavelet
Toolbox wdenoise() was used for the figures; this Python port matches it in
spirit, not bit-for-bit -- flagged in the output).

Usage:
    python audit_fft_wavelet_variant_v2.py synthetic
    python audit_fft_wavelet_variant_v2.py drives
    python audit_fft_wavelet_variant_v2.py release
    python audit_fft_wavelet_variant_v2.py all
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01_annotation"))
import label_rule_variants as L  # noqa: E402
from label_rule_variants import RuleParams, events, count_events, runs  # noqa: E402
from audit_label_robustness_v2 import synthetic_cases, _load_drive  # noqa: E402

OUT = HERE / "validation_output"; OUT.mkdir(exist_ok=True)
FS = 25.0

RELEASE_V3 = Path(os.environ.get("PREPROCESSED_ROOT", "") or
    r"C:\Users\halha\OneDrive - Durham University\Documents\DriveSense_Packages"
    r"\1_LATEST__labels_v2__release_v3__paper_v6\data"
    r"\Published_Dataset_Final_v3_VIDEO_ALIGNED\Preprocessed_Dataset")
CACHE_FILE = OUT / "release_columns_cache_v3.npz"
V2_EVENTS_CSV = RELEASE_V3.parent / "harsh_events_v2.csv"

ADOPTED = RuleParams(smooth_n=13, min_n=13, close_n=5, lateral="mean_then_abs", median_n=5,
                     brake_g=0.35, accel_g=0.20, turn_g=0.45, min_speed_kmh=15.0,
                     min_speed_turn_kmh=30.0, throttle_min_pct=25.0, brake_can_ms2=-0.3,
                     can_window_n=25, turn_min_heading_deg=5.0, heading_margin_n=3)

_ORIG_SMOOTHED_SIGNALS = L.smoothed_signals


# ============================================================ replacement smoothers
def _fft_lowpass_1d(x: np.ndarray, cutoff_hz: float) -> np.ndarray:
    n = len(x)
    X = np.fft.fft(x)
    f = np.fft.fftfreq(n, d=1.0 / FS)
    X[np.abs(f) > cutoff_hz] = 0.0
    return np.real(np.fft.ifft(X))


def _haar_dwt(x, level):
    A = x.astype(np.float64); Ds = []
    for _ in range(level):
        n = len(A) - (len(A) % 2)
        a = (A[0:n:2] + A[1:n:2]) / np.sqrt(2)
        d = (A[0:n:2] - A[1:n:2]) / np.sqrt(2)
        Ds.append(d); A = a
    return A, Ds


def _haar_idwt(A, Ds):
    for d in reversed(Ds):
        n = min(len(A), len(d))
        x = np.empty(2 * n, dtype=np.float64)
        x[0::2] = (A[:n] + d[:n]) / np.sqrt(2)
        x[1::2] = (A[:n] - d[:n]) / np.sqrt(2)
        A = x
    return A


def _wavelet_denoise_1d(x: np.ndarray, level: int = 4) -> np.ndarray:
    n0 = len(x)
    pad = (-n0) % (2 ** level)
    xp = np.r_[x, np.full(pad, x[-1])] if pad else x.copy()
    A, Ds = _haar_dwt(xp, level)
    sigma = np.median(np.abs(Ds[0] - np.median(Ds[0]))) / 0.6745
    thr = sigma * np.sqrt(2 * np.log(max(len(xp), 2)))
    Ds = [np.sign(d) * np.maximum(np.abs(d) - thr, 0.0) for d in Ds]
    xr = _haar_idwt(A, Ds)
    return xr[:n0]


def make_fft_smoother(cutoff_hz: float):
    def smoothed_signals_fft(df: pd.DataFrame, p: RuleParams):
        lon_raw = pd.to_numeric(df["lon_acc_g"], errors="coerce").to_numpy(float)
        lat_raw = pd.to_numeric(df["lat_acc_g"], errors="coerce").to_numpy(float)
        lon_raw = pd.Series(lon_raw).interpolate(limit_direction="both").to_numpy()
        lat_raw = pd.Series(lat_raw).interpolate(limit_direction="both").to_numpy()
        lon_f = _fft_lowpass_1d(lon_raw, cutoff_hz)
        lat_f = _fft_lowpass_1d(lat_raw, cutoff_hz)
        return pd.Series(lon_f, index=df.index), pd.Series(np.abs(lat_f), index=df.index)
    return smoothed_signals_fft


def smoothed_signals_wavelet(df: pd.DataFrame, p: RuleParams):
    lon_raw = pd.to_numeric(df["lon_acc_g"], errors="coerce").to_numpy(float)
    lat_raw = pd.to_numeric(df["lat_acc_g"], errors="coerce").to_numpy(float)
    lon_raw = pd.Series(lon_raw).interpolate(limit_direction="both").to_numpy()
    lat_raw = pd.Series(lat_raw).interpolate(limit_direction="both").to_numpy()
    lon_f = _wavelet_denoise_1d(lon_raw); lat_f = _wavelet_denoise_1d(lat_raw)
    return pd.Series(lon_f, index=df.index), pd.Series(np.abs(lat_f), index=df.index)


VARIANTS = {
    "ADOPTED median(5)+mean(13)": (_ORIG_SMOOTHED_SIGNALS, ADOPTED),
    "FFT low-pass 1 Hz":          (make_fft_smoother(1.0), ADOPTED),
    "FFT low-pass 2 Hz":          (make_fft_smoother(2.0), ADOPTED),
    "FFT low-pass 3 Hz":          (make_fft_smoother(3.0), ADOPTED),
    "Wavelet denoise (Haar/4, soft)": (smoothed_signals_wavelet, ADOPTED),
}


def label_with(df: pd.DataFrame, smoother, p: RuleParams) -> pd.Series:
    L.smoothed_signals = smoother
    try:
        return L.label_df(df, p)
    finally:
        L.smoothed_signals = _ORIG_SMOOTHED_SIGNALS


# ============================================================ Part A: synthetic
def part_synthetic():
    rows = []
    for name, d in synthetic_cases().items():
        r = {"case": name}
        for vn, (sm, p) in VARIANTS.items():
            c = count_events(label_with(d, sm, p))
            r[vn] = f"T{c['Harsh Turning']} B{c['Harsh Braking']} A{c['Harsh Acceleration']}"
        rows.append(r)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "fft_wavelet_variant_synthetic.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_colwidth", 55):
        print(R.to_string(index=False))
    return R


# ============================================================ Part B: calibration drives
def part_drives():
    truth = {"validation_session.csv": "S1: 10 brakes, 10 accels, 10 attempted turns (failed)",
             "validation_session2.csv": "S2: accels + attempted turns (failed)",
             "validation_session3.csv": "S3: 11 turns"}
    rows = []
    for f, t in truth.items():
        d = _load_drive(f); r = {"drive": t}
        for vn, (sm, p) in VARIANTS.items():
            c = count_events(label_with(d, sm, p))
            r[vn] = f"T{c['Harsh Turning']} B{c['Harsh Braking']} A{c['Harsh Acceleration']}"
        rows.append(r)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "fft_wavelet_variant_drives.csv", index=False)
    with pd.option_context("display.width", 220):
        print(R.to_string(index=False))
    return R


# ============================================================ Part C: full release
COLS = ["elapsed_s", "gps_speed_kmh", "speed_kph", "throttle_pct", "lat_acc_g", "lon_acc_g", "heading_deg"]


def _build_or_load_cache():
    if CACHE_FILE.exists():
        z = np.load(CACHE_FILE, allow_pickle=True)
        df = pd.DataFrame({k: z[k] for k in ["session"] + COLS})
        return df, list(z["tags"])
    files = sorted(RELEASE_V3.rglob("*_fused.csv"))
    parts, tags = [], []
    for i, f in enumerate(files):
        d = pd.read_csv(f, usecols=COLS); d["session"] = i
        tags.append(f.stem.replace("_fused", "")); parts.append(d)
        print(f"  cached {tags[-1]} ({len(d)} rows)", flush=True)
    df = pd.concat(parts, ignore_index=True)
    np.savez_compressed(CACHE_FILE, session=df.session.to_numpy(np.int16),
                        **{c: df[c].to_numpy(np.float64) for c in COLS}, tags=np.array(tags, dtype=object))
    return df, tags


def _label_all(df, tags, smoother, p):
    out = np.full(len(df), "Normal", dtype=object)
    sess = df["session"].to_numpy()
    for si, tag in enumerate(tags):
        idx = np.flatnonzero(sess == si)
        d = df.iloc[idx].reset_index(drop=True)
        out[idx] = label_with(d, smoother, p).to_numpy()
    return out


def _events_per_session(lab, sess, tags):
    """[(tag, cls, start_local, end_local)] -- events(), split by session boundary."""
    out = []
    for si, tag in enumerate(tags):
        idx = np.flatnonzero(sess == si)
        if len(idx) == 0:
            continue
        sub = lab[idx]
        for cls, a, b in events(sub):
            out.append((tag, cls, a, b))
    return out


def part_release():
    df, tags = _build_or_load_cache()
    sess = df["session"].to_numpy()
    official = pd.read_csv(V2_EVENTS_CSV) if V2_EVENTS_CSV.exists() else None

    rows = []
    ev_by_variant = {}
    for vn, (sm, p) in VARIANTS.items():
        lab = _label_all(df, tags, sm, p)
        c = count_events(lab)
        ev = _events_per_session(lab, sess, tags)
        ev_by_variant[vn] = ev
        rows.append({"variant": vn, **c, "total": sum(c.values()), "nonnormal_rows": int((lab != "Normal").sum())})
        print(rows[-1], flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "fft_wavelet_variant_release_totals.csv", index=False)
    print("\n" + R.to_string(index=False))

    # ---- event-by-event overlap of every variant against ADOPTED
    adopted_ev = ev_by_variant["ADOPTED median(5)+mean(13)"]
    adopted_set = {(t, c, a, b) for t, c, a, b in adopted_ev}
    overlap_rows = []
    for vn, ev in ev_by_variant.items():
        if vn == "ADOPTED median(5)+mean(13)":
            continue
        by_tag = {}
        for t, c, a, b in ev:
            by_tag.setdefault(t, []).append((c, a, b))
        adopted_by_tag = {}
        for t, c, a, b in adopted_ev:
            adopted_by_tag.setdefault(t, []).append((c, a, b))
        n_same_class_overlap = 0; n_class_change = 0; n_new = 0; n_missed = 0
        matched_adopted = set()
        for t, c, a, b in ev:
            hits = [(c2, a2, b2) for (c2, a2, b2) in adopted_by_tag.get(t, []) if a2 <= b and a <= b2]
            if not hits:
                n_new += 1
            elif any(c2 == c for c2, a2, b2 in hits):
                n_same_class_overlap += 1
                matched_adopted.add((t, hits[0]))
            else:
                n_class_change += 1
        n_total_adopted = len(adopted_ev)
        n_missed = n_total_adopted - len(matched_adopted)  # approximate: adopted events with no overlapping event of any class in this variant
        # recompute n_missed exactly
        n_missed = 0
        for t, c, a, b in adopted_ev:
            hits = [(c2, a2, b2) for (c2, a2, b2) in by_tag.get(t, []) if a2 <= b and a <= b2]
            if not hits:
                n_missed += 1
        overlap_rows.append({"variant": vn, "n_events_this_variant": len(ev), "n_events_adopted": n_total_adopted,
                             "same_class_overlap": n_same_class_overlap, "class_changed_on_overlap": n_class_change,
                             "new_event_not_in_adopted": n_new, "adopted_event_missed_entirely": n_missed})
    OV = pd.DataFrame(overlap_rows)
    OV.to_csv(OUT / "fft_wavelet_variant_overlap_vs_adopted.csv", index=False)
    print("\nOverlap against the ADOPTED (released) event set:")
    print(OV.to_string(index=False))
    return R, OV


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("synthetic", "all"):
        print("\n=== A. Synthetic stress cases ===")
        part_synthetic()
    if which in ("drives", "all"):
        print("\n=== B. Calibration drives (deliberate manoeuvres) ===")
        part_drives()
    if which in ("release", "all"):
        print("\n=== C. Full release (79 sessions) ===")
        part_release()
