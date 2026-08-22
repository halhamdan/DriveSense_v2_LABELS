"""
Stage 1 Data Fusion Pipeline — D2 Project
==========================================

Fuses VBOX telemetry (behavioral, labeled) with EmotiBit biometric data.

Per-session steps:
  1. Load LABELED CSV (VBOX)     → telemetry + labels at 25 Hz
  2. Load + sync EmotiBit        → biometric signals (various rates)
  3. Run data checks             → alignment, gaps, ranges, labels
  4. Clip to overlap window      → common start/end for both streams
  5. Resample everything to 4 Hz → uniform time grid
  6. Merge on unix_time          → one row per 0.25 s timestep
  7. Export fused CSV            → output/D2_S{n}_fused_stage1.csv

Usage:
  python fuse_stage1.py [--data-root <path>] [--out-root <path>] [--smoke]
  --smoke: process only Session 1

Output columns:
  unix_time           absolute Unix epoch (s)
  elapsed_s           seconds from session start (0-based)
  session             session number
  subject             subject ID
  label               Normal | Harsh Braking | Harsh Acceleration | Harsh Turning
  --- VBOX telemetry ---
  gps_speed_kmh, speed_kph, engine_rpm, throttle_pct, brake_pct,
  lat_acc_g, lon_acc_g, gradient_pct, heading_deg, height_m
  --- EmotiBit biometrics ---
  HR, EDA, EDA_LEVEL, TEMP_CONTACT, TEMP_THERMOPILE,
  PPG_IR, PPG_RED, PPG_GREEN, IBI, SCR_FREQ
  ACC_mag, ACC_x, ACC_y, ACC_z
  GYRO_mag, GYRO_x, GYRO_y, GYRO_z
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

# ── Module imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
import parse_labeled as pl
import emotibit_sync as es
import data_checks as dc

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).parent.parent.parent   # Data_Fusion/
DATA_ROOT  = REPO_ROOT / "D2"
OUT_ROOT   = Path(__file__).parent.parent           # D2_fusion/

FUSE_FS    = 25.0   # Hz — output sample rate (25 Hz captures vehicle dynamics onset)

# ── Butterworth LPF for anti-aliasing before downsampling ────────────────────
def _lowpass(data: np.ndarray, fs_in: float, cutoff: float = 1.9, order: int = 3) -> np.ndarray:
    nyq = fs_in / 2.0
    if cutoff >= nyq or len(data) < 3 * (order + 1):
        return data.copy()
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, data)


# ── Despiking (impulse-noise removal) ─────────────────────────────────────────
def _hampel(x: np.ndarray, win: int = 7, n_sigma: float = 3.0, abs_threshold: float | None = None) -> np.ndarray:
    """
    Hampel filter: replaces samples that deviate from their local median by
    more than n_sigma robust standard deviations (median absolute deviation)
    with that local median; everything else is left untouched. This removes
    isolated single/dual-sample sensor glitches (e.g. a lone multi-g spike
    sitting in an otherwise flat stretch) while leaving genuine sustained
    signal changes -- such as a real 0.4s+ harsh-braking/turning acceleration
    ramp -- completely unaffected, since a real event is not an outlier
    relative to its own immediate neighbours.

    A genuine event's ONSET, however, is by definition a sharp local jump --
    a purely statistical local-outlier test cannot distinguish that from an
    isolated sensor glitch, and a deterministic-oracle check (re-running the
    released harsh-event labelling algorithm directly on this despiked signal)
    found this was measurably attenuating real events: a magnitude-gated
    variant, applied only to lat_acc_g/lon_acc_g (the channels driving the
    harsh-event g-force thresholds), only despikes a statistical outlier if
    it ALSO exceeds abs_threshold -- a value clearly beyond any real vehicle
    dynamics (documented plausible range +-1.2g; real severe bumps/potholes
    reach roughly 1.5-2g) but far below the ~28g glitch that motivated
    despiking in the first place. This closed a 93.81%->98.03% released-label
    reproducibility gap on the deterministic oracle check while still
    correctly flagging that original glitch (verified directly). abs_threshold
    is None (no magnitude gate) for every other despiked channel.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2 * win + 1:
        return x.copy()
    out = x.copy()
    for i in range(n):
        lo, hi = max(0, i - win), min(n, i + win + 1)
        w = x[lo:hi]
        med = np.median(w)
        mad = np.median(np.abs(w - med)) * 1.4826
        if mad < 1e-6:
            mad = 1e-6
        is_stat_outlier = abs(x[i] - med) > n_sigma * mad
        is_gated_in = abs_threshold is None or abs(x[i]) > abs_threshold
        if is_stat_outlier and is_gated_in:
            out[i] = med
    return out


# Channels magnitude-gated during despiking (see _hampel docstring): only a
# statistical outlier that ALSO exceeds this absolute value is treated as a
# glitch. Both are the direct-sensor channels driving the harsh-event
# g-force thresholds.
_MAGNITUDE_GATED_SIGNALS = {"lat_acc_g": 3.0, "lon_acc_g": 3.0}


# Signals whose "outlier" samples are a real, meaningful sensor characteristic
# (PPG beat-detection dropout reported as one long HR/IBI value) rather than
# an electrical glitch, and must NOT be despiked -- doing so would silently
# hide the dropout behaviour this release deliberately discloses instead of
# masking. Both are also 1 Hz native, where the original 1.9 Hz anti-alias
# cutoff already exceeded the Nyquist frequency and therefore never actually
# filtered them; despiking would be new behaviour, not a restoration of intent.
_NO_DESPIKE_SIGNALS = {"HR", "IBI"}


# ── Resampling ────────────────────────────────────────────────────────────────
def _resample_series(
    t_in: np.ndarray,
    v_in: np.ndarray,
    t_out: np.ndarray,
    fs_in: float,
    despike: bool = True,
    abs_threshold: float | None = None,
) -> np.ndarray:
    """
    Despike then linearly interpolate to t_out grid.

    Anti-alias lowpass filtering is only applied when actually downsampling
    (fs_in > FUSE_FS) -- every native sensor rate in this dataset is already
    <= FUSE_FS, so no channel is ever truly downsampled here; applying the
    1.9 Hz filter unconditionally previously smoothed out genuine
    short-duration acceleration peaks below the harsh-event thresholds they
    were originally labelled from.

    Removing that filter outright is not safe either: it also exposed
    isolated single-sample sensor glitches (observed up to ~28g on a
    channel that reads exactly 0 on every neighbouring sample -- a clear
    electrical/sensor artefact, not vehicle dynamics). The Hampel despiking
    step below removes that specific failure mode without the frequency-domain
    smoothing that attenuates real events. See `_NO_DESPIKE_SIGNALS` for the
    channels this must not be applied to.
    """
    mask = np.isfinite(v_in)
    t_in, v_in = t_in[mask], v_in[mask]
    if len(t_in) < 4:
        return np.full(len(t_out), np.nan)
    v_despiked = _hampel(v_in, win=7, n_sigma=3.0, abs_threshold=abs_threshold) if despike else v_in
    if fs_in > FUSE_FS:
        v_use = _lowpass(v_despiked, fs_in, cutoff=min(1.9, FUSE_FS / 2.0 * 0.9))
    else:
        v_use = v_despiked
    f = interp1d(t_in, v_use, kind="linear", bounds_error=False,
                 fill_value=(v_use[0], v_use[-1]))
    return f(t_out)


def _gap_mask(t_in: np.ndarray, t_out: np.ndarray, max_gap_s: float) -> np.ndarray:
    """
    True at every t_out position whose two bracketing native samples in t_in
    are more than max_gap_s apart (i.e. t_out falls inside a native dropout).
    Boundary points (before the first / after the last native sample) count
    as inside a gap too.
    """
    idx = np.searchsorted(t_in, t_out)
    out = np.zeros(len(t_out), dtype=bool)
    for i, j in enumerate(idx):
        if j <= 0 or j >= len(t_in):
            out[i] = True
            continue
        out[i] = (t_in[j] - t_in[j - 1]) > max_gap_s
    return out


def _resample_series_gap_aware(
    t_in: np.ndarray,
    v_in: np.ndarray,
    t_out: np.ndarray,
    native_dt_s: float,
    gap_threshold_x: float = 3.0,
) -> np.ndarray:
    """
    Like `_resample_series` (no despiking -- see `_NO_DESPIKE_SIGNALS`), but
    additionally sets NaN at every t_out sample that falls inside a native
    dropout longer than gap_threshold_x * native_dt_s (matching the same
    3x-expected-dt convention `data_checks.check_gaps` already uses to flag
    gaps). Linear interpolation across a beat-detection dropout of tens of
    seconds fabricates a smooth ramp between two real endpoints and, worse,
    stretches a single long-gap IBI/HR reading across every resampled row in
    that span; leaving those rows as a genuine missing value instead is more
    honest about what was actually measured.
    """
    mask = np.isfinite(v_in)
    t_in, v_in = t_in[mask], v_in[mask]
    if len(t_in) < 4:
        return np.full(len(t_out), np.nan)
    v = _resample_series(t_in, v_in, t_out, fs_in=1.0 / native_dt_s, despike=False)
    gap = _gap_mask(t_in, t_out, max_gap_s=gap_threshold_x * native_dt_s)
    v = v.copy()
    v[gap] = np.nan
    return v


def _resample_emotibit_signal(
    df: pd.DataFrame,
    t_out: np.ndarray,
    signal_name: str,
) -> dict[str, np.ndarray]:
    """
    Resample one EmotiBit signal to t_out.
    Returns dict col_name → array.  Handles scalar and 3-axis DataFrames.
    """
    from data_checks import EMOTIBIT_FS
    fs_in = EMOTIBIT_FS.get(signal_name, 25.0)
    despike = signal_name not in _NO_DESPIKE_SIGNALS

    if "value" in df.columns:
        t_in = df["unix_time"].to_numpy()
        v_in = df["value"].to_numpy()
        if signal_name in _NO_DESPIKE_SIGNALS:
            # Gap-aware: NaN out resampled rows that fall inside a beat-detection
            # dropout instead of linearly interpolating across it, and carry an
            # explicit per-row flag so users can tell "measured" from "no data".
            v = _resample_series_gap_aware(t_in, v_in, t_out, native_dt_s=1.0 / fs_in)
            return {signal_name: v, f"{signal_name}_dropout_flag": np.isnan(v)}
        v = _resample_series(t_in, v_in, t_out, fs_in, despike=despike)
        return {signal_name: v}

    # 3-axis
    out = {}
    mag_parts = []
    for ax in ["x", "y", "z"]:
        if ax not in df.columns:
            continue
        v = _resample_series(df["unix_time"].to_numpy(), df[ax].to_numpy(), t_out, fs_in)
        out[f"{signal_name}_{ax}"] = v
        mag_parts.append(v)
    if mag_parts:
        out[f"{signal_name}_mag"] = np.sqrt(sum(a**2 for a in mag_parts))
    return out


def _resample_vbox(vbox_df: pd.DataFrame, t_out: np.ndarray) -> dict[str, np.ndarray]:
    """Resample VBOX telemetry columns to t_out grid (25 Hz → 4 Hz)."""
    fs_in = 25.0
    vbox_cols = [
        "gps_speed_kmh", "speed_kph", "engine_rpm", "throttle_pct", "brake_pct",
        "lat_acc_g", "lon_acc_g", "gradient_pct", "heading_deg", "height_m",
        "wheel_fr_kph", "wheel_fl_kph",
    ]
    t_v = vbox_df["unix_time"].to_numpy()
    out = {}
    for col in vbox_cols:
        if col not in vbox_df.columns:
            continue
        v = vbox_df[col].to_numpy(dtype=float)
        # heading_deg is circular (0 deg == 360 deg); Hampel despiking compares
        # raw numeric differences and does not know about wraparound, so a
        # genuine smooth turn crossing the 0/360 boundary gets treated as a
        # huge outlier and corrupted into an erratic, wrong signal. Must be
        # excluded from despiking, same reasoning as `_NO_DESPIKE_SIGNALS`.
        despike = col != "heading_deg"
        abs_threshold = _MAGNITUDE_GATED_SIGNALS.get(col)
        out[col] = _resample_series(t_v, v, t_out, fs_in, despike=despike, abs_threshold=abs_threshold)

    # Label: nearest-neighbour (categorical)
    if "label" in vbox_df.columns:
        labels = vbox_df["label"].to_numpy()
        t_v_arr = t_v
        idx = np.searchsorted(t_v_arr, t_out, side="left")
        idx = np.clip(idx, 0, len(t_v_arr) - 1)
        out["label"] = labels[idx]

    return out


# ── Session discovery ─────────────────────────────────────────────────────────
def discover_sessions(data_root: Path) -> list[dict]:
    sessions = []
    for d in sorted(data_root.glob("Session *")):
        if not d.is_dir():
            continue
        labeled_csvs = list(d.glob("*LABELED*.csv")) + list(d.glob("*labeled*.csv"))
        if not labeled_csvs:
            continue
        sessions.append({
            "subject": data_root.name,
            "session_dir": d,
            "session_name": d.name,
            "labeled_csv": labeled_csvs[0],
        })
    return sessions


# ── Per-session fusion ────────────────────────────────────────────────────────
def fuse_session(
    sess: dict,
    out_dir: Path,
    report_dir: Path,
    smoke: bool = False,
) -> dict | None:
    subj = sess["subject"]
    sname = sess["session_name"]
    sess_dir = sess["session_dir"]
    labeled_csv = sess["labeled_csv"]
    tag = f"{subj}_{sname.replace(' ', '_')}"

    print(f"\n{'='*60}")
    print(f"  {subj}  |  {sname}")
    print(f"{'='*60}")

    # ── 1. Load VBOX ──────────────────────────────────────────────────────────
    print("  Loading LABELED CSV...")
    vbox_df = pl.load_labeled_csv(labeled_csv, session_dir=sess_dir)
    print(f"  VBOX: {len(vbox_df)} rows | "
          f"dur={vbox_df['elapsed_s'].max():.1f}s | "
          f"labels={vbox_df['label'].nunique()} classes")

    # ── 2. Load + sync EmotiBit ───────────────────────────────────────────────
    print("  Loading EmotiBit...")
    emotibit_signals, emb_meta = es.load_emotibit(sess_dir)
    for name, df in emotibit_signals.items():
        print(f"    {name:22s}: {len(df):7d} samples")

    # ── 3. Data checks ────────────────────────────────────────────────────────
    print("  Running data checks...")
    report = dc.run_all_checks(vbox_df, emotibit_signals, session_label=tag)
    dc.print_report(report)

    report_path = report_dir / f"{tag}_checks.txt"
    dc.write_report(report, report_path)
    print(f"  Report saved: {report_path.name}")

    cov = report["checks"]["coverage"]
    if cov["overlap_s"] < 30.0:
        print(f"  [SKIP] Overlap only {cov['overlap_s']:.1f}s — skipping fusion.")
        return None

    # ── 4. Clip to overlap window ─────────────────────────────────────────────
    t_start = cov["overlap_start"]
    t_end   = cov["overlap_end"]
    print(f"\n  Overlap window: {cov['overlap_s']:.1f} s "
          f"({cov['overlap_pct_vbox']:.1f}% of VBOX)")

    vbox_clip = vbox_df[
        (vbox_df["unix_time"] >= t_start) &
        (vbox_df["unix_time"] <= t_end)
    ].copy().reset_index(drop=True)

    emb_clip = {
        name: df[(df["unix_time"] >= t_start) & (df["unix_time"] <= t_end)].reset_index(drop=True)
        for name, df in emotibit_signals.items()
    }

    if smoke:
        # Limit to first 5 minutes for smoke test
        smoke_end = t_start + 300.0
        vbox_clip = vbox_clip[vbox_clip["unix_time"] <= smoke_end].reset_index(drop=True)
        emb_clip  = {n: d[d["unix_time"] <= smoke_end].reset_index(drop=True)
                     for n, d in emb_clip.items()}
        print("  [SMOKE] Limited to first 5 minutes.")

    # ── 5 + 6. Resample to 4 Hz and merge ────────────────────────────────────
    print(f"  Resampling to {FUSE_FS} Hz and merging...")
    dt = 1.0 / FUSE_FS
    t_out = np.arange(t_start, t_end, dt)

    fused: dict[str, np.ndarray] = {"unix_time": t_out}
    fused["elapsed_s"] = t_out - t_start

    # VBOX columns
    vbox_out = _resample_vbox(vbox_clip, t_out)
    fused.update(vbox_out)

    # EmotiBit signals
    for sig_name, df in emb_clip.items():
        if df.empty:
            continue
        emb_out = _resample_emotibit_signal(df, t_out, sig_name)
        fused.update(emb_out)

    fused_df = pd.DataFrame(fused)
    fused_df.insert(0, "session",  sname)
    fused_df.insert(0, "subject",  subj)

    # Reorder: metadata first, then label, then VBOX, then biometrics
    priority = ["subject", "session", "unix_time", "elapsed_s", "label"]
    vbox_cols = [c for c in fused_df.columns
                 if c not in priority and c not in
                 [s for s in emotibit_signals] +
                 [f"{s}_x" for s in emotibit_signals] +
                 [f"{s}_y" for s in emotibit_signals] +
                 [f"{s}_z" for s in emotibit_signals] +
                 [f"{s}_mag" for s in emotibit_signals]]
    emb_cols  = [c for c in fused_df.columns if c not in priority + vbox_cols]
    ordered   = [c for c in priority + vbox_cols + emb_cols if c in fused_df.columns]
    fused_df  = fused_df[ordered]

    # ── 7. Export ─────────────────────────────────────────────────────────────
    out_path = out_dir / f"{tag}_fused_stage1.csv"
    fused_df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"\n  Exported: {out_path.name}")
    print(f"  Shape   : {fused_df.shape[0]} rows × {fused_df.shape[1]} cols")
    print(f"  Duration: {fused_df['elapsed_s'].max():.1f} s @ {FUSE_FS} Hz")
    print(f"  Columns : {list(fused_df.columns)}")

    return {
        "tag": tag,
        "rows": fused_df.shape[0],
        "cols": fused_df.shape[1],
        "duration_s": float(fused_df["elapsed_s"].max()),
        "sync_quality_score": report["sync_quality_score"],
        "label_dist": {k: int(v) for k, v in
                       fused_df["label"].value_counts().items()
                       if isinstance(fused_df["label"].iloc[0], str)},
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="D2 Stage 1 Fusion Pipeline")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--out-root",  type=Path, default=OUT_ROOT)
    parser.add_argument("--smoke", action="store_true",
                        help="Process only Session 1, first 5 minutes")
    args = parser.parse_args()

    out_dir    = args.out_root / "output"
    report_dir = args.out_root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data root : {args.data_root}")
    print(f"Output    : {out_dir}")

    sessions = discover_sessions(args.data_root)
    print(f"Found {len(sessions)} session(s): {[s['session_name'] for s in sessions]}")

    if args.smoke:
        sessions = sessions[:1]
        print("[SMOKE] Processing Session 1 only (first 5 min).")

    summaries = []
    for sess in sessions:
        result = fuse_session(sess, out_dir, report_dir, smoke=args.smoke)
        if result:
            summaries.append(result)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  STAGE 1 FUSION SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Session':<30} {'Rows':>7} {'Cols':>5} {'Dur(s)':>8} {'QScore':>7}")
    print(f"  {'-'*57}")
    for s in summaries:
        print(f"  {s['tag']:<30} {s['rows']:>7} {s['cols']:>5} "
              f"{s['duration_s']:>8.1f} {s['sync_quality_score']:>7}")

    # Save summary JSON
    summary_path = args.out_root / "fusion_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\n  Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
