"""
Technical validation of the event-based SCR channels (SCR_AMPLITUDE,
SCR_RISE_TIME) against the continuous EDA channel and SCR_FREQ.

The two channels were added to the release on 2026-09-09 (regen_v2) and
had only been characterised descriptively. This script checks that they mean
what the manufacturer's definitions say (SA = onset-to-peak EDA change in uS;
SR = onset-to-peak time in s; SF = events/min):

  1. Recover the native events from the released fused.csv. The channels are
     event-triggered and resampled gap-aware onto the 25 Hz grid (linear
     interpolation between native samples < 3 s apart, NaN otherwise), so a
     native sample is a run boundary or an interior kink of the piecewise-
     linear signal (non-zero second difference).
  2. For each event, measure the rise on the EDA channel itself:
     max(EDA over [t, t + rise_time + 2 s]) - min(EDA over [t - 3 s, t]),
     and the same quantity at a control offset 30 s earlier.
  3. Report: fraction of events with a real EDA rise (> 0.01 uS) vs control;
     Spearman correlation and ratio between reported amplitude and measured
     rise (pooled and per session); amplitude distribution vs conventional
     SCR thresholds (0.01 / 0.05 uS); rise-time distribution; per-session
     event rate vs mean SCR_FREQ.

Usage:
    python audit_scr_channels.py
Environment:
    DATASET_ROOT   release root containing Preprocessed_Dataset/
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", ""))
OUT_DIR = Path(__file__).resolve().parent / "validation_output"
OUT_DIR.mkdir(exist_ok=True)

MIN_AMP_US = 0.01          # conventional minimum SCR amplitude
RISE_DETECT_US = 0.01      # "a real EDA rise" for the presence check
CONTROL_OFFSET_S = 30.0


def native_event_indices(t: np.ndarray, x: np.ndarray, present: np.ndarray) -> np.ndarray:
    idx = []
    n = len(x)
    i = 0
    while i < n:
        if not present[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and present[j + 1]:
            j += 1
        seg = x[i:j + 1]
        if len(seg) >= 3:
            d2 = np.abs(np.diff(seg, 2))
            kinks = np.where(d2 > 1e-7 * max(1.0, float(np.nanmax(np.abs(seg)))))[0] + 1
            pts = sorted(set([0, len(seg) - 1] + kinks.tolist()))
        else:
            pts = list(range(len(seg)))
        idx.extend(i + p for p in pts)
        i = j + 1
    idx = sorted(set(idx))
    keep = []
    for k in idx:
        if not keep or t[k] - t[keep[-1]] > 0.2:
            keep.append(k)
    return np.array(keep, dtype=int)


def window_rise(t, eda, t0, r):
    pre = eda[(t >= t0 - 3) & (t <= t0)]
    post = eda[(t >= t0) & (t <= t0 + r + 2)]
    if len(pre) < 5 or len(post) < 5 or np.all(np.isnan(pre)) or np.all(np.isnan(post)):
        return np.nan
    return float(np.nanmax(post) - np.nanmin(pre))


def main():
    if not DATASET_ROOT.exists():
        raise SystemExit(f"DATASET_ROOT not found or unset: {DATASET_ROOT!r}")
    ev, rates = [], []
    for f in sorted((DATASET_ROOT / "Preprocessed_Dataset").rglob("*_fused.csv")):
        d = pd.read_csv(f, usecols=["elapsed_s", "EDA", "SCR_FREQ", "SCR_AMPLITUDE", "SCR_RISE_TIME",
                                    "SCR_AMPLITUDE_dropout_flag"])
        t = d.elapsed_s.to_numpy(float)
        eda = d.EDA.to_numpy(float)
        sa = d.SCR_AMPLITUDE.to_numpy(float)
        sr = d.SCR_RISE_TIME.to_numpy(float)
        present = ~d.SCR_AMPLITUDE_dropout_flag.astype(bool).to_numpy()
        idx = native_event_indices(t, sa, present)
        tag = f.stem.replace("_fused", "")
        for k in idx:
            r = sr[k] if np.isfinite(sr[k]) else 1.0
            ev.append((tag, t[k], sa[k], r, window_rise(t, eda, t[k], r),
                       window_rise(t, eda, t[k] - CONTROL_OFFSET_S, r)))
        dur_min = (t[-1] - t[0]) / 60.0
        rates.append((tag, len(idx), len(idx) / dur_min if dur_min > 0 else np.nan, float(np.nanmean(d.SCR_FREQ))))

    E = pd.DataFrame(ev, columns=["tag", "t_s", "amplitude_uS", "rise_time_s", "eda_rise_uS", "eda_rise_control_uS"])
    E = E[np.isfinite(E.eda_rise_uS)]
    R = pd.DataFrame(rates, columns=["tag", "n_events", "events_per_min", "scr_freq_mean"])
    E.to_csv(OUT_DIR / "scr_events.csv", index=False)
    R.to_csv(OUT_DIR / "scr_event_rate_per_session.csv", index=False)

    strong = E[(E.amplitude_uS >= MIN_AMP_US) & (E.amplitude_uS < 10)]
    per = (strong.groupby("tag")
           .apply(lambda g: g[["amplitude_uS", "eda_rise_uS"]].corr(method="spearman").iloc[0, 1] if len(g) >= 20 else np.nan)
           .dropna())
    ratio = strong.eda_rise_uS / strong.amplitude_uS
    print(f"native SCR events: {len(E):,} in {E.tag.nunique()} sessions "
          f"({(R.n_events == 0).sum()} sessions with none)")
    print(f"amplitude (uS): median {E.amplitude_uS.median():.4f}, p75 {E.amplitude_uS.quantile(.75):.4f}, "
          f"p90 {E.amplitude_uS.quantile(.9):.3f}, p99 {E.amplitude_uS.quantile(.99):.2f}; "
          f">= {MIN_AMP_US} uS: {(E.amplitude_uS >= MIN_AMP_US).mean()*100:.1f}%; >= 0.05 uS: {(E.amplitude_uS >= 0.05).mean()*100:.1f}%")
    print(f"events >= {MIN_AMP_US} uS (n={len(strong):,}): EDA rise > {RISE_DETECT_US} uS in "
          f"{(strong.eda_rise_uS > RISE_DETECT_US).mean()*100:.1f}% (control offset: {(strong.eda_rise_control_uS > RISE_DETECT_US).mean()*100:.1f}%)")
    print(f"  Spearman rho(amplitude, EDA rise): pooled {strong[['amplitude_uS','eda_rise_uS']].corr(method='spearman').iloc[0,1]:.3f}; "
          f"per-session median {per.median():.2f} (min {per.min():.2f}, {len(per)} sessions with >= 20 events)")
    print(f"  measured EDA rise / reported amplitude: median {ratio.median():.2f} (IQR {ratio.quantile(.25):.2f}-{ratio.quantile(.75):.2f})")
    print(f"rise time (s): median {E.rise_time_s.median():.2f}, p5 {E.rise_time_s.quantile(.05):.2f}, "
          f"p95 {E.rise_time_s.quantile(.95):.2f}, max {E.rise_time_s.max():.2f}")
    rr = R[R.n_events > 0]
    print(f"event rate: median {rr.events_per_min.median():.2f} events/min reconstructed vs SCR_FREQ mean {rr.scr_freq_mean.median():.2f}; "
          f"ratio median {(rr.events_per_min / rr.scr_freq_mean).median():.2f}; Spearman across sessions "
          f"{rr[['events_per_min','scr_freq_mean']].corr(method='spearman').iloc[0,1]:.3f}")
    print(f"\nwritten: {OUT_DIR / 'scr_events.csv'}, {OUT_DIR / 'scr_event_rate_per_session.csv'}")


if __name__ == "__main__":
    main()
