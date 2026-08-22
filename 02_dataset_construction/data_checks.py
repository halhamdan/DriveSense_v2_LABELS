"""
Data quality and synchronization checks for Stage 1 fusion.

Checks performed:
  1. Temporal coverage   — overlap window between VBOX and EmotiBit
  2. Clock alignment     — offset between stream starts and max expected drift
  3. Sample-count check  — expected vs actual samples per signal (based on rate + duration)
  4. Gap detection       — consecutive timestamp jumps > 3× expected dt
  5. Signal range QC     — physiologically plausible value ranges
  6. Label distribution  — class balance across the session
  7. Sync quality score  — composite 0–100 score

All checks return a dict and optionally write a plain-text report.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ── Expected value ranges for physiological signals ──────────────────────────
SIGNAL_RANGES = {
    "HR":           (30,  220),   # bpm
    "EDA":          (0,   100),     # μS (raw EA channel)
    "EDA_LEVEL":    (-50000, 50000),  # baseline-relative ADC units (EL channel)
    "TEMP_CONTACT": (25,   46),   # °C (skin contact can exceed 42 in hot environments)
    "TEMP_THERMOPILE": (15, 45),
    "PPG_IR":       (0, 300000),
    "PPG_RED":      (0, 300000),
    "PPG_GREEN":    (0, 300000),
    "IBI":          (200, 2000),  # ms
    "engine_rpm":   (0,  8000),
    "throttle_pct": (0,   100),
    "brake_pct":    (0,   100),
    "speed_kph":    (0,   250),
    "gps_speed_kmh":(0,   250),
    "lat_acc_g":    (-5,    5),
    "lon_acc_g":    (-5,    5),
}

# Sampling rates for emotibit signals (Hz)
EMOTIBIT_FS = {
    "PPG_IR": 25, "PPG_RED": 25, "PPG_GREEN": 25,
    "EDA": 15, "EDA_LEVEL": 15, "SCR_FREQ": 3,
    "TEMP_CONTACT": 7.5, "TEMP_THERMOPILE": 7.5,
    "HR": 1, "IBI": 1,
    "ACC": 25, "GYRO": 25, "MAG": 25,
}


# ── Check 1: Temporal coverage ────────────────────────────────────────────────

def check_coverage(
    vbox_df: pd.DataFrame,
    emotibit_signals: dict[str, pd.DataFrame],
) -> dict:
    """Return start/end times and overlap for both streams."""
    vbox_start = float(vbox_df["unix_time"].iloc[0])
    vbox_end   = float(vbox_df["unix_time"].iloc[-1])
    vbox_dur   = vbox_end - vbox_start

    emb_starts = [df["unix_time"].iloc[0] for df in emotibit_signals.values() if not df.empty]
    emb_ends   = [df["unix_time"].iloc[-1] for df in emotibit_signals.values() if not df.empty]

    if not emb_starts:
        return {"ok": False, "error": "No EmotiBit signals"}

    emb_start = float(np.min(emb_starts))
    emb_end   = float(np.max(emb_ends))
    emb_dur   = emb_end - emb_start

    overlap_start = max(vbox_start, emb_start)
    overlap_end   = min(vbox_end, emb_end)
    overlap_s     = max(0.0, overlap_end - overlap_start)

    return {
        "ok": overlap_s > 60.0,
        "vbox_start":    vbox_start,
        "vbox_end":      vbox_end,
        "vbox_dur_s":    vbox_dur,
        "emb_start":     emb_start,
        "emb_end":       emb_end,
        "emb_dur_s":     emb_dur,
        "overlap_start": overlap_start,
        "overlap_end":   overlap_end,
        "overlap_s":     overlap_s,
        "overlap_pct_vbox": round(overlap_s / vbox_dur * 100, 1) if vbox_dur > 0 else 0,
        "overlap_pct_emb":  round(overlap_s / emb_dur  * 100, 1) if emb_dur  > 0 else 0,
    }


# ── Check 2: Clock alignment ─────────────────────────────────────────────────

def check_alignment(
    vbox_df: pd.DataFrame,
    emotibit_signals: dict[str, pd.DataFrame],
) -> dict:
    """
    Measure clock offset between VBOX and EmotiBit streams.
    Offset = EmotiBit_start − VBOX_start  (positive = EmotiBit started earlier).
    An offset > 120 s in either direction suggests a date or timezone error.
    """
    vbox_start = float(vbox_df["unix_time"].iloc[0])

    offsets = {}
    for name, df in emotibit_signals.items():
        if df.empty:
            continue
        emb_start = float(df["unix_time"].iloc[0])
        offsets[name] = emb_start - vbox_start

    if not offsets:
        return {"ok": False, "error": "No EmotiBit signals"}

    median_offset = float(np.median(list(offsets.values())))
    max_inter_signal = float(max(offsets.values()) - min(offsets.values()))

    ok = abs(median_offset) < 300.0 and max_inter_signal < 5.0

    return {
        "ok": ok,
        "median_offset_s": round(median_offset, 3),
        "max_inter_signal_spread_s": round(max_inter_signal, 3),
        "per_signal_offsets": {k: round(v, 3) for k, v in offsets.items()},
        "interpretation": (
            "EmotiBit started EARLIER than VBOX" if median_offset < 0
            else "EmotiBit started LATER than VBOX"
        ),
    }


# ── Check 3: Sample count ────────────────────────────────────────────────────

def check_sample_counts(
    emotibit_signals: dict[str, pd.DataFrame],
    overlap_s: float,
) -> dict:
    """Compare actual vs expected sample count for each EmotiBit signal."""
    results = {}
    all_ok = True
    for name, df in emotibit_signals.items():
        if df.empty:
            results[name] = {"ok": False, "error": "empty"}
            continue
        fs = EMOTIBIT_FS.get(name, 25.0)
        expected = int(overlap_s * fs)
        actual   = len(df)
        ratio    = actual / expected if expected > 0 else 0
        ok       = 0.80 <= ratio <= 1.20
        all_ok   = all_ok and ok
        results[name] = {
            "ok": ok,
            "fs_hz": fs,
            "expected": expected,
            "actual": actual,
            "completeness_pct": round(ratio * 100, 1),
        }
    return {"ok": all_ok, "signals": results}


# ── Check 4: Gap detection ───────────────────────────────────────────────────

def check_gaps(
    emotibit_signals: dict[str, pd.DataFrame],
    gap_threshold_x: float = 3.0,
) -> dict:
    """
    Find gaps in each signal where consecutive dt > gap_threshold_x × expected_dt.
    """
    results = {}
    any_gaps = False
    for name, df in emotibit_signals.items():
        if df.empty or len(df) < 2:
            continue
        fs = EMOTIBIT_FS.get(name, 25.0)
        expected_dt = 1.0 / fs
        threshold   = gap_threshold_x * expected_dt

        t = df["unix_time"].to_numpy()
        diffs = np.diff(t)
        gap_mask = diffs > threshold
        n_gaps = int(gap_mask.sum())
        total_gap_s = float(diffs[gap_mask].sum() - expected_dt * n_gaps)

        if n_gaps > 0:
            any_gaps = True
            gap_locs = np.where(gap_mask)[0]
            largest = float(diffs[gap_locs].max())
        else:
            largest = 0.0

        results[name] = {
            "n_gaps": n_gaps,
            "total_gap_s": round(total_gap_s, 3),
            "largest_gap_s": round(largest, 3),
            "ok": n_gaps == 0,
        }

    return {"any_gaps": any_gaps, "signals": results}


# ── Check 5: Signal value ranges ────────────────────────────────────────────

def check_ranges(
    emotibit_signals: dict[str, pd.DataFrame],
    vbox_df: pd.DataFrame,
) -> dict:
    """Check that signal values fall within physiologically plausible ranges."""
    results = {}

    def _check_series(name: str, values: pd.Series) -> dict:
        lo, hi = SIGNAL_RANGES.get(name, (None, None))
        if lo is None:
            return {"ok": True, "note": "no range defined"}
        vals = values.dropna()
        if len(vals) == 0:
            return {"ok": False, "error": "all NaN"}
        pct_out = float((~vals.between(lo, hi)).mean() * 100)
        vmin = float(vals.min())
        vmax = float(vals.max())
        return {
            "ok": pct_out < 5.0,
            "expected_range": [lo, hi],
            "actual_range": [round(vmin, 3), round(vmax, 3)],
            "pct_out_of_range": round(pct_out, 2),
        }

    for name, df in emotibit_signals.items():
        if "value" in df.columns:
            results[name] = _check_series(name, df["value"])

    for col in ["engine_rpm", "throttle_pct", "brake_pct", "speed_kph",
                "gps_speed_kmh", "lat_acc_g", "lon_acc_g"]:
        if col in vbox_df.columns:
            results[col] = _check_series(col, vbox_df[col])

    all_ok = all(v["ok"] for v in results.values())
    return {"ok": all_ok, "signals": results}


# ── Check 6: Label distribution ─────────────────────────────────────────────

def check_labels(vbox_df: pd.DataFrame) -> dict:
    """Summarise label distribution and flag severe imbalance."""
    counts = vbox_df["label"].value_counts()
    total  = len(vbox_df)
    dist   = {k: {"count": int(v), "pct": round(v / total * 100, 1)}
              for k, v in counts.items()}
    normal_pct = dist.get("Normal", {}).get("pct", 0.0)
    return {
        "ok": True,   # distribution is informational, not a pass/fail
        "total_rows": total,
        "n_classes": len(dist),
        "distribution": dist,
        "dominant_class": counts.index[0],
        "imbalance_note": (
            f"Normal dominates at {normal_pct:.1f}%"
            if normal_pct > 80 else "Reasonable balance"
        ),
    }


# ── Composite report ────────────────────────────────────────────────────────

def run_all_checks(
    vbox_df: pd.DataFrame,
    emotibit_signals: dict[str, pd.DataFrame],
    session_label: str = "",
) -> dict:
    """Run all checks and return a combined report dict."""
    cov  = check_coverage(vbox_df, emotibit_signals)
    aln  = check_alignment(vbox_df, emotibit_signals)
    cnt  = check_sample_counts(emotibit_signals, cov.get("overlap_s", 0))
    gaps = check_gaps(emotibit_signals)
    rng  = check_ranges(emotibit_signals, vbox_df)
    lbl  = check_labels(vbox_df)

    checks = {"coverage": cov, "alignment": aln, "sample_counts": cnt,
              "gaps": gaps, "ranges": rng, "labels": lbl}

    # Composite score (0–100): each check except labels contributes equally
    scored = [cov["ok"], aln["ok"], cnt["ok"], not gaps["any_gaps"], rng["ok"]]
    score  = round(sum(scored) / len(scored) * 100)

    return {
        "session": session_label,
        "sync_quality_score": score,
        "checks": checks,
    }


def print_report(report: dict) -> None:
    """Print a human-readable summary of all checks."""
    s = report["session"]
    score = report["sync_quality_score"]
    c = report["checks"]
    ok_str = lambda b: "PASS" if b else "FAIL"

    print(f"\n{'='*60}")
    print(f"  DATA CHECKS: {s}   (sync quality score: {score}/100)")
    print(f"{'='*60}")

    cov = c["coverage"]
    print(f"\n[1] Temporal Coverage          {ok_str(cov['ok'])}")
    print(f"    VBOX   : {cov['vbox_dur_s']:.1f} s")
    print(f"    EmotiBit: {cov['emb_dur_s']:.1f} s")
    print(f"    Overlap: {cov['overlap_s']:.1f} s ({cov['overlap_pct_vbox']:.1f}% of VBOX)")

    aln = c["alignment"]
    print(f"\n[2] Clock Alignment            {ok_str(aln['ok'])}")
    print(f"    Offset (EmotiBit-VBOX): {aln['median_offset_s']:+.3f} s")
    print(f"    Inter-signal spread   : {aln['max_inter_signal_spread_s']:.3f} s")
    print(f"    {aln['interpretation']}")

    cnt = c["sample_counts"]
    print(f"\n[3] Sample Counts              {ok_str(cnt['ok'])}")
    for name, r in cnt["signals"].items():
        if r.get("error"):
            print(f"    {name:22s}: ERROR - {r['error']}")
        else:
            flag = "" if r["ok"] else "  ** WARNING"
            print(f"    {name:22s}: {r['actual']:6d}/{r['expected']:6d} "
                  f"({r['completeness_pct']:.1f}%){flag}")

    gaps = c["gaps"]
    print(f"\n[4] Gap Detection              {ok_str(not gaps['any_gaps'])}")
    for name, r in gaps["signals"].items():
        if r["n_gaps"] > 0:
            print(f"    {name:22s}: {r['n_gaps']} gaps, "
                  f"total={r['total_gap_s']:.2f}s, largest={r['largest_gap_s']:.2f}s")
    if not gaps["any_gaps"]:
        print("    No gaps detected in any signal.")

    rng = c["ranges"]
    print(f"\n[5] Signal Value Ranges        {ok_str(rng['ok'])}")
    for name, r in rng["signals"].items():
        if r.get("note"):
            continue
        flag = "" if r["ok"] else f"  ** {r['pct_out_of_range']:.1f}% out of range"
        print(f"    {name:22s}: {str(r['actual_range']):20s} expected={r['expected_range']}{flag}")

    lbl = c["labels"]
    print(f"\n[6] Label Distribution")
    for k, v in lbl["distribution"].items():
        print(f"    {k:25s}: {v['count']:6d} rows ({v['pct']:5.1f}%)")
    print(f"    {lbl['imbalance_note']}")


def write_report(report: dict, out_path: str | Path) -> None:
    """Write the human-readable report to a text file."""
    import io, sys
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    print_report(report)
    sys.stdout = old_stdout
    Path(out_path).write_text(buf.getvalue(), encoding="utf-8")
