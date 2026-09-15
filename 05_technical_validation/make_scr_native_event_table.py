"""
Native skin-conductance-response events and per-beat IBI check.

1. Builds a compact table of the EmotiBit's own SCR events (SA = amplitude,
   SR = rise time; one row per device event) for every valid session, on the
   released absolute timeline (same anchor-only conversion as the release,
   via regen_v2.emotibit_sync_v2.load_emotibit_v2), restricted to the released
   session window:  scr_events_native.csv  (tag, unix_time, elapsed_s,
   SCR_AMPLITUDE, SCR_RISE_TIME).
2. Internal-consistency assessment of those events against the released EDA
   channel (same modality, same device -- not an independent validation): for
   events with amplitude >= 0.01 uS, the EDA rise over the reported rise time
   vs a control offset 30 s earlier.
3. Per-session native event rate vs the device's SCR_FREQ column.
4. Per-beat check of the native IBI file: fraction of consecutive beat reports
   whose timestamp difference equals the reported interval (within 50 ms).

Outputs: validation_output/scr_events_native.csv, scr_native_per_session.csv,
and a printed summary. Environment: STAGING_DATA_ROOT, DATASET_ROOT,
optional RELEASE_ROOT (copies scr_events_native.csv there).
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "02_dataset_construction" / "regen_v2"))
sys.path.insert(0, str(HERE.parent / "02_dataset_construction"))
from emotibit_sync_v2 import load_emotibit_v2  # noqa: E402

STAGING = Path(os.environ.get("STAGING_DATA_ROOT", "") or r"C:\Users\halha\OneDrive - Durham University\Documents\Data")
ROOT = Path(os.environ.get("DATASET_ROOT", "") or r"C:\Users\halha\OneDrive - Durham University\Documents\DriveSense_Packages\3_OLD_versions\data\Published_Dataset_Final_v1_ORIGINAL_20260909")
RELEASE = Path(os.environ.get("RELEASE_ROOT", "") or "")
OUT = HERE / "validation_output"
TRIM = HERE.parent / "04_session_trimming" / "trim_review_output" / "trim_points_final.csv"


def session_dir(d, s):
    for name in (f"Session {s}", f"Session_{s}", f"Session{s}", f"Seesion {s}"):
        p = STAGING / f"D{d}" / name
        if p.exists():
            return p
    return None


def main():
    sessions = pd.read_csv(TRIM); sessions = sessions[sessions.status == "ok"]
    ev_rows, ses_rows = [], []
    for _, s in sessions.iterrows():
        tag, d, n = s.tag, int(s.driver), int(s.session)
        sd = session_dir(d, n)
        fused = pd.read_csv(ROOT / "Preprocessed_Dataset" / f"D{d}" / f"Session_{n}" / f"{tag}_fused.csv", usecols=["unix_time", "elapsed_s", "EDA", "SCR_FREQ"])
        t0, t1 = fused.unix_time.iloc[0], fused.unix_time.iloc[-1]
        signals, meta = load_emotibit_v2(sd)
        ka = next((k for k in signals if "AMPL" in k.upper() or k.upper() == "SA"), None)
        kr = next((k for k in signals if "RISE" in k.upper() or k.upper() == "SR"), None)
        kb = next((k for k in signals if k.upper() in ("IBI", "BI")), None)
        if ka is None or kr is None:
            ses_rows.append({"tag": tag, "status": f"no SA/SR in {list(signals)}"}); print(tag, "no SA/SR", list(signals)[:8]); continue
        A = signals[ka].rename(columns={"value": "SCR_AMPLITUDE"}); R = signals[kr].rename(columns={"value": "SCR_RISE_TIME"})
        E = pd.merge_asof(A.sort_values("unix_time"), R.sort_values("unix_time"), on="unix_time", tolerance=0.01, direction="nearest")
        E = E[(E.unix_time >= t0) & (E.unix_time <= t1)].copy(); E["tag"] = tag; E["elapsed_s"] = (E.unix_time - t0).round(3)
        ev_rows.append(E[["tag", "unix_time", "elapsed_s", "SCR_AMPLITUDE", "SCR_RISE_TIME"]])
        # internal consistency vs EDA
        eda_t = fused.unix_time.to_numpy(); eda = fused.EDA.to_numpy()
        def eda_at(t):
            i = np.clip(np.searchsorted(eda_t, t), 0, len(eda) - 1); return eda[i]
        big = E[E.SCR_AMPLITUDE >= 0.01]
        rise = eda_at(big.unix_time.to_numpy() + big.SCR_RISE_TIME.to_numpy()) - eda_at(big.unix_time.to_numpy())
        ctrl_t = big.unix_time.to_numpy() - 30.0; ok_c = ctrl_t > t0
        rise_c = eda_at(ctrl_t[ok_c] + big.SCR_RISE_TIME.to_numpy()[ok_c]) - eda_at(ctrl_t[ok_c])
        dur_min = (t1 - t0) / 60
        # per-beat check
        beat = None
        if kb is not None:
            B = signals[kb].sort_values("unix_time"); dt = np.diff(B.unix_time.to_numpy()); ibi = B.value.to_numpy()[1:] / 1000.0
            beat = float(np.mean(np.abs(dt - ibi) < 0.05)) if len(dt) else np.nan
        ses_rows.append({"tag": tag, "status": "ok", "n_native_events": len(E), "n_ge_0p01uS": len(big), "events_per_min": round(len(E) / dur_min, 2),
                         "scr_freq_median": round(float(fused.SCR_FREQ.median()), 2), "scr_freq_mean": round(float(fused.SCR_FREQ.mean()), 2),
                         "eda_rise_gt_0p01_frac": round(float(np.mean(rise > 0.01)), 3) if len(big) else np.nan,
                         "control_rise_gt_0p01_frac": round(float(np.mean(rise_c > 0.01)), 3) if ok_c.sum() else np.nan,
                         "spearman_amp_vs_rise": round(float(pd.Series(big.SCR_AMPLITUDE.to_numpy()).corr(pd.Series(rise), method="spearman")), 3) if len(big) > 5 else np.nan,
                         "ibi_native_rows": int(len(signals[kb])) if kb else np.nan, "ibi_per_beat_frac": round(beat, 3) if beat is not None else np.nan,
                         "n_fused_rows_with_scr": int((fused.SCR_FREQ.notna()).sum())})
        print(ses_rows[-1], flush=True)
    EV = pd.concat(ev_rows, ignore_index=True); EV.to_csv(OUT / "scr_events_native.csv", index=False)
    S = pd.DataFrame(ses_rows); S.to_csv(OUT / "scr_native_per_session.csv", index=False)
    ok = S[S.status == "ok"]
    print(f"\n{len(ok)} sessions, {len(EV):,} native SCR events in released windows (median {ok.n_native_events.median():.0f}/session, range {ok.n_native_events.min()}-{ok.n_native_events.max()}); "
          f">= 0.01 uS: {int(ok.n_ge_0p01uS.sum()):,} ({ok.n_ge_0p01uS.sum()/len(EV)*100:.1f}%)")
    w = ok.n_ge_0p01uS.to_numpy()
    print(f"EDA rise > 0.01 uS over reported rise time (amp >= 0.01): pooled {np.average(ok.eda_rise_gt_0p01_frac.fillna(0), weights=np.maximum(w,1e-9))*100:.1f}% vs control {np.average(ok.control_rise_gt_0p01_frac.fillna(0), weights=np.maximum(w,1e-9))*100:.1f}%; per-session median Spearman {ok.spearman_amp_vs_rise.median():.2f}")
    r = (ok.events_per_min / ok.scr_freq_median.replace(0, np.nan)); r2 = (ok.events_per_min / ok.scr_freq_mean.replace(0, np.nan))
    print(f"native events/min vs SCR_FREQ: ratio to session-median SCR_FREQ median {r.median():.2f} (IQR {r.quantile(.25):.2f}-{r.quantile(.75):.2f}); to mean {r2.median():.2f}; Spearman(events/min, median SCR_FREQ) = {ok.events_per_min.corr(ok.scr_freq_median, method='spearman'):.2f}")
    print(f"IBI native rows median {ok.ibi_native_rows.median():.0f}; per-beat consistency (|dt - IBI| < 50 ms) median {ok.ibi_per_beat_frac.median():.2f}, min {ok.ibi_per_beat_frac.min():.2f}")
    if RELEASE and RELEASE.exists():
        shutil.copy(OUT / "scr_events_native.csv", RELEASE / "scr_events_native.csv"); print("copied to", RELEASE)


if __name__ == "__main__":
    main()
