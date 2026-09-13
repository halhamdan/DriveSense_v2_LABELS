"""
Attempted shared physical timing anchor between the VBOX HD2 and the EmotiBit
(Technical Validation, "Session integrity and cross-modal timing").

Hard braking is transmitted mechanically to the driver's wrist within tens of
milliseconds, so if the wrist-worn IMU responded measurably to vehicle
deceleration, the lag between the VBOX longitudinal acceleration and the wrist
accelerometer/gyroscope would estimate the residual clock offset between the
two devices per session -- independently of the sync-ping anchor and without
using any physiological channel (which would confound clock error with
genuine response latency).

Method, per session: for every Harsh Braking event onset, take a +/-3 s window
of |d/dt lon_acc_g| (vehicle) and |d/dt ACC_mag| or |d/dt GYRO_mag| (wrist),
z-score each, and compute the normalised cross-correlation over lags of
-2..+2 s; average over events; the peak lag is the offset estimate. A
bootstrap over events (200 resamples) gives the 95% interval of the peak lag;
the peak's z-score relative to the correlation function's own mean/std
indicates whether a peak exists at all.

Result on the released data (2026-09-13): no session yields a usable
estimate -- peak correlations 0.02-0.03, bootstrap intervals ~3 s wide,
accelerometer- and gyroscope-derived lags agreeing to within 0.2 s in only
17 of 79 sessions (chance level). Written to
validation_output/mechanical_anchor_check.csv.

Usage:
    python audit_mechanical_anchor.py
Environment:
    DATASET_ROOT   release root containing Preprocessed_Dataset/
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", ""))
OUT = Path(__file__).resolve().parent / "validation_output" / "mechanical_anchor_check.csv"

FS = 25
MAX_LAG_S, WIN_S = 2.0, 3.0
EVENT_CLASS = "Harsh Braking"
N_BOOT = 200


def main():
    if not DATASET_ROOT.exists():
        raise SystemExit(f"DATASET_ROOT not found or unset: {DATASET_ROOT!r}")
    L, W = int(MAX_LAG_S * FS), int(WIN_S * FS)
    lags = np.arange(-L, L + 1) / FS
    rows = []
    for f in sorted((DATASET_ROOT / "Preprocessed_Dataset").rglob("*_fused.csv")):
        tag = f.stem.replace("_fused", "")
        x = pd.read_csv(f, usecols=["lon_acc_g", "ACC_mag", "GYRO_mag", "label"])
        v = np.abs(np.gradient(np.nan_to_num(x.lon_acc_g.to_numpy(float))))
        lab = x.label.to_numpy()
        starts = np.where((lab == EVENT_CLASS) & (np.roll(lab, 1) != EVENT_CLASS))[0]
        starts = starts[(starts > W + L) & (starts < len(lab) - W - L)]
        rec = {"tag": tag, "n_events": int(len(starts))}
        for name, col in [("ACC", "ACC_mag"), ("GYRO", "GYRO_mag")]:
            b = np.abs(np.gradient(np.nan_to_num(x[col].to_numpy(float))))
            per = []
            for i in starts:
                A = v[i - W:i + W]; A = (A - A.mean()) / (A.std() + 1e-9)
                per.append([np.dot(A, ((b[i - W + lag:i + W + lag] - b[i - W + lag:i + W + lag].mean())
                                       / (b[i - W + lag:i + W + lag].std() + 1e-9))) / len(A) for lag in range(-L, L + 1)])
            per = np.array(per) if per else np.zeros((0, 2 * L + 1))
            if len(per) < 8:
                rec.update({f"{name}_lag_s": np.nan, f"{name}_peak_r": np.nan, f"{name}_peak_z": np.nan, f"{name}_ci_width_s": np.nan})
                continue
            c = per.mean(0); k = int(np.argmax(c)); z = (c[k] - c.mean()) / (c.std() + 1e-9)
            rng = np.random.default_rng(0)
            pk = [lags[np.argmax(per[rng.integers(0, len(per), len(per))].mean(0))] for _ in range(N_BOOT)]
            rec.update({f"{name}_lag_s": float(lags[k]), f"{name}_peak_r": float(c[k]), f"{name}_peak_z": float(z),
                        f"{name}_ci_width_s": float(np.percentile(pk, 97.5) - np.percentile(pk, 2.5))})
        rows.append(rec)
        print(f"  {tag}: n={rec['n_events']}  ACC lag {rec['ACC_lag_s']:+.2f}s z={rec['ACC_peak_z']:.1f}  "
              f"GYRO lag {rec['GYRO_lag_s']:+.2f}s z={rec['GYRO_peak_z']:.1f}", flush=True)
    R = pd.DataFrame(rows); R.to_csv(OUT, index=False)
    ok = R.dropna()
    print(f"\nsessions: {len(R)} | usable (>= 8 events): {len(ok)}")
    for n in ("ACC", "GYRO"):
        conf = ((ok[f"{n}_peak_z"] > 4) & (ok[f"{n}_ci_width_s"] < 0.5)).sum()
        print(f"{n}: peak r median {ok[f'{n}_peak_r'].median():.3f}, z median {ok[f'{n}_peak_z'].median():.1f}, "
              f"CI width median {ok[f'{n}_ci_width_s'].median():.2f} s, sessions with a confident peak (z>4 & CI<0.5 s): {conf}")
    agree = (np.abs(ok.ACC_lag_s - ok.GYRO_lag_s) <= 0.2).sum()
    print(f"ACC and GYRO lags agree within 0.2 s in {agree} of {len(ok)} sessions")
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
