"""Caches the columns needed by audit_label_robustness_v2.py from the 79 released
fused files (v2 label tree) into one npz, so rule variants can be evaluated in
seconds instead of re-reading 1.4 GB of CSV per variant."""
import os
from pathlib import Path

import numpy as np
import pandas as pd

V2 = Path(os.environ.get("PREPROCESSED_ROOT", "") or r"C:\Users\halha\OneDrive - Durham University\Documents\Published_Dataset_Final_v2_LABELS")
OUT = Path(__file__).resolve().parent / "validation_output" / "release_columns_cache.npz"
COLS = ["elapsed_s", "gps_speed_kmh", "speed_kph", "throttle_pct", "lat_acc_g", "lon_acc_g", "heading_deg", "label", "label_v1_peak"]

files = sorted((V2 / "Preprocessed_Dataset").rglob("*_fused.csv"))
parts = []; tags = []
for i, f in enumerate(files):
    d = pd.read_csv(f, usecols=COLS); d["session"] = i; tags.append(f.stem.replace("_fused", "")); parts.append(d)
    print(f"  {tags[-1]} {len(d)}", flush=True)
D = pd.concat(parts, ignore_index=True)
np.savez_compressed(OUT, session=D.session.to_numpy(np.int16), elapsed_s=D.elapsed_s.to_numpy(np.float64),
                    gps_speed_kmh=D.gps_speed_kmh.to_numpy(np.float32), speed_kph=D.speed_kph.to_numpy(np.float32),
                    throttle_pct=D.throttle_pct.to_numpy(np.float32), lat_acc_g=D.lat_acc_g.to_numpy(np.float64),
                    lon_acc_g=D.lon_acc_g.to_numpy(np.float64), heading_deg=D.heading_deg.to_numpy(np.float64),
                    label_draft=D.label.to_numpy(dtype=object), label_v1_peak=D.label_v1_peak.to_numpy(dtype=object), tags=np.array(tags, dtype=object))
print(f"{len(D):,} rows, {len(tags)} sessions -> {OUT}")
