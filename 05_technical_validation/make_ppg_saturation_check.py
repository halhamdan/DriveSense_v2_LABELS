"""
Technical Validation: PPG photodetector saturation check ("Physiological
signal characterisation").

Checks the raw PPG_IR/PPG_RED/PPG_GREEN channels across all released sessions
for ADC saturation -- values pinned at or near the channel's own observed
maximum reading, which would indicate photodetector overexposure/clipping.
This is a distinct failure mode from the non-negativity/hardware-range check
already reported for these channels.

Usage:
  python make_ppg_saturation_check.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import DATASET_ROOT

RAW = DATASET_ROOT / "Raw_Dataset"
OUT_DIR = Path(__file__).parent / "validation_output"
OUT_DIR.mkdir(exist_ok=True)

CHANNELS = ["PPG_IR", "PPG_RED", "PPG_GREEN"]
SATURATION_TOL = 0.001  # relative tolerance for "at the ceiling"
MIN_RUN = 25  # samples; flags a sustained (not isolated) saturation run


def main():
    files_by_chan = {c: sorted(RAW.glob(f"D*/Session_*/D*_S*_EmotiBit_{c}.csv")) for c in CHANNELS}
    for c in CHANNELS:
        print(f"{c}: {len(files_by_chan[c])} files")

    global_max = {}
    for c in CHANNELS:
        vmax = 0.0
        for fp in files_by_chan[c]:
            v = pd.to_numeric(pd.read_csv(fp, usecols=["value"])["value"], errors="coerce").dropna()
            if len(v):
                vmax = max(vmax, v.max())
        global_max[c] = vmax
    print("Observed channel maxima:", global_max)

    rows = []
    for c in CHANNELS:
        ceiling = global_max[c]
        for fp in files_by_chan[c]:
            tag = fp.stem.replace(f"_EmotiBit_{c}", "")
            v = pd.to_numeric(pd.read_csv(fp, usecols=["value"])["value"], errors="coerce")
            at_ceiling = (v >= ceiling * (1 - SATURATION_TOL)).fillna(False).values

            max_run = cur = 0
            for b in at_ceiling:
                cur = cur + 1 if b else 0
                max_run = max(max_run, cur)

            rows.append({
                "channel": c, "tag": tag, "n": len(v), "n_at_ceiling": int(at_ceiling.sum()),
                "max_consecutive_run_at_ceiling": max_run,
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "ppg_saturation_check.csv", index=False)

    print("\n=== Saturation summary (native PPG readings, all sessions) ===")
    for c in CHANNELS:
        sub = out[out["channel"] == c]
        total_n, total_sat = sub["n"].sum(), sub["n_at_ceiling"].sum()
        n_sustained = (sub["max_consecutive_run_at_ceiling"] >= MIN_RUN).sum()
        print(f"{c}: ceiling={global_max[c]:.1f}, n={total_n}, at_ceiling={total_sat} "
              f"({100*total_sat/total_n:.4f}%), sessions_with_sustained_saturation={n_sustained}/{len(sub)}")


if __name__ == "__main__":
    main()
