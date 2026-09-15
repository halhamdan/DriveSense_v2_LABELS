"""Do the released fused.csv files start and end while the vehicle is moving?
(The trim points were chosen from review frames that, because of the video
lead-in, showed the scene ~11 s earlier than the telemetry instant they were
labelled with; this checks whether that left stationary head/tail segments.)
Output: validation_output/released_trim_endpoint_speeds.csv"""
from pathlib import Path
import numpy as np, pandas as pd

R = Path(r"C:\Users\halha\OneDrive - Durham University\Documents\DriveSense_Packages\2_PREVIOUS__labels_v1__release_v1__paper_v5\data\Published_Dataset_Final\Preprocessed_Dataset")
rows = []
for f in sorted(R.rglob("*_fused.csv")):
    d = pd.read_csv(f, usecols=["elapsed_s", "gps_speed_kmh"]); v = d.gps_speed_kmh.to_numpy(); e = d.elapsed_s.to_numpy()
    mv = v > 5
    rows.append({"tag": f.stem.replace("_fused", ""), "start_v": v[:25].mean(), "end_v": v[-25:].mean(),
                 "stationary_head_s": float(e[np.argmax(mv)]) if mv.any() else np.nan,
                 "stationary_tail_s": float(e[-1] - e[len(v) - 1 - np.argmax(mv[::-1])]) if mv.any() else np.nan})
T = pd.DataFrame(rows); T.to_csv(Path(__file__).resolve().parent / "validation_output" / "released_trim_endpoint_speeds.csv", index=False)
print("sessions starting < 5 km/h:", int((T.start_v < 5).sum()), "| ending < 5 km/h:", int((T.end_v < 5).sum()))
for c in ["stationary_head_s", "stationary_tail_s"]:
    print(f"{c}: median {T[c].median():.1f}, p90 {T[c].quantile(.9):.1f}, max {T[c].max():.1f} ({T.loc[T[c].idxmax(), 'tag']}); sessions > 11 s: {int((T[c] > 11).sum())}")
print(T.sort_values("stationary_head_s", ascending=False).head(5)[["tag", "start_v", "stationary_head_s"]].round(1).to_string(index=False))
print(T.sort_values("stationary_tail_s", ascending=False).head(5)[["tag", "end_v", "stationary_tail_s"]].round(1).to_string(index=False))
