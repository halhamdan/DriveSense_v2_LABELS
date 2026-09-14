"""
Scores the released hands_on_wheel_proxy columns against blind human ratings of the
100-frame pack (validation_output/hands_on_wheel_rating/ratings_*.csv; 0/1/2 hands on the
wheel, 'u' = cannot judge) and reports inter-rater agreement when several raters exist.
Usage: python score_hands_on_wheel_ratings.py
"""
from pathlib import Path
import numpy as np, pandas as pd

D = Path(__file__).resolve().parent / "validation_output" / "hands_on_wheel_rating"
key = pd.read_csv(D / "rating_key.csv").set_index("frame_id")
raters = {f.stem.replace("ratings_", ""): pd.read_csv(f).set_index("frame_id")["hands_on_wheel_rater"] for f in sorted(D.glob("ratings_*.csv"))}


def kappa(a, b, w=None):
    cats = [0, 1, 2]; n = len(a); O = np.zeros((3, 3))
    for x, y in zip(a, b): O[cats.index(x), cats.index(y)] += 1
    E = np.outer(O.sum(1), O.sum(0)) / n
    W = np.array([[abs(i - j) ** 2 for j in range(3)] for i in range(3)]) if w == "quadratic" else 1 - np.eye(3)
    return 1 - (W * O).sum() / (W * E).sum()


for name, rr in raters.items():
    ok = rr.astype(str).str.strip().str.lower() != "u"; r = rr[ok].astype(int); k = key.loc[r.index]
    print(f"\n=== rater {name}: {len(r)} rated frames ({(~ok).sum()} unjudgeable) | hands on wheel by rater: {r.value_counts().sort_index().to_dict()}")
    for col, label in (("released_proxy", "released v3 (fixed measured region)"), ("released_proxy_v1", "v1 (original region)")):
        p = (k[col] * 2).round().astype(int)
        agree = (p == r).mean() * 100; within = ((p - r).abs() <= 1).mean() * 100
        print(f"  {label}: exact agreement {agree:.0f}% ({(p == r).sum()}/{len(r)}), within one hand {within:.0f}%; under {(p < r).sum()}, over {(p > r).sum()}; "
              f"kappa {kappa(r.tolist(), p.tolist()):.2f} (quadratic {kappa(r.tolist(), p.tolist(), 'quadratic'):.2f})")
        ct = pd.crosstab(r.rename("rater"), p.rename("proxy")); print(ct.to_string())
        for cond in ("moving", "stopped"):
            m = k.condition == cond; print(f"    {cond}: exact {(p[m] == r[m]).mean()*100:.0f}% (n={m.sum()})")
names = list(raters)
if len(names) > 1:
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = raters[names[i]], raters[names[j]]; ok = (a.astype(str) != "u") & (b.astype(str) != "u"); a, b = a[ok].astype(int), b[ok].astype(int)
            print(f"\ninter-rater {names[i]} vs {names[j]}: exact {(a == b).mean()*100:.0f}% (n={ok.sum()}), kappa {kappa(a.tolist(), b.tolist()):.2f}, quadratic {kappa(a.tolist(), b.tolist(), 'quadratic'):.2f}")
