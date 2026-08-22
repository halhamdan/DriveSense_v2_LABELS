"""
Finalizes trim_points_for_review.csv: fills manual_start_s/manual_end_s from
the auto-detected values for all 79 sessions, except the 3 sessions where
visual review (full contact sheets + a wide 300s-window check) showed the
auto-detected END trim was a false positive -- the video/telemetry simply
ends mid-drive with no arrival ever visible, so those get manual_end_s=0
instead of trusting the heuristic's spurious "sustained near-base" trigger.

Usage:
  python finalize_trim_points.py
"""
from pathlib import Path

import pandas as pd

OUT_DIR = Path(__file__).parent / "trim_review_output"

# Sessions where the auto end-trim was confirmed a false positive via the
# wide end-review (D4_S1, D9_S3, D13_S4): driving continues uninterrupted
# all the way to the literal last frame of both video and telemetry.
END_TRIM_OVERRIDE_ZERO = {(4, 1), (9, 3), (13, 4)}


def main():
    df = pd.read_csv(OUT_DIR / "trim_points_for_review.csv")

    df["manual_start_s"] = df["auto_start_s"]
    df["manual_end_s"] = df["auto_end_s"]

    for driver, session in END_TRIM_OVERRIDE_ZERO:
        mask = (df["driver"] == driver) & (df["session"] == session)
        df.loc[mask, "manual_end_s"] = 0.0

    out_path = OUT_DIR / "trim_points_final.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(df)} sessions)")
    print(df[["tag", "orig_duration_s", "manual_start_s", "manual_end_s"]].to_string(index=False))


if __name__ == "__main__":
    main()
