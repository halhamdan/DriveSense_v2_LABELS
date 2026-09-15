"""
Precursor-analysis sensitivity check: does the EmotiBit sync/anchor
correction (regen_v2) change any of the companion paper's 5 FDR-significant
matched-pairs findings, relative to the CURRENT release (Published_Dataset_Final)?

Deliberately isolates ONE variable: both runs below use the same fused.csv
schema, same trim windows, same VBOX/label columns, same emotions/pose
files -- the ONLY difference between "before" and "after" is which fused.csv
tree supplies the biometric columns (current release vs. sync-corrected
regeneration). This is NOT a re-run against the original precursor_results/
baseline, which was computed against a separate, older snapshot
(Published_Dataset, not Published_Dataset_Final -- confirmed materially
different, e.g. D1_S1 median HR 133 vs 139 bpm) -- that is a real, separate
staleness issue, orthogonal to this specific correction, and out of scope
here.

Reuses precursor_analysis_matched.py's actual methodology and constants
(imported, not reimplemented) via monkey-patched session-directory
resolution, so both runs go through the exact same matching/testing code.

Usage:
    python precursor_sensitivity_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

MODELLING_GPU = Path(r"C:\Users\halha\OneDrive - Durham University\Documents\Code\Modelling_GPU")
sys.path.insert(0, str(MODELLING_GPU))

import precursor_analysis as pa           # noqa: E402
import precursor_analysis_matched as pam  # noqa: E402

import os

RELEASE_ROOT = Path(r"C:\Users\halha\OneDrive - Durham University\Documents\DriveSense_Packages\3_OLD_versions\data\Published_Dataset_Final_v1_ORIGINAL_20260909\Preprocessed_Dataset")
# "After" tree: REGEN_ROOT env var (e.g. regenerated_v3 for the anchor-only
# model); defaults to the 2026-09-08 regenerated_v2 tree.
V2_ROOT = Path(os.environ.get("REGEN_ROOT", "") or r"C:\Users\halha\Desktop\Nature Data Paper\regenerated_v2\Preprocessed_Dataset")
AFTER_LABEL = os.environ.get("AFTER_LABEL", "after_v2_corrected")
# "Before" tree: BEFORE_ROOT env var; defaults to the current release. Once a
# correction has been deployed, point this at the preserved pre-fix backup
# (flat layout, {tag}_fused.csv directly under the folder -- handled below).
BEFORE_ROOT = Path(os.environ.get("BEFORE_ROOT", "") or str(RELEASE_ROOT))
BEFORE_LABEL = os.environ.get("BEFORE_LABEL", "before_release")

OUT_DIR = Path(__file__).resolve().parent / os.environ.get("SENS_OUT_DIR", "precursor_sensitivity_output")
OUT_DIR.mkdir(exist_ok=True)


def make_session_data_fn(fused_root: Path):
    """
    Returns a drop-in replacement for precursor_analysis_matched._session_data
    that reads {tag}_fused.csv from fused_root (either the release or the v2
    regeneration) but Front_emotions.csv/Side_pose.csv ALWAYS from the
    release (unaffected by the EmotiBit correction, and not present in the
    v2 tree, which only contains regenerated fused.csv files).
    """
    import numpy as np

    def _session_data(driver_n: int, session_n: int):
        tag = f"D{driver_n}_S{session_n}"
        fused_dir = fused_root / f"D{driver_n}" / f"Session_{session_n}"
        release_dir = RELEASE_ROOT / f"D{driver_n}" / f"Session_{session_n}"
        fpath = fused_dir / f"{tag}_fused.csv"
        if not fpath.exists():
            fpath = fused_root / f"{tag}_fused.csv"   # flat *_BACKUP layout
        if not fpath.exists():
            return None
        df = pd.read_csv(fpath)
        if "label" not in df.columns or len(df) < 25 * 40:
            return None
        df = pam._filter_plausible(df)
        df = pa._add_stress_column(df)
        df = pam._merge_nearest(df, release_dir / f"{tag}_Front_emotions.csv", "timestamp_s",
                                 pam.FACIAL_SIGNALS + pam.FACE_POSE_COLS)
        if (driver_n, session_n) in pam.INVALID_CABIN_SESSIONS:
            for c in pam.CABIN_SIGNALS + pam.WRIST_COLS:
                df[c] = np.nan
        else:
            df = pam._merge_nearest(df, release_dir / f"{tag}_Side_pose.csv", "timestamp_s",
                                     pam.CABIN_SIGNALS + pam.WRIST_COLS)
        df["wrist_asymmetry_y"] = df["wrist_l_y"] - df["wrist_r_y"]
        return df

    return _session_data


def run_full_analysis(fused_root: Path, label: str) -> pd.DataFrame:
    pam._session_data = make_session_data_fn(fused_root)

    all_pairs = []
    for d in range(1, 21):
        for s in range(1, 5):
            all_pairs.extend(pam.match_session(d, s))

    pairs_df = pd.DataFrame(all_pairs)
    print(f"[{label}] matched {len(pairs_df)} pairs total")
    if pairs_df.empty:
        return pd.DataFrame()

    results = pam.run_wilcoxon_grid(pairs_df, signals=pam.CORE_SIGNALS)
    results.to_csv(OUT_DIR / f"results_{label}.csv", index=False)
    n_sig = int(results["significant_fdr"].sum()) if not results.empty else 0
    print(f"[{label}] {n_sig} / {len(results)} tests significant after FDR")
    return results


def main():
    results_before = run_full_analysis(BEFORE_ROOT, BEFORE_LABEL)
    results_after = run_full_analysis(V2_ROOT, AFTER_LABEL)

    key_cols = ["window_type", "event_type", "lead_time_s", "signal"]
    before_sig = results_before[results_before["significant_fdr"]][key_cols + ["wilcoxon_p", "p_fdr_bh", "rank_biserial_effect", "n_pairs"]]
    after_sig = results_after[results_after["significant_fdr"]][key_cols + ["wilcoxon_p", "p_fdr_bh", "rank_biserial_effect", "n_pairs"]]

    print("\n=== Significant (FDR) tests -- BEFORE (current release) ===")
    print(before_sig.to_string(index=False))
    print("\n=== Significant (FDR) tests -- AFTER (v2, sync-corrected) ===")
    print(after_sig.to_string(index=False))

    merged = pd.merge(results_before[key_cols + ["wilcoxon_p", "p_fdr_bh", "significant_fdr", "rank_biserial_effect", "n_pairs"]],
                       results_after[key_cols + ["wilcoxon_p", "p_fdr_bh", "significant_fdr", "rank_biserial_effect", "n_pairs"]],
                       on=key_cols, suffixes=("_before", "_after"), how="outer")
    merged["sig_changed"] = merged["significant_fdr_before"].fillna(False) != merged["significant_fdr_after"].fillna(False)
    merged.to_csv(OUT_DIR / "before_after_comparison.csv", index=False)

    print(f"\n=== Tests where FDR significance FLIPPED (either direction): {merged['sig_changed'].sum()} ===")
    if merged["sig_changed"].any():
        print(merged[merged["sig_changed"]].to_string(index=False))
    else:
        print("None -- FDR-significance pattern is identical before/after the EmotiBit correction.")

    print(f"\nFull output: {OUT_DIR}")


if __name__ == "__main__":
    main()
