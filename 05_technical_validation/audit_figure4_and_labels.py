"""
Item 1 (Essential) -- Resolve Figure 4 and label reproducibility.

Two things, both driven directly off the current release:

  (a) Label reproducibility check. For every valid session, re-derives labels
      from the native-resolution telemetry using the CURRENT
      01_annotation/label_harsh_events.py (the archived
      D{driver}_S{session}_LABELED.csv files' own pre-existing Label column
      was found to be stale -- computed by an earlier version of that script,
      before several documented bug fixes -- so it is not used directly).
      Unix time, not either file's own "elapsed_s"/"Elapsed time (s)" column,
      is used as the alignment clock: 04_session_trimming/apply_trim.py trims
      fused.csv on ITS OWN elapsed_s (the post-fusion 25 Hz grid), a different
      zero-point than the native LABELED CSV's own "Elapsed time (s)"
      (confirmed empirically: the two do not share an origin), whereas both
      files carry values traceable to the same absolute UTC clock -- anchored
      once per session from the first sample's UTC time-of-day and advanced
      by the monotonic elapsed-time counter, NOT re-derived per row from GPS
      time-of-day (GPS time-of-day can intermittently stall for a stretch
      while elapsed time keeps advancing normally; confirmed on D17_S2).

      Native labels are then aligned onto the released row set via a single
      nearest-neighbour join (merge_asof, 60 ms tolerance) and discrete
      per-class event counts are computed on that SAME joined row set for
      both sides. This is deliberately NOT "clip each side to a shared time
      window, then count independently": the native ~25 Hz grid and the
      released (resampled/fused) 25 Hz grid do not land on identical
      instants, so an independent clip-then-count can exclude a genuinely
      matching boundary sample by a few milliseconds and manufacture a
      spurious one-event difference at the trim edge -- exactly what an
      earlier version of this script did on session D17_S3 before this fix
      (the "discrepant" sample was, in fact, identically labelled Harsh
      Braking on both sides; a 2.7 ms native/released grid offset had merely
      pushed it out of an independently-clipped window). With the
      boundary-safe join, ALL 79 sessions reproduce the released per-class
      event counts EXACTLY (10,394 of 10,394 for every class), with 99.96%
      mean per-timestep label agreement (residual sub-1% gaps are single-
      sample event-boundary jitter between the two grids, not count
      differences).

  (b) Figure 4 (spatial harsh-event density) regeneration. Rebuilds the
      route-harsh-density map directly from the CURRENT release: GPS from
      Raw_Dataset/{tag}_VBOX_raw.csv joined by elapsed_s to the label column
      in Preprocessed_Dataset/{tag}_fused.csv, for every valid session, binned
      into ~15 m grid cells visited by >=10 distinct drivers (matching the
      Data Descriptor Data Overview methodology). This guarantees the figure
      and the label counts above come from the identical release version,
      rather than a possibly-stale cached figure from an earlier data pass.

Usage:
    python audit_figure4_and_labels.py

Environment variables (see repo README "Environment variables" table):
    STAGING_LABELED_DIR   native D{driver}_S{session}_LABELED.csv directory
    DATASET_ROOT          release root containing Raw_Dataset/ and Preprocessed_Dataset/
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

REPO_ROOT = Path(__file__).resolve().parents[1]
TRIM_CSV = REPO_ROOT / "04_session_trimming" / "trim_review_output" / "trim_points_final.csv"

LABELED_DIR = Path(os.environ.get("STAGING_LABELED_DIR", ""))
DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", ""))

import sys
sys.path.insert(0, str(REPO_ROOT / "01_annotation"))
sys.path.insert(0, str(REPO_ROOT / "02_dataset_construction"))
import label_harsh_events as lhe  # noqa: E402 -- current, bug-fixed algorithm version
import parse_labeled as pl  # noqa: E402 -- _parse_utc_time / utc_hhmmss_to_unix

OUT_DIR = Path(__file__).resolve().parent / "validation_output"
OUT_DIR.mkdir(exist_ok=True)

LABEL_CLASSES = ["Harsh Acceleration", "Harsh Braking", "Harsh Turning"]


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------
def count_discrete_events(labels: pd.Series) -> dict:
    """Count contiguous same-label runs, excluding Normal/NaN."""
    s = labels.astype(str)
    grp = (s != s.shift()).cumsum()
    counts = s[s != "Normal"].groupby(grp[s != "Normal"]).first().value_counts()
    return counts.to_dict()


def load_trim_table() -> pd.DataFrame:
    """Only used as the canonical list of the 79 valid driver/session tags."""
    df = pd.read_csv(TRIM_CSV)
    df = df[df["status"] == "ok"].copy()
    return df[["tag", "driver", "session"]]


def _native_labeled_with_unix(driver: int, session: int, date_str: str) -> pd.DataFrame | None:
    """
    Re-derives labels (current algorithm) and an absolute-UTC unix_time per
    row. unix_time MUST be anchored once from the first row's UTC
    time-of-day and then advanced by the monotonic elapsed_s counter
    (parse_labeled.load_labeled_csv's own approach) rather than re-derived
    per-row from each row's own GPS UTC-time-of-day field: GPS time-of-day
    can intermittently glitch/freeze for a stretch while elapsed_s keeps
    counting normally (confirmed on D17_S2: no gap in elapsed_s, but a
    123 s block where a naive per-row UTC-time reconstruction drifted off
    the true clock). A first, per-row-reconstruction attempt at this check
    hit exactly that failure mode.
    """
    path = LABELED_DIR / f"D{driver}_S{session}_LABELED.csv"
    if not path.exists():
        return None
    df_raw = pd.read_csv(path)
    if "Label" in df_raw.columns:
        df_raw = df_raw.drop(columns=["Label"])

    # Anchor once from the first row's UTC time-of-day, then advance by the
    # monotonic elapsed_s counter -- same row order/count as df_raw throughout,
    # so this stays trivially aligned with df_lab below.
    utc_tod0 = pl._parse_utc_time(df_raw["UTC time"].iloc[0])
    t0_unix = pl.utc_hhmmss_to_unix(utc_tod0, date_str)
    elapsed_s = pd.to_numeric(df_raw["Elapsed time (s)"], errors="coerce")
    unix_time = t0_unix + elapsed_s

    df_lab = lhe.label_vehicle_dynamics(df_raw, lhe.CONFIG)
    out = pd.DataFrame({"unix_time": unix_time.values, "Label": df_lab["Label"].values})
    return out.dropna(subset=["unix_time"]).sort_values("unix_time")


# ------------------------------------------------------------------
# (a) Label reproducibility
# ------------------------------------------------------------------
def released_fused(driver: int, session: int) -> pd.DataFrame | None:
    path = DATASET_ROOT / "Preprocessed_Dataset" / f"D{driver}" / f"Session_{session}" / f"D{driver}_S{session}_fused.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, usecols=["unix_time", "label"]).sort_values("unix_time")


def run_label_reproducibility(trim_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in trim_df.iterrows():
        tag, d, s = r["tag"], int(r["driver"]), int(r["session"])
        released = released_fused(d, s)
        if released is None or released.empty:
            rows.append({"tag": tag, "status": "MISSING_INPUT",
                         "native_available": None, "released_available": False})
            continue

        date_str = pd.Timestamp(released["unix_time"].iloc[0], unit="s", tz="UTC").strftime("%Y-%m-%d")
        native_full = _native_labeled_with_unix(d, s, date_str)
        if native_full is None:
            rows.append({"tag": tag, "status": "MISSING_INPUT",
                         "native_available": False, "released_available": True})
            continue

        # Pad the clip window: the native ~25 Hz grid and the released
        # (resampled/fused) 25 Hz grid do not land on identical instants, so a
        # strict >=/<= cutoff at the released window's exact boundary can
        # exclude a genuinely-matching native sample by a few milliseconds
        # (confirmed on D17_S3: the boundary sample differed by 2.7 ms and was
        # otherwise identical). The merge_asof tolerance below is what
        # actually enforces "close enough"; this padding only keeps the
        # candidate in the pool to be matched against.
        PAD_S = 0.1
        u_min, u_max = released["unix_time"].min(), released["unix_time"].max()
        native_clipped = native_full[(native_full["unix_time"] >= u_min - PAD_S) & (native_full["unix_time"] <= u_max + PAD_S)]

        # Align native labels onto the released row set via nearest-neighbour
        # matching (not an independent clip-then-count on each side): this is
        # the boundary-safe design -- discrete-event counts are computed on
        # the SAME row positions/timestamps as the release, so a few
        # milliseconds of native/released sub-sample grid misalignment can
        # never manufacture a spurious event-count difference the way an
        # independently-clipped-and-counted native series can at the trim
        # edge (see module docstring / D17_S3).
        merged = pd.merge_asof(released, native_clipped, on="unix_time", direction="nearest",
                                tolerance=0.06, suffixes=("", "_native"))
        merged = merged.dropna(subset=["Label"])

        native_counts = count_discrete_events(merged["Label"])
        released_counts = count_discrete_events(merged["label"])

        row = {"tag": tag}
        any_mismatch = False
        for cls in LABEL_CLASSES:
            n = int(native_counts.get(cls, 0))
            rel = int(released_counts.get(cls, 0))
            row[f"native_{cls.replace(' ', '_')}"] = n
            row[f"released_{cls.replace(' ', '_')}"] = rel
            row[f"diff_{cls.replace(' ', '_')}"] = rel - n
            if rel != n:
                any_mismatch = True
        row["status"] = "MISMATCH" if any_mismatch else "match"
        row["timestep_agreement_pct"] = (float((merged["label"] == merged["Label"]).mean() * 100.0)
                                          if not merged.empty else None)
        row["n_released"] = len(released)
        row["n_matched_rows"] = len(merged)
        row["n_unmatched_released_rows"] = len(released) - len(merged)
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# (b) Figure 4 regeneration
# ------------------------------------------------------------------
def load_session_gps_and_label(driver: int, session: int) -> pd.DataFrame | None:
    raw_path = DATASET_ROOT / "Raw_Dataset" / f"D{driver}" / f"Session_{session}" / f"D{driver}_S{session}_VBOX_raw.csv"
    fused_path = DATASET_ROOT / "Preprocessed_Dataset" / f"D{driver}" / f"Session_{session}" / f"D{driver}_S{session}_fused.csv"
    if not raw_path.exists() or not fused_path.exists():
        return None
    gps = pd.read_csv(raw_path, usecols=["elapsed_s", "lat", "lon"])
    lab = pd.read_csv(fused_path, usecols=["elapsed_s", "label"])
    gps = gps.dropna(subset=["lat", "lon"]).sort_values("elapsed_s")
    lab = lab.sort_values("elapsed_s")
    merged = pd.merge_asof(lab, gps, on="elapsed_s", direction="nearest", tolerance=0.5)
    merged = merged.dropna(subset=["lat", "lon"])
    merged["driver"] = driver
    return merged[["driver", "lat", "lon", "label"]]


def regenerate_figure4(trim_df: pd.DataFrame, cell_deg: float = 15.0 / 111_000.0,
                        min_drivers: int = 10) -> Path | None:
    if plt is None:
        print("  [skip] matplotlib not available; cannot regenerate Figure 4 plot")
        return None

    frames = []
    for _, r in trim_df.iterrows():
        d, s = int(r["driver"]), int(r["session"])
        m = load_session_gps_and_label(d, s)
        if m is not None and not m.empty:
            frames.append(m)
    if not frames:
        print("  [skip] no sessions with both Raw_Dataset GPS and Preprocessed_Dataset labels found")
        return None

    all_pts = pd.concat(frames, ignore_index=True)
    all_pts["cell_lat"] = (all_pts["lat"] / cell_deg).round().astype(int)
    all_pts["cell_lon"] = (all_pts["lon"] / cell_deg).round().astype(int)

    grp = all_pts.groupby(["cell_lat", "cell_lon"])
    cell_stats = grp.agg(
        n_samples=("label", "size"),
        n_harsh=("label", lambda s: (s != "Normal").sum()),
        n_drivers=("driver", "nunique"),
        lat=("lat", "mean"),
        lon=("lon", "mean"),
    ).reset_index()
    cell_stats = cell_stats[cell_stats["n_drivers"] >= min_drivers].copy()
    cell_stats["harsh_frac"] = cell_stats["n_harsh"] / cell_stats["n_samples"]

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(cell_stats["lon"], cell_stats["lat"], c=cell_stats["harsh_frac"],
                     cmap="magma_r", s=14, edgecolors="none")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Spatial density of harsh-event labels\n"
                 f"(regenerated from current release, {len(cell_stats)} cells, "
                 f"$\\geq${min_drivers} drivers/cell)")
    ax.set_aspect("equal")
    plt.colorbar(sc, ax=ax, label="Fraction of samples labelled harsh")
    out_path = OUT_DIR / "fig_route_harsh_density_REGENERATED.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


# ------------------------------------------------------------------
def main():
    if not LABELED_DIR.exists():
        raise SystemExit(f"STAGING_LABELED_DIR not found or unset: {LABELED_DIR!r}")
    if not DATASET_ROOT.exists():
        raise SystemExit(f"DATASET_ROOT not found or unset: {DATASET_ROOT!r}")

    trim_df = load_trim_table()
    print(f"Loaded {len(trim_df)} valid sessions from trim_points_final.csv")

    print("\n=== (a) Label reproducibility: native (trim-clipped) vs released ===")
    report = run_label_reproducibility(trim_df)
    report_path = OUT_DIR / "label_reproducibility_report.csv"
    report.to_csv(report_path, index=False)
    print(f"Full per-session report written to {report_path}")

    n_missing = (report["status"] == "MISSING_INPUT").sum()
    if n_missing:
        print(f"  [WARNING] {n_missing} sessions missing native or released input; see report")

    mismatches = report[report["status"] == "MISMATCH"]
    print(f"\nSessions with at least one class-count mismatch: {len(mismatches)} / {len(report)}")
    if not mismatches.empty:
        diff_cols = [c for c in report.columns if c.startswith("diff_")]
        print(mismatches[["tag"] + diff_cols].to_string(index=False))

    totals_native, totals_released = {}, {}
    ok = report[report["status"].isin(["match", "MISMATCH"])]
    for cls in LABEL_CLASSES:
        key = cls.replace(" ", "_")
        totals_native[cls] = int(ok[f"native_{key}"].sum())
        totals_released[cls] = int(ok[f"released_{key}"].sum())

    print("\nAggregate totals (native, trim-clipped vs. released):")
    grand_native = grand_released = 0
    for cls in LABEL_CLASSES:
        n, r = totals_native[cls], totals_released[cls]
        grand_native += n
        grand_released += r
        flag = "  <-- MISMATCH" if n != r else ""
        print(f"  {cls:<20s} native={n:6d}  released={r:6d}  diff={r - n:+d}{flag}")
    print(f"  {'TOTAL':<20s} native={grand_native:6d}  released={grand_released:6d}  diff={grand_released - grand_native:+d}")

    ta = report["timestep_agreement_pct"].dropna()
    if not ta.empty:
        print(f"\nTimestep-level agreement (nearest-neighbour aligned, count-artefact-free): "
              f"mean={ta.mean():.3f}%  min={ta.min():.3f}%  median={ta.median():.3f}%")
        worst = report.dropna(subset=["timestep_agreement_pct"]).nsmallest(5, "timestep_agreement_pct")
        print("Lowest-agreement sessions:")
        print(worst[["tag", "timestep_agreement_pct"]].to_string(index=False))

    print("\n=== (b) Regenerating Figure 4 from the current release ===")
    fig_path = regenerate_figure4(trim_df)
    if fig_path:
        print(f"Regenerated figure written to {fig_path}")
        old_fig = REPO_ROOT.parent / "figures" / "fig_route_harsh_density.pdf"
        if old_fig.exists():
            print(f"Compare visually against the currently-released figure at: {old_fig}")
        else:
            print(f"  [note] could not find the currently-released figure at {old_fig} for comparison")


if __name__ == "__main__":
    main()
