"""
Item 2 (Essential) -- Check acceleration artefacts and their effect on events.

Automatically flags implausible-magnitude lateral/longitudinal acceleration
samples (outside the documented plausible range, +/-1.2g -- Data Descriptor
Technical Validation, Table "telemetryanomalies") in every valid released
session, characterises each flagged run (duration, sign-alternation pattern),
and reports how many discrete harsh-event runs contain at least one flagged
sample. A handful of the most suspicious cases (by magnitude and by the
"rapidly alternating multi-sign" pattern flagged as an open question in the
companion multimodal-validation analysis) are extracted with surrounding
context for manual inspection, and classified as a plausible genuine
physical impact vs. an unresolved sensor artefact.

Usage:
    python audit_acceleration_artifacts.py

Environment variables:
    DATASET_ROOT   release root containing Raw_Dataset/ and Preprocessed_Dataset/
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TRIM_CSV = REPO_ROOT / "04_session_trimming" / "trim_review_output" / "trim_points_final.csv"
DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", ""))

OUT_DIR = Path(__file__).resolve().parent / "validation_output"
OUT_DIR.mkdir(exist_ok=True)

PLAUSIBLE_G = 1.2
BRIEF_RUN_MAX_SAMPLES = 3      # <=0.12 s at 25 Hz: classic isolated-glitch duration
SIGN_ALT_WINDOW = 5            # samples either side, for sign-alternation check
FS = 25.0


def load_sessions() -> list[tuple[str, int, int]]:
    df = pd.read_csv(TRIM_CSV)
    df = df[df["status"] == "ok"]
    return list(df[["tag", "driver", "session"]].itertuples(index=False, name=None))


def count_discrete_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return (start_idx, end_idx_exclusive) for every contiguous True run."""
    padded = np.concatenate(([False], mask, [False]))
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts, ends))


def discrete_label_runs(labels: np.ndarray) -> list[tuple[int, int, str]]:
    """
    Contiguous same-label runs (a Harsh Braking run directly adjacent to a
    Harsh Turning run, with no Normal in between, is TWO events, matching
    01_annotation/label_harsh_events.py's own count_distinct_events -- unlike
    grouping on label != "Normal" alone, which would merge them into one).
    """
    change = np.concatenate(([True], labels[1:] != labels[:-1]))
    runs = []
    boundaries = np.where(change)[0].tolist() + [len(labels)]
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        if labels[s] != "Normal":
            runs.append((s, e, labels[s]))
    return runs


def sign_alternation_count(values: np.ndarray) -> int:
    """Number of sign changes in a short value sequence (0 excluded)."""
    s = np.sign(values)
    s = s[s != 0]
    if len(s) < 2:
        return 0
    return int(np.sum(s[1:] != s[:-1]))


def analyse_session(tag: str, driver: int, session: int) -> dict | None:
    path = DATASET_ROOT / "Preprocessed_Dataset" / f"D{driver}" / f"Session_{session}" / f"D{driver}_S{session}_fused.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, usecols=["elapsed_s", "lat_acc_g", "lon_acc_g", "label", "speed_kph"])
    lat = df["lat_acc_g"].to_numpy()
    lon = df["lon_acc_g"].to_numpy()
    implausible = (np.abs(lat) > PLAUSIBLE_G) | (np.abs(lon) > PLAUSIBLE_G)

    n_flagged = int(implausible.sum())
    if n_flagged == 0:
        labels0 = df["label"].to_numpy()
        n_events_total0 = len(discrete_label_runs(labels0))
        return {"tag": tag, "n_flagged": 0, "n_runs": 0, "n_brief_runs": 0,
                "n_events_total": n_events_total0, "n_events_affected": 0,
                "n_affected_accel": 0, "n_affected_brake": 0, "n_affected_turn": 0,
                "max_abs_lat": float(np.nanmax(np.abs(lat))), "max_abs_lon": float(np.nanmax(np.abs(lon))),
                "cases": []}

    runs = count_discrete_runs(implausible)
    n_brief_runs = sum(1 for s, e in runs if (e - s) <= BRIEF_RUN_MAX_SAMPLES)

    # Discrete per-class harsh-event runs (matching label_harsh_events.py's own
    # count_distinct_events: a same-timestep class change with no Normal in
    # between is TWO events, not one) that contain >=1 flagged sample.
    labels = df["label"].to_numpy()
    harsh_runs = discrete_label_runs(labels)
    n_events_total = len(harsh_runs)
    events_affected_by_class = {}
    n_events_affected = 0
    for s, e, cls in harsh_runs:
        if implausible[s:e].any():
            n_events_affected += 1
            events_affected_by_class[cls] = events_affected_by_class.get(cls, 0) + 1

    cases = []
    for s, e in runs:
        run_len = e - s
        lo, hi = max(0, s - SIGN_ALT_WINDOW), min(len(lat), e + SIGN_ALT_WINDOW)
        lat_ctx, lon_ctx = lat[lo:hi], lon[lo:hi]
        peak_g = float(np.max(np.maximum(np.abs(lat[s:e]), np.abs(lon[s:e]))))
        alt = max(sign_alternation_count(lat_ctx), sign_alternation_count(lon_ctx))
        label_here = labels[s:e]
        cases.append({
            "tag": tag, "start_idx": int(s), "end_idx": int(e), "run_len": int(run_len),
            "elapsed_s": float(df["elapsed_s"].iloc[s]), "peak_g": peak_g,
            "sign_alternations_nearby": int(alt),
            "labels_in_run": sorted(set(label_here.tolist())),
            "speed_kph_at_start": float(df["speed_kph"].iloc[s]) if not pd.isna(df["speed_kph"].iloc[s]) else None,
        })

    return {
        "tag": tag, "n_flagged": n_flagged, "n_runs": len(runs), "n_brief_runs": n_brief_runs,
        "n_events_total": n_events_total, "n_events_affected": n_events_affected,
        "n_affected_accel": events_affected_by_class.get("Harsh Acceleration", 0),
        "n_affected_brake": events_affected_by_class.get("Harsh Braking", 0),
        "n_affected_turn": events_affected_by_class.get("Harsh Turning", 0),
        "max_abs_lat": float(np.nanmax(np.abs(lat))), "max_abs_lon": float(np.nanmax(np.abs(lon))),
        "cases": cases,
    }


def main():
    if not DATASET_ROOT.exists():
        raise SystemExit(f"DATASET_ROOT not found or unset: {DATASET_ROOT!r}")

    sessions = load_sessions()
    print(f"Scanning {len(sessions)} valid sessions for |lat_acc_g| or |lon_acc_g| > {PLAUSIBLE_G} g ...")

    session_rows, all_cases = [], []
    for tag, d, s in sessions:
        res = analyse_session(tag, int(d), int(s))
        if res is None:
            continue
        session_rows.append({k: v for k, v in res.items() if k != "cases"})
        all_cases.extend(res["cases"])
    TOTAL_RELEASED_SAMPLES = 3_504_071  # verified released row total (Data Descriptor, Table "classdist")

    session_df = pd.DataFrame(session_rows)
    session_df.to_csv(OUT_DIR / "acceleration_artifact_sessions.csv", index=False)
    cases_df = pd.DataFrame(all_cases)
    cases_df.to_csv(OUT_DIR / "acceleration_artifact_cases.csv", index=False)

    total_flagged = int(session_df["n_flagged"].sum())
    total_runs = int(session_df.get("n_runs", pd.Series(dtype=int)).sum())
    total_brief_runs = int(session_df["n_brief_runs"].sum())
    total_events = int(session_df.get("n_events_total", pd.Series(dtype=int)).sum())
    total_events_affected = int(session_df.get("n_events_affected", pd.Series(dtype=int)).sum())
    n_sessions_with_flags = int((session_df["n_flagged"] > 0).sum())

    print(f"\nSessions with >=1 implausible-magnitude sample: {n_sessions_with_flags} / {len(session_df)}")
    print(f"Total flagged samples (|lat| or |lon| > {PLAUSIBLE_G} g): {total_flagged} "
          f"({100*total_flagged/TOTAL_RELEASED_SAMPLES:.4f}% of all released samples)")
    print(f"Total discrete implausible-value runs: {total_runs} (brief, <= {BRIEF_RUN_MAX_SAMPLES} samples: {total_brief_runs})")
    print(f"\nDiscrete harsh-event runs total (all 79 sessions): {total_events}")
    print(f"Discrete harsh-event runs containing >=1 implausible-magnitude sample: "
          f"{total_events_affected} ({100*total_events_affected/total_events:.3f}% of all events)")
    print(f"  by class -- Harsh Acceleration: {int(session_df['n_affected_accel'].sum())}, "
          f"Harsh Braking: {int(session_df['n_affected_brake'].sum())}, "
          f"Harsh Turning: {int(session_df['n_affected_turn'].sum())}")

    if not cases_df.empty:
        print("\n=== Top 8 cases by peak magnitude (for manual inspection) ===")
        top_mag = cases_df.nlargest(8, "peak_g")
        print(top_mag[["tag", "elapsed_s", "run_len", "peak_g", "sign_alternations_nearby",
                        "labels_in_run", "speed_kph_at_start"]].to_string(index=False))

        print("\n=== Top 8 cases by nearby sign-alternation count (candidate sensor-artefact pattern) ===")
        top_alt = cases_df.nlargest(8, "sign_alternations_nearby")
        print(top_alt[["tag", "elapsed_s", "run_len", "peak_g", "sign_alternations_nearby",
                        "labels_in_run", "speed_kph_at_start"]].to_string(index=False))

    print(f"\nFull per-session summary: {OUT_DIR / 'acceleration_artifact_sessions.csv'}")
    print(f"Full per-case detail:     {OUT_DIR / 'acceleration_artifact_cases.csv'}")


if __name__ == "__main__":
    main()
