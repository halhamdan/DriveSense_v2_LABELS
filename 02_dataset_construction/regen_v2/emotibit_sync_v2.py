"""
Corrected EmotiBit sync loader (v2). Two changes over the original
emotibit_sync.py, both reusing its existing, unmodified parsing/merging
functions rather than duplicating them:

  1. Absolute-time anchor: uses a robust, full-session linear calibration
     fit from timesyncs.csv (timesync_calibration.py) when enough sync pings
     are available, injected via a synthetic 2-point sync_map that the
     original _load_raw_csv()/_emotibit_ts_to_unix() already know how to
     consume -- so the actual per-row timestamp arithmetic is untouched,
     original code. Falls back to the original filename-based anchor
     (unchanged behaviour) for the few sessions without enough sync pings.

  2. Two additional channels captured: SA (SkinConductanceResponseAmplitude,
     uS) and SR (SkinConductanceResponseRiseTime, s) -- present in the raw
     device data on every checked session but never recognised by the
     original TAG_TO_SIGNAL/SAMPLE_RATES tables, so silently dropped before
     now. Both are event-triggered (recorded once per detected skin-
     conductance response), like the existing HR/IBI channels.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "02_dataset_construction"))
import emotibit_sync as es  # noqa: E402 -- original module, reused not duplicated

sys.path.insert(0, str(Path(__file__).resolve().parent))
from timesync_calibration import fit_calibration  # noqa: E402

# Extend the original module's tag tables in place -- _load_raw_csv() reads
# these module-level dicts directly, so this is the only change needed to
# make it recognise SA/SR without touching its own code.
es.SAMPLE_RATES.setdefault("SA", 1.0)
es.SAMPLE_RATES.setdefault("SR", 1.0)
es.TAG_TO_SIGNAL.setdefault("SA", "SCR_AMPLITUDE")
es.TAG_TO_SIGNAL.setdefault("SR", "SCR_RISE_TIME")
es.PER_CHANNEL_TAG_COL.setdefault("SA", "SA")
es.PER_CHANNEL_TAG_COL.setdefault("SR", "SR")

NEW_CHANNELS = ("SCR_AMPLITUDE", "SCR_RISE_TIME")


def find_emotibit_dir(session_dir: Path) -> Path | None:
    d = session_dir / "EmotiBit"
    if d.exists():
        return d
    d = session_dir / "MotiBit"
    if d.exists():
        return d
    return None


def load_emotibit_v2(session_dir: str | Path, sync_model: str = "anchor_only") -> tuple[dict[str, pd.DataFrame], dict]:
    """sync_model: "anchor_only" (release default) or "linear" -- see timesync_calibration.py."""
    session_dir = Path(session_dir)
    emb_dir = find_emotibit_dir(session_dir)
    if emb_dir is None:
        raise FileNotFoundError(f"No EmotiBit/MotiBit folder under {session_dir}")

    # Some sessions have MORE THAN ONE main recording file -- a brief aborted
    # connection attempt (a few seconds to a few dozen lines) followed, a
    # short while later, by the real full-length recording (confirmed on
    # D1_S1, D3_S2, D4_S1, D16_S3: in every case the smaller file is the
    # aborted attempt and the larger one is the genuine session). Picking by
    # file size handles that correctly. The bug this replaces was pairing
    # with *any* "*_timesyncs.csv" match via glob() (filesystem/alphabetical
    # order, i.e. effectively "whichever recording started first") instead
    # of the specific file belonging to the CHOSEN (largest) main CSV -- for
    # 3 of those 4 sessions this silently calibrated against the aborted
    # attempt's few seconds of sync pings while resampling the real
    # recording's data, an anchor/data mismatch. Pairing is now done
    # explicitly by matching filename stem.
    # Exclude every recognised per-channel/status suffix, not just the ones in
    # TAG_TO_SIGNAL -- SKIP_TAGS (EM, RB, BV, B%, DC, DO, D+, D-, plus
    # AK/RD/TL) are device-status exports, not main recording files, and were
    # previously leaking into this candidate list (harmless to selection
    # itself, since they're always far smaller than a real main CSV, but
    # polluted the diagnostic "other candidates" listing).
    excluded_suffixes = set(es.TAG_TO_SIGNAL) | set(es.SKIP_TAGS) | {"timeSyncMap", "timesyncs"}
    main_csvs = sorted(
        [f for f in emb_dir.glob("*.csv")
         if not any(f.name.endswith(f"_{tag}.csv") for tag in excluded_suffixes)
         and "_info" not in f.name],
        key=lambda f: f.stat().st_size, reverse=True,
    )
    if not main_csvs:
        raise FileNotFoundError(f"No main EmotiBit CSV found in {emb_dir}")
    main_csv = main_csvs[0]

    meta: dict = {"session_dir": str(session_dir), "main_csv": main_csv.name}
    meta["n_main_csv_candidates"] = len(main_csvs)
    if len(main_csvs) > 1:
        meta["other_main_csv_files"] = [f.name for f in main_csvs[1:]]

    # Fallback anchor (original behaviour, needed even with a sync_map since
    # _load_raw_csv's signature requires it, though it's unused when sync_map
    # is not None).
    t_start_unix_fallback = es._filename_to_unix(main_csv)
    if t_start_unix_fallback is None:
        raise ValueError(f"Cannot parse datetime from filename: {main_csv.name}")

    # Pair explicitly with the timesyncs.csv belonging to THIS main_csv, not
    # just any file the glob happens to return first.
    paired_timesyncs = emb_dir / f"{main_csv.stem}_timesyncs.csv"
    timesyncs_files = [paired_timesyncs] if paired_timesyncs.exists() else []
    calibration = None
    if timesyncs_files:
        calibration = fit_calibration(timesyncs_files[0], model=sync_model)

    if calibration is not None:
        sync_map = calibration.as_two_point_sync_map()
        meta["sync_method"] = ("timesyncs_anchor_only_median" if calibration.model == "anchor_only"
                               else "timesyncs_robust_fit")
        meta["ppm_drift"] = calibration.ppm_drift
        meta["linear_slope_ppm"] = calibration.linear_slope_ppm
        meta["median_offset_s"] = calibration.median_offset_s
        meta["max_step_s"] = calibration.max_step_s
        meta["step_at_s"] = calibration.step_at_s
        meta["n_pings_used"] = calibration.n_pings_used
        meta["calibration_span_s"] = calibration.span_s
        meta["max_fit_residual_ms"] = calibration.max_resid_ms
    else:
        sync_map = None
        meta["sync_method"] = "filename_UTC+3_fallback"

    meta["t_start_unix"] = t_start_unix_fallback
    meta["t_start_local"] = main_csv.name[:19]

    signals = es._load_raw_csv(main_csv, t_start_unix_fallback, sync_map)

    for prefix, (x, y, z) in {
        "ACC": ("ACC_x", "ACC_y", "ACC_z"),
        "GYRO": ("GYRO_x", "GYRO_y", "GYRO_z"),
        "MAG": ("MAG_x", "MAG_y", "MAG_z"),
    }.items():
        signals = es._merge_xyz_signals(signals, x, y, z, prefix)

    meta["signals"] = {name: len(df) for name, df in signals.items()}
    if signals:
        all_starts = [df["unix_time"].iloc[0] for df in signals.values() if not df.empty]
        all_ends = [df["unix_time"].iloc[-1] for df in signals.values() if not df.empty]
        if all_starts:
            meta["emotibit_start_unix"] = float(min(all_starts))
            meta["emotibit_end_unix"] = float(max(all_ends))

    return signals, meta
