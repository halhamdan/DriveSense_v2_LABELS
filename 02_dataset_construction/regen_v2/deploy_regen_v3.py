"""
Deploys the anchor-only regeneration (regenerated_v3) into the release.

Steps, in order, aborting on the first failure:
  1. Verify every one of the 79 regenerated files against the current release:
     same row count, same column set, and every NON-biometric column
     (driver/session/unix_time/elapsed_s/all VBOX/GPS columns/label) byte-
     identical -- the regeneration only recomputes the EmotiBit-derived
     columns, so anything else differing means something is wrong.
  2. Back up the current release's 79 fused.csv files (flat layout, same
     convention as the earlier *_BACKUP folders) to
     Preprocessed_Dataset_PRE_ANCHOR_ONLY_FIX_BACKUP/.
  3. Copy the regenerated files over the release files.
  4. Re-hash: update size_bytes / rows / sha256 for those 79 rows in
     manifest.csv (all other manifest rows untouched).
  5. Append a changelog entry to data_schema.json and bump "generated".

Usage:
    python deploy_regen_v3.py            # dry run: steps 1 only + report
    python deploy_regen_v3.py --apply    # steps 1-5

Environment variables:
    DATASET_ROOT   release root (default: the OneDrive Published_Dataset_Final)
    REGEN_OUT_ROOT regenerated tree root (default: ../../../regenerated_v3)
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", "") or
                    r"C:\Users\halha\OneDrive - Durham University\Documents\Published_Dataset_Final")
REGEN_ROOT = Path(os.environ.get("REGEN_OUT_ROOT", "") or (Path(__file__).resolve().parents[3] / "regenerated_v3"))
BACKUP_DIR = DATASET_ROOT / "Preprocessed_Dataset_PRE_ANCHOR_ONLY_FIX_BACKUP"
REPORT = REGEN_ROOT / "regeneration_report.csv"

BIO_PREFIXES = ["EDA", "TEMP_CONTACT", "TEMP_THERMOPILE", "PPG_IR", "PPG_RED", "PPG_GREEN",
                "SCR_FREQ", "SCR_AMPLITUDE", "SCR_RISE_TIME", "IBI", "HR", "ACC", "GYRO", "MAG"]


def is_bio(col: str) -> bool:
    return any(col == p or col.startswith(p + "_") for p in BIO_PREFIXES)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify() -> list[tuple[str, Path, Path]]:
    rep = pd.read_csv(REPORT)
    bad = rep[rep["status"] != "ok"]
    if len(bad):
        raise SystemExit(f"regeneration report has {len(bad)} non-ok sessions:\n{bad[['tag','status']]}")
    if len(rep) != 79:
        raise SystemExit(f"expected 79 sessions in report, found {len(rep)}")
    if set(rep["sync_method"]) != {"timesyncs_anchor_only_median"}:
        raise SystemExit(f"unexpected sync methods: {rep['sync_method'].value_counts().to_dict()}")

    pairs = []
    for _, r in rep.iterrows():
        tag = r["tag"]
        d, s = tag[1:].split("_S")
        rel = DATASET_ROOT / "Preprocessed_Dataset" / f"D{d}" / f"Session_{s}" / f"{tag}_fused.csv"
        new = Path(r["out_path"])
        a = pd.read_csv(rel)
        b = pd.read_csv(new)
        if a.shape[0] != b.shape[0]:
            raise SystemExit(f"{tag}: row count differs {a.shape[0]} vs {b.shape[0]}")
        if list(a.columns) != list(b.columns):
            raise SystemExit(f"{tag}: column set differs\n  release={list(a.columns)}\n  new={list(b.columns)}")
        for c in a.columns:
            if is_bio(c):
                continue
            av, bv = a[c].to_numpy(), b[c].to_numpy()
            if av.dtype.kind in "fc" or bv.dtype.kind in "fc":
                ok = np.allclose(av.astype(float), bv.astype(float), equal_nan=True, rtol=0, atol=0)
            else:
                ok = (av == bv).all()
            if not ok:
                raise SystemExit(f"{tag}: non-biometric column {c} differs -- aborting")
        n_bio_changed = sum(
            1 for c in a.columns if is_bio(c) and not np.allclose(
                a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float), equal_nan=True))
        pairs.append((tag, rel, new))
        print(f"  {tag}: OK  ({n_bio_changed} biometric columns changed)")
    return pairs


def apply(pairs):
    BACKUP_DIR.mkdir(exist_ok=True)
    for tag, rel, _ in pairs:
        dst = BACKUP_DIR / rel.name
        if not dst.exists():
            shutil.copy2(rel, dst)
    print(f"backed up {len(pairs)} files to {BACKUP_DIR}")

    for tag, rel, new in pairs:
        shutil.copy2(new, rel)
    print(f"copied {len(pairs)} regenerated files into the release")

    man_path = DATASET_ROOT / "manifest.csv"
    man = pd.read_csv(man_path)
    updated = 0
    for tag, rel, _ in pairs:
        relpath = str(rel.relative_to(DATASET_ROOT))
        mask = man["relative_path"].str.replace("/", "\\") == relpath.replace("/", "\\")
        if mask.sum() != 1:
            raise SystemExit(f"{tag}: expected exactly one manifest row for {relpath}, found {mask.sum()}")
        n_rows = sum(1 for _ in open(rel, "rb")) - 1
        man.loc[mask, "size_bytes"] = rel.stat().st_size
        man.loc[mask, "rows"] = float(n_rows)
        man.loc[mask, "sha256"] = sha256(rel)
        updated += 1
    man.to_csv(man_path, index=False)
    print(f"manifest.csv: updated {updated} rows")

    schema_path = DATASET_ROOT / "data_schema.json"
    schema = json.load(open(schema_path, encoding="utf-8"))
    schema["changelog"].append({
        "date": date.today().isoformat(),
        "change": (
            "EmotiBit-to-VBOX time alignment model changed from a fitted linear calibration "
            "(anchor + clock-rate slope, 2026-09-09 morning) to an anchor-only calibration "
            "(slope fixed at exactly 1, session-median offset over all logged sync pings). "
            "Reason: the sync-ping logs show the device clock is stable to a few ppm within a "
            "session (median true drift ~4 ppm), but 12 of 79 sessions contain one or more "
            "discrete offset steps of 0.15-1.0 s (most plausibly host-clock adjustments on the "
            "logging laptop). A fitted slope cannot represent a step: it tilts through it, "
            "reports a spurious clock-rate error of up to several hundred ppm (D4_S1 +552, D4_S4 "
            "+295, D13_S1 -344 ppm), and smears the step across the whole session, misplacing the "
            "session end by up to ~1 s. The session-median offset is insensitive to such steps; "
            "the largest sustained step per session is the honest bound on residual cross-device "
            "alignment error and is released per session in the regeneration report. Only "
            "EmotiBit-derived columns changed; unix_time, elapsed_s, all VBOX/GPS columns and "
            "label are byte-identical. Previous files preserved at "
            "Preprocessed_Dataset_PRE_ANCHOR_ONLY_FIX_BACKUP/."
        ),
    })
    schema["generated"] = date.today().isoformat()
    json.dump(schema, open(schema_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("data_schema.json: changelog entry appended")

    # Keep the repo's read-only mirror of the deposit's descriptive files in step
    # (release_snapshot/README.md explains the mirror; hashes there must be
    # refreshed by hand or by re-running its generation snippet after this).
    snap = Path(__file__).resolve().parents[2] / "release_snapshot"
    if snap.exists():
        for name in ("data_schema.json", "dataset_metadata.json", "manifest.csv"):
            shutil.copy2(DATASET_ROOT / name, snap / name)
        print(f"release_snapshot/: refreshed 3 files (update README.md hashes before tagging)")


def main():
    print(f"release: {DATASET_ROOT}\nregen:   {REGEN_ROOT}\n")
    pairs = verify()
    print(f"\nverification passed for {len(pairs)} sessions")
    if "--apply" in sys.argv:
        apply(pairs)
        print("\nDEPLOYED.")
    else:
        print("\nDry run only. Re-run with --apply to deploy.")


if __name__ == "__main__":
    main()
