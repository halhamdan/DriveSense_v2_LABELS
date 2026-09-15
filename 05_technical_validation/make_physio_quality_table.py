"""
Generates Supplementary Table S2: per-session physiological signal quality.

Reads validation_output/physiological_quality_per_session.csv (written by
audit_physiological_quality_summary.py -- run that first) and writes a
longtable to validation_output/si_physio_quality_table.tex, one row per valid
session, plus pooled dataset-wide figures for the main text.

Screening rules reported in the table (the ones stated in the manuscript):
  * flatline    : run of >= 10 consecutive identical 25 Hz samples (0.4 s)
  * EDA usable  : flatline fraction < 30 %
  * HR/IBI usable: dropout-flag fraction < 30 %
Rows failing either screen are set in bold.

Usage:
  python make_physio_quality_table.py
"""
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
SRC = BASE / "validation_output" / "physiological_quality_per_session.csv"
OUT = BASE / "validation_output" / "si_physio_quality_table.tex"

FLAT_SCREEN = 30.0
DROPOUT_SCREEN = 30.0


def esc(x):
    return str(x).replace("_", "\\_")


def ck(b):
    return r"\checkmark" if b else r"$\times$"


long = pd.read_csv(SRC)
long["driver"] = long["tag"].str.extract(r"D(\d+)_").astype(int)
long["session"] = long["tag"].str.extract(r"_S(\d+)").astype(int)


def col(channel, field):
    return long[long["channel"] == channel].set_index("tag")[field]


tbl = pd.DataFrame({
    "eda_oor": col("EDA", "oor_pct"),
    "eda_flat": col("EDA", "pct_in_flatline"),
    "scr_flat": col("SCR_FREQ", "pct_in_flatline"),
    "hr_drop": col("IBI", "missing_pct"),          # HR and IBI share the same dropout pattern
    "ibi_oor": col("IBI", "oor_pct"),
    "hr_gap": col("IBI", "longest_gap_s"),
})
tbl["driver"] = tbl.index.str.extract(r"D(\d+)_")[0].astype(int).values
tbl["session"] = tbl.index.str.extract(r"_S(\d+)")[0].astype(int).values
tbl = tbl.sort_values(["driver", "session"])
tbl["eda_ok"] = tbl["eda_flat"] < FLAT_SCREEN
tbl["scr_ok"] = tbl["scr_flat"] < FLAT_SCREEN
tbl["hr_ok"] = tbl["hr_drop"] < DROPOUT_SCREEN
# Availability of the released IBI channel: rows that are both present (not dropout-flagged)
# and inside the plausible 300-1500 ms range, as a fraction of all rows and in minutes.
import json, os  # noqa: E402
_root = os.environ.get("DATASET_ROOT", "") or r"C:\Users\halha\OneDrive - Durham University\Documents\DriveSense_Packages\2_PREVIOUS__labels_v1__release_v1__paper_v5\data\Published_Dataset_Final"
_meta = {s["tag"]: s for s in json.load(open(os.path.join(_root, "dataset_metadata.json"), encoding="utf-8"))["sessions"]}
tbl["dur_min"] = [(_meta[t]["drive_duration_s"] / 60.0) if t in _meta else float("nan") for t in tbl.index]
tbl["ibi_avail_pct"] = (1 - tbl["hr_drop"] / 100) * (1 - tbl["ibi_oor"] / 100) * 100
tbl["ibi_avail_min"] = tbl["ibi_avail_pct"] / 100 * tbl["dur_min"]

hdr = (r"Session & EDA OOR & EDA flat & SCR flat & HR/IBI drop & IBI OOR & IBI avail.\ (\%) & IBI avail.\ (min) & Gap (s) & EDA flat$<$30 & SCR flat$<$30 & HR/IBI drop$<$30 \\")

lines = [
    r"\footnotesize",
    r"\setlength{\tabcolsep}{3.5pt}",
    r"\begin{longtable}{@{}lrrrrrrrrccc@{}}",
    r"\caption{Per-session physiological signal characteristics and the three descriptive screens. Columns: EDA OOR = EDA out-of-range (\%); EDA flat / SCR flat "
    r"= constant-value fraction (\%) of EDA / \texttt{SCR\_FREQ}; HR/IBI drop = dropout-flagged rows (\%); IBI OOR = IBI "
    r"out-of-range (\%) among present rows; IBI avail.\ = rows that are both present and within 300--1500\,ms, as a percentage of "
    r"all rows and in minutes of the session -- the availability of the released 25\,Hz IBI channel, not a statement of its "
    r"suitability for beat-to-beat heart-rate-variability analysis, for which the native per-beat file in \texttt{Raw\_Dataset} should be used; "
    r"Gap = longest HR/IBI dropout (s). The last three columns state which descriptive screen a session passes, named by the quantity "
    r"tested: EDA constant-value fraction below 30\%, \texttt{SCR\_FREQ} constant-value fraction below 30\%, HR/IBI dropout fraction below 30\%. "
    r"Passing a screen certifies only that quantity; it does not certify general signal usability (e.g.\ D11\_S3 passes the dropout screen with "
    r"an IBI availability of about 40\%). Rows are set in bold if they fail the EDA or HR/IBI screen. The table covers the four biometric channels that show any non-trivial "
    r"quality issue. All other biometric channels -- temperature, PPG, accelerometer, gyroscope, magnetometer -- have "
    r"0\% missing and 0\% out-of-range samples in every session and are omitted; the magnetometer axes do exceed the "
    r"constant-value threshold in most sessions because their readings are coarsely quantised and slowly varying, "
    r"so this metric is not informative for them. The event-based "
    r"\texttt{SCR\_AMPLITUDE}/\texttt{SCR\_RISE\_TIME} channels carry a value on 12.03\% of rows by construction (Technical "
    r"Validation) and are likewise omitted. Out-of-range uses the plausible "
    r"ranges published in \texttt{data\_schema.json} (EDA 0--50\,\textmu S; IBI 300--1500\,ms, equivalent to HR "
    r"40--200\,bpm) and is computed over present (non-dropout) samples. The constant-value fraction is the fraction of present samples "
    r"lying in a run of at least 10 consecutive identical 25\,Hz samples (0.4\,s); it is a descriptive flag -- electrode contact loss, "
    r"sensor quantisation and the device's value-reporting behaviour are all possible causes and were not separated. HR/IBI dropout is the fraction of "
    r"rows carrying \texttt{IBI\_dropout\_flag} (identical for HR); the longest gap is the longest such dropout in "
    r"seconds. All figures are produced by a single released script "
    r"(\texttt{05\_technical\_validation/audit\_physiological\_quality\_summary.py}).}",
    r"\label{tab:si-physio-quality}\\",
    r"\toprule", hdr, r"\midrule", r"\endfirsthead",
    r"\multicolumn{12}{c}{\tablename\ \thetable{} -- continued}\\",
    r"\toprule", hdr, r"\midrule", r"\endhead",
    r"\bottomrule", r"\endfoot",
]

for tag, r in tbl.iterrows():
    cells = [esc(tag), f"{r.eda_oor:.1f}", f"{r.eda_flat:.1f}", f"{r.scr_flat:.1f}",
             f"{r.hr_drop:.1f}", f"{r.ibi_oor:.1f}", f"{r.ibi_avail_pct:.0f}", f"{r.ibi_avail_min:.0f}", f"{r.hr_gap:.1f}", ck(r.eda_ok), ck(r.scr_ok), ck(r.hr_ok)]
    if not (r.eda_ok and r.hr_ok):
        cells = [rf"\textbf{{{c}}}" for c in cells]
    lines.append(" & ".join(cells) + r" \\")

lines.append(r"\end{longtable}")
OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {len(tbl)} rows to {OUT}")

print("\n=== Dataset-wide summary (for main text) ===")
print(f"Sessions passing EDA screen (flatline < {FLAT_SCREEN:.0f}%): {tbl.eda_ok.sum()} "
      f"(failing: drivers {sorted(tbl[~tbl.eda_ok].driver.unique())})")
print(f"Sessions passing HR/IBI screen (dropout < {DROPOUT_SCREEN:.0f}%): {tbl.hr_ok.sum()} "
      f"(failing: drivers {sorted(tbl[~tbl.hr_ok].driver.unique())})")
print(f"Sessions passing SCR_FREQ screen (flatline < {FLAT_SCREEN:.0f}%): {tbl.scr_ok.sum()}")
print(f"Sessions passing EDA and HR/IBI screens: {(tbl.eda_ok & tbl.hr_ok).sum()} / {len(tbl)}")
print(f"Sessions passing all three screens: {(tbl.eda_ok & tbl.hr_ok & tbl.scr_ok).sum()} / {len(tbl)} "
      f"(pass EDA+HR/IBI but fail SCR: {sorted(tbl[tbl.eda_ok & tbl.hr_ok & ~tbl.scr_ok].index)})")
print(f"Longest HR/IBI gap: {tbl.hr_gap.max():.1f} s ({tbl.hr_gap.idxmax()}); sessions with longest gap > 40 s: "
      f"{(tbl.hr_gap > 40).sum()}; median longest gap {tbl.hr_gap.median():.1f} s")
print(f"IBI availability (present & in range): median {tbl.ibi_avail_pct.median():.0f}% (range {tbl.ibi_avail_pct.min():.0f}-{tbl.ibi_avail_pct.max():.0f}%), "
      f"median {tbl.ibi_avail_min.median():.0f} min; sessions < 50%: {(tbl.ibi_avail_pct < 50).sum()} ({sorted(tbl[tbl.ibi_avail_pct < 50].index)}); "
      f"D11_S3: {tbl.loc['D11_S3','ibi_avail_pct']:.0f}% / hr_ok={tbl.loc['D11_S3','hr_ok']}")
tbl.round(2).to_csv(BASE / "validation_output" / "si_physio_quality_table.csv")
