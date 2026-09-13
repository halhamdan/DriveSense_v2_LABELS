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

hdr = (r"Session & EDA OOR & EDA flat & SCR flat & HR/IBI drop & IBI OOR & Gap (s) & EDA ok & SCR ok & HR ok \\")

lines = [
    r"\footnotesize",
    r"\setlength{\tabcolsep}{3.5pt}",
    r"\begin{longtable}{@{}lrrrrrrccc@{}}",
    r"\caption{Per-session physiological signal quality. Columns: EDA OOR = EDA out-of-range (\%); EDA flat / SCR flat "
    r"= flatline fraction (\%) of EDA / \texttt{SCR\_FREQ}; HR/IBI drop = dropout-flagged rows (\%); IBI OOR = IBI "
    r"out-of-range (\%) among present rows; Gap = longest HR/IBI dropout (s); EDA ok / SCR ok / HR ok = passes the "
    r"corresponding session-level screen (flatline $<$\,30\% for EDA and \texttt{SCR\_FREQ}; dropout $<$\,30\% for "
    r"HR/IBI). Rows are set in bold if they fail the EDA or HR/IBI screen. The table covers the four biometric channels that show any non-trivial "
    r"quality issue. All other biometric channels -- temperature, PPG, accelerometer, gyroscope, magnetometer -- have "
    r"0\% missing and 0\% out-of-range samples in every session and are omitted; the magnetometer axes do exceed the "
    r"flatline threshold in most sessions, but because their readings are coarsely quantised and slowly varying rather "
    r"than through contact loss, so this metric is not informative for them. The event-triggered "
    r"\texttt{SCR\_AMPLITUDE}/\texttt{SCR\_RISE\_TIME} channels are present on 12.02\% of rows by construction (Technical "
    r"Validation) and are likewise omitted. Out-of-range uses the plausible "
    r"ranges published in \texttt{data\_schema.json} (EDA 0--50\,\textmu S; IBI 300--1500\,ms, equivalent to HR "
    r"40--200\,bpm) and is computed over present (non-dropout) samples. Flatline is the fraction of present samples "
    r"lying in a run of at least 10 consecutive identical 25\,Hz samples (0.4\,s). HR/IBI dropout is the fraction of "
    r"rows carrying \texttt{IBI\_dropout\_flag} (identical for HR); the longest gap is the longest such dropout in "
    r"seconds. The two screening columns apply the session-level rules suggested in Technical Validation: EDA/SCR "
    r"channels usable where flatline $<$\,30\%, HR/IBI usable where dropout $<$\,30\%; rows failing either screen are "
    r"set in bold. All figures are produced by a single released script "
    r"(\texttt{05\_technical\_validation/audit\_physiological\_quality\_summary.py}).}",
    r"\label{tab:si-physio-quality}\\",
    r"\toprule", hdr, r"\midrule", r"\endfirsthead",
    r"\multicolumn{9}{c}{\tablename\ \thetable{} -- continued}\\",
    r"\toprule", hdr, r"\midrule", r"\endhead",
    r"\bottomrule", r"\endfoot",
]

for tag, r in tbl.iterrows():
    cells = [esc(tag), f"{r.eda_oor:.1f}", f"{r.eda_flat:.1f}", f"{r.scr_flat:.1f}",
             f"{r.hr_drop:.1f}", f"{r.ibi_oor:.1f}", f"{r.hr_gap:.1f}", ck(r.eda_ok), ck(r.scr_ok), ck(r.hr_ok)]
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
