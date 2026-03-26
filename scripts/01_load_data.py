"""
01_load_data.py

Extracts a patient cohort from MIMIC-III and saves it as a flat CSV.
Each row is one ICU stay with first-24h vitals + mortality label.
"""

import os
import pandas as pd
import numpy as np

# Paths
MIMIC_PATH  = "/Users/rohan/Desktop/ML Projects/Healthcare-Research/data/raw"
OUTPUT_FILE = "data/cohort.csv"

# Vital ITEMIDs (CareVue + MetaVision)
VITAL_ITEMIDS = {
    "heart_rate": [211, 220045],
    "sbp":        [51, 442, 455, 6701, 220179, 220050],
    "dbp":        [8368, 8440, 8441, 8555, 220180, 220051],
    "resp_rate":  [615, 618, 220210, 224690],
    "temp_c":     [223761, 678],
    "spo2":       [646, 220277],
    "glucose":    [807, 811, 1529, 3745, 3744, 225664, 220621, 226537],
}

ALL_ITEMIDS = [iid for ids in VITAL_ITEMIDS.values() for iid in ids]

ITEMID_TO_VITAL = {
    iid: vital
    for vital, ids in VITAL_ITEMIDS.items()
    for iid in ids
}

os.makedirs("data", exist_ok=True)


# Load ICU stays (first stay per admission)
print("Loading ICUSTAYS...")

icustays = pd.read_csv(
    os.path.join(MIMIC_PATH, "ICUSTAYS.csv"),
    usecols=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME"],
    parse_dates=["INTIME", "OUTTIME"]
)

icustays = (
    icustays
    .sort_values("INTIME")
    .groupby("HADM_ID")
    .first()
    .reset_index()
)

print(f"  {len(icustays):,} unique ICU stays")


# Add mortality label
print("Loading ADMISSIONS...")

admissions = pd.read_csv(
    os.path.join(MIMIC_PATH, "ADMISSIONS.csv"),
    usecols=["HADM_ID", "HOSPITAL_EXPIRE_FLAG"]
)

icustays = icustays.merge(admissions, on="HADM_ID", how="left")
icustays = icustays.rename(columns={"HOSPITAL_EXPIRE_FLAG": "died"})

print(f"  Mortality rate: {icustays['died'].mean():.1%}")


# Extract first-24h vitals
print("Reading CHARTEVENTS in chunks...")

valid_icustay_ids = set(icustays["ICUSTAY_ID"])
intime_lookup    = icustays.set_index("ICUSTAY_ID")["INTIME"].to_dict()
all_itemid_set   = set(ALL_ITEMIDS)

chunks = []
rows_read = 0

for chunk in pd.read_csv(
    os.path.join(MIMIC_PATH, "CHARTEVENTS.csv"),
    usecols=["ICUSTAY_ID", "ITEMID", "CHARTTIME", "VALUENUM"],
    parse_dates=["CHARTTIME"],
    chunksize=1_000_000,
    low_memory=False
):
    rows_read += len(chunk)

    chunk = chunk[
        chunk["ICUSTAY_ID"].isin(valid_icustay_ids) &
        chunk["ITEMID"].isin(all_itemid_set)
    ].copy()

    chunk = chunk.dropna(subset=["VALUENUM", "ICUSTAY_ID"])

    chunk["INTIME"] = chunk["ICUSTAY_ID"].map(intime_lookup)
    chunk["hours_since_admit"] = (
        (chunk["CHARTTIME"] - chunk["INTIME"]).dt.total_seconds() / 3600
    )

    chunk = chunk[
        (chunk["hours_since_admit"] >= 0) &
        (chunk["hours_since_admit"] <= 24)
    ]

    if len(chunk) > 0:
        chunks.append(chunk)

    if rows_read % 5_000_000 == 0:
        print(f"  {rows_read:,} rows processed...")

print(f"  Done — {rows_read:,} total rows scanned")

chartevents = pd.concat(chunks, ignore_index=True)
chartevents["vital"] = chartevents["ITEMID"].map(ITEMID_TO_VITAL)


# Aggregate to one row per ICU stay
print("Aggregating vitals...")

vitals = (
    chartevents
    .groupby(["ICUSTAY_ID", "vital"])["VALUENUM"]
    .mean()
    .unstack(level="vital")
    .reset_index()
)


# Save cohort
FEATURE_COLS = list(VITAL_ITEMIDS.keys())

cohort = icustays.merge(vitals, on="ICUSTAY_ID", how="left")
cohort = cohort.dropna(subset=FEATURE_COLS, how="all")
cohort = cohort[["ICUSTAY_ID", "HADM_ID", "SUBJECT_ID", "died"] + FEATURE_COLS]

cohort.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved {len(cohort):,} patients → {OUTPUT_FILE}")
print(f"Mortality rate: {cohort['died'].mean():.1%}")

print("\nMissing values per feature:")
for col in FEATURE_COLS:
    pct_missing = cohort[col].isna().mean()
    print(f"  {col:<15} {pct_missing:.1%} missing")