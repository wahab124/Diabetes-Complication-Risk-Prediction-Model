"""
feature_engineering.py
======================
Phase 2 – Feature Engineering
- Derives three binary complication labels from ICD-9 diagnosis codes
- Creates a longitudinal visit-count feature
- Drops raw diagnosis columns and readmitted after label creation

ICD-9 Ranges Used:
    Kidney Disease   : 580 – 589  (nephritis, nephrosis, renal failure)
    Neuropathy       : 354 – 357  (peripheral neuropathy group)
    Cardiovascular   : 390 – 459  (circulatory system diseases)

Usage:
    python data_scripts/feature_engineering.py

Input  : data/processed/diabetic_preprocessed.csv
Output : data/processed/diabetic_final.csv
"""

import os
import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH   = os.path.join(PROJECT_ROOT, "data", "processed", "diabetic_preprocessed.csv")
OUTPUT_PATH  = os.path.join(PROJECT_ROOT, "data", "processed", "diabetic_final.csv")


def safe_float(value) -> float | None:
    """Convert a value to float; return None on failure (handles V/E codes)."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def classify_complications(row) -> pd.Series:
    """
    Return binary flags for each complication based on ICD-9 codes in
    diag_1, diag_2, diag_3. A single matching code in any column is sufficient.
    """
    diagnoses = [row["diag_1"], row["diag_2"], row["diag_3"]]

    cardiovascular = 0
    kidney         = 0
    neuropathy     = 0

    for diag in diagnoses:
        val = safe_float(diag)
        if val is None:
            continue

        if 390 <= val <= 459:   # Cardiovascular / circulatory
            cardiovascular = 1

        if 580 <= val <= 589:   # Kidney disease (corrected: was 580-629)
            kidney = 1

        if 354 <= val <= 357:   # Peripheral neuropathy group (corrected: was just 357)
            neuropathy = 1

    return pd.Series([cardiovascular, kidney, neuropathy])


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    df[["cardiovascular_complication",
        "kidney_complication",
        "neuropathy_complication"]] = df.apply(classify_complications, axis=1)

    labels = ["cardiovascular_complication", "kidney_complication", "neuropathy_complication"]
    print("[INFO] Complication labels created:")
    for col in labels:
        n   = df[col].sum()
        pct = df[col].mean() * 100
        print(f"       {col:<30} {n:>6,} positive ({pct:.1f}%)")
    return df


def create_longitudinal_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Total prior healthcare visits is a proxy for disease chronicity.
    Aggregates outpatient + emergency + inpatient visit counts.
    Note: these columns have already been scaled; the sum still preserves
    relative differences and is interpretable as a composite utilisation score.
    """
    visit_cols = ["number_outpatient", "number_emergency", "number_inpatient"]
    present    = [c for c in visit_cols if c in df.columns]
    df["total_prior_visits"] = df[present].sum(axis=1)
    print(f"[INFO] Longitudinal feature 'total_prior_visits' created from: {present}")
    return df


def drop_source_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop raw diagnosis codes now that labels have been derived."""
    to_drop = ["diag_1", "diag_2", "diag_3"]
    existing = [c for c in to_drop if c in df.columns]
    df.drop(columns=existing, inplace=True)
    print(f"[INFO] Dropped source columns: {existing}")
    return df


def main():
    print("=" * 55)
    print("  FEATURE ENGINEERING")
    print("=" * 55)

    df = pd.read_csv(INPUT_PATH)
    print(f"[INFO] Loaded {df.shape[0]:,} rows × {df.shape[1]} columns\n")

    df = create_labels(df)
    df = create_longitudinal_feature(df)
    df = drop_source_columns(df)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[INFO] Saved final dataset → {OUTPUT_PATH}  ({df.shape[0]:,} × {df.shape[1]})")
    print("\n[DONE] Feature engineering complete.")


if __name__ == "__main__":
    main()
