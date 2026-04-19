"""
cleaning.py
===========
Phase 2 – Data Cleaning
- Replaces '?' sentinel values with NaN
- Drops columns with excessive missing data
- Removes duplicate patient records (keeping first encounter per patient)
- Removes invalid gender entries
- Fills remaining missing values with sensible defaults
- Detects and caps outliers using IQR

Usage:
    python data_scripts/cleaning.py

Input  : data/raw/diabetic_data.csv
Output : data/processed/diabetic_cleaned.csv
"""

import os
import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH   = os.path.join(PROJECT_ROOT, "data", "raw", "diabetic_data.csv")
OUTPUT_PATH  = os.path.join(PROJECT_ROOT, "data", "processed", "diabetic_cleaned.csv")


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.replace("?", np.nan, inplace=True)
    print(f"[INFO] Loaded {df.shape[0]:,} rows × {df.shape[1]} columns from {path}")
    return df


def report_missing(df: pd.DataFrame):
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
    print("\n[INFO] Missing value report (%):")
    print(missing_pct.to_string() if not missing_pct.empty else "  None")
    print()


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns with high missingness or low clinical value.
      - weight          : 97% missing
      - payer_code      : 40% missing, not a clinical predictor
      - medical_specialty: 50% missing
      - encounter_id    : row identifier, not a feature
    Note: patient_nbr is kept here and removed AFTER deduplication.
    """
    cols = ["weight", "payer_code", "medical_specialty", "encounter_id"]
    existing = [c for c in cols if c in df.columns]
    df.drop(columns=existing, inplace=True)
    print(f"[INFO] Dropped columns: {existing}")
    return df


def remove_duplicate_patients(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the FIRST encounter per patient.
    Simply calling drop_duplicates() on all columns would keep multiple
    visits for the same patient, causing data leakage in model evaluation.
    """
    before = len(df)
    df.drop_duplicates(subset="patient_nbr", keep="first", inplace=True)
    after = len(df)
    # Now safe to drop the patient ID column
    df.drop(columns=["patient_nbr"], inplace=True)
    print(f"[INFO] Removed {before - after:,} duplicate patient records. "
          f"Unique patients: {after:,}")
    return df


def remove_invalid_gender(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["gender"] != "Unknown/Invalid"].copy()
    print(f"[INFO] Removed {before - len(df)} rows with invalid gender.")
    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill remaining missing values with context-appropriate defaults."""
    # Race: treat unknown as a separate category
    df["race"] = df["race"].fillna("Unknown")

    # Diagnosis codes: '0' means no secondary/tertiary diagnosis recorded
    for col in ["diag_1", "diag_2", "diag_3"]:
        df[col] = df[col].fillna("0")

    # Lab test results: 'None' means the test was not performed
    df["max_glu_serum"] = df["max_glu_serum"].fillna("None")
    df["A1Cresult"]     = df["A1Cresult"].fillna("None")

    print("[INFO] Missing values filled.")
    return df


def cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Cap outliers in clinical numeric columns using the IQR method."""
    numeric_cols = [
        "time_in_hospital", "num_lab_procedures", "num_procedures",
        "num_medications", "number_diagnoses",
        "number_outpatient", "number_emergency", "number_inpatient",
    ]
    print("[INFO] Capping outliers (IQR method):")
    for col in numeric_cols:
        if col not in df.columns:
            continue
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR     = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        n = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        print(f"       {col:<25} {n} outliers capped  [{lower:.1f} – {upper:.1f}]")
    return df


def save_data(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\n[INFO] Saved cleaned data → {path}  ({df.shape[0]:,} rows × {df.shape[1]} cols)")


def main():
    print("=" * 55)
    print("  DATA CLEANING")
    print("=" * 55)
    df = load_dataset(INPUT_PATH)
    report_missing(df)
    df = drop_columns(df)
    df = remove_duplicate_patients(df)
    df = remove_invalid_gender(df)
    df = fill_missing_values(df)
    df = cap_outliers(df)
    save_data(df, OUTPUT_PATH)
    print("\n[DONE] Cleaning complete.")


if __name__ == "__main__":
    main()
