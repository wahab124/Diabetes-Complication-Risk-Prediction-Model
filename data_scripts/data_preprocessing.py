"""
data_preprocessing.py
=====================
Phase 2 – Data Preprocessing
- Converts age brackets to numeric midpoints
- Encodes gender, binary, and medication columns
- One-hot encodes race, glucose serum, and A1C result
- Scales numeric features with StandardScaler
- Saves the fitted scaler for use in EDA / inference

Usage:
    python data_scripts/data_preprocessing.py

Input  : data/processed/diabetic_cleaned.csv
Output : data/processed/diabetic_preprocessed.csv
         data/processed/scaler.pkl
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH    = os.path.join(PROJECT_ROOT, "data", "processed", "diabetic_cleaned.csv")
OUTPUT_PATH   = os.path.join(PROJECT_ROOT, "data", "processed", "diabetic_preprocessed.csv")
SCALER_PATH   = os.path.join(PROJECT_ROOT, "data", "processed", "scaler.pkl")

NUMERIC_COLS  = [
    "age", "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]

MEDICATION_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]


def convert_age(age_range: str) -> float:
    """
    Convert age bracket string to numeric midpoint.
    e.g. '[10-20)' → 15.0
    Uses split on '-' after stripping all bracket characters.
    """
    cleaned = age_range.replace("[", "").replace("(", "").replace("]", "").replace(")", "")
    parts   = cleaned.split("-")
    return (int(parts[0]) + int(parts[1])) / 2


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    # ── Age ──────────────────────────────────────────────────────────────────
    df["age"] = df["age"].apply(convert_age)

    # ── Gender ───────────────────────────────────────────────────────────────
    # Rows with Unknown/Invalid were removed in cleaning; map remaining values
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

    # ── Binary clinical flags ────────────────────────────────────────────────
    df["change"]      = df["change"].map({"Ch": 1, "No": 0})
    df["diabetesMed"] = df["diabetesMed"].map({"Yes": 1, "No": 0})

    # ── Readmission: ordinal ─────────────────────────────────────────────────
    df["readmitted"] = df["readmitted"].map({"NO": 0, ">30": 1, "<30": 2}).fillna(0)

    # ── Medication columns: ordinal ──────────────────────────────────────────
    med_map = {"No": 0, "Steady": 1, "Up": 2, "Down": 3}
    for col in MEDICATION_COLS:
        if col in df.columns:
            df[col] = df[col].map(med_map).fillna(0).astype(int)

    # ── One-hot encode categorical columns ───────────────────────────────────
    categorical_cols = ["race", "max_glu_serum", "A1Cresult"]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    df = df.replace({True: 1, False: 0})

    # ── Scale numeric columns ────────────────────────────────────────────────
    numeric_present = [c for c in NUMERIC_COLS if c in df.columns]
    scaler = StandardScaler()
    df[numeric_present] = scaler.fit_transform(df[numeric_present])

    print(f"[INFO] Preprocessing complete. Shape: {df.shape}")
    return df, scaler


def main():
    print("=" * 55)
    print("  DATA PREPROCESSING")
    print("=" * 55)

    df = pd.read_csv(INPUT_PATH)
    print(f"[INFO] Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

    df, scaler = preprocess(df)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[INFO] Saved preprocessed data → {OUTPUT_PATH}")

    # Save the scaler so EDA notebook can inverse-transform for readable plots
    with open(SCALER_PATH, "wb") as f:
        pickle.dump({"scaler": scaler, "numeric_cols": NUMERIC_COLS}, f)
    print(f"[INFO] Saved scaler → {SCALER_PATH}")

    print("\n[DONE] Preprocessing complete.")


if __name__ == "__main__":
    main()
